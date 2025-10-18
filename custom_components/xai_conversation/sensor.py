"""Token usage sensors for xAI Conversation integration."""
from __future__ import annotations

import datetime
import json
from typing import TYPE_CHECKING

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.util import dt as dt_util

from .const import (
    CONF_CHAT_MODEL,
    DEFAULT_MANUFACTURER,
    DEFAULT_MODEL_PRICING,
    DEFAULT_SENSORS_NAME,
    DOMAIN,
    LOGGER,
    RECOMMENDED_CHAT_MODEL,
)

if TYPE_CHECKING:
    from . import XAIConfigEntry
    from .entity import XAIBaseLLMEntity


async def async_setup_entry(
    hass: HomeAssistant,
    entry: XAIConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up xAI token sensors for the sensors subentry."""
    # Iterate through subentries to find the sensors subentry
    for subentry in entry.subentries.values():
        if subentry.subentry_type != "sensors":
            continue

        # Create 10 sensors: 6 per-service + 4 aggregated
        sensors = [
            # Per-service sensors (6)
            XAIServiceLastTokensSensor(entry, subentry, "conversation"),
            XAIServiceCacheRatioSensor(entry, subentry, "conversation"),
            XAIServiceLastTokensSensor(entry, subentry, "ai_task"),
            XAIServiceCacheRatioSensor(entry, subentry, "ai_task"),
            XAIServiceLastTokensSensor(entry, subentry, "code_fast"),
            XAIServiceCacheRatioSensor(entry, subentry, "code_fast"),
            # Aggregated sensors (4)
            XAITotalTokensSensor(entry, subentry),
            XAIAvgTokensSensor(entry, subentry),
            XAICostSensor(entry, subentry),
            XAIResetTimestampSensor(entry, subentry),
        ]

        async_add_entities(sensors, config_subentry_id=subentry.subentry_id)

        # Store sensor references in hass.data so entities can access them
        if DOMAIN not in hass.data:
            hass.data[DOMAIN] = {}
        hass.data[DOMAIN][f"{entry.entry_id}_sensors"] = sensors

        LOGGER.debug("sensor: created %d token sensors for sensors subentry %s", len(sensors), subentry.subentry_id)


class XAITokenSensorBase(RestoreEntity, SensorEntity):
    """Base class for xAI token sensors with persistent storage."""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, entry: XAIConfigEntry, subentry) -> None:
        """Initialize the sensor."""
        self._entry = entry
        self._subentry = subentry
        # Single device for all sensors
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=DEFAULT_SENSORS_NAME,
            manufacturer=DEFAULT_MANUFACTURER,
            model="Token Tracking",
            entry_type=dr.DeviceEntryType.SERVICE,
        )
        # Storage for cumulative data per model (persists across restarts)
        self._tokens_by_model = {}
        # Global counters
        self._cumulative_completion_tokens = 0
        self._cumulative_prompt_tokens = 0
        self._cumulative_cached_tokens = 0
        self._cumulative_reasoning_tokens = 0
        self._message_count = 0
        # Last message data
        self._last_completion_tokens = 0
        self._last_prompt_tokens = 0
        self._last_cached_tokens = 0
        self._last_reasoning_tokens = 0
        self._last_model = None
        self._last_timestamp = None
        # Reset timestamp
        self._reset_timestamp: datetime.datetime | None = None

    async def async_added_to_hass(self) -> None:
        """Restore state when entity is added."""
        await super().async_added_to_hass()

        last_state = await self.async_get_last_state()
        if last_state and last_state.attributes:
            attrs = last_state.attributes

            # Restore per-model token data
            tokens_by_model_str = attrs.get("tokens_by_model")
            if tokens_by_model_str:
                try:
                    self._tokens_by_model = json.loads(tokens_by_model_str) if isinstance(tokens_by_model_str, str) else tokens_by_model_str
                except (ValueError, TypeError):
                    self._tokens_by_model = {}

            # Restore global counters
            self._cumulative_completion_tokens = attrs.get("cumulative_completion_tokens", 0)
            self._cumulative_prompt_tokens = attrs.get("cumulative_prompt_tokens", 0)
            self._cumulative_cached_tokens = attrs.get("cumulative_cached_tokens", 0)
            self._cumulative_reasoning_tokens = attrs.get("cumulative_reasoning_tokens", 0)
            self._message_count = attrs.get("message_count", 0)

            # Restore last message data
            self._last_completion_tokens = attrs.get("last_completion_tokens", 0)
            self._last_prompt_tokens = attrs.get("last_prompt_tokens", 0)
            self._last_cached_tokens = attrs.get("last_cached_tokens", 0)
            self._last_reasoning_tokens = attrs.get("last_reasoning_tokens", 0)
            self._last_model = attrs.get("last_model")

            last_ts_str = attrs.get("last_timestamp")
            if last_ts_str:
                try:
                    self._last_timestamp = dt_util.parse_datetime(last_ts_str)
                except (ValueError, TypeError):
                    self._last_timestamp = None

            # Restore reset timestamp
            reset_ts_str = attrs.get("reset_timestamp")
            if reset_ts_str:
                try:
                    self._reset_timestamp = dt_util.parse_datetime(reset_ts_str)
                except (ValueError, TypeError):
                    self._reset_timestamp = None

            LOGGER.debug("sensor: restored %s data: messages=%d total_tokens=%d models=%s",
                        self.name, self._message_count,
                        self._cumulative_completion_tokens + self._cumulative_prompt_tokens,
                        list(self._tokens_by_model.keys()))

    @callback
    def update_token_usage(
        self,
        usage,
        model: str,
        mode: str = "pipeline",
        is_fallback: bool = False,
        store_messages: bool = True
    ) -> None:
        """Update sensor with new token usage data.

        Args:
            usage: xAI response.usage object with:
                - completion_tokens
                - prompt_tokens
                - cached_prompt_text_tokens (optional)
                - reasoning_tokens (optional)
            model: The model name from xAI response (NOT config)
            mode: "pipeline" or "tools" (only used by conversation sensors)
            is_fallback: True if fallback from pipeline to tools (only used by conversation)
            store_messages: True (server-side) or False (client-side) (only used by conversation)
        """
        # Extract token counts
        completion = getattr(usage, "completion_tokens", 0) or 0
        prompt = getattr(usage, "prompt_tokens", 0) or 0
        cached = getattr(usage, "cached_prompt_text_tokens", 0) or 0
        reasoning = getattr(usage, "reasoning_tokens", 0) or 0

        # Update last message stats
        self._last_completion_tokens = completion
        self._last_prompt_tokens = prompt
        self._last_cached_tokens = cached
        self._last_reasoning_tokens = reasoning
        self._last_model = model
        self._last_timestamp = dt_util.now()

        # Initialize model entry if doesn't exist
        if model not in self._tokens_by_model:
            self._tokens_by_model[model] = {
                "completion": 0,
                "prompt": 0,
                "cached": 0,
                "reasoning": 0,
                "count": 0
            }

        # Update per-model tokens
        self._tokens_by_model[model]["completion"] += completion
        self._tokens_by_model[model]["prompt"] += prompt
        self._tokens_by_model[model]["cached"] += cached
        self._tokens_by_model[model]["reasoning"] += reasoning
        self._tokens_by_model[model]["count"] += 1

        # Update cumulative stats
        self._cumulative_completion_tokens += completion
        self._cumulative_prompt_tokens += prompt
        self._cumulative_cached_tokens += cached
        self._cumulative_reasoning_tokens += reasoning
        self._message_count += 1

        LOGGER.debug("sensor: updated %s (model=%s): completion=%d prompt=%d cached=%d reasoning=%d",
                    self.entity_id, model, completion, prompt, cached, reasoning)

        # Trigger state update
        self.async_write_ha_state()

    @callback
    def reset_statistics(self) -> None:
        """Reset cumulative token counters to zero."""
        self._cumulative_completion_tokens = 0
        self._cumulative_prompt_tokens = 0
        self._cumulative_cached_tokens = 0
        self._cumulative_reasoning_tokens = 0
        self._message_count = 0
        self._tokens_by_model = {}
        self._last_model = None
        self._last_timestamp = None

        # Store reset timestamp
        self._reset_timestamp = dt_util.now()

        LOGGER.info("sensor: reset token statistics for %s at %s",
                   self.entity_id, self._reset_timestamp)

        # Trigger state update
        self.async_write_ha_state()


# ==============================================================================
# PER-SERVICE SENSORS (6 sensors)
# ==============================================================================

class XAIServiceLastTokensSensor(XAITokenSensorBase):
    """Sensor for last message tokens per service."""

    def __init__(self, entry: XAIConfigEntry, subentry, service_type: str) -> None:
        """Initialize the sensor.

        Args:
            service_type: "conversation", "ai_task", or "code_fast"
        """
        super().__init__(entry, subentry)
        self._service_type = service_type
        self._attr_unique_id = f"{entry.entry_id}_{service_type}_last_tokens"
        self._attr_name = f"{service_type.replace('_', ' ').title()} last tokens"
        self._attr_native_unit_of_measurement = "tokens"
        self._attr_icon = "mdi:message-text"

        # For conversation: track last mode and memory type
        self._last_mode = None  # "pipeline" or "tools"
        self._last_store_messages = None  # True (server-side) or False (client-side)

    def update_token_usage(
        self,
        usage,
        model: str,
        mode: str = "pipeline",
        is_fallback: bool = False,
        store_messages: bool = True
    ) -> None:
        """Update sensor with new token usage data.

        Args:
            usage: xAI response.usage object
            model: Model name from xAI response
            mode: "pipeline" or "tools"
            is_fallback: True if fallback from pipeline to tools
            store_messages: True (server-side) or False (client-side)
        """
        # Extract token counts
        completion = getattr(usage, "completion_tokens", 0) or 0
        prompt = getattr(usage, "prompt_tokens", 0) or 0
        cached = getattr(usage, "cached_prompt_text_tokens", 0) or 0
        reasoning = getattr(usage, "reasoning_tokens", 0) or 0

        # Update last message stats
        self._last_completion_tokens = completion
        self._last_prompt_tokens = prompt
        self._last_cached_tokens = cached
        self._last_reasoning_tokens = reasoning
        self._last_model = model
        self._last_timestamp = dt_util.now()

        # For conversation: save mode and memory type
        if self._service_type == "conversation":
            # If fallback, mode becomes "tools"
            self._last_mode = "tools" if is_fallback else mode
            self._last_store_messages = store_messages

        # Initialize model entry if doesn't exist
        if model not in self._tokens_by_model:
            self._tokens_by_model[model] = {
                "completion": 0,
                "prompt": 0,
                "cached": 0,
                "reasoning": 0,
                "count": 0
            }

        # Update per-model tokens
        self._tokens_by_model[model]["completion"] += completion
        self._tokens_by_model[model]["prompt"] += prompt
        self._tokens_by_model[model]["cached"] += cached
        self._tokens_by_model[model]["reasoning"] += reasoning
        self._tokens_by_model[model]["count"] += 1

        # Update cumulative stats
        self._cumulative_completion_tokens += completion
        self._cumulative_prompt_tokens += prompt
        self._cumulative_cached_tokens += cached
        self._cumulative_reasoning_tokens += reasoning
        self._message_count += 1

        LOGGER.debug("sensor: updated %s (model=%s, mode=%s, fallback=%s, store=%s): completion=%d prompt=%d cached=%d",
                    self.entity_id, model, mode, is_fallback, store_messages, completion, prompt, cached)

        # Trigger state update
        self.async_write_ha_state()

    @property
    def native_value(self) -> int:
        """Return the last message total tokens."""
        return self._last_completion_tokens + self._last_prompt_tokens

    @property
    def extra_state_attributes(self) -> dict:
        """Return additional attributes."""
        attrs = {
            "input_tokens": self._last_prompt_tokens,
            "output_tokens": self._last_completion_tokens,
            "cached_tokens": self._last_cached_tokens,
            "reasoning_tokens": self._last_reasoning_tokens,
            "model": self._last_model,
            "service_type": self._service_type,
            "tokens_by_model": json.dumps(self._tokens_by_model),
        }

        # For conversation: expose mode and memory type
        if self._service_type == "conversation" and self._last_mode is not None:
            attrs["mode"] = self._last_mode
            attrs["memory"] = "server-side" if self._last_store_messages else "client-side"

        if self._last_timestamp:
            attrs["timestamp"] = self._last_timestamp.isoformat()
        if self._reset_timestamp:
            attrs["reset_timestamp"] = self._reset_timestamp.isoformat()
        return attrs


class XAIServiceCacheRatioSensor(XAITokenSensorBase):
    """Sensor for cache hit ratio per service."""

    def __init__(self, entry: XAIConfigEntry, subentry, service_type: str) -> None:
        """Initialize the sensor.

        Args:
            service_type: "conversation", "ai_task", or "code_fast"
        """
        super().__init__(entry, subentry)
        self._service_type = service_type
        self._attr_unique_id = f"{entry.entry_id}_{service_type}_cache_ratio"
        self._attr_name = f"{service_type.replace('_', ' ').title()} cache ratio"
        self._attr_native_unit_of_measurement = "%"
        self._attr_icon = "mdi:cached"
        self._attr_suggested_display_precision = 1

        # For conversation: internal storage for mode and memory tracking
        if service_type == "conversation":
            self._tokens_pipeline_server = {"prompt": 0, "cached": 0, "count": 0}
            self._tokens_pipeline_client = {"prompt": 0, "cached": 0, "count": 0}
            self._tokens_tools_server = {"prompt": 0, "cached": 0, "count": 0}
            self._tokens_tools_client = {"prompt": 0, "cached": 0, "count": 0}

    def update_token_usage(
        self,
        usage,
        model: str,
        mode: str = "pipeline",
        is_fallback: bool = False,
        store_messages: bool = True
    ) -> None:
        """Update sensor with new token usage data.

        Args:
            usage: xAI response.usage object
            model: Model name from xAI response
            mode: "pipeline" or "tools"
            is_fallback: True if fallback from pipeline to tools
            store_messages: True (server-side) or False (client-side)
        """
        # Extract token counts
        completion = getattr(usage, "completion_tokens", 0) or 0
        prompt = getattr(usage, "prompt_tokens", 0) or 0
        cached = getattr(usage, "cached_prompt_text_tokens", 0) or 0
        reasoning = getattr(usage, "reasoning_tokens", 0) or 0

        # Update last message stats
        self._last_completion_tokens = completion
        self._last_prompt_tokens = prompt
        self._last_cached_tokens = cached
        self._last_reasoning_tokens = reasoning
        self._last_model = model
        self._last_timestamp = dt_util.now()

        # For conversation: route to internal storage
        if self._service_type == "conversation":
            # Determine target storage
            if mode == "pipeline" and not is_fallback:
                # Pipeline mode
                target = self._tokens_pipeline_server if store_messages else self._tokens_pipeline_client
            else:
                # Tools mode or fallback
                target = self._tokens_tools_server if store_messages else self._tokens_tools_client

            target["prompt"] += prompt
            target["cached"] += cached
            target["count"] += 1

        # Initialize model entry if doesn't exist
        if model not in self._tokens_by_model:
            self._tokens_by_model[model] = {
                "completion": 0,
                "prompt": 0,
                "cached": 0,
                "reasoning": 0,
                "count": 0
            }

        # Update per-model tokens
        self._tokens_by_model[model]["completion"] += completion
        self._tokens_by_model[model]["prompt"] += prompt
        self._tokens_by_model[model]["cached"] += cached
        self._tokens_by_model[model]["reasoning"] += reasoning
        self._tokens_by_model[model]["count"] += 1

        # ALWAYS update cumulative stats (public values)
        self._cumulative_completion_tokens += completion
        self._cumulative_prompt_tokens += prompt
        self._cumulative_cached_tokens += cached
        self._cumulative_reasoning_tokens += reasoning
        self._message_count += 1

        LOGGER.debug("sensor: updated %s (model=%s, mode=%s, fallback=%s, store=%s): cached=%d ratio=%.1f%%",
                    self.entity_id, model, mode, is_fallback, store_messages, cached, self.native_value)

        # Trigger state update
        self.async_write_ha_state()

    @property
    def native_value(self) -> float:
        """Return the cache hit ratio as percentage."""
        total_input = self._cumulative_prompt_tokens + self._cumulative_cached_tokens
        if total_input == 0:
            return 0.0
        ratio = (self._cumulative_cached_tokens / total_input) * 100
        return round(ratio, 1)

    @property
    def extra_state_attributes(self) -> dict:
        """Return additional attributes."""
        attrs = {
            "cached_tokens": self._cumulative_cached_tokens,
            "total_input_tokens": self._cumulative_prompt_tokens + self._cumulative_cached_tokens,
            "non_cached_tokens": self._cumulative_prompt_tokens,
            "service_type": self._service_type,
        }

        # For conversation: expose breakdown for transparency
        if self._service_type == "conversation":
            # Calculate ratios for each category
            ps = self._tokens_pipeline_server
            pc = self._tokens_pipeline_client
            ts = self._tokens_tools_server
            tc = self._tokens_tools_client

            attrs.update({
                "pipeline_server_cached": ps["cached"],
                "pipeline_server_total": ps["prompt"] + ps["cached"],
                "pipeline_server_count": ps["count"],
                "pipeline_client_cached": pc["cached"],
                "pipeline_client_total": pc["prompt"] + pc["cached"],
                "pipeline_client_count": pc["count"],
                "tools_server_cached": ts["cached"],
                "tools_server_total": ts["prompt"] + ts["cached"],
                "tools_server_count": ts["count"],
                "tools_client_cached": tc["cached"],
                "tools_client_total": tc["prompt"] + tc["cached"],
                "tools_client_count": tc["count"],
            })

        if self._reset_timestamp:
            attrs["reset_timestamp"] = self._reset_timestamp.isoformat()
        return attrs


# ==============================================================================
# AGGREGATED SENSORS (4 sensors)
# ==============================================================================

class XAITotalTokensSensor(XAITokenSensorBase):
    """Sensor for total cumulative tokens (all services)."""

    def __init__(self, entry: XAIConfigEntry, subentry) -> None:
        """Initialize the sensor."""
        super().__init__(entry, subentry)
        self._attr_unique_id = f"{entry.entry_id}_total_tokens"
        self._attr_name = "Total tokens"
        self._attr_state_class = SensorStateClass.TOTAL
        self._attr_native_unit_of_measurement = "tokens"
        self._attr_icon = "mdi:counter"

    @property
    def native_value(self) -> int:
        """Return the total cumulative tokens."""
        return self._cumulative_completion_tokens + self._cumulative_prompt_tokens

    @property
    def extra_state_attributes(self) -> dict:
        """Return additional attributes."""
        attrs = {
            "completion_tokens": self._cumulative_completion_tokens,
            "prompt_tokens": self._cumulative_prompt_tokens,
            "cached_tokens": self._cumulative_cached_tokens,
            "reasoning_tokens": self._cumulative_reasoning_tokens,
            "message_count": self._message_count,
            # Store for restoration
            "cumulative_completion_tokens": self._cumulative_completion_tokens,
            "cumulative_prompt_tokens": self._cumulative_prompt_tokens,
            "cumulative_cached_tokens": self._cumulative_cached_tokens,
            "cumulative_reasoning_tokens": self._cumulative_reasoning_tokens,
            "last_completion_tokens": self._last_completion_tokens,
            "last_prompt_tokens": self._last_prompt_tokens,
            "last_cached_tokens": self._last_cached_tokens,
            "last_reasoning_tokens": self._last_reasoning_tokens,
            "tokens_by_model": json.dumps(self._tokens_by_model),
            "last_model": self._last_model,
        }
        if self._last_timestamp:
            attrs["last_timestamp"] = self._last_timestamp.isoformat()
        if self._reset_timestamp:
            attrs["reset_timestamp"] = self._reset_timestamp.isoformat()
        return attrs


class XAIAvgTokensSensor(XAITokenSensorBase):
    """Sensor for average tokens per message (all services)."""

    def __init__(self, entry: XAIConfigEntry, subentry) -> None:
        """Initialize the sensor."""
        super().__init__(entry, subentry)
        self._attr_unique_id = f"{entry.entry_id}_avg_tokens"
        self._attr_name = "Average tokens per message"
        self._attr_native_unit_of_measurement = "tokens"
        self._attr_icon = "mdi:chart-line"
        self._attr_suggested_display_precision = 1

    @property
    def native_value(self) -> float:
        """Return the average tokens per message."""
        if self._message_count == 0:
            return 0.0
        total = self._cumulative_completion_tokens + self._cumulative_prompt_tokens
        return round(total / self._message_count, 1)

    @property
    def extra_state_attributes(self) -> dict:
        """Return additional attributes."""
        attrs = {}
        if self._message_count == 0:
            attrs["message_count"] = 0
        else:
            avg_completion = round(self._cumulative_completion_tokens / self._message_count, 1)
            avg_prompt = round(self._cumulative_prompt_tokens / self._message_count, 1)
            attrs.update({
                "avg_completion_tokens": avg_completion,
                "avg_prompt_tokens": avg_prompt,
                "message_count": self._message_count,
            })

        if self._reset_timestamp:
            attrs["reset_timestamp"] = self._reset_timestamp.isoformat()
        return attrs


class XAICostSensor(XAITokenSensorBase):
    """Sensor for estimated cost in USD (all services)."""

    def __init__(self, entry: XAIConfigEntry, subentry) -> None:
        """Initialize the sensor."""
        super().__init__(entry, subentry)
        self._attr_unique_id = f"{entry.entry_id}_cost"
        self._attr_name = "Estimated cost"
        self._attr_device_class = SensorDeviceClass.MONETARY
        self._attr_state_class = SensorStateClass.TOTAL
        self._attr_native_unit_of_measurement = "USD"
        self._attr_icon = "mdi:currency-usd"
        self._attr_suggested_display_precision = 4

    def _get_pricing_for_model(self, model: str) -> dict:
        """Get pricing for a specific model."""
        # Try custom pricing from config
        model_key = model.replace("-", "_")
        input_price = self._subentry.data.get(f"{model_key}_input_price")
        cached_input_price = self._subentry.data.get(f"{model_key}_cached_input_price")
        output_price = self._subentry.data.get(f"{model_key}_output_price")

        if input_price is not None and output_price is not None:
            if cached_input_price is None:
                cached_input_price = input_price * 0.25
            return {
                "input": input_price,
                "cached_input": cached_input_price,
                "output": output_price
            }

        # Use default pricing
        default_pricing = DEFAULT_MODEL_PRICING.get(model)
        if not default_pricing:
            LOGGER.warning("sensor: no pricing found for model %s, using grok-4-fast-non-reasoning", model)
            default_pricing = DEFAULT_MODEL_PRICING["grok-4-fast-non-reasoning"]

        return default_pricing

    @property
    def native_value(self) -> float:
        """Return the estimated cost in USD."""
        total_cost = 0.0

        for model, tokens in self._tokens_by_model.items():
            pricing = self._get_pricing_for_model(model)

            # FIX: Subtract cached from prompt to avoid double counting
            non_cached_prompt = tokens["prompt"] - tokens["cached"]

            prompt_cost = (non_cached_prompt / 1_000_000) * pricing["input"]
            cached_cost = (tokens["cached"] / 1_000_000) * pricing["cached_input"]
            completion_cost = (tokens["completion"] / 1_000_000) * pricing["output"]
            reasoning_cost = (tokens["reasoning"] / 1_000_000) * pricing["output"]

            model_cost = prompt_cost + cached_cost + completion_cost + reasoning_cost
            total_cost += model_cost

            LOGGER.debug("sensor: cost for model %s = $%.4f (prompt=$%.4f, cached=$%.4f, completion=$%.4f, reasoning=$%.4f)",
                        model, model_cost, prompt_cost, cached_cost, completion_cost, reasoning_cost)

        return round(total_cost, 4)

    @property
    def extra_state_attributes(self) -> dict:
        """Return additional attributes with per-model cost breakdown."""
        cost_by_model = {}
        total_cost = 0.0

        for model, tokens in self._tokens_by_model.items():
            pricing = self._get_pricing_for_model(model)

            non_cached_prompt = tokens["prompt"] - tokens["cached"]
            prompt_cost = (non_cached_prompt / 1_000_000) * pricing["input"]
            cached_cost = (tokens["cached"] / 1_000_000) * pricing["cached_input"]
            completion_cost = (tokens["completion"] / 1_000_000) * pricing["output"]
            reasoning_cost = (tokens["reasoning"] / 1_000_000) * pricing["output"]
            model_total = prompt_cost + cached_cost + completion_cost + reasoning_cost

            cost_by_model[model] = {
                "prompt_cost": round(prompt_cost, 4),
                "cached_cost": round(cached_cost, 4),
                "completion_cost": round(completion_cost, 4),
                "reasoning_cost": round(reasoning_cost, 4),
                "total_cost": round(model_total, 4),
                "input_price_per_1m": pricing["input"],
                "cached_input_price_per_1m": pricing["cached_input"],
                "output_price_per_1m": pricing["output"],
                "tokens": tokens,
            }
            total_cost += model_total

        attrs = {
            "total_cost": round(total_cost, 4),
            "cost_by_model": json.dumps(cost_by_model),
            "tokens_by_model": json.dumps(self._tokens_by_model),
            "total_tokens": self._cumulative_completion_tokens + self._cumulative_prompt_tokens,
            "last_model": self._last_model,
        }

        if self._reset_timestamp:
            attrs["reset_timestamp"] = self._reset_timestamp.isoformat()

        return attrs


class XAIResetTimestampSensor(XAITokenSensorBase):
    """Sensor showing when token statistics were last reset."""

    def __init__(self, entry: XAIConfigEntry, subentry) -> None:
        """Initialize the sensor."""
        super().__init__(entry, subentry)
        self._attr_unique_id = f"{entry.entry_id}_reset_timestamp"
        self._attr_name = "Stats reset at"
        self._attr_device_class = SensorDeviceClass.TIMESTAMP
        self._attr_icon = "mdi:restart"

    @property
    def native_value(self) -> datetime.datetime | None:
        """Return the reset timestamp."""
        return self._reset_timestamp

    @property
    def extra_state_attributes(self) -> dict:
        """Return additional attributes."""
        if not self._reset_timestamp:
            return {"status": "never_reset"}

        # Calculate time since reset
        now = dt_util.now()
        delta = now - self._reset_timestamp
        days = delta.days
        hours = delta.seconds // 3600
        minutes = (delta.seconds % 3600) // 60

        return {
            "days_since_reset": days,
            "hours_since_reset": hours,
            "minutes_since_reset": minutes,
            "total_hours_since_reset": round(delta.total_seconds() / 3600, 1),
        }
