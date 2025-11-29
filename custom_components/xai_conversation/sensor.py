"""Token usage sensors for xAI Conversation integration."""

from __future__ import annotations

from collections import defaultdict
import json
import time
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.helpers.entity import EntityCategory
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.util import dt as dt_util

from .const import (
    CONF_CHAT_MODEL,
    DEFAULT_MANUFACTURER,
    DEFAULT_SENSORS_NAME,
    DOMAIN,
    LOGGER,
    NEW_MODEL_NOTIFICATION_DAYS,
    PRICING_UPDATE_INTERVAL_HOURS,
    RECOMMENDED_CHAT_MODEL,
    TOKENS_PER_MILLION,
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

        # Collect active service types from configured subentries
        active_service_types = set()
        for se in entry.subentries.values():
            if se.subentry_type in ("conversation", "ai_task", "code_fast"):
                active_service_types.add(se.subentry_type)

        LOGGER.debug(
            "Creating sensors for active services: %s", sorted(active_service_types)
        )

        # Create per-service sensors dynamically based on active services
        sensors = []
        for service_type in sorted(active_service_types):
            sensors.extend(
                [
                    XAIServiceLastTokensSensor(entry, subentry, service_type),
                    XAIServiceCacheRatioSensor(entry, subentry, service_type),
                ]
            )

        # Add aggregated sensors (always created)
        sensors.extend(
            [
                XAITotalTokensSensor(entry, subentry),
                XAIAvgTokensSensor(entry, subentry),
                XAICostSensor(entry, subentry),
                XAIResetTimestampSensor(entry, subentry),
                # New models detector
                XAINewModelsDetectorSensor(hass, entry),
            ]
        )

        # Dynamically create pricing sensors for each model
        xai_models_data = hass.data[DOMAIN].get("xai_models_data")
        if xai_models_data:
            for model_name, model_data in xai_models_data.items():
                # Only create pricing sensors for primary model names, not aliases
                if model_data["name"] == model_name:
                    # Create sensors without initial price; they will fetch it themselves
                    if model_data.get("input_price_per_million", 0.0) > 0:
                        sensors.append(
                            XAIPricingSensor(
                                hass, entry, subentry, model_name, "input_price"
                            )
                        )

                    if model_data.get("output_price_per_million", 0.0) > 0:
                        sensors.append(
                            XAIPricingSensor(
                                hass, entry, subentry, model_name, "output_price"
                            )
                        )

                    if model_data.get("cached_input_price_per_million", 0.0) > 0:
                        sensors.append(
                            XAIPricingSensor(
                                hass, entry, subentry, model_name, "cached_input_price"
                            )
                        )

        async_add_entities(sensors, config_subentry_id=subentry.subentry_id)

        # Store sensor references in hass.data so entities can access them
        if DOMAIN not in hass.data:
            hass.data[DOMAIN] = {}
        hass.data[DOMAIN][f"{entry.entry_id}_sensors"] = sensors

        # Schedule periodic pricing and cost updates (chained for correct sequencing)
        # Pricing must be fetched before costs can be calculated with latest prices
        async def _chained_periodic_update(now):
            """Update pricing first, then costs to ensure correct sequencing."""
            # First update pricing from API
            await async_update_pricing_sensors_periodically(hass, entry)
            # Then recalculate costs with new pricing
            await async_update_cost_sensors_periodically(hass, entry.entry_id)

        async_track_time_interval(
            hass,
            _chained_periodic_update,
            timedelta(hours=PRICING_UPDATE_INTERVAL_HOURS),
        )
        LOGGER.debug(
            "sensor: created %d token sensors for sensors subentry %s",
            len(sensors),
            subentry.subentry_id,
        )


# ==============================================================================
# HELPER CLASSES
# ==============================================================================


from .helpers.model_manager import XAIModelManager


# ==============================================================================
# BASE CLASS
# ==============================================================================


class XAITokenSensorBase(SensorEntity):
    """Base class for xAI token sensors with persistent storage via TokenStatsStorage."""

    _attr_has_entity_name = True
    _attr_should_poll = False
    _attr_entity_registry_enabled_default = True
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, entry: XAIConfigEntry, subentry) -> None:
        """Initialize the sensor."""
        self._entry = entry
        self._subentry = subentry
        # Single device for all sensors
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=DEFAULT_SENSORS_NAME,
            manufacturer=DEFAULT_MANUFACTURER,
            model="Diagnostics",
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

    def _restore_json_attr(self, attrs: dict, key: str, default):
        """Restore JSON attribute safely from state attributes.

        Args:
            attrs: State attributes dictionary
            key: Attribute key to restore
            default: Default value if restoration fails

        Returns:
            Restored value or default
        """
        value = attrs.get(key)
        if not value:
            return default
        try:
            return json.loads(value) if isinstance(value, str) else value
        except (ValueError, TypeError) as e:
            LOGGER.warning("Failed to restore %s for %s: %s", key, self.entity_id, e)
            return default

    def _restore_datetime_attr(self, attrs: dict, key: str) -> datetime.datetime | None:
        """Restore datetime attribute safely from state attributes.

        Args:
            attrs: State attributes dictionary
            key: Attribute key to restore

        Returns:
            Parsed datetime or None if restoration fails
        """
        value_str = attrs.get(key)
        if not value_str:
            return None
        try:
            return dt_util.parse_datetime(value_str)
        except (ValueError, TypeError) as e:
            LOGGER.warning("Failed to restore %s for %s: %s", key, self.entity_id, e)
            return None

    @staticmethod
    def _safe_get_token_count(usage, attr_name: str) -> int:
        """Safely extract token count from usage object.

        Args:
            usage: xAI response.usage object
            attr_name: Token attribute name (e.g., 'completion_tokens')

        Returns:
            Token count as integer, 0 if not found or invalid
        """
        value = getattr(usage, attr_name, None)
        return 0 if value is None else int(value)

    async def async_added_to_hass(self) -> None:
        """Load state from TokenStatsStorage when entity is added."""
        await super().async_added_to_hass()

        # Get TokenStatsStorage instance
        storage = self.hass.data[DOMAIN].get("token_stats_storage")
        if not storage:
            LOGGER.warning(
                "TokenStatsStorage not found for %s, starting with fresh state",
                self.name,
            )
            return

        # Load aggregated stats (used by all base sensors)
        stats = await storage.get_aggregated_stats()
        if stats:
            # Restore per-model token data
            raw_tokens = stats.get("tokens_by_model", {})

            # Sanitize: Remove invalid model keys (JSON converts None → "null" string)
            self._tokens_by_model = {
                k: v for k, v in raw_tokens.items() if k and k != "null"
            }
            if (removed := len(raw_tokens) - len(self._tokens_by_model)) > 0:
                LOGGER.info("Cleaned %d invalid model entry(ies) from storage", removed)

            # Restore global counters
            self._cumulative_completion_tokens = stats.get(
                "cumulative_completion_tokens", 0
            )
            self._cumulative_prompt_tokens = stats.get("cumulative_prompt_tokens", 0)
            self._cumulative_cached_tokens = stats.get("cumulative_cached_tokens", 0)
            self._cumulative_reasoning_tokens = stats.get(
                "cumulative_reasoning_tokens", 0
            )
            self._message_count = stats.get("message_count", 0)

            # Restore last message data
            self._last_completion_tokens = stats.get("last_completion_tokens", 0)
            self._last_prompt_tokens = stats.get("last_prompt_tokens", 0)
            self._last_cached_tokens = stats.get("last_cached_tokens", 0)
            self._last_reasoning_tokens = stats.get("last_reasoning_tokens", 0)
            self._last_model = stats.get("last_model")

            # Restore timestamps
            last_ts = stats.get("last_timestamp")
            if last_ts:
                self._last_timestamp = (
                    dt_util.parse_datetime(last_ts)
                    if isinstance(last_ts, str)
                    else last_ts
                )

            reset_ts = stats.get("reset_timestamp")
            if reset_ts:
                self._reset_timestamp = (
                    dt_util.parse_datetime(reset_ts)
                    if isinstance(reset_ts, str)
                    else reset_ts
                )

            LOGGER.debug(
                "sensor: loaded %s data from storage: messages=%d total_tokens=%d models=%s",
                self.name,
                self._message_count,
                self._cumulative_completion_tokens + self._cumulative_prompt_tokens,
                list(self._tokens_by_model.keys()),
            )

    @callback
    def update_token_usage(
        self,
        usage,
        model: str,
        mode: str = "pipeline",
        is_fallback: bool = False,
        store_messages: bool = True,
        skip_save: bool = False,
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
            skip_save: If True, skip saving to storage (for batch updates)
        """
        completion = self._safe_get_token_count(usage, "completion_tokens")
        prompt = self._safe_get_token_count(usage, "prompt_tokens")
        cached = self._safe_get_token_count(usage, "cached_prompt_text_tokens")
        reasoning = self._safe_get_token_count(usage, "reasoning_tokens")

        # Validate model name FIRST (before updating any state)
        if not model or model == "null":
            LOGGER.warning("Skipping token update: invalid model name '%s'", model)
            return

        # Update last message stats (only after validation passes)
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
                "count": 0,
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

        # Save to storage asynchronously (unless skip_save is True for batch updates)
        if not skip_save:
            self.hass.async_create_task(self._async_save_to_storage())

        # Trigger state update
        self.async_write_ha_state()

    async def _async_save_to_storage(self) -> None:
        """Save current stats to TokenStatsStorage."""
        storage = self.hass.data[DOMAIN].get("token_stats_storage")
        if not storage:
            return

        LOGGER.debug("_async_save_to_storage called from sensor: %s", self.entity_id)

        stats = {
            "tokens_by_model": self._tokens_by_model,
            "cumulative_completion_tokens": self._cumulative_completion_tokens,
            "cumulative_prompt_tokens": self._cumulative_prompt_tokens,
            "cumulative_cached_tokens": self._cumulative_cached_tokens,
            "cumulative_reasoning_tokens": self._cumulative_reasoning_tokens,
            "message_count": self._message_count,
            "last_completion_tokens": self._last_completion_tokens,
            "last_prompt_tokens": self._last_prompt_tokens,
            "last_cached_tokens": self._last_cached_tokens,
            "last_reasoning_tokens": self._last_reasoning_tokens,
            "last_model": self._last_model,
            "last_timestamp": self._last_timestamp.isoformat()
            if self._last_timestamp
            else None,
            "reset_timestamp": self._reset_timestamp.isoformat()
            if self._reset_timestamp
            else None,
        }
        await storage.save_aggregated_stats(stats)

    @callback
    def reset_statistics(self, skip_save: bool = False) -> None:
        """Reset cumulative token counters to zero."""
        self._cumulative_completion_tokens = 0
        self._cumulative_prompt_tokens = 0
        self._cumulative_cached_tokens = 0
        self._cumulative_reasoning_tokens = 0
        self._message_count = 0
        self._tokens_by_model = {}
        self._last_model = None
        self._last_timestamp = None
        # Reset last message data (fixes UI display)
        self._last_completion_tokens = 0
        self._last_prompt_tokens = 0
        self._last_cached_tokens = 0
        self._last_reasoning_tokens = 0

        # Store reset timestamp
        self._reset_timestamp = dt_util.now()

        LOGGER.info(
            "sensor: reset token statistics for %s at %s",
            self.entity_id,
            self._reset_timestamp,
        )

        # Save reset state to storage unless skipped
        if not skip_save:
            self.hass.async_create_task(self._async_save_to_storage())

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
        store_messages: bool = True,
        skip_save: bool = False,
    ) -> None:
        """Update sensor with new token usage data.

        Args:
            usage: xAI response.usage object
            model: Model name from xAI response
            mode: "pipeline" or "tools"
            is_fallback: True if fallback from pipeline to tools
            store_messages: True (server-side) or False (client-side)
            skip_save: If True, skip saving to storage (for batch updates)
        """
        # Call parent to handle common token tracking
        super().update_token_usage(
            usage, model, mode, is_fallback, store_messages, skip_save
        )

        # For conversation: save mode and memory type
        if self._service_type == "conversation":
            # If fallback, mode becomes "tools"
            self._last_mode = "tools" if is_fallback else mode
            self._last_store_messages = store_messages

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
            "tokens_by_model": self._tokens_by_model,
        }

        # For conversation: expose mode and memory type
        if self._service_type == "conversation" and self._last_mode is not None:
            attrs["mode"] = self._last_mode
            attrs["memory"] = (
                "server-side" if self._last_store_messages else "client-side"
            )

        if self._last_timestamp:
            attrs["timestamp"] = self._last_timestamp.isoformat()
        if self._reset_timestamp:
            attrs["reset_timestamp"] = self._reset_timestamp.isoformat()
        return attrs

    @callback
    def reset_statistics(self) -> None:
        """Reset statistics including service-specific fields."""
        super().reset_statistics()
        # Reset service-specific fields (conversation only)
        if self._service_type == "conversation":
            self._last_mode = None
            self._last_store_messages = None


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
        store_messages: bool = True,
        skip_save: bool = False,
    ) -> None:
        """Update sensor with new token usage data.

        Args:
            usage: xAI response.usage object
            model: Model name from xAI response
            mode: "pipeline" or "tools"
            is_fallback: True if fallback from pipeline to tools
            store_messages: True (server-side) or False (client-side)
            skip_save: If True, skip saving to storage (for batch updates)
        """
        # CRITICAL: Call parent FIRST to update base state
        super().update_token_usage(
            usage, model, mode, is_fallback, store_messages, skip_save
        )

        # THEN extract token counts and update local state
        prompt = self._safe_get_token_count(usage, "prompt_tokens")
        cached = self._safe_get_token_count(usage, "cached_prompt_text_tokens")

        # For conversation: route to internal storage
        if self._service_type == "conversation":
            # Determine target storage
            if mode == "pipeline" and not is_fallback:
                # Pipeline mode
                target = (
                    self._tokens_pipeline_server
                    if store_messages
                    else self._tokens_pipeline_client
                )
            else:
                # Tools mode or fallback
                target = (
                    self._tokens_tools_server
                    if store_messages
                    else self._tokens_tools_client
                )

            target["prompt"] += prompt
            target["cached"] += cached
            target["count"] += 1

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
            "total_input_tokens": self._cumulative_prompt_tokens
            + self._cumulative_cached_tokens,
            "non_cached_tokens": self._cumulative_prompt_tokens,
            "service_type": self._service_type,
        }

        # For conversation: expose breakdown for transparency
        if self._service_type == "conversation":
            # Cache dict references to reduce attribute lookups
            ps = self._tokens_pipeline_server
            pc = self._tokens_pipeline_client
            ts = self._tokens_tools_server
            tc = self._tokens_tools_client

            attrs.update(
                {
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
                }
            )

        if self._reset_timestamp:
            attrs["reset_timestamp"] = self._reset_timestamp.isoformat()
        return attrs

    @callback
    def reset_statistics(self) -> None:
        """Reset statistics including service-specific storage buckets."""
        super().reset_statistics()
        # Reset service-specific storage (conversation only)
        if self._service_type == "conversation":
            self._tokens_pipeline_server = {"prompt": 0, "cached": 0, "count": 0}
            self._tokens_pipeline_client = {"prompt": 0, "cached": 0, "count": 0}
            self._tokens_tools_server = {"prompt": 0, "cached": 0, "count": 0}
            self._tokens_tools_client = {"prompt": 0, "cached": 0, "count": 0}


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
            "message_count": self._message_count,
            # Cumulative tokens (stored with full names for restoration)
            "cumulative_completion_tokens": self._cumulative_completion_tokens,
            "cumulative_prompt_tokens": self._cumulative_prompt_tokens,
            "cumulative_cached_tokens": self._cumulative_cached_tokens,
            "cumulative_reasoning_tokens": self._cumulative_reasoning_tokens,
            # Last message tokens
            "last_completion_tokens": self._last_completion_tokens,
            "last_prompt_tokens": self._last_prompt_tokens,
            "last_cached_tokens": self._last_cached_tokens,
            "last_reasoning_tokens": self._last_reasoning_tokens,
            # Model tracking
            "tokens_by_model": self._tokens_by_model,
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
            avg_completion = round(
                self._cumulative_completion_tokens / self._message_count, 1
            )
            avg_prompt = round(self._cumulative_prompt_tokens / self._message_count, 1)
            attrs.update(
                {
                    "avg_completion_tokens": avg_completion,
                    "avg_prompt_tokens": avg_prompt,
                    "message_count": self._message_count,
                }
            )

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
        self._current_total_cost = 0.0
        self._current_cost_by_model_attrs = {}
        self._current_model_pricing_breakdown_attrs = {}

    @callback
    def reset_statistics(self, skip_save: bool = False) -> None:
        """Reset cumulative token counters and cost data to zero."""
        # Call parent reset
        super().reset_statistics(skip_save=skip_save)
        # Reset cost-specific data
        self._current_total_cost = 0.0
        self._current_cost_by_model_attrs = {}
        self._current_model_pricing_breakdown_attrs = {}

    @callback
    def update_token_usage(
        self,
        usage,
        model: str,
        mode: str = "pipeline",
        is_fallback: bool = False,
        store_messages: bool = True,
        skip_save: bool = False,
    ) -> None:
        """Update sensor with new token usage data and recalculate costs."""
        # CRITICAL: Calculate costs BEFORE the parent method writes the state.
        # This ensures the new cost is reflected in the same state update.
        self._calculate_costs()
        super().update_token_usage(
            usage, model, mode, is_fallback, store_messages, skip_save
        )
        self.async_write_ha_state()  # Trigger state update after cost calculation

    def _get_pricing_map(self) -> dict[str, dict[str, float]]:
        """Extract pricing data from all XAIPricingSensor instances.

        Returns:
            Dictionary mapping model names to their pricing details:
            {
                "model-name": {
                    "input_price": 1.23,
                    "output_price": 4.56,
                    "cached_input_price": 0.12
                }
            }
        """
        pricing_map = defaultdict(dict)
        all_sensors = self.hass.data.get(DOMAIN, {}).get(
            f"{self._entry.entry_id}_sensors", []
        )

        for sensor in all_sensors:
            if not isinstance(sensor, XAIPricingSensor):
                continue

            model_name = sensor._model_name
            price_type = sensor._price_type

            try:
                price_value = (
                    float(sensor.native_value)
                    if sensor.native_value is not None
                    else 0.0
                )
            except (ValueError, TypeError):
                LOGGER.warning(
                    "Invalid price value for sensor %s, using 0.0",
                    getattr(sensor, "entity_id", "unknown"),
                )
                price_value = 0.0

            pricing_map[model_name][price_type] = price_value

        return dict(pricing_map)

    def _calculate_model_cost(
        self, model: str, tokens: dict, pricing: dict | None
    ) -> dict | None:
        """Calculate cost breakdown for a single model.

        Args:
            model: Model name
            tokens: Token usage dict with keys: prompt, completion, cached, reasoning, count
            pricing: Pricing dict with keys: input_price, output_price, cached_input_price

        Returns:
            Cost breakdown dict or None if pricing unavailable:
            {
                "prompt_cost": float,
                "cached_cost": float,
                "completion_cost": float,
                "reasoning_cost": float,
                "total_cost": float,
                "tokens": dict
            }
        """
        if not pricing:
            LOGGER.warning(
                "sensor: no pricing data found for model '%s' in XAIPricingSensors. Cannot calculate cost.",
                model,
            )
            return None

        # Get prices per million tokens, with fallback for cached price
        input_price = pricing.get("input_price", 0.0)
        output_price = pricing.get("output_price", 0.0)
        cached_input_price = pricing.get("cached_input_price", input_price)

        # Check if model is an image model to adjust calculation
        # Image models use per-image pricing (1 image = 1 token), so we don't divide by TOKENS_PER_MILLION
        is_image_model = False
        xai_models_data = self.hass.data.get(DOMAIN, {}).get("xai_models_data", {})
        if model in xai_models_data and xai_models_data[model].get("type") == "image":
            is_image_model = True

        # Calculate individual cost components
        prompt_cost = (tokens["prompt"] / TOKENS_PER_MILLION) * input_price

        if is_image_model:
            # For image models, output_price is price per image, and completion tokens are number of images
            completion_cost = tokens["completion"] * output_price
        else:
            completion_cost = (tokens["completion"] / TOKENS_PER_MILLION) * output_price

        cached_cost = (tokens["cached"] / TOKENS_PER_MILLION) * cached_input_price
        reasoning_cost = (tokens["reasoning"] / TOKENS_PER_MILLION) * input_price

        # The total cost should NOT include reasoning_cost if they are part of prompt_tokens
        model_total = prompt_cost + completion_cost + cached_cost

        return {
            "prompt_cost": round(prompt_cost, 4),
            "cached_cost": round(cached_cost, 4),
            "completion_cost": round(completion_cost, 4),
            "reasoning_cost": round(reasoning_cost, 4),  # Still shown as attribute
            "total_cost": round(model_total, 4),
            "tokens": tokens,
        }

    @callback
    def _calculate_costs(self) -> None:
        """Calculate and store costs using dynamic pricing from sensors.

        This method orchestrates the cost calculation process:
        1. Extracts pricing data from all pricing sensors
        2. Calculates costs per model
        3. Aggregates total costs
        4. Updates sensor state
        """
        if not self.hass:
            LOGGER.warning(
                "Cannot calculate costs for %s: sensor not yet added to hass",
                getattr(self, "entity_id", "unknown"),
            )
            return

        # Step 1: Get pricing data from sensors
        pricing_map = self._get_pricing_map()

        # Step 2 & 3: Calculate costs for each model and aggregate
        total_cost = 0.0
        cost_by_model = {}

        for model, tokens in self._tokens_by_model.items():
            model_cost = self._calculate_model_cost(
                model, tokens, pricing_map.get(model)
            )
            if model_cost:
                cost_by_model[model] = model_cost
                total_cost += model_cost["total_cost"]

        # Step 4: Update sensor state
        self._current_total_cost = round(total_cost, 4)
        self._current_cost_by_model_attrs = cost_by_model
        self._current_model_pricing_breakdown_attrs = pricing_map

    @callback
    def update_costs(self) -> None:
        """Public method to trigger cost recalculation and state update."""
        self._calculate_costs()
        self.async_write_ha_state()

    @property
    def native_value(self) -> float:
        """Return the estimated cost in USD."""
        return self._current_total_cost if self._current_total_cost is not None else 0.0

    @property
    def extra_state_attributes(self) -> dict:
        """Return additional attributes with per-model cost breakdown."""
        attrs = {
            "total_cost": self._current_total_cost,
            "cost_by_model": self._current_cost_by_model_attrs,
            "tokens_by_model": self._tokens_by_model,
            "total_tokens": self._cumulative_completion_tokens
            + self._cumulative_prompt_tokens,
            "last_model": self._last_model,
            "model_pricing_breakdown": self._current_model_pricing_breakdown_attrs,
        }

        if self._last_timestamp:
            attrs["last_timestamp"] = self._last_timestamp.isoformat()
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


# ==============================================================================
# SPECIAL SENSORS (pricing and new models detector)
# ==============================================================================


class XAIPricingSensor(SensorEntity):
    """Sensor for displaying xAI model pricing via TokenStatsStorage."""

    _attr_has_entity_name = True
    _attr_should_poll = False
    _attr_entity_registry_enabled_default = True
    _attr_device_class = SensorDeviceClass.MONETARY
    _attr_native_unit_of_measurement = "USD"
    _attr_icon = "mdi:cash-sync"
    _attr_suggested_display_precision = 2

    def __init__(
        self,
        hass: HomeAssistant,
        entry: XAIConfigEntry,
        subentry,
        model_name: str,
        price_type: str,
    ) -> None:
        """Initialize the pricing sensor."""
        self.hass = hass
        self._entry = entry
        self._subentry = subentry
        self._model_name = model_name
        self._price_type = (
            price_type  # "input_price", "output_price", or "cached_input_price"
        )

        self._attr_unique_id = f"{entry.entry_id}_{model_name}_{price_type}"
        # Format the name to include "per 1M tokens" for clarity
        price_label = price_type.replace("_", " ").replace("price", "").strip()
        self._attr_name = (
            f"{model_name.replace('-', ' ').title()} {price_label} (per 1M tokens)"
        )

        # All sensors are part of the same device
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=DEFAULT_SENSORS_NAME,
            manufacturer=DEFAULT_MANUFACTURER,
            model="Pricing",
            entry_type=dr.DeviceEntryType.SERVICE,
        )

    async def async_added_to_hass(self) -> None:
        """Load state from TokenStatsStorage when entity is added."""
        await super().async_added_to_hass()

        # Load from TokenStatsStorage
        storage = self.hass.data[DOMAIN].get("token_stats_storage")
        if storage:
            price = await storage.get_pricing(self._model_name, self._price_type)
            if price is not None:
                self._attr_native_value = price
                LOGGER.debug(
                    "Loaded pricing for %s %s: %s",
                    self._model_name,
                    self._price_type,
                    price,
                )
                return

        # On first run or if not in storage, trigger an immediate update to get the initial value
        # Use skip_save=False to ensure prices are persisted even if periodic update fails
        self.update_price(skip_save=False)

    @callback
    def update_price(self, skip_save: bool = False) -> float | None:
        """Update the sensor's state from the latest data stored in hass.data.

        Args:
            skip_save: If True, skip saving to storage (for batch updates)

        Returns:
            Price value if updated successfully, None otherwise
        """
        xai_models_data = self.hass.data[DOMAIN].get("xai_models_data", {})
        model_data = xai_models_data.get(self._model_name)

        if not model_data:
            LOGGER.warning(
                "No pricing data found for model %s during update.", self._model_name
            )
            # Do not change the state if new data is not available
            return None

        price_key = f"{self._price_type}_per_million"
        price_per_million = model_data.get(price_key)

        if price_per_million is not None:
            # The price is already per million, so no division is needed here
            self._attr_native_value = price_per_million
            # Save to storage asynchronously only if not skipping
            if not skip_save:
                self.hass.async_create_task(self._async_save_price(price_per_million))
            self.async_write_ha_state()
            return price_per_million
        else:
            LOGGER.warning(
                "Price type '%s' not found for model %s.",
                self._price_type,
                self._model_name,
            )
            return None

    async def _async_save_price(self, price: float) -> None:
        """Save price to TokenStatsStorage."""
        storage = self.hass.data[DOMAIN].get("token_stats_storage")
        if storage:
            await storage.save_pricing(self._model_name, self._price_type, price)


# XAINewModelsDetectorSensor is at the end of the file after async functions
# It will be moved here in the final cleanup


# ==============================================================================
# ASYNC HELPER FUNCTIONS
# ==============================================================================


async def _async_batch_save_pricing(storage, pricing_data: dict) -> None:
    """Batch save pricing data to storage.

    Args:
        storage: TokenStatsStorage instance
        pricing_data: Dict of {model_name: {price_type: price}}
    """
    # CRITICAL: Ensure data is loaded before modifying _data
    await storage._ensure_loaded()

    # Initialize pricing_data if needed
    if "pricing_data" not in storage._data:
        storage._data["pricing_data"] = {}

    current_time = time.time()
    pricing_storage = storage._data["pricing_data"]

    for model_name, prices in pricing_data.items():
        # Initialize model data if needed
        if model_name not in pricing_storage:
            pricing_storage[model_name] = {}

        # Update all prices for this model
        pricing_storage[model_name].update(prices)
        pricing_storage[model_name]["last_updated"] = current_time

    # Single atomic save operation for all pricing data
    await storage.async_save()
    LOGGER.debug("Batch saved pricing for %d models to storage", len(pricing_data))


async def async_update_cost_sensors_periodically(
    hass: HomeAssistant, entry_id: str
) -> None:
    """Update all XAICostSensor entities periodically."""
    sensors = hass.data.get(DOMAIN, {}).get(f"{entry_id}_sensors")
    if not sensors:
        LOGGER.debug(
            "No sensors found for entry %s during periodic cost update", entry_id
        )
        return

    updated_count = 0
    for sensor in sensors:
        if isinstance(sensor, XAICostSensor):
            sensor.update_costs()
            updated_count += 1

    if updated_count > 0:
        LOGGER.debug("Updated %d cost sensor(s) for entry %s", updated_count, entry_id)


async def async_update_pricing_sensors_periodically(
    hass: HomeAssistant, entry: XAIConfigEntry
) -> None:
    """Fetch latest model pricing from the xAI API and update all XAIPricingSensor entities."""
    # Instantiate the model manager directly
    manager = XAIModelManager(hass)

    # Get known models from persistent storage (new models detector sensor state)
    sensors = hass.data.get(DOMAIN, {}).get(f"{entry.entry_id}_sensors")
    detector_sensor = None
    if sensors:
        for sensor in sensors:
            if isinstance(sensor, XAINewModelsDetectorSensor):
                detector_sensor = sensor
                break

    # Get list of models that have already been acknowledged
    acknowledged_models = set()
    if detector_sensor and hasattr(detector_sensor, "_acknowledged_models"):
        acknowledged_models = detector_sensor._acknowledged_models
    else:
        # First run: populate from current state to avoid false positives
        acknowledged_models = set(
            hass.data.get(DOMAIN, {}).get("xai_models_data", {}).keys()
        )

    try:
        # Check if data is fresh (to avoid double fetch on startup)
        last_update = hass.data.get(DOMAIN, {}).get("xai_models_data_timestamp", 0)
        now = time.time()
        fresh_data_threshold = 60.0  # 1 minute

        if now - last_update < fresh_data_threshold:
            LOGGER.debug("Skipping API fetch in periodic update (data is fresh)")
            xai_models_data = hass.data.get(DOMAIN, {}).get("xai_models_data", {})
        else:
            # Fetch data using the manager
            api_key = entry.data["api_key"]
            xai_models_data = await manager.async_get_models_data(api_key)

            if xai_models_data:
                # Store in hass.data for global access
                if DOMAIN not in hass.data:
                    hass.data[DOMAIN] = {}
                hass.data[DOMAIN]["xai_models_data"] = xai_models_data
                hass.data[DOMAIN]["xai_models_data_timestamp"] = now
                LOGGER.info(
                    "Successfully updated xAI model pricing data via periodic task."
                )
            else:
                LOGGER.warning("Periodic pricing update failed: No data returned.")
                # Even if API fails, update detector sensor with existing data to prevent false positives
                xai_models_data_existing = hass.data.get(DOMAIN, {}).get(
                    "xai_models_data", {}
                )
                if detector_sensor and xai_models_data_existing:
                    # Filter to include only canonical models (exclude aliases)
                    canonical_existing = {
                        k
                        for k, v in xai_models_data_existing.items()
                        if v.get("name") == k
                    }
                    current_models = canonical_existing
                    detector_sensor.update_new_models(
                        [], xai_models_data_existing, current_models
                    )
                return

    except Exception as err:
        LOGGER.error("Failed to update xAI model pricing data: %s", err, exc_info=True)
        return

    # Detect new models by comparing with acknowledged models
    xai_models_data_after = hass.data.get(DOMAIN, {}).get("xai_models_data", {})

    # Filter to include only canonical models (exclude aliases) to prevent duplicate counting
    # Canonical models have their key equal to the "name" field in their data
    canonical_models = {
        k for k, v in xai_models_data_after.items() if v.get("name") == k
    }
    current_models = canonical_models
    new_models = sorted(list(current_models - acknowledged_models))

    # Get the list of sensors for this entry
    sensors = hass.data.get(DOMAIN, {}).get(f"{entry.entry_id}_sensors")
    if not sensors:
        LOGGER.debug(
            "No sensors found for entry %s during periodic pricing update",
            entry.entry_id,
        )
        return

    pricing_count = 0
    pricing_data_to_save = defaultdict(
        dict
    )  # Collect all pricing updates for batch save

    for sensor in sensors:
        if isinstance(sensor, XAIPricingSensor):
            # Update price but collect data instead of saving individually
            price = sensor.update_price(skip_save=True)
            if price is not None:
                pricing_data_to_save[sensor._model_name][sensor._price_type] = price
            pricing_count += 1
        elif isinstance(sensor, XAINewModelsDetectorSensor):
            sensor.update_new_models(new_models, xai_models_data_after, current_models)

    # Batch save: save all pricing data in one operation
    if pricing_data_to_save:
        storage = hass.data.get(DOMAIN, {}).get("token_stats_storage")
        if storage:
            hass.async_create_task(
                _async_batch_save_pricing(storage, pricing_data_to_save)
            )
            LOGGER.debug(
                "Batch saving pricing data for %d model(s), %d sensor(s) total",
                len(pricing_data_to_save),
                pricing_count,
            )


class XAINewModelsDetectorSensor(SensorEntity):
    """Sensor that detects when new xAI models become available via TokenStatsStorage."""

    _attr_has_entity_name = True
    _attr_should_poll = False
    _attr_entity_registry_enabled_default = True
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_icon = "mdi:new-box"
    _attr_state_class = SensorStateClass.MEASUREMENT

    def __init__(self, hass: HomeAssistant, entry: XAIConfigEntry) -> None:
        """Initialize the new models detector sensor."""
        self.hass = hass
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_new_models_detector"
        self._attr_name = "New Grok Models Detector"
        self._attr_native_value = 0
        self._new_models: list[str] = []
        self._detected_at: datetime.datetime | None = None
        self._expires_at: datetime.datetime | None = None
        self._pricing: dict[str, dict] = {}
        self._acknowledged_models: set[str] = set()  # Models we've already seen

    @property
    def device_info(self) -> DeviceInfo:
        """Return device information."""
        return DeviceInfo(
            identifiers={(DOMAIN, f"{self._entry.entry_id}_notifications")},
            name="xAI Notifications",
            manufacturer=DEFAULT_MANUFACTURER,
            model="xAI Notifications",
            sw_version="1.0",
        )

    def _check_expiration(self) -> None:
        """Check if notification has expired and clear if needed."""
        if self._expires_at and dt_util.now() > self._expires_at:
            self._new_models = []
            self._detected_at = None
            self._expires_at = None
            self._pricing = {}
            # Schedule a state write to persist the expiration
            self.async_write_ha_state()

    @property
    def native_value(self) -> int:
        """Return the number of new models detected."""
        if self._expires_at and dt_util.now() > self._expires_at:
            return 0
        return len(self._new_models)

    @property
    def extra_state_attributes(self) -> dict:
        """Return additional attributes."""
        attrs = {
            "new_models": self._new_models,
            "acknowledged_models": sorted(list(self._acknowledged_models)),
        }

        if self._detected_at:
            attrs["detected_at"] = self._detected_at.isoformat()

        if self._expires_at:
            attrs["expires_at"] = self._expires_at.isoformat()

        if self._pricing:
            attrs["pricing"] = self._pricing

        return attrs

    @callback
    def update_new_models(
        self, new_models: list[str], pricing_data: dict, all_current_models: set[str]
    ) -> None:
        """Update the sensor with newly detected models.

        Args:
            new_models: List of newly detected model names
            pricing_data: Pricing information for all models
            all_current_models: Complete set of all current models from API
        """
        # Update acknowledged models: add current models but keep previously acknowledged ones
        if all_current_models:
            self._acknowledged_models.update(all_current_models)

        if not new_models:
            self._check_expiration()
            return

        now = dt_util.now()
        self._new_models = new_models
        self._detected_at = now
        self._expires_at = now + timedelta(days=NEW_MODEL_NOTIFICATION_DAYS)

        # Extract pricing for new models
        self._pricing = {}
        for model_name in new_models:
            if model_name in pricing_data:
                model_data = pricing_data[model_name]
                self._pricing[model_name] = {
                    "input": model_data.get("input_price_per_million", 0.0),
                    "output": model_data.get("output_price_per_million", 0.0),
                    "cached_input": model_data.get(
                        "cached_input_price_per_million", 0.0
                    ),
                }

        # Save to storage
        self.hass.async_create_task(self._async_save_new_models_data())

        self.async_write_ha_state()
        LOGGER.info(
            "Detected %d new xAI model(s): %s", len(new_models), ", ".join(new_models)
        )

    @callback
    def dismiss_new_models(self) -> None:
        """Manually dismiss the new models notification."""
        self._new_models = []
        self._detected_at = None
        self._expires_at = None
        self._pricing = {}

        # Save to storage
        self.hass.async_create_task(self._async_save_new_models_data())

        self.async_write_ha_state()
        LOGGER.info("New models notification dismissed manually")

    async def _async_save_new_models_data(self) -> None:
        """Save new models data to TokenStatsStorage."""
        storage = self.hass.data[DOMAIN].get("token_stats_storage")
        if not storage:
            return

        data = {
            "detected_models": self._new_models,
            "acknowledged_models": sorted(list(self._acknowledged_models)),
            "detected_at": self._detected_at.isoformat() if self._detected_at else None,
            "expires_at": self._expires_at.isoformat() if self._expires_at else None,
            "pricing": self._pricing,
        }
        await storage.save_new_models_data(data)

    async def async_added_to_hass(self) -> None:
        """Load state from TokenStatsStorage when added to hass."""
        await super().async_added_to_hass()

        # Load from TokenStatsStorage
        storage = self.hass.data[DOMAIN].get("token_stats_storage")
        if not storage:
            return

        data = await storage.get_new_models_data()
        if not data:
            return

        # Restore new models list and acknowledged models
        self._new_models = data.get("detected_models", [])
        self._acknowledged_models = set(data.get("acknowledged_models", []))

        if detected_str := data.get("detected_at"):
            self._detected_at = (
                dt_util.parse_datetime(detected_str)
                if isinstance(detected_str, str)
                else detected_str
            )

        if expires_str := data.get("expires_at"):
            self._expires_at = (
                dt_util.parse_datetime(expires_str)
                if isinstance(expires_str, str)
                else expires_str
            )

        self._pricing = data.get("pricing", {})
