"""Token usage sensors for xAI Conversation integration - V2 Simplified.

This version uses the simplified TokenStats class from helpers/sensors.py.
Sensors are pure presentation layer - no business logic, no calculations.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.helpers.entity import EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.util import dt as dt_util

from .const import (
    CONF_COST_PER_TOOL_CALL,
    DEFAULT_MANUFACTURER,
    DEFAULT_SENSORS_NAME,
    DOMAIN,
    LOGGER,
    RECOMMENDED_COST_PER_TOOL_CALL,
    TOOL_PRICING,
)

if TYPE_CHECKING:
    from . import XAIConfigEntry


async def async_setup_entry(
    hass: HomeAssistant,
    entry: XAIConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up xAI token sensors for the sensors subentry."""
    for subentry in entry.subentries.values():
        if subentry.subentry_type != "sensors":
            continue

        # Clean up deprecated New Models Detector sensor
        from homeassistant.helpers import entity_registry as er
        from homeassistant.const import Platform

        ent_reg = er.async_get(hass)
        old_sensor_unique_id = f"{entry.entry_id}_new_models_detector"
        old_entity_id = ent_reg.async_get_entity_id(
            Platform.SENSOR, DOMAIN, old_sensor_unique_id
        )
        if old_entity_id:
            LOGGER.info("Removing deprecated sensor entity: %s", old_entity_id)
            ent_reg.async_remove(old_entity_id)

        active_service_types = set()
        for se in entry.subentries.values():
            if se.subentry_type in ("conversation", "ai_task", "code_fast"):
                active_service_types.add(se.subentry_type)

        LOGGER.debug(
            "Creating sensors for active services: %s", sorted(active_service_types)
        )

        sensors = []
        for service_type in sorted(active_service_types):
            sensors.extend(
                [
                    XAIServiceLastTokensSensor(entry, subentry, service_type),
                    XAIServiceCacheRatioSensor(entry, subentry, service_type),
                ]
            )

        sensors.extend(
            [
                XAITotalTokensSensor(entry, subentry),
                XAIAvgTokensSensor(entry, subentry),
                XAICostSensor(entry, subentry),
                XAIServerToolUsageSensor(entry, subentry),
                XAIAvailableModelsSensor(hass, entry),
                XAIResetTimestampSensor(entry, subentry),
            ]
        )

        # Pricing sensors - create from xai_models_data (populated at startup)
        # Sensors will fetch current prices from TokenStats when added to hass
        xai_models_data = hass.data[DOMAIN].get("xai_models_data", {})
        for model_name, model_data in xai_models_data.items():
            # Only create sensors for primary model names, not aliases
            if model_data.get("name") == model_name:
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

        if DOMAIN not in hass.data:
            hass.data[DOMAIN] = {}
        hass.data[DOMAIN][f"{entry.entry_id}_sensors"] = sensors

        LOGGER.debug("sensor: created %d token sensors", len(sensors))


# ==============================================================================
# BASE CLASS (OBSERVER PATTERN)
# ==============================================================================


class XAITokenSensorBase(SensorEntity):
    """Base class for xAI token sensors acting as observers."""

    _attr_has_entity_name = True
    _attr_should_poll = False
    _attr_entity_registry_enabled_default = True
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, entry: XAIConfigEntry, subentry) -> None:
        """Initialize the sensor."""
        self._entry = entry
        self._subentry = subentry
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=DEFAULT_SENSORS_NAME,
            manufacturer=DEFAULT_MANUFACTURER,
            model="Diagnostics",
            entry_type=dr.DeviceEntryType.SERVICE,
        )
        self._unsubscribe_listener = None
        # Local cache of data for display
        self._stats = {}

    async def async_added_to_hass(self) -> None:
        """Register as listener when added."""
        await super().async_added_to_hass()
        # V2: Use new TokenStats class
        storage = self.hass.data[DOMAIN].get("token_stats")
        if storage:
            self._unsubscribe_listener = storage.register_listener(self._handle_update)
            # Initial fetch
            await self._handle_update()

    async def async_will_remove_from_hass(self) -> None:
        """Unregister listener when removed."""
        if self._unsubscribe_listener:
            self._unsubscribe_listener()
        await super().async_will_remove_from_hass()

    async def _handle_update(self) -> None:
        """Fetch new data from storage and update state."""
        # V2: Use new TokenStats class
        storage = self.hass.data[DOMAIN].get("token_stats")
        if not storage:
            return

        # Fetch data specific to the sensor type (subclasses override _fetch_data)
        self._stats = await self._fetch_data(storage)
        self.async_write_ha_state()

    async def _fetch_data(self, storage) -> dict:
        """Fetch relevant data from storage. To be implemented by subclasses."""
        # V2: Method signature unchanged, uses TokenStats.get_aggregated_stats()
        return await storage.get_aggregated_stats()


class XAIServerToolUsageSensor(XAITokenSensorBase):
    """Cumulative sensor for server-side tool invocations."""

    def __init__(self, entry, subentry) -> None:
        super().__init__(entry, subentry)
        self._attr_unique_id = f"{entry.entry_id}_server_tool_usage"
        self._attr_name = "Server tool invocations"
        self._attr_native_unit_of_measurement = "invocations"
        self._attr_icon = "mdi:tools"
        self._attr_state_class = SensorStateClass.TOTAL_INCREASING
        self._cost_per_source = 0.0

    async def _fetch_data(self, storage) -> dict:
        # Override to fetch server tool stats
        # Get default cost per call from config (used as fallback)
        for subentry in self._entry.subentries.values():
            if subentry.subentry_type == "sensors":
                self._cost_per_source = float(
                    subentry.data.get(
                        CONF_COST_PER_TOOL_CALL, RECOMMENDED_COST_PER_TOOL_CALL
                    )
                )
                break

        return await storage.get_server_tool_stats()

    @property
    def native_value(self) -> int:
        """Return total tool invocations."""
        if not self._stats:
            return 0
        return self._stats.get("total_invocations", 0)

    @property
    def extra_state_attributes(self) -> dict:
        """Return detailed breakdown of tool usage and costs."""
        if not self._stats:
            return {
                "by_tool": {},
                "by_service": {},
                "tool_cost_usd": 0.0,
                "tool_pricing": TOOL_PRICING,
            }

        by_tool = self._stats.get("by_tool", {})

        # Calculate cost for each tool using official pricing
        tool_cost = 0.0
        tool_cost_details = {}

        for tool_name, invocations in by_tool.items():
            price = TOOL_PRICING.get(tool_name, self._cost_per_source)
            cost = invocations * price
            tool_cost += cost
            tool_cost_details[tool_name] = {
                "invocations": invocations,
                "cost_per_call": price,
                "total_cost": round(cost, 4),
            }

        return {
            "by_tool": by_tool,
            "by_service": self._stats.get("by_service", {}),
            "total_sources": self._stats.get("total_sources", 0),
            "tool_cost_usd": round(tool_cost, 4),
            "tool_cost_details": tool_cost_details,
            "tool_pricing": TOOL_PRICING,
        }


# ==============================================================================
# PER-SERVICE SENSORS
# ==============================================================================


class XAIServiceLastTokensSensor(XAITokenSensorBase):
    """Sensor for last message tokens per service."""

    def __init__(self, entry: XAIConfigEntry, subentry, service_type: str) -> None:
        super().__init__(entry, subentry)
        self._service_type = service_type
        self._attr_unique_id = f"{entry.entry_id}_{service_type}_last_tokens"
        self._attr_name = f"{service_type.replace('_', ' ').title()} last tokens"
        self._attr_native_unit_of_measurement = "tokens"
        self._attr_icon = "mdi:message-text"

    async def _fetch_data(self, storage) -> dict:
        # V2: Uses TokenStats.get_service_stats()
        return await storage.get_service_stats(self._service_type)

    @property
    def native_value(self) -> int:
        completion = self._stats.get("last_completion_tokens", 0)
        prompt = self._stats.get("last_prompt_tokens", 0)
        return completion + prompt

    @property
    def extra_state_attributes(self) -> dict:
        attrs = {
            "input_tokens": self._stats.get("last_prompt_tokens", 0),
            "output_tokens": self._stats.get("last_completion_tokens", 0),
            "cached_tokens": self._stats.get("last_cached_tokens", 0),
            "reasoning_tokens": self._stats.get("last_reasoning_tokens", 0),
            "model": self._stats.get("last_model"),
            "service_type": self._service_type,
        }

        if self._service_type == "conversation":
            attrs["mode"] = self._stats.get("last_mode")
            attrs["memory"] = (
                "server-side"
                if self._stats.get("last_store_messages")
                else "client-side"
            )

        if ts := self._stats.get("last_timestamp"):
            attrs["timestamp"] = ts

        return attrs


class XAIServiceCacheRatioSensor(XAITokenSensorBase):
    """Sensor for cache hit ratio per service."""

    def __init__(self, entry: XAIConfigEntry, subentry, service_type: str) -> None:
        super().__init__(entry, subentry)
        self._service_type = service_type
        self._attr_unique_id = f"{entry.entry_id}_{service_type}_cache_ratio"
        self._attr_name = f"{service_type.replace('_', ' ').title()} cache ratio"
        self._attr_native_unit_of_measurement = "%"
        self._attr_icon = "mdi:cached"
        self._attr_suggested_display_precision = 1

    async def _fetch_data(self, storage) -> dict:
        # V2: Uses TokenStats.get_service_stats()
        return await storage.get_service_stats(self._service_type)

    @property
    def native_value(self) -> float:
        prompt = self._stats.get("cumulative_prompt_tokens", 0)
        cached = self._stats.get("cumulative_cached_tokens", 0)
        total = prompt + cached
        if total == 0:
            return 0.0
        return round((cached / total) * 100, 1)

    @property
    def extra_state_attributes(self) -> dict:
        prompt = self._stats.get("cumulative_prompt_tokens", 0)
        cached = self._stats.get("cumulative_cached_tokens", 0)

        attrs = {
            "cached_tokens": cached,
            "total_input_tokens": prompt + cached,
            "non_cached_tokens": prompt,
            "service_type": self._service_type,
        }

        if self._service_type == "conversation":
            # Flatten detailed buckets if available
            for key in [
                "tokens_pipeline_server",
                "tokens_pipeline_client",
                "tokens_tools_server",
                "tokens_tools_client",
            ]:
                bucket = self._stats.get(key, {"prompt": 0, "cached": 0, "count": 0})
                suffix = key.replace("tokens_", "")  # e.g. pipeline_server
                attrs[f"{suffix}_cached"] = bucket.get("cached", 0)
                attrs[f"{suffix}_total"] = bucket.get("prompt", 0) + bucket.get(
                    "cached", 0
                )
                attrs[f"{suffix}_count"] = bucket.get("count", 0)

        return attrs


# ==============================================================================
# AGGREGATED SENSORS
# ==============================================================================


class XAITotalTokensSensor(XAITokenSensorBase):
    """Sensor for total cumulative tokens (all services)."""

    def __init__(self, entry: XAIConfigEntry, subentry) -> None:
        super().__init__(entry, subentry)
        self._attr_unique_id = f"{entry.entry_id}_total_tokens"
        self._attr_name = "Total tokens"
        self._attr_state_class = SensorStateClass.TOTAL
        self._attr_native_unit_of_measurement = "tokens"
        self._attr_icon = "mdi:counter"

    @property
    def native_value(self) -> int:
        return self._stats.get("cumulative_completion_tokens", 0) + self._stats.get(
            "cumulative_prompt_tokens", 0
        )

    @property
    def extra_state_attributes(self) -> dict:
        return {
            "message_count": self._stats.get("message_count", 0),
            "cumulative_completion_tokens": self._stats.get(
                "cumulative_completion_tokens", 0
            ),
            "cumulative_prompt_tokens": self._stats.get("cumulative_prompt_tokens", 0),
            "cumulative_cached_tokens": self._stats.get("cumulative_cached_tokens", 0),
            "cumulative_reasoning_tokens": self._stats.get(
                "cumulative_reasoning_tokens", 0
            ),
            "last_completion_tokens": self._stats.get("last_completion_tokens", 0),
            "last_prompt_tokens": self._stats.get("last_prompt_tokens", 0),
            "last_cached_tokens": self._stats.get("last_cached_tokens", 0),
            "last_reasoning_tokens": self._stats.get("last_reasoning_tokens", 0),
            "tokens_by_model": self._stats.get("tokens_by_model", {}),
            "last_model": self._stats.get("last_model"),
            "last_timestamp": self._stats.get("last_timestamp"),
        }


class XAIAvgTokensSensor(XAITokenSensorBase):
    """Sensor for average tokens per message."""

    def __init__(self, entry: XAIConfigEntry, subentry) -> None:
        super().__init__(entry, subentry)
        self._attr_unique_id = f"{entry.entry_id}_avg_tokens"
        self._attr_name = "Average tokens per message"
        self._attr_native_unit_of_measurement = "tokens"
        self._attr_icon = "mdi:chart-line"
        self._attr_suggested_display_precision = 1

    @property
    def native_value(self) -> float:
        count = self._stats.get("message_count", 0)
        if count == 0:
            return 0.0
        total = self._stats.get("cumulative_completion_tokens", 0) + self._stats.get(
            "cumulative_prompt_tokens", 0
        )
        return round(total / count, 1)

    @property
    def extra_state_attributes(self) -> dict:
        count = self._stats.get("message_count", 0)
        attrs = {"message_count": count}
        if count > 0:
            comp = self._stats.get("cumulative_completion_tokens", 0)
            prompt = self._stats.get("cumulative_prompt_tokens", 0)
            attrs["avg_completion_tokens"] = round(comp / count, 1)
            attrs["avg_prompt_tokens"] = round(prompt / count, 1)
        return attrs


class XAICostSensor(XAITokenSensorBase):
    """Sensor for estimated cost in USD - V2 Pure Frontend.

    This sensor is a pure presentation layer:
    - Receives pre-calculated costs from TokenStats
    - Displays them in Home Assistant
    - NO calculations, NO business logic
    """

    def __init__(self, entry: XAIConfigEntry, subentry) -> None:
        super().__init__(entry, subentry)
        self._attr_unique_id = f"{entry.entry_id}_cost"
        self._attr_name = "Estimated cost"
        self._attr_device_class = SensorDeviceClass.MONETARY
        self._attr_state_class = SensorStateClass.TOTAL
        self._attr_native_unit_of_measurement = "USD"
        self._attr_icon = "mdi:currency-usd"
        self._attr_suggested_display_precision = 4
        self._current_total_cost = 0.0
        self._current_search_cost = 0.0
        self._current_cost_by_model_attrs = {}
        self._current_model_pricing_breakdown_attrs = {}
        # Cache for cost data (received from TokenStats)
        self._cost_data = {}

    async def _handle_update(self) -> None:
        """Fetch pre-calculated costs from TokenStats."""
        # V2: Get costs already calculated by TokenStats
        storage = self.hass.data[DOMAIN].get("token_stats")
        if not storage:
            return

        # Get costs (TokenStats calculates on-demand)
        self._cost_data = await storage.get_costs()
        self.async_write_ha_state()

    @property
    def native_value(self) -> float:
        """Return total cost."""
        return self._cost_data.get("total_cost", 0.0)

    @property
    def extra_state_attributes(self) -> dict:
        """Return cost breakdown attributes."""
        return {
            "total_cost": self._cost_data.get("total_cost", 0.0),
            "tool_cost": self._cost_data.get("tool_cost", 0.0),
            "tool_cost_breakdown": self._cost_data.get("tool_cost_breakdown", {}),
            "cost_by_model": self._cost_data.get("cost_by_model", {}),
            "tokens_by_model": self._cost_data.get("tokens_by_model", {}),
        }


# ==============================================================================
# SPECIAL SENSORS
# ==============================================================================


class XAIResetTimestampSensor(XAITokenSensorBase):
    """Sensor showing when token statistics were last reset."""

    def __init__(self, entry: XAIConfigEntry, subentry) -> None:
        """Initialize the sensor."""
        super().__init__(entry, subentry)
        self._attr_unique_id = f"{entry.entry_id}_reset_timestamp"
        self._attr_name = "Stats reset at"
        self._attr_device_class = SensorDeviceClass.TIMESTAMP
        self._attr_icon = "mdi:restart"

    async def _fetch_data(self, storage) -> dict:
        """Fetch aggregated stats which includes reset_timestamp."""
        return await storage.get_aggregated_stats()

    @property
    def native_value(self) -> datetime | None:
        """Return the reset timestamp."""
        ts = self._stats.get("reset_timestamp")
        if ts:
            # Convert timestamp to datetime
            if isinstance(ts, (int, float)):
                return dt_util.utc_from_timestamp(ts)
            elif isinstance(ts, str):
                return dt_util.parse_datetime(ts)
            elif isinstance(ts, datetime):
                return ts
        return None

    @property
    def extra_state_attributes(self) -> dict:
        """Return additional attributes."""
        reset_ts = self.native_value
        if not reset_ts:
            return {"status": "never_reset"}

        # Calculate time since reset
        now = dt_util.now()
        # Ensure timezones match if present
        if reset_ts.tzinfo is None:
             reset_ts = reset_ts.replace(tzinfo=now.tzinfo)
        
        delta = now - reset_ts
        days = delta.days
        hours = delta.seconds // 3600
        minutes = (delta.seconds % 3600) // 60

        return {
            "days_since_reset": days,
            "hours_since_reset": hours,
            "minutes_since_reset": minutes,
            "total_hours_since_reset": round(delta.total_seconds() / 3600, 1),
        }


class XAIPricingSensor(SensorEntity):
    """Sensor for displaying xAI model pricing (Observer)."""

    _attr_has_entity_name = True
    _attr_should_poll = False
    _attr_entity_registry_enabled_default = True
    _attr_device_class = SensorDeviceClass.MONETARY
    _attr_native_unit_of_measurement = "USD"
    _attr_icon = "mdi:cash-sync"
    _attr_suggested_display_precision = 2

    def __init__(self, hass, entry, subentry, model_name, price_type) -> None:
        self.hass = hass
        self._entry = entry
        self._model_name = model_name
        self._price_type = price_type
        self._attr_unique_id = f"{entry.entry_id}_{model_name}_{price_type}"
        label = price_type.replace("_", " ").replace("price", "").strip()
        self._attr_name = (
            f"{model_name.replace('-', ' ').title()} {label} (per 1M tokens)"
        )
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=DEFAULT_SENSORS_NAME,
            manufacturer=DEFAULT_MANUFACTURER,
            model="Pricing",
            entry_type=dr.DeviceEntryType.SERVICE,
        )
        self._unsubscribe = None
        # Initialize with price from xai_models_data if available
        xai_models_data = hass.data[DOMAIN].get("xai_models_data", {})
        model_data = xai_models_data.get(model_name, {})
        price_key = f"{price_type}_per_million"
        initial_price = model_data.get(price_key, 0.0)
        self._attr_native_value = initial_price if initial_price > 0 else 0.0

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        # V2: Use new TokenStats class
        storage = self.hass.data[DOMAIN].get("token_stats")
        if storage:
            self._unsubscribe = storage.register_listener(self._update_from_storage)
            await self._update_from_storage()

    async def async_will_remove_from_hass(self) -> None:
        if self._unsubscribe:
            self._unsubscribe()
        await super().async_will_remove_from_hass()

    async def _update_from_storage(self) -> None:
        # V2: Use new TokenStats class
        storage = self.hass.data[DOMAIN].get("token_stats")
        if storage:
            price = await storage.get_pricing(self._model_name, self._price_type)
            if price is not None:
                self._attr_native_value = price
                self.async_write_ha_state()
            elif self._attr_native_value == 0.0:
                # Fallback: try to get from xai_models_data if storage is empty
                xai_models_data = self.hass.data[DOMAIN].get("xai_models_data", {})
                model_data = xai_models_data.get(self._model_name, {})
                price_key = f"{self._price_type}_per_million"
                fallback_price = model_data.get(price_key, 0.0)
                if fallback_price > 0:
                    self._attr_native_value = fallback_price
                    self.async_write_ha_state()


class XAIAvailableModelsSensor(SensorEntity):
    """Sensor that reports the count and list of available xAI models."""

    _attr_has_entity_name = True
    _attr_should_poll = True  # Poll periodically to catch updates to hass.data
    # Ideally this would be push-based, but we don't have a listener on hass.data
    # Since we have a periodic task updating hass.data, polling this sensor is acceptable
    # or we could trigger an update from ModelManager. But simple polling is safer for now.
    _attr_entity_registry_enabled_default = True
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_icon = "mdi:robot-happy-outline"
    _attr_state_class = SensorStateClass.MEASUREMENT

    def __init__(self, hass: HomeAssistant, entry: XAIConfigEntry) -> None:
        """Initialize the available models sensor."""
        self.hass = hass
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_available_models"
        self._attr_name = "Available Models"
        self._attr_native_unit_of_measurement = "models"
        self._attr_native_value = 0
        self._available_models: list[str] = []

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

    async def async_update(self) -> None:
        """Update the sensor state."""
        xai_models_data = self.hass.data[DOMAIN].get("xai_models_data", {})
        if xai_models_data:
            # Filter only distinct models (not aliases)
            distinct_models = [
                m for m, d in xai_models_data.items() if d.get("name") == m
            ]
            self._available_models = sorted(distinct_models)
            self._attr_native_value = len(self._available_models)

    @property
    def extra_state_attributes(self) -> dict:
        """Return additional attributes."""
        return {
            "models_list": self._available_models,
            "last_updated": dt_util.now().isoformat(),
        }
