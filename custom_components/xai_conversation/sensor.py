"""Token usage sensors for xAI Conversation integration - V2 Simplified.

This version uses the simplified TokenStats class from helpers/sensors.py.
Sensors are pure presentation layer - no business logic, no calculations.
"""

from __future__ import annotations

import re
from datetime import datetime

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.entity import DeviceInfo, EntityCategory
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.util import dt as dt_util

from .const import (
    DEFAULT_MANUFACTURER,
    DEFAULT_SENSORS_NAME,
    DOMAIN,
    LOGGER,
    DEFAULT_TOOL_PRICE_RAW,
    CLEARER_VISION_LABEL,
    CLEARER_SEARCH_LABEL,
    CLEARER_CACHED_LABEL,
    CHAT_MODE_PIPELINE,
    CHAT_MODE_TOOLS,
    XAIConfigEntry,
)
from .helpers import (
    get_pricing_conversion_factor,
    get_tokens_per_million,
    MemoryManager,
    async_get_user_display_name,
    get_device_display_name,
)


def _raw_to_display_usd(
    raw_value: float,
    conversion_factor: float,
    tokens_per_million: int,
    is_unit_price: bool = False,
) -> float:
    """Convert raw API value to display USD.

    Args:
        raw_value: Raw value from API (price * tokens or price * units)
        conversion_factor: Pricing conversion factor from config
        tokens_per_million: Tokens per million divisor
        is_unit_price: True for per-image or per-call prices

    Returns:
        USD value for display (not rounded - caller decides precision)
    """
    usd = raw_value / conversion_factor
    divisor = max(tokens_per_million, 1.0) if is_unit_price else tokens_per_million
    return usd / divisor


def _format_model_name(model_name: str) -> str:
    """Format model name for display.

    Converts patterns like 'grok-4-1-fast' to 'Grok 4.1 Fast'.
    Handles version numbers separated by hyphens (e.g., -4-1- becomes 4.1).
    """
    # Convert version patterns: -X-Y- to -X.Y- (e.g., -4-1- to -4.1-)
    formatted = re.sub(r"-(\d+)-(\d+)(-|$)", r"-\1.\2\3", model_name)
    # Replace remaining hyphens with spaces and title case
    return formatted.replace("-", " ").title()


class XAITurnSensorsManager:
    """Manage dynamic creation of user chat turn sensors."""

    def __init__(
        self,
        hass: HomeAssistant,
        entry: XAIConfigEntry,
        subentry,
        async_add_entities: AddEntitiesCallback,
    ) -> None:
        self.hass = hass
        self._entry = entry
        self._subentry = subentry
        self._async_add_entities = async_add_entities
        self._created: set[tuple[str, str, str, str]] = set()
        self._unsubscribe = None

    async def async_start(self) -> None:
        memory = self.hass.data[DOMAIN].get("conversation_memory")
        if memory and not self._unsubscribe:
            self._unsubscribe = memory.register_listener(self._schedule_sync)
        await self.async_sync_from_memory()

    def async_stop(self) -> None:
        if self._unsubscribe:
            self._unsubscribe()
            self._unsubscribe = None

    def _schedule_sync(self):
        return self.async_sync_from_memory()

    async def async_sync_from_memory(self) -> None:
        memory = self.hass.data[DOMAIN].get("conversation_memory")
        if not memory:
            return

        try:
            turn_counts = await memory.async_get_turn_counts()
        except Exception as err:
            LOGGER.warning("sensor: failed to load turn counts - %s", err)
            return

        if not turn_counts:
            return

        subentry_titles = {
            se.subentry_id: se.title
            for se in self._entry.subentries.values()
            if se.subentry_type == "conversation"
        }

        sensors_to_add = []
        for scope, identifier, subentry_id, mode, turns in turn_counts:
            if scope not in ("user", "device"):
                continue
            if mode not in (CHAT_MODE_PIPELINE, CHAT_MODE_TOOLS):
                continue
            if turns <= 0:
                continue

            key = (scope, identifier, subentry_id, mode)
            if key in self._created:
                continue

            subentry_title = subentry_titles.get(subentry_id)
            if subentry_title is None:
                # Turn counts should only exist for conversation subentries.
                # Skip anything else (like AI Task which is stateless).
                continue
            if scope == "user":
                display_name = (
                    await async_get_user_display_name(self.hass, identifier) or "User"
                )
            else:
                display_name = (
                    get_device_display_name(self.hass, identifier) or "Device"
                )
            sensors_to_add.append(
                XAIChatTurnsSensor(
                    self.hass,
                    self._entry,
                    scope,
                    identifier,
                    display_name,
                    subentry_id,
                    subentry_title,
                    mode,
                )
            )
            self._created.add(key)

        if sensors_to_add:
            self._async_add_entities(
                sensors_to_add, config_subentry_id=self._subentry.subentry_id
            )


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
            LOGGER.debug("Removing deprecated sensor entity: %s", old_entity_id)
            ent_reg.async_remove(old_entity_id)

        # Migration: Attempt to migrate old cost sensor unique ID to new format
        # Old: {subentry_id}_cost (e.g. xai_conversation:sensors_cost)
        # New: {entry_id}_cost
        # Only migrate if we are processing the sensors subentry
        if subentry.subentry_type == "sensors":
            old_cost_uid = f"{subentry.subentry_id}_cost"
            new_cost_uid = f"{entry.entry_id}_cost"

            old_ent_id = ent_reg.async_get_entity_id(
                Platform.SENSOR, DOMAIN, old_cost_uid
            )
            new_ent_id = ent_reg.async_get_entity_id(
                Platform.SENSOR, DOMAIN, new_cost_uid
            )

            if old_ent_id and not new_ent_id:
                LOGGER.debug(
                    "Migrating cost sensor unique_id from %s to %s",
                    old_cost_uid,
                    new_cost_uid,
                )
                ent_reg.async_update_entity(old_ent_id, new_unique_id=new_cost_uid)

        active_service_types = set()
        for se in entry.subentries.values():
            if se.subentry_type in ("conversation", "ai_task"):
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
                if model_data.get("input_price", 0.0) > 0:
                    sensors.append(
                        XAIPricingSensor(
                            hass, entry, subentry, model_name, "input_price"
                        )
                    )
                if model_data.get("output_price", 0.0) > 0:
                    sensors.append(
                        XAIPricingSensor(
                            hass, entry, subentry, model_name, "output_price"
                        )
                    )
                if model_data.get("cached_input_price", 0.0) > 0:
                    sensors.append(
                        XAIPricingSensor(
                            hass, entry, subentry, model_name, "cached_input_price"
                        )
                    )
                if model_data.get("input_image_price", 0.0) > 0:
                    sensors.append(
                        XAIPricingSensor(
                            hass, entry, subentry, model_name, "input_image_price"
                        )
                    )
                if model_data.get("search_price", 0.0) > 0:
                    sensors.append(
                        XAIPricingSensor(
                            hass, entry, subentry, model_name, "search_price"
                        )
                    )

        async_add_entities(sensors, config_subentry_id=subentry.subentry_id)

        if DOMAIN not in hass.data:
            hass.data[DOMAIN] = {}
        hass.data[DOMAIN][f"{entry.entry_id}_sensors"] = sensors
        manager = XAITurnSensorsManager(hass, entry, subentry, async_add_entities)
        hass.data[DOMAIN][f"{entry.entry_id}_turn_sensors_manager"] = manager
        await manager.async_start()

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

    async def _fetch_data(self, storage) -> dict:
        # Override to fetch server tool stats
        return await storage.get_server_tool_stats()

    @property
    def native_value(self) -> int:
        """Return total tool invocations."""
        if not self._stats:
            return 0
        return self._stats.get("total_invocations", 0)

    @property
    def extra_state_attributes(self) -> dict:
        """Return detailed breakdown of tool usage."""
        if not self._stats:
            return {
                "by_tool": {},
                "by_service": {},
                "total_sources": 0,
            }

        return {
            "by_tool": self._stats.get("by_tool", {}),
            "by_service": self._stats.get("by_service", {}),
            "total_sources": self._stats.get("total_sources", 0),
            "default_pricing_fallback_raw": DEFAULT_TOOL_PRICE_RAW,
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
        """Return total cost in USD (calculated from raw data)."""
        if not self._cost_data:
            return 0.0

        factor = get_pricing_conversion_factor(self._entry)
        tpm = get_tokens_per_million(self._entry)

        # Model costs (token-based)
        raw_total = self._cost_data.get("total_raw", 0.0)
        tool_raw = self._cost_data.get("tool_raw", 0.0)
        model_raw = raw_total - tool_raw
        usd_model = _raw_to_display_usd(model_raw, factor, tpm)

        # Tool costs (unit-based: per call)
        usd_tools = 0.0
        for data in self._cost_data.get("tool_cost_breakdown", {}).values():
            tool_cost_raw = data.get("invocations", 0) * data.get("price_raw", 0.0)
            usd_tools += _raw_to_display_usd(
                tool_cost_raw, factor, tpm, is_unit_price=True
            )

        return round(usd_model + usd_tools, 4)

    @property
    def extra_state_attributes(self) -> dict:
        """Return cost breakdown attributes in USD."""
        if not self._cost_data:
            return {}

        factor = get_pricing_conversion_factor(self._entry)
        tpm = get_tokens_per_million(self._entry)

        # Build USD breakdown per model
        usd_by_model = {}
        for model, data in self._cost_data.get("cost_by_model", {}).items():
            usd_by_model[model] = {
                "total_cost": round(
                    _raw_to_display_usd(data.get("total_raw", 0.0), factor, tpm), 4
                ),
                "prompt_cost": round(
                    _raw_to_display_usd(data.get("prompt_raw", 0.0), factor, tpm), 4
                ),
                "cached_cost": round(
                    _raw_to_display_usd(data.get("cached_raw", 0.0), factor, tpm), 4
                ),
                "completion_cost": round(
                    _raw_to_display_usd(data.get("completion_raw", 0.0), factor, tpm), 4
                ),
                "tokens": data.get("tokens", {}),
            }

        # Build USD breakdown per tool
        usd_tool_breakdown = {}
        total_tool_usd = 0.0
        for tool_name, data in self._cost_data.get("tool_cost_breakdown", {}).items():
            tool_cost_raw = data.get("invocations", 0) * data.get("price_raw", 0.0)
            cost = _raw_to_display_usd(tool_cost_raw, factor, tpm, is_unit_price=True)
            total_tool_usd += cost
            usd_tool_breakdown[tool_name] = {
                "invocations": data.get("invocations", 0),
                "total_cost": round(cost, 4),
            }

        return {
            "total_cost": self.native_value,
            "tool_cost": round(total_tool_usd, 4),
            "tool_cost_breakdown": usd_tool_breakdown,
            "cost_by_model": usd_by_model,
            "tokens_by_model": self._cost_data.get("tokens_by_model", {}),
            "conversion_factor": factor,
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

        # Determine model type to format label and value correctly
        xai_models_data = hass.data[DOMAIN].get("xai_models_data", {})
        model_data = xai_models_data.get(model_name, {})
        m_lower = model_name.lower()
        self._is_image_model = (
            model_data.get("type") == "image"
            or "image" in m_lower
            or "aurora" in m_lower
        )

        # Custom labeling mapping
        label_map = {
            "input_price": "Input",
            "output_price": "Output",
            "cached_input_price": CLEARER_CACHED_LABEL,
            "input_image_price": "Image Input"
            if self._is_image_model
            else CLEARER_VISION_LABEL,
            "search_price": CLEARER_SEARCH_LABEL,
        }
        label = label_map.get(price_type, price_type.replace("_", " ").strip())

        display_name = _format_model_name(model_name)
        if self._is_image_model and self._price_type in (
            "output_price",
            "input_image_price",
        ):
            self._attr_name = f"{display_name} {label} (per image)"
        elif price_type == "search_price":
            self._attr_name = f"{display_name} {label} (per call)"
        else:
            self._attr_name = f"{display_name} {label} (per 1M tokens)"

        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=DEFAULT_SENSORS_NAME,
            manufacturer=DEFAULT_MANUFACTURER,
            model="Pricing",
            entry_type=dr.DeviceEntryType.SERVICE,
        )
        self._unsubscribe = None

        # Initial value from model_data
        initial_price = model_data.get(price_type, 0.0)
        self._attr_native_value = self._convert_price(initial_price)

    def _is_unit_price(self) -> bool:
        """Check if this is a unit price (per image/call) vs token price."""
        return (
            self._is_image_model
            and self._price_type in ("output_price", "input_image_price")
        ) or self._price_type == "search_price"

    def _convert_price(self, raw_price: float) -> float:
        """Convert raw API price to display USD."""
        if raw_price <= 0:
            return 0.0

        factor = get_pricing_conversion_factor(self._entry)
        tpm = get_tokens_per_million(self._entry)
        usd = raw_price / factor

        if self._is_unit_price():
            return usd / max(tpm, 1.0)
        return usd

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        storage = self.hass.data[DOMAIN].get("token_stats")
        if storage:
            self._unsubscribe = storage.register_listener(self._update_from_storage)
            await self._update_from_storage()

    async def async_will_remove_from_hass(self) -> None:
        if self._unsubscribe:
            self._unsubscribe()
        await super().async_will_remove_from_hass()

    async def _update_from_storage(self) -> None:
        storage = self.hass.data[DOMAIN].get("token_stats")
        if not storage:
            return

        price_raw = await storage.get_pricing(self._model_name, self._price_type)
        if price_raw is not None:
            self._attr_native_value = self._convert_price(price_raw)
            self.async_write_ha_state()
        elif self._attr_native_value == 0.0:
            # Fallback: try to get from xai_models_data if storage is empty
            xai_models_data = self.hass.data[DOMAIN].get("xai_models_data", {})
            model_data = xai_models_data.get(self._model_name, {})
            fallback_price_raw = model_data.get(self._price_type, 0.0)
            if fallback_price_raw > 0:
                self._attr_native_value = self._convert_price(fallback_price_raw)
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


class XAIChatTurnsSensor(SensorEntity):
    """Sensor that reports chat turn count for a user or device per subentry/mode."""

    _attr_has_entity_name = True
    _attr_should_poll = False
    _attr_entity_registry_enabled_default = True
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = "turns"

    _ICONS = {"user": "mdi:message-text-clock", "device": "mdi:tablet-dashboard"}

    def __init__(
        self,
        hass: HomeAssistant,
        entry: XAIConfigEntry,
        scope: str,
        identifier: str,
        display_name: str,
        subentry_id: str,
        subentry_title: str,
        mode: str,
    ) -> None:
        self.hass = hass
        self._entry = entry
        self._scope = scope
        self._identifier = identifier
        self._display_name = display_name
        self._subentry_id = subentry_id
        self._subentry_title = subentry_title
        self._mode = mode
        uid_prefix = "turns_device_" if scope == "device" else "turns_"
        self._attr_unique_id = (
            f"{entry.entry_id}_{uid_prefix}{identifier}_{subentry_id}_{mode}"
        )
        self._attr_name = f"{display_name} - {subentry_title} ({mode})"
        self._attr_icon = self._ICONS.get(scope, "mdi:message-text-clock")
        self._attr_native_value = 0
        self._unsubscribe = None

    @property
    def device_info(self) -> DeviceInfo:
        return DeviceInfo(
            identifiers={(DOMAIN, f"{self._entry.entry_id}_notifications")},
            name="xAI Notifications",
            manufacturer=DEFAULT_MANUFACTURER,
            model="xAI Notifications",
            sw_version="1.0",
        )

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        memory = self.hass.data[DOMAIN].get("conversation_memory")
        if memory:
            self._unsubscribe = memory.register_listener(self._schedule_update)
        await self._update_from_memory()

    async def async_will_remove_from_hass(self) -> None:
        if self._unsubscribe:
            self._unsubscribe()
            self._unsubscribe = None
        await super().async_will_remove_from_hass()

    def _schedule_update(self):
        return self._update_from_memory()

    async def _update_from_memory(self) -> None:
        memory = self.hass.data[DOMAIN].get("conversation_memory")
        if not memory:
            self._attr_native_value = 0
            self.async_write_ha_state()
            return

        key = MemoryManager.generate_key(
            self._scope, self._identifier, self._mode, self._subentry_id
        )
        try:
            self._attr_native_value = await memory.async_get_turn_count(key)
        except Exception:
            self._attr_native_value = 0
        self.async_write_ha_state()

        if self._scope == "user":
            name = await async_get_user_display_name(self.hass, self._identifier)
        else:
            name = get_device_display_name(self.hass, self._identifier)
        if name and name != self._display_name:
            self._display_name = name
            self._attr_name = (
                f"{self._display_name} - {self._subentry_title} ({self._mode})"
            )
