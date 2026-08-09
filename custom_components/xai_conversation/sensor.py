"""Token usage and pricing sensors for xAI Conversation integration.

Provides diagnostic sensors for tracking API token usage, cache hit ratio,
estimated monetary costs, API model pricing, and chat turn counts.
Uses SensorEntityDescription to minimize code duplication and conform to HA guidelines.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from datetime import datetime
import re
from typing import Any

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorEntityDescription,
    SensorStateClass,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr, entity_registry as er
from homeassistant.helpers.entity import DeviceInfo, EntityCategory
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import StateType
from homeassistant.util import dt as dt_util

from .const import (
    CHAT_MODE_PIPELINE,
    CHAT_MODE_TOOLS,
    CLEARER_CACHED_LABEL,
    CLEARER_SEARCH_LABEL,
    CLEARER_VISION_LABEL,
    DEFAULT_MANUFACTURER,
    DEFAULT_SENSORS_NAME,
    DEFAULT_TOOL_PRICE_RAW,
    DOMAIN,
    LOGGER,
    XAIConfigEntry,
)
from .helpers import (
    MemoryManager,
    async_get_user_display_name,
    get_device_display_name,
    get_pricing_conversion_factor,
    get_tokens_per_million,
)


def _raw_to_display_usd(
    raw_value: float,
    conversion_factor: float,
    tokens_per_million: int,
    is_unit_price: bool = False,
) -> float:
    """Convert raw API value to display USD."""
    factor = conversion_factor or 10000.0
    tpm = tokens_per_million or 1000000
    usd = raw_value / factor
    divisor = max(tpm, 1.0) if is_unit_price else tpm
    return usd / divisor


def _format_model_name(model_name: str) -> str:
    """Format model name for display."""
    formatted = re.sub(r"-(\d+)-(\d+)(-|$)", r"-\1.\2\3", model_name)
    return formatted.replace("-", " ").title()


# ==============================================================================
# SENSOR DESCRIPTIONS FOR UNIFIED STATS
# ==============================================================================


@dataclass(frozen=True)
class XAITokenSensorEntityDescription(SensorEntityDescription):
    """Describe an XAI token sensor."""

    value_fn: Callable[[dict[str, Any], XAITokenSensor], StateType] | None = None
    attrs_fn: Callable[[dict[str, Any], XAITokenSensor], dict[str, Any]] | None = None
    fetch_fn: (
        Callable[[Any, str | None], Coroutine[Any, Any, dict[str, Any]]] | None
    ) = None
    service_type: str | None = None


# Generic fetch functions
async def _fetch_aggregated(storage, service_type: str | None) -> dict[str, Any]:
    return await storage.get_aggregated_stats()


async def _fetch_service(storage, service_type: str | None) -> dict[str, Any]:
    return await storage.get_service_stats(service_type or "conversation")


async def _fetch_tool_usage(storage, service_type: str | None) -> dict[str, Any]:
    return await storage.get_server_tool_stats()


async def _fetch_costs(storage, service_type: str | None) -> dict[str, Any]:
    return await storage.get_costs()


# Helper value / attributes callbacks
def _val_last_tokens(stats: dict[str, Any], entity: XAITokenSensor) -> int:
    return stats.get("last_completion_tokens", 0) + stats.get("last_prompt_tokens", 0)


def _attrs_last_tokens(stats: dict[str, Any], entity: XAITokenSensor) -> dict[str, Any]:
    service = entity.entity_description.service_type
    attrs = {
        "input_tokens": stats.get("last_prompt_tokens", 0),
        "output_tokens": stats.get("last_completion_tokens", 0),
        "cached_tokens": stats.get("last_cached_tokens", 0),
        "reasoning_tokens": stats.get("last_reasoning_tokens", 0),
        "model": stats.get("last_model"),
        "service_type": service,
    }
    if service == "conversation":
        attrs["mode"] = stats.get("last_mode")
        attrs["memory"] = (
            "server-side" if stats.get("last_store_messages") else "client-side"
        )
    if ts := stats.get("last_timestamp"):
        attrs["timestamp"] = ts
    return attrs


def _val_cache_ratio(stats: dict[str, Any], entity: XAITokenSensor) -> float:
    prompt = stats.get("cumulative_prompt_tokens", 0)
    cached = stats.get("cumulative_cached_tokens", 0)
    total = prompt + cached
    if total == 0:
        return 0.0
    return round((cached / total) * 100, 1)


def _attrs_cache_ratio(stats: dict[str, Any], entity: XAITokenSensor) -> dict[str, Any]:
    return {
        "cached_prompt_tokens": stats.get("cumulative_cached_tokens", 0),
        "total_prompt_tokens_excluding_cached": stats.get(
            "cumulative_prompt_tokens", 0
        ),
        "service_type": entity.entity_description.service_type,
    }


def _val_total_tokens(stats: dict[str, Any], entity: XAITokenSensor) -> int:
    return stats.get("cumulative_prompt_tokens", 0) + stats.get(
        "cumulative_completion_tokens", 0
    )


def _attrs_total_tokens(
    stats: dict[str, Any], entity: XAITokenSensor
) -> dict[str, Any]:
    return {
        "total_input_tokens": stats.get("cumulative_prompt_tokens", 0),
        "total_output_tokens": stats.get("cumulative_completion_tokens", 0),
        "total_cached_input_tokens": stats.get("cumulative_cached_tokens", 0),
        "total_reasoning_tokens": stats.get("cumulative_reasoning_tokens", 0),
        "total_image_input_tokens": stats.get("cumulative_image_tokens", 0),
    }


def _val_avg_tokens(stats: dict[str, Any], entity: XAITokenSensor) -> float:
    count = stats.get("message_count", 0)
    total = stats.get("cumulative_prompt_tokens", 0) + stats.get(
        "cumulative_completion_tokens", 0
    )
    return round(total / count, 1) if count > 0 else 0.0


def _attrs_avg_tokens(stats: dict[str, Any], entity: XAITokenSensor) -> dict[str, Any]:
    count = stats.get("message_count", 0)
    comp = stats.get("cumulative_completion_tokens", 0)
    prompt = stats.get("cumulative_prompt_tokens", 0)
    if count == 0:
        return {
            "total_calls": 0,
            "avg_completion_tokens": 0.0,
            "avg_prompt_tokens": 0.0,
        }
    return {
        "total_calls": count,
        "avg_completion_tokens": round(comp / count, 1),
        "avg_prompt_tokens": round(prompt / count, 1),
    }


def _val_cost(stats: dict[str, Any], entity: XAITokenSensor) -> float:
    factor = get_pricing_conversion_factor(entity._entry)
    tpm = get_tokens_per_million(entity._entry)

    raw_total = stats.get("total_raw", 0.0)
    tool_raw = stats.get("tool_raw", 0.0)
    model_raw = raw_total - tool_raw
    usd_model = _raw_to_display_usd(model_raw, factor, tpm)

    usd_tools = 0.0
    for data in stats.get("tool_cost_breakdown", {}).values():
        tool_cost_raw = data.get("invocations", 0) * data.get("price_raw", 0.0)
        usd_tools += _raw_to_display_usd(tool_cost_raw, factor, tpm, is_unit_price=True)

    return round(usd_model + usd_tools, 4)


def _attrs_cost(stats: dict[str, Any], entity: XAITokenSensor) -> dict[str, Any]:
    factor = get_pricing_conversion_factor(entity._entry)
    tpm = get_tokens_per_million(entity._entry)

    usd_by_model = {}
    for model, data in stats.get("cost_by_model", {}).items():
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

    usd_tool_breakdown = {}
    total_tool_usd = 0.0
    for tool_name, data in stats.get("tool_cost_breakdown", {}).items():
        tool_cost_raw = data.get("invocations", 0) * data.get("price_raw", 0.0)
        cost = _raw_to_display_usd(tool_cost_raw, factor, tpm, is_unit_price=True)
        total_tool_usd += cost
        usd_tool_breakdown[tool_name] = {
            "invocations": data.get("invocations", 0),
            "total_cost": round(cost, 4),
        }

    total_cost = _val_cost(stats, entity)
    return {
        "total_cost": total_cost,
        "tool_cost": round(total_tool_usd, 4),
        "tool_cost_breakdown": usd_tool_breakdown,
        "cost_by_model": usd_by_model,
        "tokens_by_model": stats.get("tokens_by_model", {}),
        "conversion_factor": factor,
    }


def _val_tool_usage(stats: dict[str, Any], entity: XAITokenSensor) -> int:
    return stats.get("total_invocations", 0)


def _attrs_tool_usage(stats: dict[str, Any], entity: XAITokenSensor) -> dict[str, Any]:
    return {
        "by_tool": stats.get("by_tool", {}),
        "by_service": stats.get("by_service", {}),
        "total_sources": stats.get("total_sources", 0),
        "default_pricing_fallback_raw": DEFAULT_TOOL_PRICE_RAW,
    }


def _val_reset_time(stats: dict[str, Any], entity: XAITokenSensor) -> datetime | None:
    ts = stats.get("reset_timestamp")
    if ts:
        if isinstance(ts, (int, float)):
            return dt_util.utc_from_timestamp(ts)
        elif isinstance(ts, str):
            return dt_util.parse_datetime(ts)
        elif isinstance(ts, datetime):
            return ts
    return None


def _attrs_reset_time(stats: dict[str, Any], entity: XAITokenSensor) -> dict[str, Any]:
    val = _val_reset_time(stats, entity)
    if not val:
        return {"status": "never_reset"}

    now = dt_util.now()
    if val.tzinfo is None:
        val = val.replace(tzinfo=now.tzinfo)

    delta = now - val
    return {
        "days_since_reset": delta.days,
        "hours_since_reset": delta.seconds // 3600,
        "minutes_since_reset": (delta.seconds % 3600) // 60,
        "total_hours_since_reset": round(delta.total_seconds() / 3600, 1),
    }


CORE_SENSOR_DESCRIPTIONS = [
    XAITokenSensorEntityDescription(
        key="total_tokens",
        name="Total tokens",
        native_unit_of_measurement="tokens",
        icon="mdi:counter",
        state_class=SensorStateClass.TOTAL_INCREASING,
        fetch_fn=_fetch_aggregated,
        value_fn=_val_total_tokens,
        attrs_fn=_attrs_total_tokens,
    ),
    XAITokenSensorEntityDescription(
        key="avg_tokens",
        name="Average tokens per call",
        native_unit_of_measurement="tokens",
        icon="mdi:chart-bell-curve",
        suggested_display_precision=1,
        fetch_fn=_fetch_aggregated,
        value_fn=_val_avg_tokens,
        attrs_fn=_attrs_avg_tokens,
    ),
    XAITokenSensorEntityDescription(
        key="cost",
        name="Estimated cost",
        device_class=SensorDeviceClass.MONETARY,
        state_class=SensorStateClass.TOTAL,
        native_unit_of_measurement="USD",
        icon="mdi:currency-usd",
        suggested_display_precision=4,
        fetch_fn=_fetch_costs,
        value_fn=_val_cost,
        attrs_fn=_attrs_cost,
    ),
    XAITokenSensorEntityDescription(
        key="server_tool_usage",
        name="Server tool invocations",
        native_unit_of_measurement="invocations",
        icon="mdi:tools",
        state_class=SensorStateClass.TOTAL_INCREASING,
        fetch_fn=_fetch_tool_usage,
        value_fn=_val_tool_usage,
        attrs_fn=_attrs_tool_usage,
    ),
    XAITokenSensorEntityDescription(
        key="reset_timestamp",
        name="Stats reset at",
        device_class=SensorDeviceClass.TIMESTAMP,
        icon="mdi:restart",
        fetch_fn=_fetch_aggregated,
        value_fn=_val_reset_time,
        attrs_fn=_attrs_reset_time,
    ),
]

# ==============================================================================
# ENTRY POINT SETUP
# ==============================================================================


async def async_setup_entry(
    hass: HomeAssistant,
    entry: XAIConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up xAI token and diagnostics sensors."""
    for subentry in entry.subentries.values():
        if subentry.subentry_type != "sensors":
            continue

        # Migration: remove deprecated models detector sensor and update old cost unique ID
        ent_reg = er.async_get(hass)
        old_detector_uid = f"{entry.entry_id}_new_models_detector"
        if old_detector := ent_reg.async_get_entity_id(
            "sensor", DOMAIN, old_detector_uid
        ):
            LOGGER.debug("Removing deprecated sensor entity: %s", old_detector)
            ent_reg.async_remove(old_detector)

        old_cost_uid = f"{subentry.subentry_id}_cost"
        new_cost_uid = f"{entry.entry_id}_cost"
        if (
            old_cost_ent := ent_reg.async_get_entity_id("sensor", DOMAIN, old_cost_uid)
        ) and not ent_reg.async_get_entity_id("sensor", DOMAIN, new_cost_uid):
            LOGGER.debug(
                "Migrating cost sensor unique_id from %s to %s",
                old_cost_uid,
                new_cost_uid,
            )
            ent_reg.async_update_entity(old_cost_ent, new_unique_id=new_cost_uid)

        active_service_types = {
            se.subentry_type
            for se in entry.subentries.values()
            if se.subentry_type in ("conversation", "ai_task")
        }

        LOGGER.debug(
            "Creating sensors for active services: %s", sorted(active_service_types)
        )

        sensors: list[SensorEntity] = []

        # 1. Create service-specific sensors (Last tokens, Cache Hit Ratio)
        for service_type in sorted(active_service_types):
            service_label = service_type.replace("_", " ").title()
            sensors.append(
                XAITokenSensor(
                    entry,
                    subentry,
                    XAITokenSensorEntityDescription(
                        key=f"{service_type}_last_tokens",
                        name=f"{service_label} last tokens",
                        native_unit_of_measurement="tokens",
                        icon="mdi:message-text",
                        fetch_fn=_fetch_service,
                        value_fn=_val_last_tokens,
                        attrs_fn=_attrs_last_tokens,
                        service_type=service_type,
                    ),
                )
            )
            sensors.append(
                XAITokenSensor(
                    entry,
                    subentry,
                    XAITokenSensorEntityDescription(
                        key=f"{service_type}_cache_ratio",
                        name=f"{service_label} cache ratio",
                        native_unit_of_measurement="%",
                        icon="mdi:cached",
                        suggested_display_precision=1,
                        fetch_fn=_fetch_service,
                        value_fn=_val_cache_ratio,
                        attrs_fn=_attrs_cache_ratio,
                        service_type=service_type,
                    ),
                )
            )

        # 2. Create core aggregated sensors
        for desc in CORE_SENSOR_DESCRIPTIONS:
            sensors.append(XAITokenSensor(entry, subentry, desc))

        # 3. Create notifications & available models sensor
        sensors.append(XAIAvailableModelsSensor(hass, entry))

        async_add_entities(sensors, config_subentry_id=subentry.subentry_id)

        hass.data[DOMAIN][f"{entry.entry_id}_sensors"] = sensors

        # 4. Start managers for dynamic turns and pricing sensors
        turn_manager = XAIChatTurnsSensorManager(
            hass, entry, subentry, async_add_entities
        )
        hass.data[DOMAIN][f"{entry.entry_id}_turn_sensors_manager"] = turn_manager
        await turn_manager.async_start()
        entry.async_on_unload(turn_manager.async_stop)

        pricing_manager = XAIPricingSensorManager(
            hass, entry, subentry, async_add_entities
        )
        hass.data[DOMAIN][f"{entry.entry_id}_pricing_sensors_manager"] = pricing_manager
        await pricing_manager.async_start()
        entry.async_on_unload(pricing_manager.async_stop)

        LOGGER.debug("sensor: created %d token and pricing sensors", len(sensors))


# ==============================================================================
# UNIFIED BASE OBSERVABLE CLASS
# ==============================================================================


class XAITokenSensorBase(SensorEntity):
    """Base class for xAI token sensors acting as observers on TokenStats."""

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
        self._stats: dict[str, Any] = {}

    async def async_added_to_hass(self) -> None:
        """Register as listener when added to Home Assistant."""
        await super().async_added_to_hass()
        storage = self.hass.data[DOMAIN].get("token_stats")
        if storage:
            self._unsubscribe_listener = storage.register_listener(self._handle_update)
            self._stats = await self._fetch_data(storage)

    async def async_will_remove_from_hass(self) -> None:
        """Unregister listener when removed."""
        if self._unsubscribe_listener:
            self._unsubscribe_listener()
        await super().async_will_remove_from_hass()

    async def _handle_update(self) -> None:
        """Fetch new data from storage and trigger state update."""
        storage = self.hass.data[DOMAIN].get("token_stats")
        if not storage:
            return

        self._stats = await self._fetch_data(storage)
        self.async_write_ha_state()

    async def _fetch_data(self, storage: Any) -> dict[str, Any]:
        """Fetch relevant data. To be overridden by subclasses."""
        return await storage.get_aggregated_stats()


# ==============================================================================
# UNIFIED DYNAMIC SENSOR (REPLACES 7 INDIVIDUAL CLASSES)
# ==============================================================================


class XAITokenSensor(XAITokenSensorBase):
    """Unified sensor class that dynamically formats its state using description callbacks."""

    entity_description: XAITokenSensorEntityDescription

    def __init__(
        self,
        entry: XAIConfigEntry,
        subentry,
        description: XAITokenSensorEntityDescription,
    ) -> None:
        """Initialize unified token sensor."""
        super().__init__(entry, subentry)
        self.entity_description = description
        self._attr_unique_id = f"{entry.entry_id}_{description.key}"

    async def _fetch_data(self, storage: Any) -> dict[str, Any]:
        """Fetch data dynamically using the description callback if defined."""
        if self.entity_description.fetch_fn:
            return await self.entity_description.fetch_fn(
                storage, self.entity_description.service_type
            )
        return await super()._fetch_data(storage)

    @property
    def native_value(self) -> StateType:
        """Return the processed native state value."""
        if self.entity_description.value_fn is None:
            return None
        return self.entity_description.value_fn(self._stats or {}, self)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return processed extra state attributes."""
        if self.entity_description.attrs_fn is None:
            return {}
        return self.entity_description.attrs_fn(self._stats or {}, self)


# ==============================================================================
# SPECIFIC DIAGNOSTIC & TURN SENSORS
# ==============================================================================


class XAIPricingSensor(SensorEntity):
    """Sensor for displaying current xAI model API pricing."""

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

        xai_models_data = hass.data[DOMAIN].get("xai_models_data", {})
        model_data = xai_models_data.get(model_name, {})
        m_lower = model_name.lower()
        self._is_image_model = (
            model_data.get("type") == "image"
            or "image" in m_lower
            or "aurora" in m_lower
        )

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
        self._attr_native_value = self._convert_price(model_data.get(price_type, 0.0))

    def _is_unit_price(self) -> bool:
        return (
            self._is_image_model
            and self._price_type in ("output_price", "input_image_price")
        ) or self._price_type == "search_price"

    def _convert_price(self, raw_price: float) -> float:
        if raw_price <= 0:
            return 0.0
        factor = get_pricing_conversion_factor(self._entry) or 10000.0
        tpm = get_tokens_per_million(self._entry) or 1000000
        usd = raw_price / factor
        return usd / max(tpm, 1.0) if self._is_unit_price() else usd

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        storage = self.hass.data[DOMAIN].get("token_stats")
        if storage:
            self._unsubscribe = storage.register_listener(self._update_from_storage)
            price_raw = await storage.get_pricing(self._model_name, self._price_type)
            if price_raw is not None:
                self._attr_native_value = self._convert_price(price_raw)

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


class XAIAvailableModelsSensor(SensorEntity):
    """Sensor reporting the count and list of available xAI models."""

    _attr_has_entity_name = True
    _attr_should_poll = False
    _attr_entity_registry_enabled_default = True
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_icon = "mdi:robot-happy-outline"
    _attr_state_class = SensorStateClass.MEASUREMENT

    def __init__(self, hass: HomeAssistant, entry: XAIConfigEntry) -> None:
        self.hass = hass
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_available_models"
        self._attr_name = "Available Models"
        self._attr_native_unit_of_measurement = "models"
        self._attr_native_value = 0
        self._available_models: list[str] = []
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
        storage = self.hass.data[DOMAIN].get("token_stats")
        if storage:
            self._unsubscribe = storage.register_listener(self._update_sensor)
        await self._update_sensor()

    async def async_will_remove_from_hass(self) -> None:
        if self._unsubscribe:
            self._unsubscribe()
            self._unsubscribe = None
        await super().async_will_remove_from_hass()

    async def _update_sensor(self) -> None:
        xai_models_data = self.hass.data[DOMAIN].get("xai_models_data", {})

        # Fallback to persistent pricing data in token_stats if memory cache is empty
        if not xai_models_data:
            storage = self.hass.data[DOMAIN].get("token_stats")
            if storage:
                try:
                    db_pricing = await storage.get_pricing_data()
                    xai_models_data = {
                        m: {"name": m, **d} for m, d in db_pricing.items()
                    }
                except Exception as err:
                    LOGGER.warning(
                        "sensor: failed to load pricing data from storage - %s", err
                    )

        if xai_models_data:
            distinct_models = [
                m for m, d in xai_models_data.items() if d.get("name") == m
            ]
            self._available_models = sorted(distinct_models)
            self._attr_native_value = len(self._available_models)
        else:
            self._available_models = []
            self._attr_native_value = 0
        self.async_write_ha_state()

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        return {
            "models": self._available_models,
            "last_updated": dt_util.now().isoformat(),
        }


class XAIChatTurnsSensorManager:
    """Manages creation and runtime tracking of user/device chat turns sensors."""

    def __init__(
        self,
        hass: HomeAssistant,
        entry: XAIConfigEntry,
        subentry: Any,
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
        active_keys = set()
        for scope, identifier, subentry_id, mode, turns in turn_counts:
            if scope not in ("user", "device") or mode not in (
                CHAT_MODE_PIPELINE,
                CHAT_MODE_TOOLS,
            ):
                continue
            if turns <= 0:
                continue

            key = (scope, identifier, subentry_id, mode)
            active_keys.add(key)
            if key in self._created:
                continue

            subentry_title = subentry_titles.get(subentry_id)
            if subentry_title is None:
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

        # Cleanup runtime orphaned turn sensors that hit zero turns
        orphaned_keys = self._created - active_keys
        if orphaned_keys:
            ent_reg = er.async_get(self.hass)
            for key in orphaned_keys:
                scope, identifier, subentry_id, mode = key
                uid_prefix = "turns_device_" if scope == "device" else "turns_"
                unique_id = f"{self._entry.entry_id}_{uid_prefix}{identifier}_{subentry_id}_{mode}"

                entity_id = ent_reg.async_get_entity_id("sensor", DOMAIN, unique_id)
                if entity_id:
                    LOGGER.debug("Removing reset/orphaned turn sensor: %s", entity_id)
                    ent_reg.async_remove(entity_id)

                self._created.remove(key)


class XAIPricingSensorManager:
    """Manages dynamic creation of pricing sensors when model data is fetched."""

    def __init__(
        self,
        hass: HomeAssistant,
        entry: XAIConfigEntry,
        subentry: Any,
        async_add_entities: AddEntitiesCallback,
    ) -> None:
        self.hass = hass
        self._entry = entry
        self._subentry = subentry
        self._async_add_entities = async_add_entities
        self._created: set[str] = set()
        self._unsubscribe = None

    async def async_start(self) -> None:
        """Start monitoring model pricing updates."""
        storage = self.hass.data[DOMAIN].get("token_stats")
        if storage:
            self._unsubscribe = storage.register_listener(self._schedule_sync)
        await self.async_sync_pricing()

    def async_stop(self) -> None:
        """Stop monitoring."""
        if self._unsubscribe:
            self._unsubscribe()
            self._unsubscribe = None

    def _schedule_sync(self) -> None:
        self.hass.async_create_task(self.async_sync_pricing())

    async def async_sync_pricing(self) -> None:
        """Create new pricing sensors for newly discovered models/prices."""
        xai_models_data = self.hass.data[DOMAIN].get("xai_models_data", {})

        # Fallback to persistent pricing data in token_stats if memory cache is empty
        if not xai_models_data:
            storage = self.hass.data[DOMAIN].get("token_stats")
            if storage:
                try:
                    db_pricing = await storage.get_pricing_data()
                    xai_models_data = {
                        m: {"name": m, **d} for m, d in db_pricing.items()
                    }
                except Exception as err:
                    LOGGER.warning(
                        "sensor: failed to load pricing data from storage - %s", err
                    )

        if not xai_models_data:
            return

        sensors_to_add = []
        for model_name, model_data in xai_models_data.items():
            if model_data.get("name") != model_name:
                continue

            for price_type in (
                "input_price",
                "output_price",
                "cached_input_price",
                "input_image_price",
                "search_price",
            ):
                if model_data.get(price_type, 0.0) > 0:
                    unique_id = f"{self._entry.entry_id}_{model_name}_{price_type}"
                    if unique_id in self._created:
                        continue

                    sensors_to_add.append(
                        XAIPricingSensor(
                            self.hass,
                            self._entry,
                            self._subentry,
                            model_name,
                            price_type,
                        )
                    )
                    self._created.add(unique_id)

        if sensors_to_add:
            LOGGER.debug(
                "sensor: dynamically adding %d pricing sensors", len(sensors_to_add)
            )
            self._async_add_entities(
                sensors_to_add, config_subentry_id=self._subentry.subentry_id
            )


class XAIChatTurnsSensor(SensorEntity):
    """Sensor reporting turn count for a user or device conversation flow."""

    _attr_has_entity_name = True
    _attr_should_poll = False
    _attr_entity_registry_enabled_default = True
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = "turns"

    _ICONS = {
        "user": "mdi:message-text-clock",
        "device": "mdi:tablet-dashboard",
    }

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
            key = MemoryManager.generate_key(
                self._scope, self._identifier, self._mode, self._subentry_id
            )
            try:
                self._attr_native_value = await memory.async_get_turn_count(key)
            except Exception:
                self._attr_native_value = 0

            if self._scope == "user":
                name = await async_get_user_display_name(self.hass, self._identifier)
            else:
                name = get_device_display_name(self.hass, self._identifier)

            if name and name != self._display_name:
                self._display_name = name
                self._attr_name = (
                    f"{self._display_name} - {self._subentry_title} ({self._mode})"
                )

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

        if self._scope == "user":
            name = await async_get_user_display_name(self.hass, self._identifier)
        else:
            name = get_device_display_name(self.hass, self._identifier)

        if name and name != self._display_name:
            self._display_name = name
            self._attr_name = (
                f"{self._display_name} - {self._subentry_title} ({self._mode})"
            )

        self.async_write_ha_state()
