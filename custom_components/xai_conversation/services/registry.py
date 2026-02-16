"""Service Registry and Management."""

from __future__ import annotations
from typing import Any

from homeassistant.core import HomeAssistant, SupportsResponse
from homeassistant.config_entries import ConfigEntry, ConfigEntryState

from ..const import (
    DOMAIN,
)
from ..exceptions import raise_generic_error
from .ask import AskService
from .image import PhotoAnalysisService
from .memory import (
    ClearMemoryService,
)
from .stats import ManageSensorsService

# Tuple format: (Service Name, Service Class, Requires ConfigEntry)
SERVICES_METADATA: list[tuple[str, Any, bool]] = [
    ("ask", AskService, True),
    ("photo_analysis", PhotoAnalysisService, True),
    ("clear_memory", ClearMemoryService, True),
    ("manage_sensors", ManageSensorsService, True),
]

_SERVICE_REGISTRY_KEY = "service_registry"
_SERVICE_REFCOUNT_KEY = "refcount"
_SERVICE_PRIMARY_ENTRY_KEY = "primary_entry_id"


def _get_registry(hass: HomeAssistant) -> dict[str, Any]:
    domain_data = hass.data.setdefault(DOMAIN, {})
    registry = domain_data.setdefault(
        _SERVICE_REGISTRY_KEY,
        {_SERVICE_REFCOUNT_KEY: 0, _SERVICE_PRIMARY_ENTRY_KEY: None},
    )
    return registry


def _get_loaded_entry_ids(hass: HomeAssistant) -> list[str]:
    entries = hass.config_entries.async_entries(DOMAIN)
    return [e.entry_id for e in entries if e.state == ConfigEntryState.LOADED]


def _resolve_primary_entry_id(hass: HomeAssistant) -> str | None:
    registry = _get_registry(hass)
    entry_id = registry.get(_SERVICE_PRIMARY_ENTRY_KEY)
    if entry_id and entry_id in _get_loaded_entry_ids(hass):
        return entry_id

    loaded_entry_ids = _get_loaded_entry_ids(hass)
    if loaded_entry_ids:
        entry_id = loaded_entry_ids[0]
        registry[_SERVICE_PRIMARY_ENTRY_KEY] = entry_id
        return entry_id

    registry[_SERVICE_PRIMARY_ENTRY_KEY] = None
    return None


def _build_service_handler(hass: HomeAssistant, service_cls: Any, requires_entry: bool):
    async def _handle(call):
        if requires_entry:
            entry_id = _resolve_primary_entry_id(hass)
            if not entry_id:
                raise_generic_error(
                    "No loaded xAI Conversation entry is available for this service."
                )
            entry = hass.config_entries.async_get_entry(entry_id)
            if not entry:
                raise_generic_error(
                    "xAI Conversation entry not found while handling service."
                )
            handler = service_cls(hass, entry)
        else:
            handler = service_cls(hass)

        return await handler.async_handle(call)

    return _handle


def register_services(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Register all xAI Conversation services.

    Dynamically registers services based on SERVICES_METADATA.
    Handles dependencies (some services need entry, others just hass).
    """
    registry = _get_registry(hass)
    if registry[_SERVICE_REFCOUNT_KEY] == 0:
        for service_name, service_cls, requires_entry in SERVICES_METADATA:
            handler = _build_service_handler(hass, service_cls, requires_entry)
            hass.services.async_register(
                DOMAIN,
                service_name,
                handler,
                supports_response=SupportsResponse.OPTIONAL,
            )
        registry[_SERVICE_PRIMARY_ENTRY_KEY] = entry.entry_id
    elif registry.get(_SERVICE_PRIMARY_ENTRY_KEY) is None:
        registry[_SERVICE_PRIMARY_ENTRY_KEY] = entry.entry_id

    registry[_SERVICE_REFCOUNT_KEY] += 1


def unregister_services(hass: HomeAssistant, entry_id: str | None = None) -> None:
    """Unregister all xAI Conversation services."""
    registry = _get_registry(hass)
    if registry[_SERVICE_REFCOUNT_KEY] <= 0:
        registry[_SERVICE_REFCOUNT_KEY] = 0
        return

    registry[_SERVICE_REFCOUNT_KEY] -= 1

    if entry_id and registry.get(_SERVICE_PRIMARY_ENTRY_KEY) == entry_id:
        loaded_entry_ids = [e for e in _get_loaded_entry_ids(hass) if e != entry_id]
        registry[_SERVICE_PRIMARY_ENTRY_KEY] = (
            loaded_entry_ids[0] if loaded_entry_ids else None
        )

    if registry[_SERVICE_REFCOUNT_KEY] == 0:
        for service_name, _, _ in SERVICES_METADATA:
            hass.services.async_remove(DOMAIN, service_name)
        registry[_SERVICE_PRIMARY_ENTRY_KEY] = None
