"""Service Registry and Management."""

from __future__ import annotations
from typing import Any

from homeassistant.core import HomeAssistant, SupportsResponse
from homeassistant.config_entries import ConfigEntry

from ..const import (
    DOMAIN,
)
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


def register_services(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Register all xAI Conversation services.

    Dynamically registers services based on SERVICES_METADATA.
    Handles dependencies (some services need entry, others just hass).
    """
    for service_name, service_cls, requires_entry in SERVICES_METADATA:
        # Instantiate service handler
        if requires_entry:
            handler = service_cls(hass, entry)
        else:
            handler = service_cls(hass)

        # Register service
        hass.services.async_register(
            DOMAIN,
            service_name,
            handler.async_handle,
            supports_response=SupportsResponse.OPTIONAL,
        )


def unregister_services(hass: HomeAssistant) -> None:
    """Unregister all xAI Conversation services."""
    for service_name, _, _ in SERVICES_METADATA:
        hass.services.async_remove(DOMAIN, service_name)
