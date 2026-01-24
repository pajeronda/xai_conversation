"""Base mixin for services."""

from __future__ import annotations
from typing import TYPE_CHECKING
from ..const import DOMAIN

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant
    from homeassistant.config_entries import ConfigEntry
    from ..xai_gateway import XAIGateway


class GatewayMixin:
    """Mixin providing shared gateway property for services that require entry."""

    hass: HomeAssistant
    entry: ConfigEntry

    @property
    def gateway(self) -> XAIGateway:
        """Get the shared gateway from the coordinator."""
        coordinator = self.hass.data[DOMAIN][self.entry.entry_id]
        return coordinator.gateway
