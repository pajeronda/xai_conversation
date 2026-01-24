"""Statistics and Maintenance Services."""

from __future__ import annotations
from typing import TYPE_CHECKING

from ..const import DOMAIN, LOGGER, STATUS_OK, STATUS_ERROR
from ..exceptions import raise_validation_error
from ..xai_gateway import XAIGateway

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant, ServiceCall, ServiceResponse
    from homeassistant.config_entries import ConfigEntry
    from ..init_manager import XaiInitManager


class ManageSensorsService:
    """Service handler for manage_sensors - Proxy for sensor management operations."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry):
        """Initialize the service."""
        self.hass = hass
        self.entry = entry

    async def async_handle(self, call: ServiceCall) -> ServiceResponse:
        """Handle the manage_sensors service call."""
        reload_pricing = call.data.get("reload_pricing", False)
        reset_stats = call.data.get("reset_stats", False)

        if not reload_pricing and not reset_stats:
            raise_validation_error("Select at least one action to perform")

        results = []

        if reload_pricing:
            result = await self._reload_pricing()
            results.append(f"Pricing: {result.get('status', 'unknown')}")

        if reset_stats:
            result = await self._reset_stats()
            results.append(f"Stats: {result.get('status', 'unknown')}")

        return {
            "status": STATUS_OK,
            "message": " | ".join(results),
        }

    async def _reload_pricing(self) -> ServiceResponse:
        """Force refresh of model pricing data."""
        LOGGER.debug("manage_sensors: reload_pricing")

        try:
            coordinator: XaiInitManager | None = self.hass.data[DOMAIN].get(
                self.entry.entry_id
            )

            if coordinator:
                # Uses coordinator which includes cleanup after update
                await coordinator._async_update_models_with_retry(
                    max_retries=1, context="manual_reload"
                )
            else:
                # Fallback: direct gateway call (no cleanup possible)
                gateway = XAIGateway(self.hass, self.entry)
                await gateway.async_update_models()

            return {
                "status": STATUS_OK,
                "message": "Pricing data reloaded successfully",
            }

        except Exception as err:
            LOGGER.error("Failed to reload pricing: %s", err, exc_info=True)
            return {
                "status": STATUS_ERROR,
                "message": f"Failed to reload pricing: {err}",
            }

    async def _reset_stats(self) -> ServiceResponse:
        """Reset token statistics sensors."""
        LOGGER.debug("manage_sensors: reset_stats")

        storage = self.hass.data.get(DOMAIN, {}).get("token_stats")

        try:
            await storage.reset_stats()

            return {
                "status": STATUS_OK,
                "message": "Token statistics reset successfully",
            }

        except Exception as err:
            LOGGER.error("Failed to reset token stats: %s", err, exc_info=True)
            return {
                "status": STATUS_ERROR,
                "message": f"Failed to reset token stats: {err}",
            }
