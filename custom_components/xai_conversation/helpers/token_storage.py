"""Token statistics storage manager for xAI Conversation integration."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from ..const import LOGGER

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant
    from .. import XAIConfigEntry


class TokenStatsStorage:
    """Persistent storage for token statistics and pricing data.

    This class provides dedicated storage for sensor data separate from
    Home Assistant's RestoreEntity system, allowing for clean removal
    when the integration is uninstalled.

    Storage structure:
    {
        "token_stats": {
            "service_type": {
                "tokens_by_model": {...},
                "cumulative_completion_tokens": 0,
                "cumulative_prompt_tokens": 0,
                "cumulative_cached_tokens": 0,
                "cumulative_reasoning_tokens": 0,
                "message_count": 0,
                "last_completion_tokens": 0,
                "last_prompt_tokens": 0,
                "last_cached_tokens": 0,
                "last_reasoning_tokens": 0,
                "last_model": null,
                "last_timestamp": null,
                "reset_timestamp": null,
                "last_mode": null,  # conversation only
                "last_store_messages": null,  # conversation only
                "tokens_pipeline_server": {...},  # conversation cache ratio only
                "tokens_pipeline_client": {...},
                "tokens_tools_server": {...},
                "tokens_tools_client": {...}
            }
        },
        "pricing_data": {
            "model_name": {
                "input_price": 0.0,
                "output_price": 0.0,
                "cached_input_price": 0.0,
                "last_updated": timestamp
            }
        },
        "cost_data": {
            "total_cost": 0.0,
            "cost_by_model": {...},
            "model_pricing_breakdown": {...}
        },
        "new_models": {
            "detected_models": [],
            "acknowledged_models": [],
            "detected_at": null,
            "expires_at": null,
            "pricing": {}
        }
    }
    """

    def __init__(
        self, hass: HomeAssistant, storage_path: str, entry: XAIConfigEntry
    ) -> None:
        """Initialize token stats storage.

        Args:
            hass: Home Assistant instance
            storage_path: Path for storage file (e.g., "xai_conversation/token_stats")
            entry: Config entry
        """
        from homeassistant.helpers.storage import Store

        self.hass = hass
        self.entry = entry
        self._store = Store(hass, 1, storage_path)
        self._data: dict = {}
        self._loaded = False

    async def _ensure_loaded(self) -> None:
        """Ensure data is loaded from storage."""
        if self._loaded:
            return

        try:
            data = await self._store.async_load()
            if isinstance(data, dict):
                self._data = data
            else:
                self._data = {}
            self._loaded = True
            LOGGER.debug(
                "Loaded token stats storage: %d token_stats, %d pricing entries",
                len(self._data.get("token_stats", {})),
                len(self._data.get("pricing_data", {})),
            )
        except Exception as err:
            LOGGER.warning("Failed to load token stats storage: %s", err)
            self._data = {}
            self._loaded = True

    async def async_save(self) -> None:
        """Save data to storage."""
        await self._ensure_loaded()

        try:
            await self._store.async_save(self._data)
            LOGGER.debug("Saved token stats storage")
        except Exception as err:
            LOGGER.error("Failed to save token stats storage: %s", err)

    async def get_token_stats(self, service_type: str) -> dict:
        """Get token statistics for a service type.

        Args:
            service_type: "conversation", "ai_task", or "code_fast"

        Returns:
            Dictionary with token statistics, empty dict if not found
        """
        await self._ensure_loaded()
        return self._data.get("token_stats", {}).get(service_type, {})

    async def save_token_stats(self, service_type: str, stats: dict) -> None:
        """Save token statistics for a service type.

        Args:
            service_type: "conversation", "ai_task", or "code_fast"
            stats: Dictionary with token statistics
        """
        await self._ensure_loaded()
        if "token_stats" not in self._data:
            self._data["token_stats"] = {}
        self._data["token_stats"][service_type] = stats
        await self.async_save()

    async def get_aggregated_stats(self) -> dict:
        """Get aggregated statistics across all services.

        Returns:
            Dictionary with aggregated token statistics
        """
        await self._ensure_loaded()
        return self._data.get("token_stats", {}).get("aggregated", {})

    async def save_aggregated_stats(self, stats: dict) -> None:
        """Save aggregated statistics.

        Args:
            stats: Dictionary with aggregated token statistics
        """
        await self._ensure_loaded()
        if "token_stats" not in self._data:
            self._data["token_stats"] = {}
        self._data["token_stats"]["aggregated"] = stats
        await self.async_save()

    async def get_pricing(self, model_name: str, price_type: str) -> float | None:
        """Get pricing for a model and price type.

        Args:
            model_name: Model name (e.g., "grok-beta")
            price_type: "input_price", "output_price", or "cached_input_price"

        Returns:
            Price per million tokens, or None if not found
        """
        await self._ensure_loaded()
        pricing_data = self._data.get("pricing_data", {}).get(model_name, {})
        return pricing_data.get(price_type)

    async def save_pricing(
        self, model_name: str, price_type: str, price: float
    ) -> None:
        """Save pricing for a model and price type.

        Args:
            model_name: Model name
            price_type: "input_price", "output_price", or "cached_input_price"
            price: Price per million tokens
        """
        await self._ensure_loaded()
        if "pricing_data" not in self._data:
            self._data["pricing_data"] = {}
        if model_name not in self._data["pricing_data"]:
            self._data["pricing_data"][model_name] = {}

        self._data["pricing_data"][model_name][price_type] = price
        self._data["pricing_data"][model_name]["last_updated"] = time.time()
        await self.async_save()

    async def get_cost_data(self) -> dict:
        """Get cost data.

        Returns:
            Dictionary with cost data
        """
        await self._ensure_loaded()
        return self._data.get("cost_data", {})

    async def save_cost_data(self, cost_data: dict) -> None:
        """Save cost data.

        Args:
            cost_data: Dictionary with cost information
        """
        await self._ensure_loaded()
        self._data["cost_data"] = cost_data
        await self.async_save()

    async def get_new_models_data(self) -> dict:
        """Get new models detection data.

        Returns:
            Dictionary with new models data
        """
        await self._ensure_loaded()
        return self._data.get("new_models", {})

    async def save_new_models_data(self, new_models_data: dict) -> None:
        """Save new models detection data.

        Args:
            new_models_data: Dictionary with new models information
        """
        await self._ensure_loaded()
        self._data["new_models"] = new_models_data
        await self.async_save()

    async def clear_all(self) -> None:
        """Clear all stored data."""
        self._data = {}
        self._loaded = True
        await self.async_save()
        LOGGER.info("Cleared all token stats storage")

    async def async_remove(self) -> None:
        """Remove storage file completely."""
        try:
            await self._store.async_remove()
            self._data = {}
            self._loaded = False
            LOGGER.info("Removed token stats storage file")
        except Exception as err:
            LOGGER.error("Failed to remove token stats storage file: %s", err)
