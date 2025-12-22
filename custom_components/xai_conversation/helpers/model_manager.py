"""Model management utilities for xAI conversation.

This module centralizes all logic related to fetching, parsing, and managing
xAI models and their pricing. It serves as the single source of truth for
model data.

Responsibilities:
1. Fetch model list and pricing from xAI API.
2. Calculate standardized pricing (USD per million tokens).
3. Periodically update model data during runtime.
4. Detect newly added models and notify the user.
5. Push pricing updates to TokenStats.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from homeassistant.core import HomeAssistant
from homeassistant.components import persistent_notification
from homeassistant.util import dt as dt_util

if TYPE_CHECKING:
    from ..xai_gateway import XAIGateway

from ..const import (
    CONF_XAI_PRICING_CONVERSION_FACTOR,
    LOGGER,
    DOMAIN,
    RECOMMENDED_IMAGE_MODEL,
    RECOMMENDED_XAI_PRICING_CONVERSION_FACTOR,
    SUPPORTED_MODELS,
    REASONING_EFFORT_MODELS,
)


class XAIModelManager:
    """Central manager for xAI models and pricing data.

    This class is responsible for:
    - Interacting with the xAI API to retrieve model information.
    - normalizing pricing data.
    - Detecting changes in available models (new releases).
    - Notifying the user of new models via persistent notifications.
    - Updating the central TokenStats pricing database.
    """

    def __init__(self, hass: HomeAssistant):
        """Initialize the model manager."""
        self.hass = hass

    def _get_pricing_conversion_factor(self) -> float:
        """Get pricing conversion factor from sensors subentry config."""
        # Search for sensors subentry in all config entries
        for entry in self.hass.config_entries.async_entries(DOMAIN):
            for subentry in entry.subentries.values():
                if subentry.subentry_type == "sensors":
                    return subentry.data.get(
                        CONF_XAI_PRICING_CONVERSION_FACTOR,
                        RECOMMENDED_XAI_PRICING_CONVERSION_FACTOR,
                    )
        return RECOMMENDED_XAI_PRICING_CONVERSION_FACTOR

    def _build_model_data_entry(
        self,
        model: Any,
        model_type: str,
        conversion_factor: float,
        price_key: str | None = "prompt_text_token_price",
        image_price_key: str | None = None,
    ) -> dict:
        """Helper to build a single model data entry."""
        entry = {
            "name": model.name,
            "type": model_type,
            "input_price_per_million": 0.0,
            "output_price_per_million": 0.0,
            "cached_input_price_per_million": 0.0,
            "context_window": getattr(model, "context_window", 0),
            "aliases": getattr(model, "aliases", []),
        }

        if image_price_key:
            raw_price = getattr(model, image_price_key, 0.0)
            if raw_price is None:
                raw_price = 0.0
            entry["output_price_per_million"] = raw_price / conversion_factor
        else:
            if price_key:
                val = getattr(model, price_key, 0.0)
                if val is None:
                    val = 0.0
                entry["input_price_per_million"] = val / conversion_factor

            completion_price = getattr(model, "completion_text_token_price", 0.0)
            if completion_price is None:
                completion_price = 0.0
            entry["output_price_per_million"] = completion_price / conversion_factor

            cached_price = getattr(model, "cached_prompt_token_price", 0.0)
            if cached_price is None:
                cached_price = 0.0
            entry["cached_input_price_per_million"] = cached_price / conversion_factor

        return entry

    async def async_get_models_data(self, gateway: XAIGateway) -> dict[str, Any] | None:
        """Fetch available xAI models and their pricing asynchronously.

        This method handles:
        1. Connecting to xAI API via the provided Gateway
        2. Fetching language, image, and embedding models
        3. Applying fallback logic for known issues (e.g., missing image models)
        4. Converting raw API pricing to USD per million tokens using standardized factors
        5. Populating global SUPPORTED_MODELS and REASONING_EFFORT_MODELS lists

        Args:
            gateway: The XAIGateway instance to use for API communication.

        Returns:
            A dictionary where keys are model names and values are their data (including pricing),
            or None if fetching fails.
        """
        try:
            # Use gateway to create client (handles auth, host, and SDK availability)
            client = gateway.create_client()

            # Get conversion factor from config (cached for this call)
            conversion_factor = self._get_pricing_conversion_factor()

            models_data = {}

            # --- 1. Fetch Language Models ---
            language_models = await client.models.list_language_models()
            LOGGER.debug("Fetched %d language models", len(language_models))

            for model in language_models:
                entry = self._build_model_data_entry(
                    model, "language", conversion_factor
                )
                models_data[model.name] = entry
                for alias in entry["aliases"]:
                    models_data[alias] = entry

            # --- 2. Fetch Image Generation Models ---
            image_models = await client.models.list_image_generation_models()
            LOGGER.debug("Fetched %d image models", len(image_models))

            # Fallback Logic: If API returns empty list, manually add the known image model
            if not image_models:
                LOGGER.info(
                    "No image models returned by API. Adding fallback model: %s",
                    RECOMMENDED_IMAGE_MODEL,
                )
                # Map image price ($0.07 per image = 1 token)
                # Note: We use 70000.0 because cost calculation divides tokens by 1,000,000.
                # Price per million tokens = $0.07 * 1,000,000 = $70,000
                fallback_data = {
                    "name": RECOMMENDED_IMAGE_MODEL,
                    "type": "image",
                    "input_price_per_million": 0.0,
                    "output_price_per_million": 70000.0,
                    "cached_input_price_per_million": 0.0,
                    "context_window": 0,
                    "aliases": [],
                }
                models_data[RECOMMENDED_IMAGE_MODEL] = fallback_data

            for model in image_models:
                entry = self._build_model_data_entry(
                    model, "image", conversion_factor, image_price_key="image_price"
                )

                # Fix: If API returns the model but with 0 price, force our $0.07/image fallback
                if (
                    entry["name"] == RECOMMENDED_IMAGE_MODEL
                    and entry["output_price_per_million"] == 0
                ):
                    entry["output_price_per_million"] = 70000.0

                models_data[model.name] = entry
                for alias in entry["aliases"]:
                    models_data[alias] = entry

            # --- 3. Fetch Embedding Models ---
            embedding_models = await client.models.list_embedding_models()
            LOGGER.debug("Fetched %d embedding models", len(embedding_models))

            for model in embedding_models:
                entry = self._build_model_data_entry(
                    model,
                    "embedding",
                    conversion_factor,
                    price_key="prompt_text_token_price",
                )
                # Embedding models typically don't have completion_text_token_price
                entry["output_price_per_million"] = 0.0
                models_data[model.name] = entry
                for alias in entry["aliases"]:
                    models_data[alias] = entry

            # --- 4. Populate global SUPPORTED_MODELS and REASONING_EFFORT_MODELS ---
            self._populate_supported_models(models_data)

            return models_data

        except Exception as exc:
            LOGGER.error("Failed to fetch xAI model data: %s", exc)
            return None

    async def async_update_models(self, gateway: XAIGateway) -> None:
        """Update model data, detect new models, and update pricing.

        This method is called both at startup and periodically. It:
        1. Fetches fresh data from API using the provided Gateway.
        2. Compares with currently stored known models from TokenStats.
        3. Identifies truly new models (not just new to current runtime session).
        4. Sends notifications for truly new models.
        5. Updates hass.data and TokenStats pricing.
        6. Updates TokenStats's persistent list of known models.

        Args:
            gateway: The XAIGateway instance to use for API communication.
        """
        LOGGER.debug("Starting model update check...")

        # 1. Get TokenStats instance
        token_stats = self.hass.data.get(DOMAIN, {}).get("token_stats")
        if not token_stats:
            LOGGER.warning(
                "TokenStats not available during model update, skipping new model detection."
            )
            # Continue with basic update even if TokenStats is not available
            new_data = await self.async_get_models_data(gateway)
            if new_data:
                self.hass.data[DOMAIN]["xai_models_data"] = new_data
                self.hass.data[DOMAIN]["xai_models_data_timestamp"] = (
                    dt_util.now().timestamp()
                )
                LOGGER.debug(
                    "Updated hass.data.xai_models_data (no TokenStats for pricing/known models)."
                )
            return

        new_data = await self.async_get_models_data(gateway)

        if not new_data:
            LOGGER.warning("Model update failed: could not fetch data.")
            return

        # 2. Get known models from TokenStats (persistent memory)
        known_models = await token_stats.get_known_models()
        known_models_set = set(known_models)

        # Extract only primary model names (exclude aliases) for known_models comparison
        # This ensures consistency with SUPPORTED_MODELS which also excludes aliases
        primary_models_set = {
            model_name
            for model_name, model_data in new_data.items()
            if model_data["name"] == model_name  # Only primary names, not aliases
        }

        # 3. Detect truly new models (not in known_models)
        truly_new_models = sorted(list(primary_models_set - known_models_set))

        if truly_new_models:
            LOGGER.info("Detected truly new xAI models: %s", truly_new_models)
            self._notify_new_models(truly_new_models)
            await token_stats.add_known_models(
                truly_new_models
            )  # Add to persistent known_models
        else:
            LOGGER.debug("No truly new models detected.")

        # 4. Update hass.data with fresh data
        self.hass.data[DOMAIN]["xai_models_data"] = new_data
        self.hass.data[DOMAIN]["xai_models_data_timestamp"] = dt_util.now().timestamp()

        # 5. Update TokenStats pricing
        await self._update_token_stats_pricing(new_data)

        # 6. Ensure all primary models are in known_models for next comparison
        # This covers cases where some models might be fetched, but not marked as 'new'
        # (e.g., if known_models was empty initially)
        # Note: We only store primary model names, not aliases, for consistency with SUPPORTED_MODELS
        await token_stats.add_known_models(list(primary_models_set))

        LOGGER.debug("Model update completed successfully.")

    def _notify_new_models(self, new_models: list[str]) -> None:
        """Send a persistent notification for new models.

        Args:
            new_models: List of new model names.
        """
        # Filter out likely aliases to reduce noise if desired,
        # but reporting everything is safer for now.
        message = (
            "New models detected from xAI API:\n\n"
            + "\n".join([f"- {m}" for m in new_models])
            + "\n\nPlease check the integration options if you wish to use them."
        )

        persistent_notification.async_create(
            self.hass,
            message,
            "New xAI Models Detected",
            f"{DOMAIN}_new_models",
        )

    async def _update_token_stats_pricing(self, models_data: dict[str, Any]) -> None:
        """Push updated pricing data to TokenStats using bulk update.

        Args:
            models_data: The fresh model data dictionary.
        """
        token_stats = self.hass.data.get(DOMAIN, {}).get("token_stats")
        if not token_stats:
            LOGGER.warning("TokenStats not available during model update.")
            return

        pricing_updates = {}

        for model_name, model_data in models_data.items():
            try:
                prices = {}
                if model_data.get("input_price_per_million") is not None:
                    prices["input_price"] = model_data["input_price_per_million"]

                if model_data.get("output_price_per_million") is not None:
                    prices["output_price"] = model_data["output_price_per_million"]

                if model_data.get("cached_input_price_per_million") is not None:
                    prices["cached_input_price"] = model_data[
                        "cached_input_price_per_million"
                    ]

                if prices:
                    pricing_updates[model_name] = prices

            except Exception as err:
                LOGGER.error("Error parsing pricing for %s: %s", model_name, err)

        if pricing_updates:
            try:
                await token_stats.save_pricing_batch(pricing_updates)
                LOGGER.debug(
                    "Updated pricing for %d models in TokenStats (Batch)",
                    len(pricing_updates),
                )
            except Exception as err:
                LOGGER.error("Failed to batch update pricing in TokenStats: %s", err)

    def _populate_supported_models(self, models_data: dict[str, Any]) -> None:
        """Populate global SUPPORTED_MODELS and REASONING_EFFORT_MODELS from fetched data.

        Args:
            models_data: Dictionary of model data from async_get_models_data
        """
        SUPPORTED_MODELS.clear()
        REASONING_EFFORT_MODELS.clear()

        dynamic_supported = set()
        dynamic_reasoning = set()

        for model_name, model_data in models_data.items():
            # Only add primary model name, not aliases
            if model_data["name"] == model_name:
                # Add language and image models to SUPPORTED_MODELS
                if model_data["type"] in ["language", "image"]:
                    dynamic_supported.add(model_name)
                    # Hardcoded reasoning effort models (until SDK provides a flag)
                    if model_name in ["grok-3", "grok-3-mini"]:
                        dynamic_reasoning.add(model_name)

        # Sort for consistency
        SUPPORTED_MODELS.extend(sorted(list(dynamic_supported)))
        REASONING_EFFORT_MODELS.extend(sorted(list(dynamic_reasoning)))

        LOGGER.debug("Populated SUPPORTED_MODELS: %s", SUPPORTED_MODELS)
        LOGGER.debug("Populated REASONING_EFFORT_MODELS: %s", REASONING_EFFORT_MODELS)
