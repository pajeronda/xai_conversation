"""Model management utilities for xAI conversation.

This module centralizes all logic related to fetching, parsing, and managing
xAI models and their pricing. It serves as the single source of truth for
model data, used by both the configuration flow (startup) and the gateway
(runtime sensors).
"""

from __future__ import annotations

from typing import Any

from ..const import (
    LOGGER,
    RECOMMENDED_TIMEOUT,
    RECOMMENDED_IMAGE_MODEL,
    XAI_PRICING_CONVERSION_FACTOR,
)


class XAIModelManager:
    """Manager for xAI models and pricing data."""

    def __init__(self, hass):
        """Initialize the model manager."""
        self.hass = hass

    async def async_get_models_data(self, api_key: str) -> dict[str, Any] | None:
        """Fetch available xAI models and their pricing asynchronously.

        This method handles:
        1. Connecting to xAI API
        2. Fetching language, image, and embedding models
        3. Applying fallback logic for known issues (e.g., missing image models)
        4. Converting raw API pricing to USD per million tokens using standardized factors
        5. Populating global SUPPORTED_MODELS and REASONING_EFFORT_MODELS lists

        Args:
            api_key: The xAI API key.

        Returns:
            A dictionary where keys are model names and values are their data (including pricing),
            or None if fetching fails.
        """
        try:
            # Use gateway to create client (handles SDK availability check)
            from .xai_gateway import XAIGateway

            client = XAIGateway.create_standalone_client(
                api_key=api_key, timeout=float(RECOMMENDED_TIMEOUT)
            )

            models_data = {}

            # --- 1. Fetch Language Models ---
            language_models = await client.models.list_language_models()
            LOGGER.debug("Fetched %d language models", len(language_models))

            for model in language_models:
                # API returns prices in units of 0.0001 USD per 1M tokens.
                # We divide by XAI_PRICING_CONVERSION_FACTOR (10000.0) to get USD per 1M tokens.
                models_data[model.name] = {
                    "name": model.name,
                    "type": "language",
                    "input_price_per_million": getattr(
                        model, "prompt_text_token_price", 0.0
                    )
                    / XAI_PRICING_CONVERSION_FACTOR,
                    "output_price_per_million": getattr(
                        model, "completion_text_token_price", 0.0
                    )
                    / XAI_PRICING_CONVERSION_FACTOR,
                    "cached_input_price_per_million": getattr(
                        model, "cached_prompt_token_price", 0.0
                    )
                    / XAI_PRICING_CONVERSION_FACTOR,
                    "context_window": getattr(model, "context_window", 0),
                    "aliases": getattr(model, "aliases", []),
                }
                # Register aliases pointing to the same data
                for alias in models_data[model.name]["aliases"]:
                    models_data[alias] = models_data[model.name]

            # --- 2. Fetch Image Generation Models ---
            image_models = await client.models.list_image_generation_models()
            LOGGER.debug("Fetched %d image models", len(image_models))

            # Fallback Logic: If API returns empty list, manually add the known image model
            if not image_models:
                LOGGER.info(
                    "No image models returned by API. Adding fallback model: %s",
                    RECOMMENDED_IMAGE_MODEL,
                )
                # Map image price ($0.07) directly (no per-million conversion for images)
                fallback_data = {
                    "name": RECOMMENDED_IMAGE_MODEL,
                    "type": "image",
                    "input_price_per_million": 0.0,
                    "output_price_per_million": 0.07,
                    "cached_input_price_per_million": 0.0,
                    "context_window": 0,
                    "aliases": [],
                }
                models_data[RECOMMENDED_IMAGE_MODEL] = fallback_data

            for model in image_models:
                # Image models return 'image_price' (e.g., 700 for $0.07).
                # Convert raw price to USD: raw / 10000.0
                raw_price = getattr(model, "image_price", 0.0)
                price_usd = raw_price / XAI_PRICING_CONVERSION_FACTOR

                models_data[model.name] = {
                    "name": model.name,
                    "type": "image",
                    "input_price_per_million": 0.0,
                    "output_price_per_million": price_usd,
                    "cached_input_price_per_million": 0.0,
                    "context_window": 0,
                    "aliases": getattr(model, "aliases", []),
                }
                for alias in models_data[model.name]["aliases"]:
                    models_data[alias] = models_data[model.name]

            # --- 3. Fetch Embedding Models ---
            embedding_models = await client.models.list_embedding_models()
            LOGGER.debug("Fetched %d embedding models", len(embedding_models))

            for model in embedding_models:
                models_data[model.name] = {
                    "name": model.name,
                    "type": "embedding",
                    "input_price_per_million": getattr(
                        model, "prompt_text_token_price", 0.0
                    )
                    / XAI_PRICING_CONVERSION_FACTOR,
                    "output_price_per_million": 0.0,
                    "cached_input_price_per_million": getattr(
                        model, "cached_prompt_token_price", 0.0
                    )
                    / XAI_PRICING_CONVERSION_FACTOR,
                    "aliases": getattr(model, "aliases", []),
                }
                for alias in models_data[model.name]["aliases"]:
                    models_data[alias] = models_data[model.name]

            # --- 4. Populate global SUPPORTED_MODELS and REASONING_EFFORT_MODELS ---
            self._populate_supported_models(models_data)

            return models_data

        except Exception as exc:
            LOGGER.error("Failed to fetch xAI model data: %s", exc)
            return None

    def _populate_supported_models(self, models_data: dict[str, Any]) -> None:
        """Populate global SUPPORTED_MODELS and REASONING_EFFORT_MODELS from fetched data.

        Args:
            models_data: Dictionary of model data from async_get_models_data
        """
        from ..const import SUPPORTED_MODELS, REASONING_EFFORT_MODELS

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
