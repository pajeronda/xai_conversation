"""Model management utilities for xAI conversation.

Centralizes fetching, parsing, and managing xAI models and their pricing.
Functions as the single source of truth for model data.
"""

from __future__ import annotations

from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.components import persistent_notification
from homeassistant.util import dt as dt_util

from ..const import (
    LOGGER,
    DOMAIN,
    SUPPORTED_MODELS,
    REASONING_EFFORT_MODELS,
)


class XAIModelManager:
    """Central manager for xAI models and pricing data.

    Handles API interactions, pricing normalization, fresh model detection,
    and updates to the central TokenStats pricing database.
    """

    def __init__(self, hass: HomeAssistant):
        """Initialize the model manager."""
        self.hass = hass

    def _build_model_data_entry(
        self,
        model: Any,
        model_type: str,
        price_key: str | None = "prompt_text_token_price",
        image_price_key: str | None = None,
    ) -> dict:
        """Helper to build a single model data entry."""
        model_name_lower = model.name.lower()
        # supports_vision = (
        #     model_type == "image"
        #     or "vision" in model_name_lower
        #     or "image" in model_name_lower
        # )
        supports_search = "search" in model_name_lower or model_type == "language"

        # Initialize base prices (UNIFIED NAMING FOR DIRECT SYNC)
        prices = {
            "input_price": 0.0,
            "output_price": 0.0,
            "cached_input_price": 0.0,
            "input_image_price": 0.0,
            "search_price": 0.0,
            "type": model_type,
        }

        if supports_search:
            prices["search_price"] = getattr(model, "search_price", 0.0)

        # NOTE: xAI currently uses unified input pricing for vision models ($2.00/1M tokens
        # for both text and image tokens). The API exposes separate attributes
        # (prompt_text_token_price, prompt_image_token_price) but they have the same value.
        # If xAI introduces differentiated pricing in the future, uncomment the following:
        # if supports_vision:
        #     prices["input_image_price"] = getattr(
        #         model, "prompt_image_token_price", 0.0
        #     )

        if image_price_key:
            prices["output_price"] = getattr(model, image_price_key, 0.0)
            # Capture input prices for image generation models (e.g. grok-imagine-image)
            prices["input_price"] = getattr(model, "prompt_text_token_price", 0.0)
            prices["input_image_price"] = getattr(
                model, "prompt_image_token_price", 0.0
            )
        else:
            if price_key:
                prices["input_price"] = getattr(model, price_key, 0.0)
            prices["output_price"] = getattr(model, "completion_text_token_price", 0.0)
            prices["cached_input_price"] = getattr(
                model, "cached_prompt_token_price", 0.0
            )

        return {
            "name": model.name,
            **prices,
            "context_window": getattr(model, "max_prompt_length", 0)
            or getattr(model, "context_window", 0),
            "aliases": list(getattr(model, "aliases", [])),
        }

    async def async_update_models(self, client: Any) -> None:
        """Fetch and sync available models from xAI."""
        try:
            models_data = {}

            # 1. Fetch all model types
            fetch_configs = [
                (client.models.list_language_models, "language", None),
                (client.models.list_image_generation_models, "image", "image_price"),
                (client.models.list_embedding_models, "embedding", None),
            ]

            for fetch_func, m_type, im_p_key in fetch_configs:
                models = await fetch_func()
                for m in models:
                    entry = self._build_model_data_entry(
                        m, m_type, image_price_key=im_p_key
                    )
                    if m_type == "embedding":
                        entry["output_price"] = 0.0

                    models_data[m.name] = entry
                    for alias in entry.get("aliases", []):
                        models_data[alias] = entry

            # 2. Update global lists
            self._populate_supported_models(models_data)

            # 3. Handle persistent storage and new model detection
            token_stats = self.hass.data.get(DOMAIN, {}).get("token_stats")
            if token_stats:
                known = await token_stats.get_known_models()
                known_set = set(known)

                # Identify new primary models
                primary_names = {n for n, d in models_data.items() if d["name"] == n}
                new_models = sorted(list(primary_names - known_set))

                if new_models:
                    LOGGER.info("[models] new models detected: %s", new_models)
                    self._notify_new_models(new_models)
                    await token_stats.add_known_models(new_models)

                # Identify deprecated models (known but disappeared from API)
                deprecated_models = sorted(list(known_set - primary_names))
                if deprecated_models:
                    LOGGER.info(
                        "[models] models decommissioned by xAI: %s", deprecated_models
                    )
                    self._notify_deprecated_models(deprecated_models)
                    # Sync known models to reflect current API reality
                    await token_stats.set_known_models(list(primary_names))

                # Bulk update pricing in TokenStats
                await token_stats.save_pricing_batch(models_data)

                # Prune obsolete pricing for discontinued models (including aliases)
                all_valid_names = list(models_data.keys())
                await token_stats.prune_pricing(all_valid_names)

            # 4. Update hass.data
            self.hass.data[DOMAIN]["xai_models_data"] = models_data
            self.hass.data[DOMAIN]["xai_models_data_timestamp"] = (
                dt_util.now().timestamp()
            )

            LOGGER.debug("[models] update complete: %d entries", len(models_data))

        except Exception as exc:
            LOGGER.error("[models] update failed: %s", exc, exc_info=True)

    def _notify_new_models(self, new_models: list[str]) -> None:
        """Send a persistent notification for new models."""
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

    def _notify_deprecated_models(self, deprecated_models: list[str]) -> None:
        """Send a persistent notification for decommissioned models."""
        message = (
            "The following xAI models are no longer available in the API and have been decommissioned:\n\n"
            + "\n".join([f"- {m}" for m in deprecated_models])
            + "\n\nAssociated pricing sensors will be removed. Please update your configurations if these models were in use."
        )

        persistent_notification.async_create(
            self.hass,
            message,
            "xAI Models Decommissioned",
            f"{DOMAIN}_deprecated_models",
        )

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
                m_type = model_data["type"]
                # Add language and image models to SUPPORTED_MODELS
                if m_type in ["language", "image"]:
                    dynamic_supported.add(model_name)

                    # Dynamic Reasoning Detection Strategy:
                    # 1. Models starting with 'grok-3' (Grok-3 is reasoning-first)
                    # 2. Models containing '-reasoning'
                    # 3. EXCLUDE models containing '-non-reasoning'
                    name_lower = model_name.lower()
                    # Official documentation: only grok-3-mini supports reasoning_effort
                    if "grok-3-mini" in name_lower:
                        dynamic_reasoning.add(model_name)

        # Sort for consistency
        SUPPORTED_MODELS.extend(sorted(list(dynamic_supported)))
        REASONING_EFFORT_MODELS.extend(sorted(list(dynamic_reasoning)))

        LOGGER.debug(
            "[models] loaded: %d supported, %d reasoning",
            len(SUPPORTED_MODELS),
            len(REASONING_EFFORT_MODELS),
        )
