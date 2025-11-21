"""Configuration validation utilities for xAI conversation."""
from __future__ import annotations


async def async_validate_api_key(hass, api_key: str) -> None:
    """Validate the API key by making a minimal async chat request."""
    from ..const import RECOMMENDED_CHAT_MODEL, RECOMMENDED_TIMEOUT
    from .. import XAI_SDK_AVAILABLE, XAI_CLIENT_CLASS, xai_user

    if not XAI_SDK_AVAILABLE or XAI_CLIENT_CLASS is None:
        raise ValueError("xAI SDK not installed")

    try:
        # We are in a proper async function, so no more executor job.
        client = XAI_CLIENT_CLASS(api_key=api_key, timeout=float(RECOMMENDED_TIMEOUT))
        
        # These methods are synchronous setup calls
        chat = client.chat.create(model=RECOMMENDED_CHAT_MODEL, max_tokens=1, temperature=0.1)
        chat.append(xai_user("ok"))
        
        # This is the network call that must be awaited.
        await chat.sample()

    except ImportError:
        # This is a fallback, but the check above should catch it.
        raise ValueError("xAI SDK not installed")
    except Exception as exc:
        # Any other exception (invalid key, network error) is caught here.
        raise ValueError(f"Failed to validate API credentials: {exc}") from exc


from ..const import (
    CONF_TIMEOUT,
    LOGGER,
    RECOMMENDED_TIMEOUT,
    SUPPORTED_MODELS,
)
from ..exceptions import raise_config_error


def validate_xai_configuration(entry, get_option_func, model: str) -> None:
    """Validate xAI client configuration and raise errors for invalid settings.

    Note: API key validation is performed separately via async_validate_api_key()
    during integration setup and config flow.
    """

    # Validate model using SUPPORTED_MODELS from const.py
    if model not in SUPPORTED_MODELS:
        LOGGER.warning("Unknown model '%s', proceeding anyway (supported: %s)", model, SUPPORTED_MODELS)

    # Validate timeout
    timeout = get_option_func(CONF_TIMEOUT, RECOMMENDED_TIMEOUT)
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        LOGGER.warning("Invalid timeout value '%s' (type: %s), using default %s", timeout, type(timeout).__name__, RECOMMENDED_TIMEOUT)

async def async_get_xai_models_data(hass, api_key: str) -> dict[str, Any] | None:
    """Fetch available xAI models and their pricing asynchronously.

    Args:
        hass: Home Assistant instance.
        api_key: The xAI API key.

    Returns:
        A dictionary where keys are model names and values are their data (including pricing),
        or None if fetching fails.
    """
    from ..const import LOGGER, RECOMMENDED_TIMEOUT
    from .. import XAI_SDK_AVAILABLE, XAI_CLIENT_CLASS

    if not XAI_SDK_AVAILABLE or XAI_CLIENT_CLASS is None:
        LOGGER.error("xAI SDK not installed, cannot fetch model data.")
        return None

    try:
        client = XAI_CLIENT_CLASS(api_key=api_key, timeout=float(RECOMMENDED_TIMEOUT))
        
        models_data = {}

        # Fetch language models
        language_models = await client.models.list_language_models()
        for model in language_models:
            # NOTE: xAI API returns prices in an internal unit (e.g., 500000 for $5.00 per million tokens)
            # We divide by 100000.0 to convert to USD per million tokens
            # Example: API returns 500000 → 500000 / 100000 = 5.0 USD per million tokens
            models_data[model.name] = {
                "name": model.name,
                "type": "language",
                "input_price_per_million": getattr(model, "prompt_text_token_price", 0.0) / 100000.0,
                "output_price_per_million": getattr(model, "completion_text_token_price", 0.0) / 100000.0,
                "cached_input_price_per_million": getattr(model, "cached_prompt_token_price", 0.0) / 100000.0,
                "aliases": getattr(model, "aliases", []),
            }
            for alias in models_data[model.name]["aliases"]:
                models_data[alias] = models_data[model.name]

        # Fetch image generation models
        image_models = await client.models.list_image_generation_models()
        for model in image_models:
            models_data[model.name] = {
                "name": model.name,
                "type": "image",
                "input_price_per_million": 0.0, # Image models typically have a single image price
                "output_price_per_million": getattr(model, "image_price", 0.0),
                "cached_input_price_per_million": 0.0,
                "aliases": getattr(model, "aliases", []),
            }
            for alias in models_data[model.name]["aliases"]:
                models_data[alias] = models_data[model.name]

        # Fetch embedding models
        embedding_models = await client.models.list_embedding_models()
        for model in embedding_models:
            models_data[model.name] = {
                "name": model.name,
                "type": "embedding",
                "input_price_per_million": getattr(model, "prompt_text_token_price", 0.0) / 100000.0,
                "output_price_per_million": 0.0, # Embedding models usually only have input price
                "cached_input_price_per_million": getattr(model, "cached_prompt_token_price", 0.0) / 100000.0,
                "aliases": getattr(model, "aliases", []),
            }
            for alias in models_data[model.name]["aliases"]:
                models_data[alias] = models_data[model.name]
        
        return models_data

    except Exception as exc:
        LOGGER.error("Failed to fetch xAI model data: %s", exc)
        return None
