"""Configuration validation utilities for xAI conversation."""
from __future__ import annotations

from ..const import (
    CONF_TIMEOUT,
    LOGGER,
    RECOMMENDED_TIMEOUT,
    SUPPORTED_MODELS,
)
from ..exceptions import raise_config_error


def validate_xai_configuration(entry, get_option_func, model: str) -> None:
    """Validate xAI client configuration and raise errors for invalid settings."""

    # Validate API key
    api_key = entry.data.get("api_key")
    if not api_key:
        LOGGER.error("Configuration validation failed: API key is missing")
        raise_config_error("API key", "not configured")

    # Validate model using SUPPORTED_MODELS from const.py
    if model not in SUPPORTED_MODELS:
        LOGGER.warning("Unknown model '%s', proceeding anyway (supported: %s)", model, SUPPORTED_MODELS)

    # Validate timeout
    timeout = get_option_func(CONF_TIMEOUT, RECOMMENDED_TIMEOUT)
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        LOGGER.warning("Invalid timeout value '%s' (type: %s), using default %s", timeout, type(timeout).__name__, RECOMMENDED_TIMEOUT)
