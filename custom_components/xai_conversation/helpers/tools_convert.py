"""Tool conversion utilities between HA and xAI formats."""
from __future__ import annotations

import json
from typing import Any

from homeassistant.core import HomeAssistant as HA_HomeAssistant
from homeassistant.helpers import llm as ha_llm

from ..const import LOGGER


def _safe_parse_tool_arguments(raw_args) -> dict:
    """Parse tool arguments safely with comprehensive error handling."""
    if isinstance(raw_args, dict):
        LOGGER.debug("Arguments already dict: %s", raw_args)
        return raw_args

    if isinstance(raw_args, str):
        if not raw_args.strip():
            LOGGER.debug("Empty string arguments, returning empty dict")
            return {}

        try:
            parsed = json.loads(raw_args)
            if isinstance(parsed, dict):
                LOGGER.debug("Successfully parsed JSON arguments: %s", parsed)
                return parsed
            else:
                LOGGER.warning("JSON parsed to non-dict type: %s", type(parsed))
                return {}
        except json.JSONDecodeError as err:
            # Fail fast on invalid JSON to avoid executing with wrong args
            LOGGER.warning("Invalid JSON for tool arguments, ignoring: %s", err)
            return {}

    # Handle other types
    if raw_args is None:
        return {}

    # Try to convert to dict if possible
    try:
        if hasattr(raw_args, '__dict__'):
            return vars(raw_args)
        elif hasattr(raw_args, 'items'):
            return dict(raw_args)
    except Exception as err:
        LOGGER.debug("Could not convert args to dict: %s", err)

    LOGGER.warning("Unknown argument type: %s, returning empty dict", type(raw_args))
    return {}


def _sanitize_argument_value(key: str, value: Any) -> Any:
    """Sanitize individual argument values with corrected logic for lists."""
    if value is None:
        return None

    # Handle list-like values first
    if isinstance(value, (list, tuple)):
        # Specific handler for RGB color lists
        if key == 'rgb_color':
            try:
                # Ensure 3 values and convert them to integers
                return [int(float(c)) for c in value[:3]]
            except (ValueError, TypeError):
                LOGGER.debug("Invalid RGB color format for %s: %s", key, value)
                return None  # Return None if format is invalid

        # Specific handler for domain/device_class which should be strings
        if key in ('domain', 'device_class'):
            return value[0] if value else ""

        # Generic handler for other unexpected lists
        LOGGER.warning("Unexpected list value for %s: %s, taking first element", key, value)
        return value[0] if value else ""

    # For non-list values, convert to string and strip
    str_value = str(value).strip()
    if not str_value:
        return None

    # Handle boolean-like strings
    if str_value.lower() in ('true', 'false'):
        return str_value.lower() == 'true'

    # Handle numeric strings for specific keys
    if key in ('brightness_pct', 'color_temp', 'temperature'):
        try:
            return float(str_value) if '.' in str_value else int(str_value)
        except ValueError:
            LOGGER.debug("Could not convert %s to number: %s. Returning as string.", key, str_value)
            # Fall through to return the original string if conversion fails

    return str_value


def _validate_brightness(args: dict, tool_name: str) -> dict:
    """Validate and clamp brightness values for light tools."""
    for key in ("brightness", "brightness_pct"):
        if key in args:
            try:
                brightness = int(float(args[key]))
                # Clamp value to the valid 0-100 range
                args[key] = max(0, min(100, brightness))
                LOGGER.debug("Validated and clamped '%s' to %d for %s", key, args[key], tool_name)
            except (ValueError, TypeError):
                LOGGER.warning("Invalid value for '%s' in %s: %s. Removing.", key, tool_name, args[key])
                args.pop(key)
    return args


def _validate_position(args: dict, tool_name: str) -> dict:
    """Validate and clamp position values for cover/valve tools."""
    if "position" in args:
        try:
            position = int(float(args["position"]))
            # Clamp value to the valid 0-100 range
            args["position"] = max(0, min(100, position))
            LOGGER.debug("Validated and clamped 'position' to %d for %s", args["position"], tool_name)
        except (ValueError, TypeError):
            LOGGER.warning("Invalid value for 'position' in %s: %s. Removing.", tool_name, args["position"])
            args.pop("position")
    return args


def _validate_temperature(args: dict, tool_name: str) -> dict:
    """Validate temperature values for climate tools."""
    if "temperature" in args:
        try:
            # Just ensure it's a valid float, no clamping unless we know the unit
            float(args["temperature"])
        except (ValueError, TypeError):
            LOGGER.warning("Invalid value for 'temperature' in %s: %s. Removing.", tool_name, args["temperature"])
            args.pop("temperature")
    return args


# Map of tool names to their specific validation functions
_TOOL_VALIDATORS = {
    "HassLightSet": _validate_brightness,
    "HassSetPosition": _validate_position,
    "HassClimateSetTemperature": _validate_temperature,
}


def _apply_tool_specific_validation(hass: HA_HomeAssistant, tool_name: str, args: dict) -> dict:
    """Apply tool-specific validation by dispatching to the correct validator.

    This function adds a layer of validation to tool arguments to ensure they
    are within expected ranges and formats, making the integration more robust
    against malformed data from the LLM.

    Args:
        hass: Home Assistant instance
        tool_name: Name of the tool being validated
        args: Tool arguments to validate

    Returns:
        Validated and potentially modified arguments.
    """
    validator = _TOOL_VALIDATORS.get(tool_name)
    if validator:
        return validator(args, tool_name)
    return args


def _validate_and_sanitize_args(hass: HA_HomeAssistant, tool_name: str, tool_args: dict) -> dict:
    """Validate and sanitize tool arguments for HA compatibility."""
    if not isinstance(tool_args, dict):
        LOGGER.warning("Tool args not a dict for %s: %s", tool_name, type(tool_args))
        tool_args = {}

    # Clean up argument values
    sanitized_args = {}
    for key, value in tool_args.items():
        clean_key = str(key).strip()
        if clean_key:
            # Convert values to appropriate types
            sanitized_value = _sanitize_argument_value(clean_key, value)
            if sanitized_value is not None:
                sanitized_args[clean_key] = sanitized_value

    # Apply tool-specific validation and defaults
    sanitized_args = _apply_tool_specific_validation(hass, tool_name, sanitized_args)

    return sanitized_args


def convert_xai_to_ha_tool(hass: HA_HomeAssistant, xai_tool_call) -> ha_llm.ToolInput:
    """Convert xAI tool call to HA ToolInput format with robust argument parsing."""
    tool_name = xai_tool_call.function.name
    raw_args = xai_tool_call.function.arguments

    LOGGER.debug("Converting xAI tool call: %s with raw args: %s (type: %s)",
                tool_name, raw_args, type(raw_args))

    # Parse arguments safely
    tool_args = _safe_parse_tool_arguments(raw_args)

    # Validate and sanitize arguments
    tool_args = _validate_and_sanitize_args(hass, tool_name, tool_args)

    LOGGER.debug("Final tool args for %s: %s", tool_name, tool_args)

    return ha_llm.ToolInput(
        tool_name=tool_name,
        tool_args=tool_args,
    )
