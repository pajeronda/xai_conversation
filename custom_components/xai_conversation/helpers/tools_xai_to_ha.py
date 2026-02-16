"""Tool conversion utilities between HA and xAI formats."""

from __future__ import annotations

import json
import re
from typing import Any

from homeassistant.helpers import llm as ha_llm

from ..const import LOGGER


def _safe_parse_tool_arguments(raw_args) -> dict:
    """Parse tool arguments safely with robust error handling."""
    if isinstance(raw_args, dict):
        return raw_args

    if isinstance(raw_args, str):
        if not raw_args.strip():
            return {}

        try:
            parsed = json.loads(raw_args)
            if isinstance(parsed, dict):
                return parsed
            else:
                LOGGER.warning("[tool-conv] JSON non-dict type: %s", type(parsed))
                return {}
        except json.JSONDecodeError as err:
            LOGGER.warning("[tool-conv] invalid JSON: %s", err)
            return {}

    # Handle other types
    if raw_args is None:
        return {}

    # Try to convert to dict if possible
    try:
        if hasattr(raw_args, "__dict__"):
            return vars(raw_args)
        elif hasattr(raw_args, "items"):
            return dict(raw_args)
    except Exception:
        pass

    LOGGER.warning("[tool-conv] unknown type: %s", type(raw_args))
    return {}


def _sanitize_argument_value(key: str, value: Any) -> Any:
    """Sanitize individual argument values with corrected logic for lists."""
    if value is None:
        return None

    # Handle list-like values first
    if isinstance(value, (list, tuple)):
        # Specific handler for RGB color lists
        if key == "rgb_color":
            try:
                # Ensure 3 values and convert them to integers
                return [int(float(c)) for c in value[:3]]
            except (ValueError, TypeError):
                return None  # Return None if format is invalid

        # Specific handler for domain/device_class which should be strings
        if key in ("domain", "device_class"):
            return value[0] if value else ""

        # Generic handler for other unexpected lists
        LOGGER.warning("[tool-conv] list for %s, using first: %s", key, value)
        return value[0] if value else ""

    # For non-list values, convert to string and strip
    str_value = str(value).strip()
    if not str_value:
        return None

    # Handle boolean-like strings
    if str_value.lower() in ("true", "false"):
        return str_value.lower() == "true"

    # Handle numeric strings for specific keys
    if key in (
        "brightness",
        "brightness_pct",
        "color_temp",
        "color_temp_kelvin",
        "temperature",
    ):
        try:
            return float(str_value) if "." in str_value else int(str_value)
        except ValueError:
            pass  # Fall through to return the original string if conversion fails

    return str_value


def _validate_position(args: dict, tool_name: str) -> dict:
    """Validate and clamp position values for cover/valve tools."""
    if "position" in args:
        try:
            position = int(float(args["position"]))
            args["position"] = max(0, min(100, position))
        except (ValueError, TypeError):
            LOGGER.warning(
                "[tool-conv] %s invalid position: %s", tool_name, args["position"]
            )
            args.pop("position")
    return args


def _validate_temperature(args: dict, tool_name: str) -> dict:
    """Validate temperature values for climate tools."""
    if "temperature" in args:
        try:
            # Just ensure it's a valid float, no clamping unless we know the unit
            float(args["temperature"])
        except (ValueError, TypeError):
            LOGGER.warning(
                "[tool-conv] %s invalid temperature: %s", tool_name, args["temperature"]
            )
            args.pop("temperature")
    return args


def _validate_timer_duration(args: dict, tool_name: str) -> dict:
    """Validate and clamp timer duration values."""
    for key in ("hours", "minutes", "seconds"):
        if key in args:
            try:
                value = int(float(args[key]))
                if key == "hours":
                    args[key] = max(0, value)
                else:
                    args[key] = max(0, min(59, value))
            except (ValueError, TypeError):
                LOGGER.warning(
                    "[tool-conv] %s invalid %s: %s", tool_name, key, args[key]
                )
                args.pop(key)
    return args


def _validate_calendar_dates(args: dict, tool_name: str) -> dict:
    """Validate date format for calendar tools (YYYY-MM-DD)."""
    date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    for key in ("start_date", "end_date"):
        if key in args:
            date_str = str(args[key]).strip()
            if not date_pattern.match(date_str):
                LOGGER.warning(
                    "[tool-conv] %s invalid %s: %s", tool_name, key, args[key]
                )
                args.pop(key)
            else:
                args[key] = date_str
    return args


def _validate_todo_status(args: dict, tool_name: str) -> dict:
    """Validate status values for todo tools."""
    if "status" in args:
        status = str(args["status"]).strip().lower()
        valid_statuses = ("needs_action", "completed")
        if status not in valid_statuses:
            LOGGER.warning(
                "[tool-conv] %s invalid status: %s", tool_name, args["status"]
            )
            args.pop("status")
        else:
            args["status"] = status
    return args


# Map of tool names to their specific validation functions
_TOOL_VALIDATORS = {
    # Cover/valve controls
    "HassSetPosition": _validate_position,
    # Climate controls
    "HassClimateSetTemperature": _validate_temperature,
    # Timer controls
    "HassStartTimer": _validate_timer_duration,
    "HassIncreaseTimer": _validate_timer_duration,
    "HassDecreaseTimer": _validate_timer_duration,
    # Calendar tools
    "calendar_get_events": _validate_calendar_dates,
    # Todo tools
    "todo_get_items": _validate_todo_status,
}


def _apply_tool_specific_validation(tool_name: str, args: dict) -> dict:
    """Apply tool-specific validation by dispatching to the correct validator.

    This function adds a layer of validation to tool arguments to ensure they
    are within expected ranges and formats, making the integration more robust
    against malformed data from the LLM.

    Args:
        tool_name: Name of the tool being validated
        args: Tool arguments to validate

    Returns:
        Validated and potentially modified arguments.
    """
    validator = _TOOL_VALIDATORS.get(tool_name)
    if validator:
        return validator(args, tool_name)
    return args


def _validate_and_sanitize_args(tool_name: str, tool_args: dict) -> dict:
    """Validate and sanitize tool arguments for HA compatibility."""
    if not isinstance(tool_args, dict):
        LOGGER.warning("[tool-conv] %s args not dict: %s", tool_name, type(tool_args))
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
    sanitized_args = _apply_tool_specific_validation(tool_name, sanitized_args)

    return sanitized_args


def convert_xai_to_ha_tool(xai_tool_call) -> ha_llm.ToolInput:
    """Convert xAI tool call to HA ToolInput format with robust argument parsing."""
    tool_name = xai_tool_call.function.name
    raw_args = xai_tool_call.function.arguments

    # Parse arguments safely
    tool_args = _safe_parse_tool_arguments(raw_args)

    # Validate and sanitize arguments
    tool_args = _validate_and_sanitize_args(tool_name, tool_args)

    return ha_llm.ToolInput(
        tool_name=tool_name,
        tool_args=tool_args,
    )
