"""xAI tool formatting and schema conversion utilities."""
from __future__ import annotations

from typing import Any

from homeassistant.helpers import llm as ha_llm

from ..const import LOGGER
from ..exceptions import raise_tool_error


def _normalize_json_dict(data: Any) -> Any:
    """Recursively normalize a dict/list for consistent JSON serialization.

    Sorts dictionary keys alphabetically and recursively processes nested structures.
    This ensures that the same data always produces the same JSON representation,
    improving cache hit rates.

    Args:
        data: The data structure to normalize (dict, list, or primitive)

    Returns:
        Normalized version of the data with sorted keys
    """
    if isinstance(data, dict):
        # Sort keys and recursively normalize values
        return {k: _normalize_json_dict(v) for k, v in sorted(data.items())}
    elif isinstance(data, list):
        # Recursively normalize list items
        return [_normalize_json_dict(item) for item in data]
    else:
        # Return primitives as-is
        return data


def _convert_schema_to_xai(schema) -> dict:
    """Convert HA schema to xAI function parameters format with robust voluptuous handling."""
    if not schema:
        return {"type": "object", "properties": {}, "required": []}

    try:
        LOGGER.debug("Converting schema: %s (type: %s)", schema, type(schema))
        # Use the standard Home Assistant voluptuous to OpenAPI converter
        # This is the same method used by the official OpenAI integration
        result: dict[str, Any] = ha_llm.convert(
            schema,
            custom_serializer=ha_llm.selector_serializer,
        )
        LOGGER.debug("Schema converted successfully via voluptuous-openapi")
        return result
    except Exception as err:
        LOGGER.warning("Failed to convert schema using voluptuous-openapi: %s. Will attempt fallback.", err)
        return {"type": "object", "properties": {}, "required": []}


def _build_common_schema(additional_props: dict | None = None, required: list | None = None) -> dict:
    """Build JSON schema with common HA targeting properties.

    Args:
        additional_props: Additional properties to merge with common targeting props
        required: List of required field names

    Returns:
        Complete JSON schema dict with merged properties
    """
    # Complete HA built-in intent standard slots
    common_target_props = {
        "name": {"type": "string", "description": "Friendly name of the device/entity"},
        "area": {"type": "string", "description": "Area/room name"},
        "floor": {"type": "string", "description": "Floor name"},
        "domain": {"type": "string", "description": "Entity domain (light, switch, fan, climate, etc.)"},
        "device_class": {"type": "string", "description": "Device class category"},
    }

    # Merge additional properties if provided
    if additional_props:
        properties = {**common_target_props, **additional_props}
    else:
        properties = common_target_props

    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
    }


# Schemas for fallback tools, defined as constants for clarity and reuse
_LIGHT_PROPS = {
    "brightness": {"type": "integer", "minimum": 0, "maximum": 100, "description": "Brightness percentage"},
    "brightness_pct": {"type": "integer", "minimum": 0, "maximum": 100, "description": "Brightness percentage"},
    "color": {"type": "string", "description": "Color name or hex code"},
    "rgb_color": {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 255}, "minItems": 3, "maxItems": 3, "description": "RGB color values"},
    "color_temp": {"type": "integer", "description": "Color temperature in mireds"},
    "temperature": {"type": "integer", "description": "Color temperature"},
}

_MEDIA_PROPS = {
    "search_query": {"type": "string", "description": "Media search query"},
    "media_class": {"type": "string", "description": "Media class (album, movie, track, etc.)"},
}

_VOLUME_PROPS = {
    "volume_level": {"type": "integer", "minimum": 0, "maximum": 100, "description": "Volume level percentage"},
    "volume_step": {"type": "integer", "minimum": -100, "maximum": 100, "description": "Volume step change"},
}

_TIMER_PROPS = {
    "hours": {"type": "integer", "minimum": 0, "description": "Hours"},
    "minutes": {"type": "integer", "minimum": 0, "maximum": 59, "description": "Minutes"},
    "seconds": {"type": "integer", "minimum": 0, "maximum": 59, "description": "Seconds"},
    "name": {"type": "string", "description": "Timer name"},
    "area": {"type": "string", "description": "Area name"},
}


def _get_fallback_tool_schema(tool_name: str) -> dict | None:
    """Provide fallback JSON schema for known HA tools using a structured map.
    Supports all standard HA intent slots: name, area, floor, domain, device_class.
    """

    # A map of tool names to their schema generation logic.
    # Using a dictionary for direct lookups is more efficient than iterating.
    tool_schema_map = {
        # Tools with no additional properties (use common schema)
        "HassTurnOn": _build_common_schema,
        "HassTurnOff": _build_common_schema,
        "HassVolumeUp": _build_common_schema,
        "HassVolumeDown": _build_common_schema,
        "HassOpenCover": _build_common_schema,
        "HassCloseCover": _build_common_schema,
        "HassClimateGetTemperature": _build_common_schema,
        "HassVacuumStart": _build_common_schema,
        "HassVacuumReturnToBase": _build_common_schema,
        "HassVacuumPause": _build_common_schema,
        "HassVacuumStop": _build_common_schema,
        "TriggerAssistAI": _build_common_schema,
        "TriggerAssistMessage": _build_common_schema,
        "HassGetCurrentDate": lambda: {"type": "object", "properties": {}, "required": []},
        "HassGetCurrentTime": lambda: {"type": "object", "properties": {}, "required": []},
        "HassNevermind": lambda: {"type": "object", "properties": {}, "required": []},

        # Tools with additional properties
        "HassLightSet": lambda: _build_common_schema(_LIGHT_PROPS),
        "HassSetVolume": lambda: _build_common_schema(_VOLUME_PROPS),
        "HassGetState": lambda: _build_common_schema({"state": {"type": "string", "description": "State to check"}}),
        "HassSetPosition": lambda: _build_common_schema({"position": {"type": "integer", "minimum": 0, "maximum": 100, "description": "Position percentage"}}),

        # Tools with additional properties and required fields
        "HassClimateSetTemperature": lambda: _build_common_schema({"temperature": {"type": "number", "description": "Target temperature"}}, ["temperature"]),
        "HassSetVolumeRelative": lambda: _build_common_schema({"volume_step": {"type": "string", "enum": ["up", "down"], "description": "Volume direction or integer step"}}, ["volume_step"]),

        # Tools with fully custom schemas
        "HassStartTimer": lambda: {"type": "object", "properties": _TIMER_PROPS, "required": []},
        "HassCancelAllTimers": lambda: {"type": "object", "properties": {"area": {"type": "string", "description": "Area name to cancel timers in"}}, "required": []},
        "HassShoppingListAddItem": lambda: {"type": "object", "properties": {"item": {"type": "string", "description": "Item to add to list"}, "name": {"type": "string", "description": "List name (optional)"}}, "required": ["item"]},
        "HassListAddItem": lambda: {"type": "object", "properties": {"item": {"type": "string", "description": "Item to add to list"}, "name": {"type": "string", "description": "List name (optional)"}}, "required": ["item"]},
        "HassListCompleteItem": lambda: {"type": "object", "properties": {"item": {"type": "string", "description": "Item to complete"}, "name": {"type": "string", "description": "List name"}}, "required": ["item"]},
        "HassBroadcast": lambda: {"type": "object", "properties": {"message": {"type": "string", "description": "Message to broadcast"}}, "required": ["message"]},
        "HassRespond": lambda: {"type": "object", "properties": {"response": {"type": "string", "description": "Response message"}}, "required": ["response"]},
        "HassGetWeather": lambda: {"type": "object", "properties": {"name": {"type": "string", "description": "Location name"}}, "required": []},
    }

    # Handle media tools, which share properties but have different required fields
    if tool_name in ("HassMediaSearchAndPlay", "HassMediaPause", "HassMediaUnpause", "HassMediaNext", "HassMediaPrevious"):
        required = ["search_query"] if tool_name == "HassMediaSearchAndPlay" else []
        return _build_common_schema(_MEDIA_PROPS, required)

    # Look up the tool in the map
    schema_func = tool_schema_map.get(tool_name)
    if schema_func:
        return schema_func()

    # No fallback for completely unknown tools - let HA handle them
    return None


def format_tools_for_xai(ha_tools: list[ha_llm.Tool], xai_tool_constructor) -> list:
    """Convert HA LLM tools to xAI format using the SDK's tool constructor.

    Args:
        ha_tools: List of Home Assistant LLM tools
        xai_tool_constructor: xAI SDK tool constructor (from xai_sdk.chat.tool) - REQUIRED
    """
    if xai_tool_constructor is None:
        raise_tool_error("all_tools", "xAI tool constructor is required but was None")

    xai_tools = []

    # Sort tools alphabetically by name for consistent ordering
    # This improves cache hit rate by ensuring tools are always in the same order
    sorted_ha_tools = sorted(ha_tools, key=lambda t: str(t.name))

    for ha_tool in sorted_ha_tools:
        # Convert parameters to xAI JSON schema
        try:
            parameters = _convert_schema_to_xai(ha_tool.parameters)

            # If schema lacks properties, provide a minimal known schema for common HA tools
            try:
                props = parameters.get("properties", {}) if isinstance(parameters, dict) else {}
            except Exception:
                props = {}
            if not props:
                fallback = _get_fallback_tool_schema(str(ha_tool.name))
                if fallback:
                    LOGGER.debug("Tool '%s' has no properties, using fallback schema.", ha_tool.name)
                    parameters = fallback

            # Normalize parameters dict for consistent serialization
            # Sort keys recursively to ensure stable JSON representation
            parameters = _normalize_json_dict(parameters)

            xai_tool_obj = xai_tool_constructor(
                name=str(ha_tool.name),
                description=str(ha_tool.description or "Home Assistant tool"),
                parameters=parameters
            )
            xai_tools.append(xai_tool_obj)

        except Exception as err:
            LOGGER.error("Error converting tool %s: %s", ha_tool.name, err, exc_info=True)
            continue

    return xai_tools
