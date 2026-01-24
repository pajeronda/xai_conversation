"""xAI tool formatting and schema conversion utilities."""

from __future__ import annotations

import json
from typing import Any

from homeassistant.helpers import llm as ha_llm

from ..const import LOGGER


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


def convert_ha_schema_to_xai(schema, tool_name: str = "unknown") -> dict:
    """Convert HA schema to xAI function parameters format with robust voluptuous handling."""
    if not schema:
        return {"type": "object", "properties": {}, "required": []}

    try:
        # Use the standard Home Assistant voluptuous to OpenAPI converter
        # This is the same method used by the official OpenAI integration
        result: dict[str, Any] = ha_llm.convert(
            schema,
            custom_serializer=ha_llm.selector_serializer,
        )
        return result
    except Exception as err:
        LOGGER.warning("[ha-to-xai] schema convert failed '%s': %s", tool_name, err)
        return {"type": "object", "properties": {}, "required": []}


def _build_common_schema(
    additional_props: dict | None = None, required: list | None = None
) -> dict:
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
        "domain": {
            "type": "string",
            "description": "Entity domain (light, switch, fan, climate, etc.)",
        },
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
    "brightness": {
        "type": "integer",
        "minimum": 0,
        "maximum": 255,
        "description": "Sets the light brightness on a scale from 0 (off) to 255 (full brightness). For 'set light to 50%', use a value around 128.",
    },
    "color_name": {
        "type": "string",
        "description": "Sets color by name (standard CSS color names in English). For 'set color to red', use {'color_name': 'red'}.",
    },
    "rgb_color": {
        "type": "array",
        "items": {"type": "integer", "minimum": 0, "maximum": 255},
        "minItems": 3,
        "maxItems": 3,
        "description": "Sets color using RGB values [R, G, B]. For 'set color to yellow', use {'rgb_color': [255, 255, 0]}.",
    },
    "color_temp": {
        "type": "integer",
        "description": "Sets color temperature in mireds (e.g., 153-500). Use for 'warm white' or 'cool white'. DO NOT use for standard colors like 'red' or 'blue'.",
    },
}

_MEDIA_PROPS = {
    "search_query": {
        "type": "string",
        "description": "The title, artist, or name of the media to search for. Examples: 'Bohemian Rhapsody', 'The Beatles', 'The Dark Knight'.",
    },
    "media_class": {
        "type": "string",
        "description": "Category of the media. Must be one of: 'album', 'artist', 'channel', 'composer', 'contributing_artist', 'directory', 'episode', 'game', 'genre', 'image', 'movie', 'music', 'playlist', 'podcast', 'season', 'track', 'tv_show', 'url', 'video'. For a request about the movie Batman, use {'media_class': 'movie'}.",
    },
}

_VOLUME_PROPS = {
    "volume_level": {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
        "description": "Sets the volume to a specific percentage (0-100). For a 'set volume to 50%' request, use {'volume_level': 50}.",
    },
}

_TIMER_PROPS = {
    "hours": {
        "type": "integer",
        "minimum": 0,
        "description": "Number of hours for the timer's duration.",
    },
    "minutes": {
        "type": "integer",
        "minimum": 0,
        "maximum": 59,
        "description": "Number of minutes for the timer's duration.",
    },
    "seconds": {
        "type": "integer",
        "minimum": 0,
        "maximum": 59,
        "description": "Number of seconds for the timer's duration.",
    },
    "name": {
        "type": "string",
        "description": "Optional name for the new timer. Example: 'kitchen timer'.",
    },
    "area": {
        "type": "string",
        "description": "Optional area to associate with the new timer.",
    },
}

_TIMER_CONTROL_PROPS = {
    "name": {
        "type": "string",
        "description": "The name of the timer to control. Used to identify which timer to pause, cancel, etc.",
    },
    "area": {
        "type": "string",
        "description": "The area of the timer to control, used if the name is not unique.",
    },
}

_CALENDAR_PROPS = {
    "start_date": {
        "type": "string",
        "format": "date",
        "description": "The start date for the event search range, in YYYY-MM-DD format.",
    },
    "end_date": {
        "type": "string",
        "format": "date",
        "description": "The end date for the event search range, in YYYY-MM-DD format.",
    },
}

_TODO_PROPS = {
    "status": {
        "type": "string",
        "enum": ["needs_action", "completed"],
        "description": "Filters the to-do list items by their status. Use 'needs_action' for open items or 'completed' for finished items.",
    },
}


# Map of tools to their required domains.
# If a tool maps to a set of domains, AT LEAST ONE of those domains must be present
# in the exposed entities for the tool to be included.
# If a tool is not in this map, it is included by default (conservative fallback).
_TOOL_DOMAIN_REQUIREMENTS = {
    # Domain-specific tools
    "HassLightSet": {"light"},
    "HassClimateSetTemperature": {"climate"},
    "HassClimateGetTemperature": {"climate"},
    "HassVacuumStart": {"vacuum"},
    "HassVacuumReturnToBase": {"vacuum"},
    "HassVacuumPause": {"vacuum"},
    "HassVacuumStop": {"vacuum"},
    "HassMediaSearchAndPlay": {"media_player"},
    "HassMediaPause": {"media_player"},
    "HassMediaUnpause": {"media_player"},
    "HassMediaNext": {"media_player"},
    "HassMediaPrevious": {"media_player"},
    "HassSetVolume": {"media_player"},
    "HassSetVolumeRelative": {"media_player"},
    "HassVolumeUp": {"media_player"},
    "HassVolumeDown": {"media_player"},
    "HassOpenCover": {"cover"},
    "HassCloseCover": {"cover"},
    "HassSetPosition": {"cover"},
    "HassStartTimer": {"timer"},
    "HassCancelTimer": {"timer"},
    "HassCancelAllTimers": {"timer"},
    "HassIncreaseTimer": {"timer"},
    "HassDecreaseTimer": {"timer"},
    "HassPauseTimer": {"timer"},
    "HassUnpauseTimer": {"timer"},
    "HassTimerStatus": {"timer"},
    "HassGetWeather": {"weather"},
    "CalendarGetEvents": {"calendar"},
    "TodoGetItems": {"todo"},
    "HassShoppingListAddItem": {"shopping_list", "todo"},
    "HassListAddItem": {"todo"},
    "HassListCompleteItem": {"todo"},
    # Generic tools that require at least one controllable domain
    # We include these by default (empty set) as they are fundamental to HA operation
    "HassTurnOn": set(),
    "HassTurnOff": set(),
    "HassToggle": set(),
    # Tools that are always useful or don't map cleanly to a single domain
    "HassGetState": set(),  # Empty set means always include
    "HassNevermind": set(),
    "HassGetCurrentDate": set(),
    "HassGetCurrentTime": set(),
    "TriggerAssistAI": set(),
    "TriggerAssistMessage": set(),
}


# Tool schema map - defined at module level to avoid reconstruction on every call
# Maps tool names to their schema generation logic for efficient direct lookups
_TOOL_SCHEMA_MAP = {
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
    "HassGetState": lambda: _build_common_schema(
        {
            "state": {
                "type": "string",
                "description": "The state to check for, e.g., 'on', 'off', 'locked', 'unlocked', 'open', 'closed'. Example: To check if a light is on, use {'state': 'on'}.",
            }
        }
    ),
    "HassSetPosition": lambda: _build_common_schema(
        {
            "position": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100,
                "description": "The target position as a percentage from 0 (fully closed) to 100 (fully open). For 'open the blinds halfway', use {'position': 50}.",
            }
        }
    ),
    # Tools with additional properties and required fields
    "HassClimateSetTemperature": lambda: _build_common_schema(
        {
            "temperature": {
                "type": "number",
                "description": "The target temperature for a climate device. The value should be in the system's configured temperature unit (°C or °F).",
            }
        },
        ["temperature"],
    ),
    "HassSetVolumeRelative": lambda: _build_common_schema(
        {
            "volume_step": {
                "type": "string",
                "description": "Adjusts volume relatively. Use 'up' or 'down' for small increments, or an integer from -100 to 100 for a specific percentage change. For 'turn volume up', use {'volume_step': 'up'}. For 'decrease volume by 20', use {'volume_step': -20}.",
            }
        },
        ["volume_step"],
    ),
    # Tools with fully custom schemas
    "HassStartTimer": lambda: {
        "type": "object",
        "properties": _TIMER_PROPS,
        "required": [],
    },
    "HassCancelTimer": lambda: {
        "type": "object",
        "properties": _TIMER_CONTROL_PROPS,
        "required": [],
    },
    "HassCancelAllTimers": lambda: {
        "type": "object",
        "properties": {
            "area": {"type": "string", "description": "Area name to cancel timers in"}
        },
        "required": [],
    },
    "HassIncreaseTimer": lambda: {
        "type": "object",
        "properties": _TIMER_CONTROL_PROPS,
        "required": [],
    },
    "HassDecreaseTimer": lambda: {
        "type": "object",
        "properties": _TIMER_CONTROL_PROPS,
        "required": [],
    },
    "HassPauseTimer": lambda: {
        "type": "object",
        "properties": _TIMER_CONTROL_PROPS,
        "required": [],
    },
    "HassUnpauseTimer": lambda: {
        "type": "object",
        "properties": _TIMER_CONTROL_PROPS,
        "required": [],
    },
    "HassTimerStatus": lambda: {
        "type": "object",
        "properties": _TIMER_CONTROL_PROPS,
        "required": [],
    },
    "HassShoppingListAddItem": lambda: {
        "type": "object",
        "properties": {
            "item": {
                "type": "string",
                "description": "The item to add to the shopping list. Example: 'milk'.",
            },
            "name": {
                "type": "string",
                "description": "The name of the list. Defaults to the primary shopping list if not provided.",
            },
        },
        "required": ["item"],
    },
    "HassListAddItem": lambda: {
        "type": "object",
        "properties": {
            "item": {
                "type": "string",
                "description": "The item to add to the to-do list. Example: 'call the doctor'.",
            },
            "name": {
                "type": "string",
                "description": "The name of the list. Defaults to the primary to-do list if not provided.",
            },
        },
        "required": ["item"],
    },
    "HassListCompleteItem": lambda: {
        "type": "object",
        "properties": {
            "item": {
                "type": "string",
                "description": "The name of the item on the list to mark as complete. Example: 'milk'.",
            },
            "name": {
                "type": "string",
                "description": "The name of the specific list where the item is located. Defaults to the primary list.",
            },
        },
        "required": ["item"],
    },
    "HassBroadcast": lambda: {
        "type": "object",
        "properties": {
            "message": {"type": "string", "description": "Message to broadcast"}
        },
        "required": ["message"],
    },
    "HassRespond": lambda: {
        "type": "object",
        "properties": {
            "response": {"type": "string", "description": "Response message"}
        },
        "required": ["response"],
    },
    "HassGetWeather": lambda: {
        "type": "object",
        "properties": {"name": {"type": "string", "description": "Location name"}},
        "required": [],
    },
    "CalendarGetEvents": lambda: _build_common_schema(_CALENDAR_PROPS),
    "TodoGetItems": lambda: _build_common_schema(_TODO_PROPS),
}


def _get_fallback_tool_schema(tool_name: str) -> dict | None:
    """Provide fallback JSON schema for known HA tools using a structured map.
    Supports all standard HA intent slots: name, area, floor, domain, device_class.
    """
    # Handle media tools, which share properties but have different required fields
    if tool_name in (
        "HassMediaSearchAndPlay",
        "HassMediaPause",
        "HassMediaUnpause",
        "HassMediaNext",
        "HassMediaPrevious",
    ):
        required = ["search_query"] if tool_name == "HassMediaSearchAndPlay" else []
        return _build_common_schema(_MEDIA_PROPS, required)

    # Look up the tool in the module-level map
    schema_func = _TOOL_SCHEMA_MAP.get(tool_name)
    if schema_func:
        return schema_func()

    # No fallback for completely unknown tools - let HA handle them
    return None


def format_tools_for_xai(ha_tools: list[ha_llm.Tool]) -> tuple[list[dict], dict]:
    """Convert HA LLM tools to xAI format (neutral dict format).

    Args:
        ha_tools: List of Home Assistant LLM tools
    """
    xai_tools = []

    # Track schema conversion statistics
    schema_stats = {
        "total": 0,
        "converted": 0,
        "fallback_used": 0,
        "failed": 0,
    }

    # Sort tools alphabetically by name for consistent ordering
    # This improves cache hit rate by ensuring tools are always in the same order
    sorted_ha_tools = sorted(ha_tools, key=lambda t: str(t.name))

    for ha_tool in sorted_ha_tools:
        # Convert parameters to xAI JSON schema
        try:
            schema_stats["total"] += 1
            parameters = convert_ha_schema_to_xai(ha_tool.parameters, str(ha_tool.name))

            # If schema lacks properties, provide a minimal known schema for common HA tools
            try:
                props = (
                    parameters.get("properties", {})
                    if isinstance(parameters, dict)
                    else {}
                )
            except Exception:
                props = {}
            if not props:
                fallback = _get_fallback_tool_schema(str(ha_tool.name))
                if fallback:
                    parameters = fallback
                    schema_stats["fallback_used"] += 1
                else:
                    schema_stats["converted"] += 1
            else:
                schema_stats["converted"] += 1

            # Normalize parameters dict for consistent serialization
            # Sort keys recursively to ensure stable JSON representation
            parameters = _normalize_json_dict(parameters)

            xai_tool_obj = {
                "name": str(ha_tool.name),
                "description": str(ha_tool.description or "Home Assistant tool"),
                "parameters": parameters,
            }
            xai_tools.append(xai_tool_obj)

        except Exception as err:
            LOGGER.error(
                "[ha-to-xai] convert '%s' failed: %s", ha_tool.name, err, exc_info=True
            )
            schema_stats["failed"] += 1
            continue

    return xai_tools, schema_stats


def convert_ha_to_xai_tool_call(tool_call: Any, chat_pb2: Any) -> Any:
    """Convert Home Assistant ToolInput to xAI ToolCall protobuf object.

    Args:
        tool_call: Home Assistant ToolInput or similar object with tool_name and tool_args.
        chat_pb2: xAI protobuf module for creating ToolCall/FunctionCall objects.
    """
    if not (hasattr(tool_call, "tool_name") and hasattr(tool_call, "tool_args")):
        return tool_call

    return chat_pb2.ToolCall(
        id=getattr(tool_call, "tool_call_id", ""),
        function=chat_pb2.FunctionCall(
            name=tool_call.tool_name,
            arguments=json.dumps(tool_call.tool_args),
        ),
    )


def _filter_tools_by_exposed_domains(
    ha_tools: list[Any], exposed_entities: dict
) -> tuple[list[Any], int]:
    """Filter HA tools based on exposed entity domains.

    Args:
        ha_tools: List of HA tool objects (must have 'name' attribute)
        exposed_entities: Dictionary of exposed entities from ha_llm._get_exposed_entities

    Returns:
        Tuple of (Filtered list of HA tool objects, dropped_count)
    """
    if not exposed_entities or "entities" not in exposed_entities:
        # If no exposed entities info available, return all tools conservatively
        return ha_tools, 0

    # Extract active domains from exposed entities
    active_domains = {
        entity_id.split(".")[0] for entity_id in exposed_entities["entities"]
    }

    filtered_tools = []
    dropped_count = 0

    for tool in ha_tools:
        required_domains = _TOOL_DOMAIN_REQUIREMENTS.get(tool.name)

        # Case 1: Tool is not in our requirement map -> Keep it (conservative)
        if required_domains is None:
            filtered_tools.append(tool)
            continue

        # Case 2: Tool maps to empty set -> Always keep (e.g. HassGetState)
        if not required_domains:
            filtered_tools.append(tool)
            continue

        # Case 3: Tool requires domains -> Keep if intersection is not empty
        # (i.e. at least one required domain is active)
        if not active_domains.isdisjoint(required_domains):
            filtered_tools.append(tool)
        else:
            dropped_count += 1

    return filtered_tools, dropped_count
