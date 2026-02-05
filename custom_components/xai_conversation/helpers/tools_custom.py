"""Custom tools for xAI Conversation to control automations, scripts, and helpers."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Callable, Coroutine

import voluptuous as vol

from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import llm as ha_llm
try:  # floor registry introduced in newer HA versions
    from homeassistant.helpers import floor_registry as fr
except Exception:  # pragma: no cover - optional dependency
    fr = None

from ..const import DOMAIN, INTENT_EXECUTION_NO_INTENT_MESSAGE, LOGGER


class Tool:
    """Simple tool implementation."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: vol.Schema,
        async_call: Callable[
            [HomeAssistant, ha_llm.ToolInput, ha_llm.LLMContext],
            Coroutine[Any, Any, Any],
        ],
        required_domain: str | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self.async_call = async_call
        self.required_domain = required_domain

    def __repr__(self) -> str:
        return f"<Tool {self.name}>"


# ==========================================================================
# Light Set Tool (direct light.turn_on with full color support)
# ==========================================================================


def _resolve_light_entity_id(
    hass: HomeAssistant, tool_args: dict
) -> str | None:
    """Resolve a light entity_id from intent targeting slots (name, area).

    Searches exposed entities for a light entity matching
    the given friendly name (or alias) and optionally area.
    """
    if entity_id := (tool_args.get("entity_id") or "").strip():
        if entity_id.startswith("light."):
            return entity_id

    name = (tool_args.get("name") or "").strip().lower()
    area = (tool_args.get("area") or "").strip().lower()
    floor = (tool_args.get("floor") or "").strip().lower()

    if not name and not area and not floor:
        return None

    exposed_result = ha_llm._get_exposed_entities(
        hass, "conversation", include_state=False
    )
    if not exposed_result or "entities" not in exposed_result:
        return None

    ent_reg = er.async_get(hass)
    area_reg = ar.async_get(hass)
    floor_reg = fr.async_get(hass) if fr else None

    candidates: list[tuple[str, bool]] = []
    for entity_id, entity_data in exposed_result["entities"].items():
        if not entity_id.startswith("light."):
            continue

        ename = (entity_data.get("name") or "").lower()

        # Get registry entry and resolve area/floor via registries
        entry = ent_reg.async_get(entity_id)
        area_name = ""
        floor_name = ""
        if entry and entry.area_id:
            area_entry = area_reg.async_get_area(entry.area_id)
            if area_entry:
                area_name = (area_entry.name or "").lower()
                floor_id = getattr(area_entry, "floor_id", None)
                if floor_id and floor_reg:
                    floor_entry = floor_reg.async_get_floor(floor_id)
                    floor_name = (floor_entry.name or "").lower() if floor_entry else ""
                else:
                    floor_name = (getattr(area_entry, "floor_name", "") or "").lower()

        entity_area = area_name or (entity_data.get("area") or "").lower()
        entity_floor = floor_name or (entity_data.get("floor") or "").lower()
        aliases = [a.lower() for a in entry.aliases] if entry and entry.aliases else []

        # Enrich name from registry if missing
        if not ename and entry:
            ename = (entry.name or entry.original_name or "").lower()

        name_match = name and (
            name == ename or name in ename or ename in name or name in aliases
        )
        area_match = area and area == entity_area
        floor_match = floor and floor == entity_floor

        if floor and not floor_match:
            continue

        if name and name_match:
            candidates.append((entity_id, area_match))
        elif not name and area and area_match:
            candidates.append((entity_id, True))
        elif not name and not area and floor and floor_match:
            candidates.append((entity_id, True))

    if not candidates:
        return None

    # Prefer candidates that also match area
    area_matched = [eid for eid, am in candidates if am]
    if area_matched:
        return area_matched[0]

    return candidates[0][0]


async def async_light_set(
    hass: HomeAssistant, tool_input: ha_llm.ToolInput, llm_context: ha_llm.LLMContext
) -> Any:
    """Set brightness, color, or color temperature of a light via direct light.turn_on."""
    tool_args = tool_input.tool_args
    entity_id = _resolve_light_entity_id(hass, tool_args)
    if not entity_id:
        return "Could not resolve light entity from the provided name/area."

    service_data: dict[str, Any] = {"entity_id": entity_id}

    if color_name := tool_args.get("color_name"):
        service_data["color_name"] = str(color_name)

    if rgb_color := tool_args.get("rgb_color"):
        service_data["rgb_color"] = rgb_color

    if color_temp := tool_args.get("color_temp_kelvin"):
        service_data["color_temp_kelvin"] = color_temp

    brightness = tool_args.get("brightness")
    if brightness is not None:
        service_data["brightness"] = int(brightness)

    brightness_pct = tool_args.get("brightness_pct")
    if brightness_pct is not None:
        service_data["brightness_pct"] = int(brightness_pct)
        service_data.pop("brightness", None)

    LOGGER.debug("[light_set] light.turn_on %s (args: %s)", entity_id, tool_args)

    await hass.services.async_call(
        "light",
        "turn_on",
        service_data,
        blocking=True,
        context=llm_context.context,
    )

    params = {k: v for k, v in service_data.items() if k != "entity_id"}
    return f"Set {entity_id}: {params}" if params else f"Turned on {entity_id}"


SPEC_LIGHT_SET = Tool(
    name="HassLightSet",
    description=(
        "Sets brightness, color, or color temperature of a light. "
        "Target by 'entity_id', or by 'name' with optional 'area'/'floor'. "
        "For brightness: use 'brightness_pct' (0-100%) - e.g. 50 for 50%. "
        "For color: use 'color_name' (e.g. 'red', 'blue') or 'rgb_color' ([R,G,B]). "
        "For warm/cool white: use 'color_temp_kelvin' (2000-6500). "
        "NEVER use rgb_color for brightness. Parameters can be combined."
    ),
    parameters=vol.Schema(
        {
            vol.Optional("entity_id"): str,
            vol.Optional("name"): str,
            vol.Optional("area"): str,
            vol.Optional("floor"): str,
            vol.Optional("brightness_pct"): vol.All(
                vol.Coerce(int), vol.Range(min=0, max=100)
            ),
            vol.Optional("brightness"): vol.All(
                vol.Coerce(int), vol.Range(min=0, max=255)
            ),
            vol.Optional("color_name"): str,
            vol.Optional("color_temp_kelvin"): vol.All(
                vol.Coerce(int), vol.Range(min=2000, max=6500)
            ),
            vol.Optional("rgb_color"): vol.All(
                [vol.All(vol.Coerce(int), vol.Range(min=0, max=255))],
                vol.Length(min=3, max=3),
            ),
        }
    ),
    async_call=async_light_set,
    required_domain="light",
)


# ==========================================================================
# Input Number Tool
# ==========================================================================


async def async_set_input_number(
    hass: HomeAssistant, tool_input: ha_llm.ToolInput, llm_context: ha_llm.LLMContext
) -> Any:
    """Set the value of an input_number."""
    entity_id = tool_input.tool_args["entity_id"]
    value = tool_input.tool_args["value"]

    await hass.services.async_call(
        "input_number",
        "set_value",
        {"entity_id": entity_id, "value": value},
        blocking=True,
        context=llm_context.context,
    )
    return f"Set {entity_id} to {value}"


SPEC_SET_INPUT_NUMBER = Tool(
    name="HassSetInputNumber",
    description="Sets the value of an input_number entity. Use this to change settings like volume, target temperature (if input_number), or other numeric configurations.",
    parameters=vol.Schema(
        {
            vol.Required("entity_id"): str,
            vol.Required("value"): vol.Coerce(float),
        }
    ),
    async_call=async_set_input_number,
    required_domain="input_number",
)


# ==========================================================================
# Input Boolean Tool
# ==========================================================================


async def async_set_input_boolean(
    hass: HomeAssistant, tool_input: ha_llm.ToolInput, llm_context: ha_llm.LLMContext
) -> Any:
    """Turn on or off an input_boolean."""
    entity_id = tool_input.tool_args["entity_id"]
    state = tool_input.tool_args["state"]

    service = "turn_on" if state else "turn_off"
    await hass.services.async_call(
        "input_boolean",
        service,
        {"entity_id": entity_id},
        blocking=True,
        context=llm_context.context,
    )
    return f"Set {entity_id} to {'on' if state else 'off'}"


SPEC_SET_INPUT_BOOLEAN = Tool(
    name="HassSetInputBoolean",
    description="Turns an input_boolean entity on or off. Use this to toggle switches, flags, or boolean settings.",
    parameters=vol.Schema(
        {
            vol.Required("entity_id"): str,
            vol.Required("state"): bool,
        }
    ),
    async_call=async_set_input_boolean,
    required_domain="input_boolean",
)


# ==========================================================================
# Input Text Tool
# ==========================================================================


async def async_set_input_text(
    hass: HomeAssistant, tool_input: ha_llm.ToolInput, llm_context: ha_llm.LLMContext
) -> Any:
    """Set the value of an input_text."""
    entity_id = tool_input.tool_args["entity_id"]
    value = tool_input.tool_args["value"]

    await hass.services.async_call(
        "input_text",
        "set_value",
        {"entity_id": entity_id, "value": value},
        blocking=True,
        context=llm_context.context,
    )
    return f"Set {entity_id} to '{value}'"


SPEC_SET_INPUT_TEXT = Tool(
    name="HassSetInputText",
    description="Sets the text value of an input_text entity.",
    parameters=vol.Schema(
        {
            vol.Required("entity_id"): str,
            vol.Required("value"): str,
        }
    ),
    async_call=async_set_input_text,
    required_domain="input_text",
)


# ==========================================================================
# Script Tool
# ==========================================================================


async def async_run_script(
    hass: HomeAssistant, tool_input: ha_llm.ToolInput, llm_context: ha_llm.LLMContext
) -> Any:
    """Run a Home Assistant script."""
    entity_id = tool_input.tool_args["entity_id"]

    # Extract script name from entity_id (script.my_script -> my_script)
    script_name = entity_id.replace("script.", "")

    await hass.services.async_call(
        "script",
        script_name,
        {},
        blocking=True,
        context=llm_context.context,
    )
    return f"Executed script {entity_id}"


SPEC_RUN_SCRIPT = Tool(
    name="HassRunScript",
    description="Executes a Home Assistant script. Use this to run predefined sequences of actions.",
    parameters=vol.Schema(
        {
            vol.Required("entity_id"): str,
        }
    ),
    async_call=async_run_script,
    required_domain="script",
)


# ==========================================================================
# Automation Tool
# ==========================================================================


async def async_trigger_automation(
    hass: HomeAssistant, tool_input: ha_llm.ToolInput, llm_context: ha_llm.LLMContext
) -> Any:
    """Trigger an automation."""
    entity_id = tool_input.tool_args["entity_id"]

    await hass.services.async_call(
        "automation",
        "trigger",
        {
            "entity_id": entity_id,
            "skip_condition": True,
        },  # Usually we want to force trigger
        blocking=True,
        context=llm_context.context,
    )
    return f"Triggered automation {entity_id}"


SPEC_TRIGGER_AUTOMATION = Tool(
    name="HassTriggerAutomation",
    description="Triggers a Home Assistant automation.",
    parameters=vol.Schema(
        {
            vol.Required("entity_id"): str,
        }
    ),
    async_call=async_trigger_automation,
    required_domain="automation",
)


# ==========================================================================
# Intent Execution Tool (conversation.process)
# ==========================================================================


async def async_intent_execution(
    hass: HomeAssistant, tool_input: ha_llm.ToolInput, llm_context: ha_llm.LLMContext
) -> Any:
    """Execute a command via conversation.process (Home Assistant intents).

    If an intent matches, returns the speech response. If none match, returns
    a message so the LLM can use other tools.
    """
    text = (tool_input.tool_args or {}).get("text", "").strip()
    if not text:
        return INTENT_EXECUTION_NO_INTENT_MESSAGE

    try:
        language = llm_context.language or hass.config.language or "en"
        data = {
            "text": text,
            "language": language,
            "agent_id": "conversation.home_assistant",
        }
        result = await hass.services.async_call(
            "conversation",
            "process",
            data,
            blocking=True,
            return_response=True,
            context=llm_context.context or Context(),
        )

        response = (result or {}).get("response", {})
        if response.get("response_type") == "error":
            return INTENT_EXECUTION_NO_INTENT_MESSAGE

        speech = response.get("speech", {}).get("plain", {}).get("speech") or "Done."
        return speech

    except Exception:
        return INTENT_EXECUTION_NO_INTENT_MESSAGE


SPEC_INTENT_EXECUTION = Tool(
    name="IntentExecution",
    description="Execute a user command via Home Assistant intents (conversation.process).",
    parameters=vol.Schema(
        {
            vol.Required("text"): str,  # the recognized command in the user's language
        }
    ),
    async_call=async_intent_execution,
)


# ==========================================================================
# Photo Analysis Tool (camera snapshot -> Grok Vision)
# ==========================================================================


async def async_photo_analysis(
    hass: HomeAssistant, tool_input: ha_llm.ToolInput, llm_context: ha_llm.LLMContext
) -> Any:
    """Take a camera snapshot and analyze it with Grok Vision."""
    entity_id = tool_input.tool_args["entity_id"]
    prompt = tool_input.tool_args.get("prompt", "Describe what you see.")

    # 1. Snapshot to /media/xai_snapshots/ (official HA media path, no allowlist needed)
    snapshot_dir = os.path.join("/media", "xai_snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    camera_name = entity_id.replace("camera.", "")
    snapshot_path = os.path.join(snapshot_dir, f"{camera_name}_{timestamp}.jpg")

    try:
        await hass.services.async_call(
            "camera",
            "snapshot",
            {"entity_id": entity_id, "filename": snapshot_path},
            blocking=True,
            context=llm_context.context or Context(),
        )
    except Exception as err:
        LOGGER.error("[photo_analysis] snapshot failed for %s: %s", entity_id, err)
        return f"Failed to capture snapshot from {entity_id}: {err}"

    # 2. Call photo_analysis service
    try:
        result = await hass.services.async_call(
            DOMAIN,
            "photo_analysis",
            {"prompt": prompt, "images": [snapshot_path]},
            blocking=True,
            return_response=True,
            context=llm_context.context or Context(),
        )
        analysis = (result or {}).get("analysis", "No analysis returned.")
        return analysis
    except Exception as err:
        LOGGER.error("[photo_analysis] analysis failed: %s", err)
        return f"Failed to analyze image: {err}"


SPEC_PHOTO_ANALYSIS = Tool(
    name="HassPhotoAnalysis",
    description="Take a snapshot from a camera entity and analyze the image using Grok Vision. Use this when the user asks to look at, check, or describe what a camera sees.",
    parameters=vol.Schema(
        {
            vol.Required("entity_id"): str,
            vol.Optional("prompt", default="Describe what you see."): str,
        }
    ),
    async_call=async_photo_analysis,
    required_domain="camera",
)


# List of all custom tools
CUSTOM_TOOLS = [
    SPEC_LIGHT_SET,
    SPEC_SET_INPUT_NUMBER,
    SPEC_SET_INPUT_BOOLEAN,
    SPEC_SET_INPUT_TEXT,
    SPEC_RUN_SCRIPT,
    SPEC_TRIGGER_AUTOMATION,
    SPEC_INTENT_EXECUTION,
    SPEC_PHOTO_ANALYSIS,
]
