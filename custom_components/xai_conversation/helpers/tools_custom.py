"""Custom tools for xAI Conversation to control automations, scripts, and helpers."""

from __future__ import annotations

from typing import Any, Callable, Coroutine

import voluptuous as vol

from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm as ha_llm


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
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self.async_call = async_call

    def __repr__(self) -> str:
        return f"<Tool {self.name}>"


# Input Number Tool
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
)


# Input Boolean Tool
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
)


# Input Text Tool
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
)


# Script Tool
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
)


# Automation Tool
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
)

# List of all custom tools
CUSTOM_TOOLS = [
    SPEC_SET_INPUT_NUMBER,
    SPEC_SET_INPUT_BOOLEAN,
    SPEC_SET_INPUT_TEXT,
    SPEC_RUN_SCRIPT,
    SPEC_TRIGGER_AUTOMATION,
]
