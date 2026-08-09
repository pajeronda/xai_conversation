"""Home Assistant Compatibility Helper for LLM API.

Centralizes all direct interactions with Home Assistant LLM core features,
providing a clean, version-agnostic interface for custom components.
Isolates the use of protected APIs and provides forward compatibility.
"""

from __future__ import annotations

import inspect
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm as ha_llm

from ..const import LOGGER


def async_get_exposed_entities(
    hass: HomeAssistant, assistant_id: str = "conversation"
) -> dict:
    """Get the exposed entities registry dictionary for a specific assistant.

    Uses the core helper if available, or falls back to querying the exposure helper
    (HA 2026.8+ compatibility).
    """
    # 1. Try legacy core helper
    get_exposed = getattr(ha_llm, "_get_exposed_entities", None)
    if get_exposed:
        try:
            res = get_exposed(hass, assistant_id, include_state=False)
            if res and isinstance(res, dict) and res.get("entities"):
                return res
        except Exception as err:
            LOGGER.debug("Legacy '_get_exposed_entities' failed: %s", err)

    # 2. Try importing exposure helper across known HA core locations
    async_should_expose = None
    try:
        from homeassistant.components.homeassistant.exposed_entities import (
            async_should_expose,
        )
    except ImportError:
        try:
            from homeassistant.components.homeassistant import (
                async_should_expose,
            )
        except ImportError:
            try:
                from homeassistant.components.conversation import (
                    async_should_expose,
                )
            except ImportError:
                async_should_expose = None

    # Fallback for Home Assistant 2026.8+
    exposed_entities: dict[str, dict[str, Any]] = {}
    try:
        from homeassistant.helpers import area_registry as ar
        from homeassistant.helpers import entity_registry as er

        ent_reg = er.async_get(hass)
        area_reg = ar.async_get(hass)

        all_entity_ids = set(hass.states.async_entity_ids())
        if ent_reg:
            all_entity_ids.update(ent_reg.entities.keys())

        for entity_id in all_entity_ids:
            is_exposed = False
            if async_should_expose is not None:
                try:
                    is_exposed = async_should_expose(hass, assistant_id, entity_id)
                except Exception:
                    is_exposed = True
            else:
                entry = ent_reg.async_get(entity_id) if ent_reg else None
                is_exposed = not (entry and entry.hidden_by is not None)

            if is_exposed:
                state = hass.states.get(entity_id)
                entry = ent_reg.async_get(entity_id) if ent_reg else None

                name = (
                    (state.name if state else None)
                    or (entry.name if entry else None)
                    or (entry.original_name if entry else None)
                    or entity_id
                )
                entity_info: dict[str, Any] = {"name": name}

                if entry and entry.area_id and area_reg:
                    area_entry = area_reg.async_get_area(entry.area_id)
                    if area_entry:
                        entity_info["area"] = area_entry.name

                exposed_entities[entity_id] = entity_info
    except Exception as err:
        LOGGER.warning(
            "Failed to retrieve exposed entities via exposure helper: %s",
            err,
            exc_info=True,
        )

    return {"entities": exposed_entities}


def _safe_get_assist_api_legacy(hass: HomeAssistant) -> Any | None:
    """Retrieve the legacy Assist API instance using protected helper."""
    get_apis = getattr(ha_llm, "_async_get_apis", None)
    if not get_apis:
        return None
    try:
        return get_apis(hass).get(getattr(ha_llm, "LLM_API_ASSIST", "assist"))
    except Exception as err:
        LOGGER.debug("Legacy Assist API retrieval failed: %s", err)
        return None


async def async_get_assist_tools(
    hass: HomeAssistant, llm_context: ha_llm.LLMContext
) -> list[ha_llm.Tool]:
    """Retrieve the list of LLM tools configured for the Assist API.

    Tries to use the public stable async_get_api method (HA 2024.6+) first.
    Falls back to legacy protected APIs on older Home Assistant versions.
    """
    llm_api_assist = getattr(ha_llm, "LLM_API_ASSIST", "assist")

    # 1. Modern approach (Home Assistant 2024.6+): public stable async_get_api
    if hasattr(ha_llm, "async_get_api"):
        try:
            api_instance = await ha_llm.async_get_api(
                hass,
                llm_api_assist,
                llm_context,
            )
            # Check for tools property or async_get_tools method
            tools_attr = getattr(api_instance, "tools", None)
            if tools_attr is not None:
                if callable(tools_attr):
                    tools_attr = tools_attr()
                if inspect.isawaitable(tools_attr):
                    tools_attr = await tools_attr
                return list(tools_attr)

            async_get_tools = getattr(api_instance, "async_get_tools", None)
            if async_get_tools:
                tools_res = async_get_tools()
                if inspect.isawaitable(tools_res):
                    tools_res = await tools_res
                return list(tools_res)
        except Exception as err:
            LOGGER.warning(
                "Failed to retrieve Assist tools via public API 'async_get_api': %s. Trying legacy fallback...",
                err,
                exc_info=True,
            )

    # 2. Legacy fallback approach (using protected APIs for older HA versions)
    assist_api = _safe_get_assist_api_legacy(hass)
    if not assist_api:
        LOGGER.error("Legacy Assist API is not available. No tools could be retrieved.")
        return []

    get_tools = getattr(assist_api, "_async_get_tools", None) or getattr(
        assist_api, "async_get_tools", None
    )
    if not get_tools:
        LOGGER.error("Legacy Assist API missing tool retrieval methods.")
        return []

    try:
        exposed_result = async_get_exposed_entities(hass, "conversation")
        sig = inspect.signature(get_tools)
        if "exposed_entities" in sig.parameters:
            tools_res = get_tools(llm_context, exposed_result)
        else:
            tools_res = get_tools(llm_context)

        if inspect.isawaitable(tools_res):
            tools_res = await tools_res
        return list(tools_res)
    except Exception as err:
        LOGGER.error(
            "Failed to retrieve tools via legacy Assist API wrapper: %s",
            err,
            exc_info=True,
        )
        return []


async def async_call_llm_tool(
    hass: HomeAssistant,
    tool: ha_llm.Tool,
    tool_input: ha_llm.ToolInput,
    llm_context: ha_llm.LLMContext,
) -> Any:
    """Call an LLM tool in a version-agnostic way."""
    try:
        return await tool.async_call(hass, tool_input, llm_context)
    except Exception as err:
        LOGGER.error(
            "Failed to execute tool '%s' call: %s", tool.name, err, exc_info=True
        )
        raise
