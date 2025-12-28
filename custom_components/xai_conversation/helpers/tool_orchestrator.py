"""Tool Orchestrator for xAI Conversation.

This module centralizes the management of:
1. Native Home Assistant Tools (via Assist API)
2. Custom Tools (locally defined helpers)
3. Extended Tools (via YAML configuration)

It handles caching, tool selection logic, execution routing, and tool conversion.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import time
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm as ha_llm
from homeassistant.helpers import entity_registry as er

from ..const import (
    CONF_ALLOW_SMART_HOME_CONTROL,
    CONF_EXTENDED_TOOLS_YAML,
    CONF_USE_EXTENDED_TOOLS,
    LOGGER,
)
from ..exceptions import raise_generic_error
from .conversation import build_llm_context
from .tools_custom import CUSTOM_TOOLS
from .tools_ha_to_xai import filter_tools_by_exposed_domains, format_tools_for_xai
from .tools_xai_to_ha import convert_xai_to_ha_tool
from .tools_extended import ExtendedToolError

if TYPE_CHECKING:
    from ..entity import XAIBaseLLMEntity


def get_exposed_entities_with_aliases(
    hass: HomeAssistant, assistant_id: str = "conversation"
) -> list[dict]:
    """Get exposed entities enriched with aliases from entity registry.

    Args:
        hass: Home Assistant instance
        assistant_id: Assistant ID to filter exposed entities (default: "conversation")

    Returns:
        List of entity data dictionaries including 'aliases' field if present.
    """
    # Get base exposed entities (no state needed for static context)
    exposed_entities_result = ha_llm._get_exposed_entities(
        hass, assistant_id, include_state=False
    )

    if not exposed_entities_result or "entities" not in exposed_entities_result:
        return []

    ent_reg = er.async_get(hass)
    enriched_entities = []

    for entity_id, entity_data in exposed_entities_result["entities"].items():
        # Create a copy to avoid mutating the cached result from HA
        entity_data_copy = entity_data.copy()
        entity_data_copy["entity_id"] = (
            entity_id  # ADDED: Ensure entity_id is explicitly present
        )

        # Enrich with aliases from registry
        entry = ent_reg.async_get(entity_id)
        if entry and entry.aliases:
            entity_data_copy["aliases"] = list(entry.aliases)

        enriched_entities.append(entity_data_copy)

    return enriched_entities


@dataclass
class ToolExecutionResult:
    """Standardized result of a tool execution."""

    result: Any
    is_error: bool = False


class ToolOrchestrator:
    """Manages tool lifecycle, selection, and execution."""

    def __init__(self, hass: HomeAssistant, entity: XAIBaseLLMEntity):
        """Initialize the orchestrator."""
        # self._hass is not stored as we use self._entity.hass to ensure it's up to date
        self._entity = entity

        # State Cache
        self._cached_xai_tools: list[dict] | None = None

        # Native Tool Cache
        self._cached_ha_tools_map: dict[str, Any] = {}
        self._cached_active_domains: set[str] | None = None
        self._cached_entity_ids: set[str] | None = None

        # Extended Tool Cache
        self._cached_extended_hash: str | None = None

        # Execution State
        self._using_extended_mode: bool = False

        # Cached Prompt Context (for PromptManager)
        self._cached_static_context_csv: str | None = None
        self._cached_enriched_entities: list[dict] | None = None

    @property
    def using_extended_mode(self) -> bool:
        """Return True if currently operating in Extended Tools mode."""
        return self._using_extended_mode

    def _clear_cache(self):
        """Clear all caches."""
        self._cached_xai_tools = None
        self._cached_ha_tools_map = {}
        self._cached_active_domains = None
        self._cached_entity_ids = None
        self._cached_extended_hash = None
        self._cached_static_context_csv = None
        self._cached_enriched_entities = None

    async def async_refresh_tools_if_needed(self, user_input) -> bool:
        """Check cache validity and rebuild tools if necessary."""
        # 1. Determine Desired Mode
        allow_control = self._entity._get_option(CONF_ALLOW_SMART_HOME_CONTROL, True)
        use_extended_config = self._entity._get_option(CONF_USE_EXTENDED_TOOLS, False)

        if not allow_control:
            # Tools disabled completely
            if self._cached_xai_tools is not None and not self._cached_xai_tools:
                return False  # Already cached as empty

            self._clear_cache()
            self._cached_xai_tools = []
            self._cached_xai_tools = []
            self._cached_static_context_csv = ""  # PromptManager will build full prompt
            return True

        # 2. Get Home Assistant Exposed Entities
        exposed_result = ha_llm._get_exposed_entities(
            self._entity.hass, "conversation", include_state=False
        )

        current_active_domains = {
            e.split(".")[0] for e in exposed_result.get("entities", [])
        }
        current_entity_ids = set(exposed_result.get("entities", {}).keys())

        # 3. Check Cache Validity
        should_rebuild = False
        rebuild_reason = ""

        if self._cached_xai_tools is None:
            should_rebuild = True
            rebuild_reason = "cache_empty"

        elif use_extended_config:
            # Extended Mode Check: Hash the YAML
            yaml_config = self._entity.entry.data.get(CONF_EXTENDED_TOOLS_YAML, "")
            current_hash = (
                hashlib.sha256(yaml_config.encode()).hexdigest()[:8]
                if yaml_config
                else None
            )

            if current_hash != self._cached_extended_hash:
                should_rebuild = True
                rebuild_reason = "extended_yaml_changed"
            # Also rebuild if we switched mode from Native to Extended
            elif not self._using_extended_mode:
                should_rebuild = True
                rebuild_reason = "mode_switch_to_extended"

        else:
            # Native Mode Check: Domains or Entities changed
            if self._using_extended_mode:
                should_rebuild = True
                rebuild_reason = "mode_switch_to_native"
            elif current_active_domains != self._cached_active_domains:
                should_rebuild = True
                rebuild_reason = "active_domains_changed"
            elif current_entity_ids != self._cached_entity_ids:
                should_rebuild = True
                rebuild_reason = "entity_ids_changed"

        if not should_rebuild:
            LOGGER.debug(
                "ToolOrchestrator: Cache hit (Extended=%s)", self._using_extended_mode
            )
            return False

        # 4. Rebuild Cache
        start_time = time.time()
        LOGGER.debug("ToolOrchestrator: Rebuilding cache. Reason: %s", rebuild_reason)

        # Update tracking state
        self._cached_active_domains = current_active_domains
        self._cached_entity_ids = current_entity_ids

        if use_extended_config:
            await self._build_extended_tools()
        else:
            await self._build_native_tools(
                user_input, exposed_result, current_active_domains
            )

        # Rebuild the full system prompt after tools are updated
        # self._cached_full_system_prompt = self._build_full_system_prompt()

        LOGGER.debug(
            "ToolOrchestrator: Rebuild complete in %.2fs. Tools: %d. Mode: %s",
            time.time() - start_time,
            len(self._cached_xai_tools),
            "Extended" if self._using_extended_mode else "Native",
        )
        return True

    async def _build_extended_tools(self):
        """Build cache for Extended Tools mode."""
        self._using_extended_mode = True  # Default to True, disable if empty
        registry = self._entity._extended_tools_registry

        if not registry or registry.is_empty:
            LOGGER.warning(
                "Extended tools enabled but registry empty. Falling back to native."
            )
            self._using_extended_mode = False
            self._cached_xai_tools = []
        self._cached_xai_tools = registry.get_specs_for_llm()

        yaml_config = self._entity.entry.data.get(CONF_EXTENDED_TOOLS_YAML, "")
        self._cached_extended_hash = (
            hashlib.sha256(yaml_config.encode()).hexdigest()[:8]
            if yaml_config
            else None
        )
        self._cached_ha_tools_map = {}

        # Build Static Context CSV for PromptManager
        self._cached_static_context_csv = self._generate_compact_csv_context()

    def _generate_compact_csv_context(self) -> str:
        """Helper to generate a compact CSV string of exposed entities."""
        allow_control = self._entity._get_option(CONF_ALLOW_SMART_HOME_CONTROL, True)
        if not allow_control:
            self._cached_enriched_entities = []
            return ""

        self._cached_enriched_entities = get_exposed_entities_with_aliases(
            self._entity.hass
        )
        output = io.StringIO()
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["id", "name", "aliases"])
        for entity in self._cached_enriched_entities:
            aliases = entity.get("aliases", [])
            aliases_str = "/".join(aliases) if aliases else ""
            writer.writerow(
                [entity.get("entity_id", ""), entity.get("name", ""), aliases_str]
            )
        return output.getvalue().strip()

    async def _build_native_tools(self, user_input, exposed_result, active_domains):
        """Build cache for Native HA Tools mode."""
        from ..xai_gateway import XAIGateway

        self._using_extended_mode = False
        self._cached_extended_hash = None

        assist_api = ha_llm._async_get_apis(self._entity.hass).get(
            ha_llm.LLM_API_ASSIST
        )
        if not assist_api:
            raise_generic_error("AssistAPI is not available in Home Assistant.")

        llm_context = build_llm_context(self._entity.hass, user_input)

        # 1. Fetch HA Tools
        ha_tools = assist_api._async_get_tools(llm_context, exposed_result)

        # 2. Filter by Domains (Using imported helper)
        ha_tools = filter_tools_by_exposed_domains(ha_tools, exposed_result)

        # 3. Inject Custom Tools
        custom_tool_requirements = {
            "HassSetInputNumber": "input_number",
            "HassSetInputBoolean": "input_boolean",
            "HassSetInputText": "input_text",
            "HassRunScript": "script",
            "HassTriggerAutomation": "automation",
        }

        for custom_tool in CUSTOM_TOOLS:
            required_domain = custom_tool_requirements.get(custom_tool.name)
            if required_domain and required_domain in active_domains:
                ha_tools.append(custom_tool)

        # 4. Update Caches (Using imported helper)
        self._cached_ha_tools_map = {tool.name: tool for tool in ha_tools}
        self._cached_xai_tools = format_tools_for_xai(ha_tools, XAIGateway.tool_def)

        # 5. Build Static Context CSV for PromptManager
        self._cached_static_context_csv = self._generate_compact_csv_context()

    def get_xai_tools(self) -> list[dict]:
        """Get the tools formatted for the xAI API."""
        return self._cached_xai_tools or []

    def get_static_context_csv(self) -> str:
        """Get the cached static context (entities, areas, etc.) for prompt building."""
        return self._cached_static_context_csv or ""

    def _parse_arguments_if_needed(self, arguments: Any) -> dict:
        """Helper to safely parse JSON arguments if passed as string."""
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str):
            try:
                if not arguments.strip():
                    return {}
                return json.loads(arguments)
            except json.JSONDecodeError:
                LOGGER.warning("Failed to parse tool arguments: %s", arguments)
                return {}
        return {}

    async def async_execute_tool(
        self,
        tool_name: str,
        arguments: Any,
        user_input: Any,  # Changed from user_context to user_input for clarity
    ) -> ToolExecutionResult:
        """Execute a tool (Native, Custom, or Extended).

        Args:
            tool_name: Name of the tool to execute.
            arguments: Arguments (dict or JSON string).
            user_input: The ConversationInput object containing the context.
        """
        allow_control = self._entity._get_option(CONF_ALLOW_SMART_HOME_CONTROL, True)
        if not allow_control:
            return ToolExecutionResult(
                result="Tool execution blocked: home control is disabled in chat-only mode.",
                is_error=True,
            )

        try:
            if self._using_extended_mode:
                registry = self._entity._extended_tools_registry
                if not registry:
                    raise ExtendedToolError("Extended Registry not initialized")

                if self._cached_enriched_entities is None:
                    self._cached_enriched_entities = get_exposed_entities_with_aliases(
                        self._entity.hass
                    )

                parsed_args = self._parse_arguments_if_needed(arguments)

                # Delay handling: wrap in composite/script delay and background
                if "delay" in parsed_args:
                    tool_data = registry.get_tool_config(tool_name)
                    if tool_data:
                        LOGGER.debug("Scheduling extended tool '%s' for background execution with delay", tool_name)
                        delayed_func = registry.get_delayed_function_config(
                            tool_data["function"], parsed_args["delay"]
                        )
                        # Execute the wrapped sequence in a background task
                        self._entity.hass.async_create_task(
                            registry.async_execute_raw_config(
                                delayed_func,
                                parsed_args,
                                user_input.context,
                                self._cached_enriched_entities,
                            )
                        )
                        return ToolExecutionResult(result="Scheduled")

                # Use user_input.context for extended tools
                result_data = await registry.execute(
                    tool_name,
                    parsed_args,
                    user_input.context,
                    self._cached_enriched_entities,
                )
                return ToolExecutionResult(result=result_data)

            else:
                ha_tool = self._cached_ha_tools_map.get(tool_name)

                # Fallback: Try case-insensitive lookup if exact match fails
                if not ha_tool:
                    # Case-insensitive map (built on demand or cached if frequent)
                    # For now, simple iteration is fast enough for <50 tools
                    for cached_name, tool_obj in self._cached_ha_tools_map.items():
                        if (
                            cached_name.lower() == tool_name.lower()
                            or cached_name.lower().replace("hass", "")
                            == tool_name.lower().replace("hass", "")
                        ):
                            ha_tool = tool_obj
                            LOGGER.warning(
                                "Tool fuzzy match: '%s' resolved to '%s'",
                                tool_name,
                                cached_name,
                            )
                            break

                if not ha_tool:
                    raise_generic_error(
                        f"Tool '{tool_name}' not found in cached native tools"
                    )

                # Native tools execution

                # Mock tool call object for converter
                class MockToolCall:
                    class Function:
                        def __init__(self, name, args):
                            self.name = name
                            self.arguments = args

                    def __init__(self, name, args):
                        self.function = self.Function(name, args)

                mock_call = MockToolCall(tool_name, arguments)

                # Use shared converter for sanitization and validation
                ha_tool_input = convert_xai_to_ha_tool(self._entity.hass, mock_call)

                # Build robust LLMContext using shared helper
                # This ensures we use the correct DOMAIN constant and handle all fields
                llm_context = build_llm_context(self._entity.hass, user_input)

                result_data = await ha_tool.async_call(
                    self._entity.hass, ha_tool_input, llm_context
                )
                return ToolExecutionResult(result=result_data)

        except Exception as err:
            return ToolExecutionResult(
                result=f"Error executing tool '{tool_name}': {err}", is_error=True
            )
