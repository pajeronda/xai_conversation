"""Tool Orchestrator for xAI Conversation.

Centralizes management of Native, Custom, and Extended tools.
Handles caching, selection, execution routing, and conversion.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import time
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from pathlib import Path

import yaml

from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm as ha_llm
from homeassistant.helpers import area_registry as ar
try:  # floor registry introduced in newer HA versions
    from homeassistant.helpers import floor_registry as fr
except Exception:  # pragma: no cover - optional dependency
    fr = None
from homeassistant.helpers import entity_registry as er

from ..const import (
    LOGGER,
    DOMAIN,
    CONF_ALLOW_SMART_HOME_CONTROL,
    CONF_USE_EXTENDED_TOOLS,
    CONF_EXTENDED_TOOLS_YAML,
)
from ..exceptions import raise_generic_error
from .utils import extract_device_id
from .tools_custom import CUSTOM_TOOLS
from .tools_ha_to_xai import _filter_tools_by_exposed_domains, format_tools_for_xai
from .tools_xai_to_ha import convert_xai_to_ha_tool, _safe_parse_tool_arguments
from .tools_extended import ExtendedToolError

if TYPE_CHECKING:
    from .tools_extended import ExtendedToolsRegistry


class _MockToolCall:
    """Mock tool call object for native tool execution."""

    class Function:
        def __init__(self, name: str, args):
            self.name = name
            self.arguments = args

    def __init__(self, name: str, args):
        self.function = self.Function(name, args)


def _get_exposed_entities_with_aliases(
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
    area_reg = ar.async_get(hass)
    floor_reg = fr.async_get(hass) if fr else None
    enriched_entities = []

    for entity_id, entity_data in exposed_entities_result["entities"].items():
        # Create a copy to avoid mutating the cached result from HA
        entity_data_copy = entity_data.copy()
        entity_data_copy["entity_id"] = entity_id

        # Get Registry Entry
        entry = ent_reg.async_get(entity_id)

        # Resolve area/floor names via registries for prompt context
        area_name = ""
        floor_name = ""
        if entry and entry.area_id:
            area_entry = area_reg.async_get_area(entry.area_id)
            if area_entry:
                area_name = area_entry.name or ""
                floor_id = getattr(area_entry, "floor_id", None)
                if floor_id and floor_reg:
                    floor_entry = floor_reg.async_get_floor(floor_id)
                    floor_name = floor_entry.name if floor_entry else ""
                else:
                    floor_name = getattr(area_entry, "floor_name", "") or ""
        if area_name:
            entity_data_copy["area"] = area_name
        if floor_name:
            entity_data_copy["floor"] = floor_name

        # 1. Enrich Name (Fallback chain: HA_LLM > Registry Name > Registry Original Name > State Name)
        if not entity_data_copy.get("name"):
            if entry and entry.name:
                entity_data_copy["name"] = entry.name
            elif entry and entry.original_name:
                entity_data_copy["name"] = entry.original_name
            else:
                state = hass.states.get(entity_id)
                if state and state.name:
                    entity_data_copy["name"] = state.name
                else:
                    entity_data_copy["name"] = entity_id  # Last resort

        # 2. Enrich Aliases
        if entry and entry.aliases:
            entity_data_copy["aliases"] = sorted(list(entry.aliases))

        enriched_entities.append(entity_data_copy)

    # 3. Sort by entity_id for stable prompt generation
    enriched_entities.sort(key=lambda x: x.get("entity_id", ""))

    return enriched_entities


@dataclass
class ToolExecutionResult:
    """Standardized result of a tool execution."""

    result: Any
    is_error: bool = False


@dataclass
class ToolSessionConfig:
    """Configuration for a tool session, decoupled from the Entity."""

    allow_control: bool = True
    use_extended_tools: bool = False
    extended_tools_yaml: str = ""
    extended_tools_registry: ExtendedToolsRegistry | None = None


def resolve_tool_session_config(
    config: dict, entry_data: dict, registry: ExtendedToolsRegistry | None = None
) -> ToolSessionConfig:
    """Build tool session configuration from raw data.

    Decouples configuration resolution from the Entity state and discovery logic.
    """
    return ToolSessionConfig(
        allow_control=config.get(CONF_ALLOW_SMART_HOME_CONTROL, True),
        use_extended_tools=config.get(CONF_USE_EXTENDED_TOOLS),
        extended_tools_yaml=entry_data.get(CONF_EXTENDED_TOOLS_YAML, ""),
        extended_tools_registry=registry,
    )


# Set of custom tool names for deduplication against native HA tools
_CUSTOM_TOOL_NAMES = {tool.name for tool in CUSTOM_TOOLS}


class ToolOrchestrator:
    """Manages tool lifecycle, selection, and execution."""

    def __init__(self, hass: HomeAssistant):
        """Initialize the orchestrator."""
        self.hass = hass

        # State Cache
        self._cached_xai_tools: list[dict] | None = None

        # Native Tool Cache
        self._cached_ha_tools_map: dict[str, Any] = {}
        self._cached_exposed_hash: str | None = None

        # Extended Tool Cache
        self._cached_extended_hash: str | None = None

        # Execution State
        self._using_extended_mode: bool = False

        # Cached Prompt Context (for PromptManager)
        self._cached_static_context_csv: str | None = None
        self._cached_custom_intents_csv: str | None = None
        self._custom_intent_names: set[str] = set()
        self._cached_intents_hash: str | None = None
        self._cached_enriched_entities: list[dict] | None = None

    def _clear_cache(self):
        """Clear all caches."""
        self._cached_xai_tools = None
        self._cached_ha_tools_map = {}
        self._cached_exposed_hash = None
        self._cached_extended_hash = None
        self._cached_static_context_csv = None
        self._cached_custom_intents_csv = None
        self._custom_intent_names = set()
        self._cached_intents_hash = None
        self._cached_enriched_entities = None

    async def async_refresh_tools_if_needed(
        self, user_input, config: ToolSessionConfig
    ) -> bool:
        """Check cache validity and rebuild tools if necessary."""
        # 1. Determine Desired Mode
        if not config.allow_control:
            # Tools disabled completely
            if self._cached_xai_tools is not None and not self._cached_xai_tools:
                return False  # Already cached as empty

            self._clear_cache()
            self._cached_xai_tools = []
            self._cached_static_context_csv = ""  # PromptManager will build full prompt
            return True

        # 2. Get Home Assistant Exposed Entities
        exposed_result = ha_llm._get_exposed_entities(
            self.hass, "conversation", include_state=False
        )

        # Hash the full exposed result to detect ANY change
        # (entity names, areas, aliases, additions, removals, domain changes)
        exposed_json = json.dumps(
            exposed_result.get("entities", {}), sort_keys=True
        )
        current_exposed_hash = hashlib.sha256(
            exposed_json.encode()
        ).hexdigest()[:12]

        current_active_domains = {
            e.split(".")[0] for e in exposed_result.get("entities", [])
        }

        # Lightweight check for custom_sentences changes (stat-based)
        current_intents_hash = await self._compute_intents_hash()

        # 3. Check Cache Validity
        should_rebuild = False
        rebuild_reason = ""

        if self._cached_xai_tools is None:
            should_rebuild = True
            rebuild_reason = "cache_empty"

        elif current_intents_hash != self._cached_intents_hash:
            should_rebuild = True
            rebuild_reason = "custom_sentences_changed"

        elif config.use_extended_tools:
            # Extended Mode Check: Hash the YAML + exposed entities
            yaml_config = config.extended_tools_yaml
            current_ext_hash = (
                hashlib.sha256(yaml_config.encode()).hexdigest()[:8]
                if yaml_config
                else None
            )

            if current_ext_hash != self._cached_extended_hash:
                should_rebuild = True
                rebuild_reason = "extended_yaml_changed"
            elif not self._using_extended_mode:
                should_rebuild = True
                rebuild_reason = "mode_switch_to_extended"
            elif current_exposed_hash != self._cached_exposed_hash:
                should_rebuild = True
                rebuild_reason = "exposed_entities_changed"

        else:
            # Native Mode Check: exposed entities changed (covers domains, IDs, names, areas, etc.)
            if self._using_extended_mode:
                should_rebuild = True
                rebuild_reason = "mode_switch_to_native"
            elif current_exposed_hash != self._cached_exposed_hash:
                should_rebuild = True
                rebuild_reason = "exposed_entities_changed"

        if not should_rebuild:
            LOGGER.debug(
                "[orchestrator] cache: HIT (%d tools)", len(self._cached_xai_tools)
            )
            return False

        # 4. Rebuild Cache
        start_time = time.time()

        # Update tracking state
        self._cached_exposed_hash = current_exposed_hash
        self._cached_intents_hash = current_intents_hash

        if config.use_extended_tools:
            await self._build_extended_tools(config)
        else:
            await self._build_native_tools(
                user_input, exposed_result, current_active_domains
            )

        LOGGER.debug(
            "[orchestrator] cache: REFRESHED (%s: %d tools in %.2fs)",
            rebuild_reason,
            len(self._cached_xai_tools),
            time.time() - start_time,
        )
        return True

    async def _build_extended_tools(self, config: ToolSessionConfig):
        """Build cache for Extended Tools mode."""
        self._using_extended_mode = True  # Default to True, disable if empty
        registry = config.extended_tools_registry

        if not registry or registry.is_empty:
            LOGGER.warning("[orchestrator] extended tools empty, fallback to native")
            self._using_extended_mode = False
            self._cached_xai_tools = []
            self._cached_ha_tools_map = {}
            self._cached_static_context_csv = ""
            self._cached_custom_intents_csv = ""
            self._custom_intent_names = set()
            return

        self._cached_xai_tools = registry.get_specs_for_llm()

        yaml_config = config.extended_tools_yaml
        self._cached_extended_hash = (
            hashlib.sha256(yaml_config.encode()).hexdigest()[:8]
            if yaml_config
            else None
        )
        self._cached_ha_tools_map = {}

        # Build Static Context CSV for PromptManager
        self._cached_static_context_csv = self._generate_compact_csv_context(
            config.allow_control
        )

        # Build Custom Intents Context for PromptManager
        self._cached_custom_intents_csv = await self._generate_custom_intents_context()

    def _generate_compact_csv_context(self, allow_control: bool = True) -> str:
        """Helper to generate a compact CSV string of exposed entities."""
        if not allow_control:
            self._cached_enriched_entities = []
            return ""

        # _get_exposed_entities_with_aliases returns a SORTED list
        self._cached_enriched_entities = _get_exposed_entities_with_aliases(self.hass)
        output = io.StringIO()
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["id", "name", "aliases"])

        for entity in self._cached_enriched_entities:
            aliases = entity.get("aliases", [])
            aliases_str = "/".join(sorted(aliases)) if aliases else ""
            writer.writerow(
                [entity.get("entity_id", ""), entity.get("name", ""), aliases_str]
            )
        return output.getvalue().strip()

    async def _compute_intents_hash(self) -> str:
        """Compute a lightweight hash of custom_sentences directory (file paths + mtimes)."""
        return await self.hass.async_add_executor_job(
            self._compute_intents_hash_sync
        )

    def _compute_intents_hash_sync(self) -> str:
        """Sync implementation of custom_sentences hash (run in executor)."""
        sentences_dir = Path(self.hass.config.config_dir) / "custom_sentences"
        if not sentences_dir.exists():
            return ""
        parts: list[str] = []
        for lang_dir in sorted(sentences_dir.iterdir()):
            if not lang_dir.is_dir():
                continue
            for yaml_file in sorted(lang_dir.glob("*.yaml")):
                try:
                    st = yaml_file.stat()
                    parts.append(f"{yaml_file}:{st.st_mtime_ns}:{st.st_size}")
                except OSError:
                    continue
        if not parts:
            return ""
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]

    async def _generate_custom_intents_context(self) -> str:
        """Read custom_sentences YAML and generate a list of custom intents for LLM."""

        def _read_custom_sentences() -> dict[str, list[str]]:
            sentences_dir = Path(self.hass.config.config_dir) / "custom_sentences"
            if not sentences_dir.exists():
                return {}

            intents_info: dict[str, list[str]] = {}
            for lang_dir in sentences_dir.iterdir():
                if not lang_dir.is_dir():
                    continue
                for yaml_file in lang_dir.glob("*.yaml"):
                    try:
                        data = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
                    except Exception:
                        continue
                    for intent_name, intent_data in (data.get("intents") or {}).items():
                        if intent_name.startswith("Hass"):
                            continue
                        topics: list[str] = []
                        for entry in intent_data.get("data") or []:
                            slots = entry.get("slots") or {}
                            for val in slots.values():
                                if isinstance(val, str) and val.startswith("call_"):
                                    topics.append(val[5:].replace("_", " "))
                        # Fallback: if no call_* slots found, use the intent name
                        if not topics:
                            # CamelCase -> spaced lowercase (e.g. SetVolume -> set volume)
                            human_name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", intent_name).lower()
                            topics.append(human_name)
                        if intent_name not in intents_info:
                            intents_info[intent_name] = []
                        intents_info[intent_name].extend(topics)
            return intents_info

        try:
            intents_info = await self.hass.async_add_executor_job(
                _read_custom_sentences
            )
        except Exception as err:
            LOGGER.debug("[orchestrator] error reading custom_sentences: %s", err)
            self._custom_intent_names = set()
            return ""

        # Store intent names for filtering them out of the Assist API tool list
        self._custom_intent_names = set(intents_info.keys())

        all_topics: list[str] = []
        for topics in intents_info.values():
            all_topics.extend(topics)
        unique_topics = list(dict.fromkeys(all_topics))

        return "\n".join(f"- {topic}" for topic in unique_topics)

    def get_custom_intents_context(self) -> str:
        """Get the cached custom intents context for prompt building."""
        return self._cached_custom_intents_csv or ""

    async def _build_native_tools(self, user_input, exposed_result, active_domains):
        """Build cache for Native HA Tools mode."""
        self._using_extended_mode = False
        self._cached_extended_hash = None

        assist_api = ha_llm._async_get_apis(self.hass).get(ha_llm.LLM_API_ASSIST)
        if not assist_api:
            raise_generic_error("AssistAPI is not available in Home Assistant.")

        llm_context = ha_llm.LLMContext(
            platform=DOMAIN,
            context=user_input.context,
            language=user_input.language,
            assistant="conversation",
            device_id=extract_device_id(user_input),
        )

        # 1. Read custom_sentences intents FIRST (needed for filtering and prompt)
        self._cached_custom_intents_csv = await self._generate_custom_intents_context()

        # 2. Fetch HA Tools
        ha_tools = assist_api._async_get_tools(llm_context, exposed_result)

        # 3. Filter by Domains (Using imported helper)
        ha_tools, dropped_count = _filter_tools_by_exposed_domains(
            ha_tools, exposed_result
        )

        # 4. Remove native HA tools that are replaced by custom tools
        ha_tools = [t for t in ha_tools if t.name not in _CUSTOM_TOOL_NAMES]

        # 5. Remove custom_sentences intents exposed as tools by the Assist API.
        #    These are handled via the IntentExecution tool, not as individual tools.
        if self._custom_intent_names:
            before = len(ha_tools)
            ha_tools = [t for t in ha_tools if t.name not in self._custom_intent_names]
            intent_dropped = before - len(ha_tools)
            if intent_dropped:
                LOGGER.debug(
                    "[orchestrator] filtered %d custom_sentences intent-tools: %s",
                    intent_dropped,
                    self._custom_intent_names,
                )

        # 6. Inject Custom Tools (using required_domain from Tool definition)
        for custom_tool in CUSTOM_TOOLS:
            if custom_tool.required_domain is None or custom_tool.required_domain in active_domains:
                ha_tools.append(custom_tool)

        # 7. Update Caches (Neutral specs)
        # SORT tools by name to ensure stable API payload for prompt caching
        ha_tools_sorted = sorted(ha_tools, key=lambda t: t.name)
        self._cached_ha_tools_map = {tool.name: tool for tool in ha_tools_sorted}
        self._cached_xai_tools, schema_stats = format_tools_for_xai(ha_tools_sorted)

        # Log consolidated tool stats
        tool_names = [t.name for t in ha_tools_sorted]
        LOGGER.debug(
            "[ha-to-xai] tools: %d kept, %d dropped | total=%d (fallback=%d, failed=%d) | names=%s",
            len(ha_tools_sorted),
            dropped_count,
            schema_stats["total"],
            schema_stats["fallback_used"],
            schema_stats["failed"],
            tool_names,
        )

        # 8. Build Static Context CSV for PromptManager
        self._cached_static_context_csv = self._generate_compact_csv_context(True)

    def get_xai_tools(self) -> list[dict]:
        """Get the tools formatted for the xAI API."""
        return self._cached_xai_tools or []

    def get_static_context_csv(self) -> str:
        """Get the cached static context (entities, areas, etc.) for prompt building."""
        return self._cached_static_context_csv or ""

    async def async_execute_tool(
        self,
        tool_name: str,
        arguments: Any,
        user_input: Any,
        config: ToolSessionConfig,
    ) -> ToolExecutionResult:
        """Execute a tool (Native, Custom, or Extended).

        Args:
            tool_name: Name of the tool to execute.
            arguments: Arguments (dict or JSON string).
            user_input: The ConversationInput object containing the context.
            config: The tool session configuration.
        """
        if not config.allow_control:
            return ToolExecutionResult(
                result="Tool execution blocked: home control is disabled in chat-only mode.",
                is_error=True,
            )

        try:
            if self._using_extended_mode:
                registry = config.extended_tools_registry
                if not registry:
                    raise ExtendedToolError("Extended Registry not initialized")

                if self._cached_enriched_entities is None:
                    self._cached_enriched_entities = _get_exposed_entities_with_aliases(
                        self.hass
                    )

                parsed_args = _safe_parse_tool_arguments(arguments)

                # Delay handling: wrap in composite/script delay and background
                if "delay" in parsed_args:
                    tool_data = registry.get_tool_config(tool_name)
                    if tool_data:
                        LOGGER.debug(
                            "[orchestrator] scheduling '%s' with delay", tool_name
                        )
                        delayed_func = registry.get_delayed_function_config(
                            tool_data["function"], parsed_args["delay"]
                        )
                        # Execute the wrapped sequence in a background task
                        self.hass.async_create_task(
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
                    for cached_name, tool_obj in self._cached_ha_tools_map.items():
                        if (
                            cached_name.lower() == tool_name.lower()
                            or cached_name.lower().replace("hass", "")
                            == tool_name.lower().replace("hass", "")
                        ):
                            ha_tool = tool_obj
                            LOGGER.warning(
                                "[orchestrator] fuzzy match: '%s' -> '%s'",
                                tool_name,
                                cached_name,
                            )
                            break

                if not ha_tool:
                    raise_generic_error(
                        f"Tool '{tool_name}' not found in cached native tools"
                    )

                # Build tool input with sanitization and validation
                mock_call = _MockToolCall(tool_name, arguments)
                ha_tool_input = convert_xai_to_ha_tool(mock_call)

                # Build LLM context for execution
                llm_context = ha_llm.LLMContext(
                    platform=DOMAIN,
                    context=user_input.context,
                    language=user_input.language,
                    assistant="conversation",
                    device_id=extract_device_id(user_input),
                )

                result_data = await ha_tool.async_call(
                    self.hass, ha_tool_input, llm_context
                )
                return ToolExecutionResult(result=result_data)

        except Exception as err:
            return ToolExecutionResult(
                result=f"Error executing tool '{tool_name}': {err}", is_error=True
            )
