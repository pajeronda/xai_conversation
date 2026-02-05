from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from homeassistant.core import HomeAssistant

from ..const import (
    CHAT_MODE_AI_TASK,
    CHAT_MODE_CHATONLY,
    CHAT_MODE_PIPELINE,
    CHAT_MODE_TOOLS,
    CONF_ALLOW_SMART_HOME_CONTROL,
    CONF_CHAT_MODEL,
    CONF_IMAGE_MODEL,
    CONF_LIVE_SEARCH,
    CONF_LOCATION_CONTEXT,
    CONF_MAX_TOKENS,
    CONF_REASONING_EFFORT,
    CONF_SEND_USER_NAME,
    CONF_SHOW_CITATIONS,
    CONF_STORE_MESSAGES,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_USE_INTELLIGENT_PIPELINE,
    CONF_VISION_MODEL,
    CONF_ZDR,
    DOMAIN,
    LIVE_SEARCH_FULL,
    LIVE_SEARCH_OFF,
    LIVE_SEARCH_WEB,
    LIVE_SEARCH_X,
    LOGGER,
    MODEL_TARGET_CHAT,
    MODEL_TARGET_IMAGE,
    MODEL_TARGET_VISION,
    REASONING_EFFORT_MODELS,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_IMAGE_MODEL,
    RECOMMENDED_LIVE_SEARCH,
    RECOMMENDED_MAX_ITERATIONS,
    RECOMMENDED_SEND_USER_NAME,
    RECOMMENDED_SHOW_CITATIONS,
    RECOMMENDED_STORE_MESSAGES,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
    RECOMMENDED_VISION_MODEL,
)
from .memory_manager import MemoryManager

# xAI SDK imports (conditional)
try:
    from xai_sdk.chat import (
        user as xai_user,
        system as xai_system,
        assistant as xai_assistant,
        tool_result as xai_tool_result,
        image as xai_image,
        text as xai_text,
        tool as xai_tool,
    )
    from xai_sdk.tools import (
        web_search,
        x_search,
        get_tool_call_type as get_tool_call_type_sdk,
    )
    from xai_sdk.proto import chat_pb2

    XAI_SDK_AVAILABLE = True
except ImportError:
    xai_user = None
    xai_system = None
    xai_assistant = None
    xai_tool_result = None
    xai_image = None
    xai_text = None
    xai_tool = None
    web_search = None
    x_search = None
    chat_pb2 = None
    get_tool_call_type_sdk = None
    XAI_SDK_AVAILABLE = False


if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant


@dataclass
class ChatOptions:
    """Class to encapsulate chat overrides and configuration options."""

    model: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    store_messages: bool | None = None
    reasoning_effort: str | None = None
    system_prompt: str | None = None
    mode_override: str | None = None
    model_target: str = MODEL_TARGET_CHAT
    client_tools: list | None = None
    scope: str | None = None
    identifier: str | None = None
    subentry_id: str | None = None  # For memory key isolation between subentries
    task_name: str | None = None
    task_structure: dict | None = None
    max_turns: int | None = None
    live_search: str | None = None
    top_p: float | None = None
    send_user_name: bool | None = None
    show_citations: bool | None = None
    mixed_content: list[Any] | None = None
    extra_content: list[Any] | None = None
    timer: Any | None = None  # LogTimeServices
    response_format: Any | None = None
    use_encrypted_content: bool | None = None
    mode: str | None = None
    config: dict | None = None
    user_input: Any | None = None  # ConversationInput for metadata extraction
    is_fallback: bool = False
    forced_last_message: str | None = None
    messages: list[dict[str, Any]] | None = None
    is_resolved: bool = False
    prompt_hash: str | None = None
    prompt_suffix: str | None = None


# ==========================================================================
# CONTEXT & PROMPT HELPERS
# ==========================================================================


def build_session_context_info(hass: HomeAssistant, config: dict | None = None) -> str:
    """Build session context information (timestamp, timezone, country)."""
    now = datetime.now()
    location_override = (
        config.get(CONF_LOCATION_CONTEXT, "").strip() if config else None
    )

    location_info = (
        location_override or f"{hass.config.time_zone} ({hass.config.country})"
    )
    return f"\nContext: {now.strftime('%Y-%m-%d %H:%M')} | {location_info}"


# ==========================================================================
# CONFIGURATION & PARAMETER RESOLUTION
# ==========================================================================


def resolve_chat_parameters(
    service_type: str,
    entry: Any,
    subentry_id: str | None = None,
    options: ChatOptions | None = None,
) -> ChatOptions:
    """Resolve all chat parameters and operational mode with hierarchical lookups.

    Hierarchy: Overrides (options) > Subentry (if found) > Entry (Global) > Defaults.
    """
    # Early return if already resolved (avoids duplicate work in gateway)
    if options and options.is_resolved:
        return options

    opts = options or ChatOptions()

    # 1. CONFIG RETRIEVAL (Internalized from dead get_service_config)
    def _merge_config(subentry: Any | None = None) -> dict:
        config = dict(entry.data)
        if hasattr(entry, "options") and entry.options:
            config.update(entry.options)
        if subentry:
            config.update(subentry.data)
            if hasattr(subentry, "options") and subentry.options:
                config.update(subentry.options)
        return config

    config = None
    # A. Match by ID (explicit)
    if subentry_id and subentry_id in getattr(entry, "subentries", {}):
        config = _merge_config(entry.subentries[subentry_id])

    # B. Match by type (implicit for standard tasks)
    if not config and service_type not in ("ask", "photo_analysis"):
        target_stype = "ai_task" if service_type == "ai_task" else "conversation"
        for subentry in getattr(entry, "subentries", {}).values():
            if subentry.subentry_type == target_stype:
                config = _merge_config(subentry)
                break

    # C. Global fallback
    if not config:
        config = _merge_config()

    # 2. PARAMETER RESOLUTION
    resolved = ChatOptions()

    # 2.1 Resolve model based on target
    if opts.model:
        resolved.model = opts.model
    elif opts.model_target == MODEL_TARGET_VISION:
        resolved.model = config.get(CONF_VISION_MODEL, RECOMMENDED_VISION_MODEL)
    elif opts.model_target == MODEL_TARGET_IMAGE:
        resolved.model = config.get(CONF_IMAGE_MODEL, RECOMMENDED_IMAGE_MODEL)
    else:
        resolved.model = config.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)

    # 2.2 Resolve operational mode
    override = opts.mode_override

    if service_type == "ai_task":
        resolved.mode = override or CHAT_MODE_AI_TASK
    else:
        resolved.mode = (
            CHAT_MODE_CHATONLY
            if not config.get(CONF_ALLOW_SMART_HOME_CONTROL, True)
            else override
            or (
                CHAT_MODE_PIPELINE
                if config.get(CONF_USE_INTELLIGENT_PIPELINE, True)
                else CHAT_MODE_TOOLS
            )
        )

    # 2.3 Automate standard parameters mapping with fallback chain
    field_map = {
        "temperature": (CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
        "top_p": (CONF_TOP_P, RECOMMENDED_TOP_P),
        "max_tokens": (CONF_MAX_TOKENS, 1000),
        "store_messages": (CONF_STORE_MESSAGES, RECOMMENDED_STORE_MESSAGES),
        "live_search": (CONF_LIVE_SEARCH, RECOMMENDED_LIVE_SEARCH),
        "show_citations": (CONF_SHOW_CITATIONS, RECOMMENDED_SHOW_CITATIONS),
        "send_user_name": (CONF_SEND_USER_NAME, RECOMMENDED_SEND_USER_NAME),
        "use_encrypted_content": (CONF_ZDR, False),
        "max_turns": ("max_turns", RECOMMENDED_MAX_ITERATIONS),
        "reasoning_effort": (CONF_REASONING_EFFORT, None),
    }

    for field, (conf_key, default) in field_map.items():
        val = getattr(opts, field)
        # Fallback chain: Option Override -> Config Value -> Recommended Default
        if val is None:
            val = config.get(conf_key, default)
        setattr(resolved, field, val)

    # 2.4 Automate scope/identifier resolution from user_input if missing
    if (resolved.scope is None or resolved.identifier is None) and opts.user_input:
        from .utils import extract_scope_and_identifier

        s, i = extract_scope_and_identifier(opts.user_input)
        if resolved.scope is None:
            resolved.scope = s
        if resolved.identifier is None:
            resolved.identifier = i

    # Copy utility fields
    resolved.model_target = opts.model_target or resolved.model_target

    # Enforce service-level contracts (Linear Data Flow)
    if service_type == "photo_analysis":
        resolved.model_target = MODEL_TARGET_VISION

    resolved.client_tools = opts.client_tools
    resolved.scope = resolved.scope or opts.scope
    resolved.identifier = resolved.identifier or opts.identifier
    # Use function parameter as primary source, fallback to opts
    resolved.subentry_id = subentry_id or opts.subentry_id
    resolved.task_name = opts.task_name
    resolved.task_structure = opts.task_structure
    resolved.mixed_content = opts.mixed_content
    resolved.extra_content = opts.extra_content
    resolved.timer = opts.timer
    resolved.response_format = opts.response_format
    resolved.system_prompt = opts.system_prompt
    resolved.user_input = opts.user_input
    resolved.is_fallback = opts.is_fallback
    resolved.forced_last_message = opts.forced_last_message
    resolved.messages = opts.messages
    resolved.prompt_hash = opts.prompt_hash
    resolved.prompt_suffix = opts.prompt_suffix

    resolved.config = config
    resolved.is_resolved = True
    return resolved


async def resolve_memory_context(
    hass: HomeAssistant,
    mode: str,
    options: ChatOptions,
    prompt_hash: str = "",
) -> tuple[str | None, str | None, str | None]:
    """Resolve conversation key, previous response ID, and stored hash.

    Args:
        hass: Home Assistant instance.
        mode: Chat mode (tools, pipeline, etc.).
        options: Chat options (contains config, store_messages, etc.).
        prompt_hash: Current hash of the prompt.

    Returns:
        A tuple of (conv_key, previous_response_id, stored_hash).
    """
    # ZDR allows memory context (for blob) even if server storage is disabled
    config = options.config or {}
    store_messages = options.store_messages

    is_zdr = options.use_encrypted_content
    if is_zdr is None:
        is_zdr = config.get(CONF_ZDR, False)

    if not (store_messages or is_zdr) or not options.scope or not options.identifier:
        return None, None, None

    # subentry_id is required for memory isolation
    if not options.subentry_id:
        LOGGER.debug("[gateway] memory: skipped - no subentry_id")
        return None, None, None

    conv_key = MemoryManager.generate_key(
        options.scope, options.identifier, mode, options.subentry_id
    )
    memory = hass.data[DOMAIN]["conversation_memory"]
    previous_response_id = None
    stored_hash = None

    if not is_zdr:
        previous_response_id = await memory.async_get_last_response_id(conv_key)
        stored_hash = await memory.async_get_stored_hash(conv_key)

    if previous_response_id:
        status = "CONTINUING CONVERSATION"
        if stored_hash and prompt_hash and stored_hash == prompt_hash:
            status += " (hash matched)"
        LOGGER.debug(
            "[gateway] memory: %s mode=%s key=%s id=%s hash=%s",
            status,
            mode,
            conv_key[:8] if conv_key else "none",
            previous_response_id[:8],
            stored_hash[:8] if stored_hash else "none",
        )
    else:
        LOGGER.debug(
            "[gateway] memory: NEW CONVERSATION mode=%s key=%s",
            mode,
            conv_key[:8] if conv_key else "none",
        )

    return conv_key, previous_response_id, stored_hash


# ==========================================================================
# LOGGING & REMOTE DELETION
# ==========================================================================


def _extract_usage(
    response: Any, service_type: str, options: ChatOptions | None
) -> tuple[Any, int, int, int, int]:
    """Extract usage info from response. Returns (usage, prompt, completion, cached, reasoning)."""
    usage = getattr(response, "usage", None)
    if usage:
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        cached_tokens = 0
        reasoning_tokens = getattr(usage, "reasoning_tokens", 0) or 0
        try:
            details = getattr(usage, "prompt_tokens_details", None)
            cached_tokens = (
                getattr(details, "cached_tokens", 0)
                if details
                else getattr(usage, "cached_prompt_text_tokens", 0) or 0
            )

            # Try extracting reasoning tokens from details if top-level failed
            if reasoning_tokens == 0:
                completion_details = getattr(usage, "completion_tokens_details", None)
                if completion_details:
                    reasoning_tokens = (
                        getattr(completion_details, "reasoning_tokens", 0) or 0
                    )

        except AttributeError:
            pass
        return usage, prompt_tokens, completion_tokens, cached_tokens, reasoning_tokens

    # Fallback for image generation
    if (
        service_type == "ai_task"
        and options
        and options.model_target == MODEL_TARGET_IMAGE
    ):
        from types import SimpleNamespace

        return SimpleNamespace(completion_tokens=1, prompt_tokens=0), 0, 1, 0, 0
    return None, 0, 0, 0, 0


async def async_log_completion(
    response: Any,
    service_type: str,
    conv_key: str | None = None,
    options: ChatOptions | None = None,
    is_fallback: bool = False,
    entity: Any | None = None,
    citations: list | None = None,
    num_sources_used: int = 0,
    model_name: str | None = None,
    encrypted_content: str | None = None,
    hass: Any | None = None,
) -> None:
    """Log completion metadata, update token stats, and save memory.

    This is the single entry point for post-response processing.
    Requires entity or hass to be provided.
    """
    if response is None:
        return

    # Get hass from entity or parameter
    if entity and hasattr(entity, "hass"):
        hass = entity.hass
    elif not hass:
        LOGGER.warning("[gateway] async_log_completion: entity or hass required")
        return

    try:
        # Extract usage and model
        usage, prompt_tokens, completion_tokens, cached_tokens, reasoning_tokens = (
            _extract_usage(response, service_type, options)
        )
        model_used = model_name or getattr(response, "model", None) or "unknown"
        mode = options.mode if options else "unknown"

        # Resolve store_messages: explicit option > entity config > False
        store_messages = False
        if options and options.store_messages is not None:
            store_messages = options.store_messages
        elif entity and hasattr(entity, "get_config_dict"):
            store_messages = entity.get_config_dict().get(CONF_STORE_MESSAGES, False)

        response_id = getattr(response, "id", None)
        server_side_tool_usage = getattr(response, "server_side_tool_usage", None)

        LOGGER.debug(
            "[gateway] tokens: in=%d out=%d cached=%d reasoning=%d%s",
            prompt_tokens,
            completion_tokens,
            cached_tokens,
            reasoning_tokens,
            " (pipeline fallback in mode tools)" if is_fallback else "",
        )

        # ZDR: Save encrypted blob if present
        if conv_key:
            zdr_blob = encrypted_content or getattr(response, "encrypted_content", None)
            if zdr_blob:
                memory = hass.data[DOMAIN]["conversation_memory"]
                hass.async_create_task(
                    memory.async_save_encrypted_blob(conv_key, zdr_blob)
                )

        # Build async tasks for parallel execution
        tasks = []

        # Task 1: Save response ID to conversation memory
        if response_id and conv_key and service_type == "conversation":

            async def _save_response_id():
                try:
                    memory = hass.data[DOMAIN]["conversation_memory"]
                    await memory.async_save_response(
                        conv_key,
                        response_id,
                        store_messages,
                        prompt_hash=options.prompt_hash if options else None,
                    )
                    LOGGER.debug(
                        "[gateway] memory_save: mode=%s key=%s id=%s store=%s",
                        mode,
                        conv_key[:8],
                        response_id[:8],
                        store_messages,
                    )
                except Exception as err:
                    LOGGER.error("Failed to store response_id: %s", err)

            tasks.append(_save_response_id())

        # Task 2: Update token stats (always)
        if usage:

            async def _update_token_stats():
                token_stats = hass.data.get(DOMAIN, {}).get("token_stats")
                if token_stats:
                    await token_stats.async_update_usage(
                        service_type=service_type,
                        model=model_used,
                        usage=usage,
                        mode=mode,
                        is_fallback=is_fallback,
                        store_messages=store_messages,
                        server_side_tool_usage=server_side_tool_usage,
                        num_sources_used=num_sources_used,
                        response_id=response_id,
                        reasoning_tokens=reasoning_tokens,
                    )

            tasks.append(_update_token_stats())

        # Execute tasks: fire-and-forget with graceful shutdown tracking
        if tasks:

            async def _background_save():
                try:
                    await asyncio.gather(*tasks, return_exceptions=True)
                except Exception as err:
                    LOGGER.error("Background save failed: %s", err)

            task = asyncio.create_task(_background_save())

            # Track for graceful shutdown
            if "pending_save_tasks" not in hass.data[DOMAIN]:
                hass.data[DOMAIN]["pending_save_tasks"] = set()
            hass.data[DOMAIN]["pending_save_tasks"].add(task)
            task.add_done_callback(
                lambda t: hass.data[DOMAIN]["pending_save_tasks"].discard(t)
            )

            # Also track in entity if provided
            if entity is not None and hasattr(entity, "_pending_save_tasks"):
                entity._pending_save_tasks.add(task)
                task.add_done_callback(lambda t: entity._pending_save_tasks.discard(t))

        # Debug log for search features (only if both present)
        if service_type == "conversation" and citations and num_sources_used > 0:
            LOGGER.debug(
                "conversation: citations=%d sources=%d",
                len(citations),
                num_sources_used,
            )

    except Exception as err:
        LOGGER.warning("[gateway] completion log failed: %s", err)


# =========================================================================
# SDK PAYLOAD BUILDERS (SRP)
# =========================================================================


def prepare_sdk_payload(
    messages: list[dict],
    params: ChatOptions,
    system_prompt: str | None = None,
    session_context: str | None = None,
    encrypted_blob: str | None = None,
) -> list[Any]:
    """Build the final SDK message list with system prompt and ZDR state."""
    sdk_messages = []

    # 1. Add System Prompt (with redundancy check)
    if system_prompt:
        has_system_in_history = any(
            m.get("role") == "system" and m.get("content") for m in messages
        )
        if not has_system_in_history:
            full_content = system_prompt
            if session_context:
                full_content = f"{system_prompt}\n{session_context}"
            sdk_messages.append(translate_system_message(full_content))

    # 2. Add History turns and apply prompt suffix if needed
    if messages:
        # Detect if we need to apply a suffix to the last user message
        final_messages = messages
        if params.prompt_suffix:
            # Create a copy to avoid mutating the input list/dicts
            final_messages = [dict(m) for m in messages]
            # Find the last USER message
            for i in range(len(final_messages) - 1, -1, -1):
                if final_messages[i].get("role") == "user":
                    content = final_messages[i].get("content", "")
                    if isinstance(content, list):
                        # Append to the first text part found, or add a new string part
                        text_part_found = False
                        for j in range(len(content)):
                            if isinstance(content[j], str) and not content[j].startswith(
                                ("data:", "http")
                            ):
                                content[j] += params.prompt_suffix
                                text_part_found = True
                                break
                        if not text_part_found:
                            content.append(params.prompt_suffix)
                        final_messages[i]["content"] = content
                    else:
                        final_messages[i]["content"] = (
                            str(content) + params.prompt_suffix
                        )
                    break

        msg_list = translate_messages_to_sdk(final_messages)
        if msg_list:
            sdk_messages.extend(msg_list)

    # 3. Inject ZDR Blob (Session restoration)
    if encrypted_blob:
        blob_msg = translate_assistant_message(
            content="", encrypted_content=encrypted_blob
        )
        # Insert after system messages but before interaction turns
        insert_idx = 0
        for i, m in enumerate(sdk_messages):
            if m.role != chat_pb2.MessageRole.ROLE_SYSTEM:
                insert_idx = i
                break
        else:
            insert_idx = len(sdk_messages)

        sdk_messages.insert(insert_idx, blob_msg)

    return sdk_messages


def get_tool_call_type(tool_call: Any) -> str | None:
    """Determine tool call type based on SDK logic."""
    if not chat_pb2 or not get_tool_call_type_sdk:
        return None

    type_id = get_tool_call_type_sdk(tool_call)

    if type_id == chat_pb2.ToolCallType.TOOL_CALL_TYPE_CLIENT_SIDE_TOOL:
        return "client_side_tool"

    # Server-side tools that we should NOT execute locally
    if type_id in (
        chat_pb2.ToolCallType.TOOL_CALL_TYPE_WEB_SEARCH_TOOL,
        chat_pb2.ToolCallType.TOOL_CALL_TYPE_X_SEARCH_TOOL,
        chat_pb2.ToolCallType.TOOL_CALL_TYPE_CODE_EXECUTION_TOOL,
        chat_pb2.ToolCallType.TOOL_CALL_TYPE_COLLECTIONS_SEARCH_TOOL,
        chat_pb2.ToolCallType.TOOL_CALL_TYPE_MCP_TOOL,
        chat_pb2.ToolCallType.TOOL_CALL_TYPE_ATTACHMENT_SEARCH_TOOL,
    ):
        return "server_side_tool"

    # Fallback/Safety: check tool name if type is unknown or sdk mismatch
    # Broadened to catch variants like x_news, x_image, x_trends, etc.
    tool_name = getattr(getattr(tool_call, "function", None), "name", "").lower()
    if tool_name and (
        tool_name.startswith(("x_", "web_", "browse"))
        or any(kw in tool_name for kw in ("search", "keyword", "semantic", "trends"))
    ):
        return "server_side_tool"

    return None


def translate_system_message(content: str) -> Any:
    """Create an SDK system message."""
    return xai_system(content) if xai_system else None


def translate_assistant_message(
    content: str | None,
    tool_calls: list | None = None,
    encrypted_content: str | None = None,
) -> Any:
    """Create an SDK assistant message with local conversion logic."""
    if not chat_pb2:
        return None

    if tool_calls or encrypted_content:
        from .tools_ha_to_xai import convert_ha_to_xai_tool_call

        processed_tool_calls = []
        if tool_calls:
            for tc in tool_calls:
                processed_tool_calls.append(convert_ha_to_xai_tool_call(tc, chat_pb2))

        return chat_pb2.Message(
            role=chat_pb2.MessageRole.ROLE_ASSISTANT,
            content=[xai_text(content or "")],
            tool_calls=processed_tool_calls,
            encrypted_content=encrypted_content,
        )
    return xai_assistant(content)


def translate_tool_message(content: str, tool_call_id: str = "") -> Any:
    """Create an SDK tool message with optional tool_call_id for proper linking.

    The tool_call_id links this result to a specific tool call (should match
    tool_call.id from the assistant's response). Supported since xai_sdk 1.6.0.
    """
    if not xai_tool_result:
        return None

    # Pass tool_call_id only if supported by the SDK version
    if tool_call_id:
        import inspect

        if "tool_call_id" in inspect.signature(xai_tool_result).parameters:
            return xai_tool_result(str(content), tool_call_id=tool_call_id)

    return xai_tool_result(str(content))


def translate_image_message(data_uri_or_url: str) -> Any:
    """Create an SDK image message."""
    return xai_image(data_uri_or_url) if xai_image else None


def translate_messages_to_sdk(messages: list[dict]) -> list[Any]:
    """Translate neutral message payload to SDK protobuf objects."""
    sdk_messages = []
    for msg_data in messages:
        role = msg_data.get("role")
        content = msg_data.get("content")

        if role == "system":
            sdk_messages.append(translate_system_message(content))
        elif role == "user":
            if isinstance(content, list):
                # Mixed content (Vision/Text)
                parts = []
                for p in content:
                    if isinstance(p, str):
                        if p.startswith(("data:", "http")):
                            parts.append(translate_image_message(p))
                        else:
                            parts.append(xai_text(p))
                sdk_messages.append(xai_user(*parts))
            else:
                sdk_messages.append(xai_user(content))
        elif role == "assistant":
            sdk_messages.append(
                translate_assistant_message(
                    content,
                    tool_calls=msg_data.get("tool_calls"),
                    encrypted_content=msg_data.get("encrypted_content"),
                )
            )
        elif role == "tool":
            sdk_messages.append(
                translate_tool_message(
                    content=content,
                    tool_call_id=msg_data.get("tool_call_id"),
                )
            )
    return sdk_messages


def _convert_dict_to_sdk_tool(tool_dict: dict) -> Any:
    """Convert a dict tool spec to xAI SDK Tool format.

    Args:
        tool_dict: Dict with 'name', 'description', 'parameters' keys.

    Returns:
        SDK Tool object created via xai_tool() helper.
    """
    if not xai_tool:
        return tool_dict  # Fallback if SDK not available

    # xai_tool() accepts parameters as dict directly (JSON schema format)
    return xai_tool(
        name=tool_dict.get("name", "unknown"),
        description=tool_dict.get("description", ""),
        parameters=tool_dict.get("parameters", {}),
    )


def assemble_chat_args(
    params: ChatOptions,
    sdk_messages: list[Any],
    store_messages: bool = False,
    previous_response_id: str | None = None,
) -> dict[str, Any]:
    """Assemble common chat arguments for the SDK."""
    args = {
        "model": params.model,
        "store_messages": store_messages,
        "messages": sdk_messages,
    }

    if previous_response_id:
        args["previous_response_id"] = previous_response_id

    if params.max_tokens:
        args["max_tokens"] = int(params.max_tokens)
    if params.temperature is not None:
        args["temperature"] = params.temperature
    if params.top_p is not None:
        args["top_p"] = params.top_p
    # Only send reasoning_effort for models that explicitly support it
    is_reasoning_model = params.model and (
        params.model in REASONING_EFFORT_MODELS
        or "reasoning" in params.model.lower()
        and "non-reasoning" not in params.model.lower()
    )
    if params.reasoning_effort and is_reasoning_model:
        args["reasoning_effort"] = params.reasoning_effort

    # Safety: ensure presence_penalty, frequency_penalty and stop are NOT sent for reasoning models
    # (they are currently not configured in ChatOptions, but this is a future-proof guard)
    if is_reasoning_model:
        args.pop("presence_penalty", None)
        args.pop("frequency_penalty", None)
        args.pop("stop", None)

    # Live search tools (Server-side)
    # SAFETY: Specialized models (Vision, Image) do NOT support server-side tools.
    search_mode = params.live_search or LIVE_SEARCH_OFF
    server_tools = []

    if search_mode != LIVE_SEARCH_OFF and params.model_target not in (
        MODEL_TARGET_VISION,
        MODEL_TARGET_IMAGE,
    ):
        # SECURITY/API FILTER: Grok-3 and older do NOT support server-side tools
        model_name = (params.model or "").lower()
        version_match = re.search(r"grok-(\d+)", model_name)
        if version_match:
            major_version = int(version_match.group(1))
            if major_version < 4:
                LOGGER.debug(
                    "[gateway] Stripping server-side tools: model %s < grok-4",
                    params.model,
                )
                search_mode = LIVE_SEARCH_OFF

        # If still enabled, build session/location options and tools
        if search_mode != LIVE_SEARCH_OFF:
            # Build user location kwargs for web_search (SDK >= 1.6.0)
            ws_kwargs: dict[str, str] = {}
            loc_str = (
                params.config.get(CONF_LOCATION_CONTEXT, "").strip()
                if params.config
                else ""
            )
            if loc_str:
                import inspect

                if "user_location_city" in inspect.signature(web_search).parameters:
                    parts = [p.strip() for p in loc_str.split(",")]
                    ws_kwargs["user_location_city"] = parts[0]
                    if len(parts) > 1 and len(parts[-1]) == 2:
                        ws_kwargs["user_location_country"] = parts[-1].upper()

            if search_mode == LIVE_SEARCH_WEB:
                server_tools.append(web_search(**ws_kwargs))
            elif search_mode == LIVE_SEARCH_X:
                server_tools.append(x_search())
            elif search_mode in [LIVE_SEARCH_FULL, "auto", "on"]:
                server_tools.extend([web_search(**ws_kwargs), x_search()])

    # User defined tools (client-side) - convert dicts to SDK protobuf format
    if params.client_tools:
        sdk_client_tools = []
        for tool in params.client_tools:
            if isinstance(tool, dict):
                sdk_client_tools.append(_convert_dict_to_sdk_tool(tool))
            else:
                sdk_client_tools.append(tool)  # Already SDK format

        if server_tools:
            args["tools"] = server_tools + sdk_client_tools
        else:
            args["tools"] = sdk_client_tools
    elif server_tools:
        args["tools"] = server_tools

    # JSON Format
    if params.response_format:
        fmt = params.response_format
        if isinstance(fmt, dict):
            import json

            args["response_format"] = {
                "format_type": chat_pb2.FormatType.FORMAT_TYPE_JSON_SCHEMA,
                "schema": json.dumps(fmt),
            }

    # Extra SDK features
    if params.use_encrypted_content:
        args["use_encrypted_content"] = True

    return args


def log_api_request(
    sdk_messages: list[Any],
    model: str,
    service_type: str,
    params: ChatOptions | None = None,
    is_stateless: bool = False,
) -> None:
    """Centralized logging for API request submission."""
    if not sdk_messages:
        return

    has_system = any(
        "ROLE_SYSTEM" in str(m).upper() or "ROLE: SYSTEM" in str(m).upper()
        for m in sdk_messages
    )

    if is_stateless:
        label = "STATELESS REQUEST"
    elif has_system:
        label = "FULL PROMPT (System + Interactions)"
    else:
        label = "NEW MESSAGES (Resuming Session)"

    temp = getattr(params, "temperature", "N/A")
    top_p = getattr(params, "top_p", "N/A")

    LOGGER.debug("[gateway] ====================================================")
    LOGGER.debug("[gateway] >>> SENDING %s TO xAI API <<<", label)
    LOGGER.debug(
        "[gateway] service=%s model=%s temp=%s top_p=%s",
        service_type,
        model,
        temp,
        top_p,
    )

    for idx, msg in enumerate(sdk_messages):
        # Protobuf repr is typically 'role: ROLE_NAME\ncontent: ...'
        role_str_upper = str(msg).upper()
        role = "UNKNOWN"
        if "ROLE_SYSTEM" in role_str_upper or "ROLE: SYSTEM" in role_str_upper:
            role = "SYSTEM"
        elif "ROLE_USER" in role_str_upper or "ROLE: USER" in role_str_upper:
            role = "USER"
        elif "ROLE_ASSISTANT" in role_str_upper or "ROLE: ASSISTANT" in role_str_upper:
            role = "ASSISTANT"
        elif "ROLE_TOOL" in role_str_upper or "ROLE: TOOL" in role_str_upper:
            role = "TOOL"

        content_text = ""
        if hasattr(msg, "content"):
            for part in msg.content:
                if hasattr(part, "text") and part.text:
                    content_text = part.text
                    break

        extra_info = []
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            extra_info.append(f"{len(msg.tool_calls)} tool calls")
        if hasattr(msg, "encrypted_content") and msg.encrypted_content:
            extra_info.append("ZDR blob")

        extra_str = f" ({', '.join(extra_info)})" if extra_info else ""

        if content_text:
            LOGGER.debug(
                '[gateway] Message[%d]: role=%s text="%s"%s',
                idx,
                role,
                content_text,
                extra_str,
            )
        else:
            LOGGER.debug("[gateway] Message[%d]: role=%s%s", idx, role, extra_str)

    LOGGER.debug("[gateway] ====================================================")
