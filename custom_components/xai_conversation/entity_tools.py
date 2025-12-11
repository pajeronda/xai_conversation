"""Tools processing logic for the xAI Conversation integration."""

from __future__ import annotations

# Standard library imports
import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import json
import time

# Home Assistant imports
from homeassistant.components import conversation as ha_conversation
from homeassistant.helpers import llm as ha_llm

# Local imports
from .const import (
    CONF_ALLOW_SMART_HOME_CONTROL,
    CONF_SEND_USER_NAME,
    CONF_SHOW_CITATIONS,
    CONF_STORE_MESSAGES,
    LOGGER,
    RECOMMENDED_HISTORY_LIMIT_TURNS,
    RECOMMENDED_SHOW_CITATIONS,
    RECOMMENDED_SEND_USER_NAME,
    RECOMMENDED_STORE_MESSAGES,
)
from .helpers import (
    add_manual_history_to_chat,
    build_llm_context,
    build_session_context_info,
    convert_xai_to_ha_tool,
    format_tools_for_xai,
    filter_tools_by_exposed_domains,
    format_user_message_with_metadata,
    get_exposed_entities_with_aliases,
    get_last_user_message,
    PromptManager,
    save_response_metadata,
    XAIGateway,
    LogTimeServices,
    timed_stream_generator,
    CUSTOM_TOOLS,  # Now imported from helpers
)

# Import helper to distinguish tool types
try:
    from xai_sdk.tools import get_tool_call_type
except ImportError:
    get_tool_call_type = None

from .exceptions import (
    raise_generic_error,
)

# gRPC imports for error handling (conditional)
try:
    from grpc import StatusCode
    from grpc._channel import _InactiveRpcError
except ImportError:
    StatusCode = None
    _InactiveRpcError = None
except Exception:
    StatusCode = None
    _InactiveRpcError = None


if TYPE_CHECKING:
    from .entity import XAIBaseLLMEntity


@dataclass
class ToolOutput:
    """Result from a tool execution."""

    tool_name: str
    tool_result: Any
    is_error: bool = False


class XAIToolsProcessor:
    """
    Handles the processing of conversations in 'tools' mode for a XAIBaseLLMEntity.

    This class encapsulates the logic for the tool-calling loop, API interactions,
    and state management required when Home Assistant's LLM API is active.
    """

    def __init__(self, entity: XAIBaseLLMEntity):
        """
        Initialize the XAIToolsProcessor.

        Args:
            entity: The parent XAIBaseLLMEntity instance.
        """
        self._entity = entity
        # Cache for static context to avoid re-computation
        self._cached_static_prompt: str | None = None
        self._cached_xai_tools: list | None = None
        self._cached_ha_tools: list | None = None  # Cache for original HA tools
        self._cached_ha_tools_dict: dict[str, Any] | None = (
            None  # Cache for O(1) tool lookup by name
        )
        self._cached_active_domains: set[str] | None = (
            None  # Cache for active domains to trigger tool rebuild
        )
        self._cached_entity_ids: set[str] | None = (
            None  # Cache for active entity IDs to trigger tool rebuild
        )
        # Instance of HA's AssistAPI
        self._assist_api: ha_llm.AssistAPI | None = None
        # Calculate and cache tools enabled/disabled state once
        allow_control = self._entity._get_option(CONF_ALLOW_SMART_HOME_CONTROL, True)
        self._use_tools = allow_control

    async def async_process_with_loop(
        self, user_input, chat_log, timer: LogTimeServices, force_tools: bool = False
    ) -> None:
        """Process a conversation with tools, using the provided timer for metrics."""
        LOGGER.debug(
            "tools_loop_start (delegated) for user_input: %s", user_input.text[:50]
        )

        # Get store_messages from entity config
        store_messages = self._entity._get_option(
            CONF_STORE_MESSAGES, RECOMMENDED_STORE_MESSAGES
        )

        # Override use_tools if force_tools is True (e.g., from fallback)
        use_tools = self._use_tools or force_tools

        # Get memory key and previous_response_id from ConversationMemory
        (
            conv_key,
            prev_id_from_memory,
        ) = await self._entity._conversation_memory.get_conv_key_and_prev_id(
            user_input, "tools", self._entity.subentry.data
        )

        max_iterations = 5  # Safety break to prevent infinite loops

        for i in range(max_iterations):
            LOGGER.debug("tools_loop: iteration %d/%d", i + 1, max_iterations)

            # Get the latest previous_response_id for this iteration
            if i == 0:
                current_prev_id = prev_id_from_memory
                LOGGER.debug(
                    "tools_loop: using memory prev_id=%s",
                    current_prev_id[:8] if current_prev_id else None,
                )
            else:
                current_prev_id = (
                    await self._entity._conversation_memory.get_response_id_by_key(
                        conv_key
                    )
                )

            # Prepare chat object (but don't call API yet - that happens in streaming)
            chat = await self._prepare_chat_for_tools(
                user_input, chat_log, current_prev_id, use_tools
            )

            # Stream the response while accumulating full response object
            response_holder = {"response": None}
            is_fallback = force_tools and i == 0

            try:
                async for _ in chat_log.async_add_delta_content_stream(
                    agent_id=self._entity.entity_id,
                    stream=self._delta_generator(chat, response_holder, timer),
                ):
                    # Yield control to event loop to allow UI updates
                    await asyncio.sleep(0)

            except Exception as err:
                # Handle gRPC NOT_FOUND errors (expired response_id)
                if _InactiveRpcError is not None and isinstance(err, _InactiveRpcError):
                    from .helpers.response import handle_response_not_found_error

                    should_retry = await handle_response_not_found_error(
                        err=err,
                        attempt=i,
                        memory=self._entity._conversation_memory,
                        conv_key=conv_key,
                        mode="tools",
                        context_id=user_input.conversation_id,
                    )

                    if should_retry:
                        # Retry iteration with fresh conversation
                        continue
                    else:
                        LOGGER.error("tools_loop: NOT_FOUND error but retry not possible")
                        break
                # Re-raise non-gRPC errors
                raise

            # Extract final response from holder
            # The _delta_generator stores the final response in the holder
            response = response_holder.get("response")
            if not response:
                LOGGER.error("tools_loop: no response from stream!")
                break

            # Save response metadata
            await save_response_metadata(
                hass=self._entity.hass,
                entry_id=self._entity.entry.entry_id,
                usage=getattr(response, "usage", None),
                model=self._entity._model,  # Explicitly pass model
                service_type="conversation",
                mode="tools",
                is_fallback=is_fallback,
                store_messages=store_messages,
                conv_key=conv_key,
                response_id=getattr(response, "id", None),
                entity=self._entity,
                citations=response_holder.get("citations"),
                num_sources_used=response_holder.get("num_sources_used", 0),
                server_side_tool_usage=response_holder.get("server_side_tool_usage"),
            )

            # Retrieve citations and num_sources_used
            citations = response_holder.get("citations")
            num_sources_used = response_holder.get("num_sources_used", 0)

            # Log search details if available
            if citations:
                LOGGER.debug("tools_loop: citations found: %d", len(citations))
                for citation in citations:
                    LOGGER.debug("Citation: %s", citation)
            if num_sources_used > 0:
                LOGGER.debug(
                    "tools_loop: unique search sources used: %d", num_sources_used
                )

            # Append citations to the chat log if enabled (useful for UI, noisy for voice)
            show_citations = self._entity._get_option(
                CONF_SHOW_CITATIONS, RECOMMENDED_SHOW_CITATIONS
            )

            if citations and show_citations:
                formatted_citations = "\n\nCitations:\n"
                for i, citation in enumerate(citations):
                    # Extract title and url - handle strings (URLs), dicts, and objects
                    if isinstance(citation, str):
                        # xAI returns citations as URL strings
                        url = citation
                        # Try to extract a meaningful title from URL
                        if 'x.com' in url or 'twitter.com' in url:
                            title = 'X/Twitter Post'
                        elif 'github.com' in url:
                            title = 'GitHub'
                        else:
                            # Use domain as title
                            from urllib.parse import urlparse
                            parsed = urlparse(url)
                            title = parsed.netloc or 'Web Source'
                    elif isinstance(citation, dict):
                        title = citation.get('title', 'No Title')
                        url = citation.get('url', 'No URL')
                    else:
                        # Object with attributes
                        title = getattr(citation, 'title', 'No Title')
                        url = getattr(citation, 'url', 'No URL')

                    formatted_citations += f"[{i + 1}] {title} - {url}\n"
                async for _ in chat_log.async_add_assistant_content(
                    ha_conversation.AssistantContent(
                        agent_id=self._entity.entity_id,
                        content=formatted_citations,
                    )
                ):
                    pass
            elif citations and not show_citations:
                LOGGER.debug(
                    "tools_loop: citations available (%d) but show_citations=False, skipping chat log append",
                    len(citations)
                )

            # Check for tool calls
            has_tool_calls = hasattr(response, "tool_calls") and response.tool_calls

            if not has_tool_calls:
                # No tool calls - streaming is complete, exit
                LOGGER.debug("tools_loop: no tool calls, final answer streamed.")
                break

            # Has tool calls - execute them
            llm_context = build_llm_context(self._entity.hass, user_input)
            tool_results = []

            for tool_call in response.tool_calls:
                # Filter out server-side tools (e.g. web_search) that are already executed by xAI
                if get_tool_call_type:
                    tool_type = get_tool_call_type(tool_call)
                    if tool_type != "client_side_tool":
                        LOGGER.debug(
                            "Skipping server-side tool call: %s (type: %s)",
                            tool_call.function.name,
                            tool_type,
                        )
                        continue

                tool_start = time.time()
                LOGGER.debug("tools_loop: executing tool '%s'", tool_call.function.name)

                # Security check
                if tool_call.function.name == "GetLiveContext" and not use_tools:
                    LOGGER.warning(
                        "Blocked attempt to call GetLiveContext while tools are disabled."
                    )
                    tool_results.append(
                        ToolOutput(
                            tool_name=tool_call.function.name,
                            tool_result="Tool execution blocked: home control is disabled in chat-only mode.",
                            is_error=True,
                        )
                    )
                    continue

                try:
                    ha_tool_input = convert_xai_to_ha_tool(self._entity.hass, tool_call)
                    if self._cached_ha_tools_dict is None:
                        raise_generic_error("Tools cache not initialized")
                    ha_tool = self._cached_ha_tools_dict.get(ha_tool_input.tool_name)
                    if not ha_tool:
                        raise_generic_error(
                            f"Tool '{ha_tool_input.tool_name}' not found in cached tools"
                        )

                    result_data = await ha_tool.async_call(
                        self._entity.hass, ha_tool_input, llm_context
                    )
                    tool_results.append(
                        ToolOutput(
                            tool_name=ha_tool_input.tool_name, tool_result=result_data
                        )
                    )
                    tool_time = time.time() - tool_start
                    LOGGER.info(
                        "tool_exec: name=%s duration=%.2fs success=True",
                        tool_call.function.name,
                        tool_time,
                    )

                except Exception as err:
                    tool_time = time.time() - tool_start
                    LOGGER.debug(
                        "tool_exec: name=%s duration=%.2fs success=False error=%s params=%s",
                        tool_call.function.name,
                        tool_time,
                        str(err)[:100],
                        str(tool_call.function.arguments)[:100],
                    )
                    tool_results.append(
                        ToolOutput(
                            tool_name=tool_call.function.name,
                            tool_result=f"An error occurred while trying to execute the tool: {err}",
                            is_error=True,
                        )
                    )

            # Stream the final answer after tool execution
            await self._stream_final_answer_after_tools(
                tool_results, response.id, conv_key, user_input, chat_log, timer
            )
            break  # Tool cycle complete

        else:
            LOGGER.warning(
                "tools_loop: reached max iterations (%d), aborting.", max_iterations
            )
            await self._entity._add_error_response_and_continue(
                user_input,
                chat_log,
                "I'm having trouble completing the request, it seems to be stuck in a loop.",
            )

    async def _prepare_chat_for_tools(
        self,
        user_input,
        chat_log,
        previous_response_id: str | None = None,
        use_tools: bool = None,
    ):
        """Prepare xAI chat object for tools mode (without calling API).

        This method builds the chat object with system prompt, history, and tools,
        but doesn't execute the API call (that happens during streaming).

        Returns:
            chat: xAI chat object ready for streaming
        """
        # Use provided use_tools or fall back to instance default
        if use_tools is None:
            use_tools = self._use_tools

        # Only use previous_response_id in server-side mode
        store_messages = self._entity._get_option(
            CONF_STORE_MESSAGES, RECOMMENDED_STORE_MESSAGES
        )
        prev_id = previous_response_id if store_messages else None

        # Handle static context (prompt and tools) with caching
        needs_cache_rebuild = self._cached_xai_tools is None

        # Build LLMContext independently from chat_log
        llm_context = build_llm_context(self._entity.hass, user_input)

        # Get exposed entities - always check this to dynamically rebuild tools if domains change
        assist_api = ha_llm._async_get_apis(self._entity.hass).get(
            ha_llm.LLM_API_ASSIST
        )
        if not assist_api:
            raise_generic_error("AssistAPI is not available in Home Assistant.")
        exposed_entities_result = ha_llm._get_exposed_entities(
            self._entity.hass, "conversation", include_state=False
        )

        # Extract current active domains for comparison
        current_active_domains = (
            {
                entity_id.split(".")[0]
                for entity_id in exposed_entities_result["entities"]
            }
            if exposed_entities_result and "entities" in exposed_entities_result
            else set()
        )

        # Extract current entity IDs for comparison
        current_entity_ids = (
            set(exposed_entities_result["entities"].keys())
            if exposed_entities_result and "entities" in exposed_entities_result
            else set()
        )

        # If cache is not empty, check if active domains or entity IDs have changed
        if (
            not needs_cache_rebuild
            and self._cached_active_domains is not None
            and self._cached_entity_ids is not None
        ):
            if current_active_domains != self._cached_active_domains:
                LOGGER.debug(
                    "cache_rebuild: Active domains changed from %s to %s",
                    self._cached_active_domains,
                    current_active_domains,
                )
                needs_cache_rebuild = True
            elif current_entity_ids != self._cached_entity_ids:
                LOGGER.debug(
                    "cache_rebuild: Exposed entity IDs changed from %s to %s",
                    self._cached_entity_ids,
                    current_entity_ids,
                )
                needs_cache_rebuild = True

        if needs_cache_rebuild:
            cache_start = time.time()
            rebuild_reason = (
                "cache_empty"
                if self._cached_xai_tools is None
                else "active_domains_changed"
                if current_active_domains != self._cached_active_domains
                else "entity_ids_changed"
            )
            LOGGER.debug(
                "cache_rebuild: reason=%s use_tools=%s", rebuild_reason, use_tools
            )

            # Update cached active domains and entity IDs
            self._cached_active_domains = current_active_domains
            self._cached_entity_ids = current_entity_ids

            if use_tools:
                # --- TOOLS ENABLED ---
                self._assist_api = ha_llm._async_get_apis(self._entity.hass).get(
                    ha_llm.LLM_API_ASSIST
                )
                if not self._assist_api:
                    raise_generic_error("AssistAPI is not available in Home Assistant.")

                # Get exposed entities
                exposed_entities_result = ha_llm._get_exposed_entities(
                    self._entity.hass, "conversation", include_state=False
                )

                # Get static list of tools
                ha_tools = self._assist_api._async_get_tools(
                    llm_context, exposed_entities_result
                )

                # Filter tools based on active domains to reduce context size
                ha_tools = filter_tools_by_exposed_domains(
                    ha_tools, exposed_entities_result
                )

                # --- INJECT CUSTOM TOOLS ---
                # Map custom tool names to required domains
                custom_tool_requirements = {
                    "HassSetInputNumber": "input_number",
                    "HassSetInputBoolean": "input_boolean",
                    "HassSetInputText": "input_text",
                    "HassRunScript": "script",
                    "HassTriggerAutomation": "automation",
                }

                for custom_tool in CUSTOM_TOOLS:
                    required_domain = custom_tool_requirements.get(custom_tool.name)
                    # Only add tool if its required domain is active (present in exposed entities)
                    if required_domain and required_domain in current_active_domains:
                        LOGGER.debug(
                            "Injecting custom tool '%s' (domain '%s' is active)",
                            custom_tool.name,
                            required_domain,
                        )
                        ha_tools.append(custom_tool)
                # ---------------------------

                self._cached_ha_tools = ha_tools
                self._cached_ha_tools_dict = {tool.name: tool for tool in ha_tools}
                # Use XAIGateway.tool_def for tool definition wrapper
                self._cached_xai_tools = format_tools_for_xai(
                    ha_tools, XAIGateway.tool_def
                )

                # Build textual description of tools
                tool_descriptions = []
                for ha_tool in ha_tools:
                    tool_desc = f"- **{ha_tool.name}**: {ha_tool.description or 'No description'}"
                    tool_descriptions.append(tool_desc)
                tool_definitions_text = "\n".join(tool_descriptions)

                # Get entities enriched with aliases for context
                enriched_entities = get_exposed_entities_with_aliases(self._entity.hass)

                # Format static context as compact JSON
                static_context_json = (
                    json.dumps(
                        enriched_entities,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    if enriched_entities
                    else "[]"
                )

                # Build the full system prompt for tools mode
                tools_prompt_mgr = PromptManager(self._entity.subentry.data, "tools")
                self._cached_static_prompt = (
                    tools_prompt_mgr.build_base_prompt_with_user_instructions(
                        static_context=static_context_json,
                        tool_definitions=tool_definitions_text,
                    )
                )
            else:
                # --- TOOLS DISABLED (SIMPLE CHAT) ---
                tools_prompt_mgr = PromptManager(self._entity.subentry.data, "tools")
                self._cached_xai_tools = []
                self._cached_static_prompt = (
                    tools_prompt_mgr.build_base_prompt_with_user_instructions()
                )

            cache_time = time.time() - cache_start
            LOGGER.debug(
                "cache_build: duration=%.2fs tools_count=%d prompt_length=%d",
                cache_time,
                len(self._cached_xai_tools),
                len(self._cached_static_prompt),
            )
        else:
            LOGGER.debug(
                "cache_hit: Reusing cached static context (tools=%d)",
                len(self._cached_xai_tools),
            )

        # Build the API call
        tools_for_call = self._cached_xai_tools

        LOGGER.debug(
            "PREPARING API CALL: prev_id=%s, tools_count=%d, has_static_prompt=%s",
            prev_id[:8] if prev_id else None,
            len(tools_for_call) if tools_for_call else 0,
            bool(self._cached_static_prompt),
        )

        chat = self._entity.gateway.create_chat(
            service_type="conversation",
            subentry_id=self._entity.subentry.subentry_id,
            client_tools=tools_for_call,
            previous_response_id=prev_id,
        )

        # On first call, send the full static prompt
        if not prev_id:
            if self._cached_static_prompt:
                # Add temporal and geographic context to system prompt
                context_info = build_session_context_info(self._entity.hass)
                system_prompt_with_context = self._cached_static_prompt + context_info

                LOGGER.debug(
                    "FIRST MESSAGE: Adding system prompt (length=%d chars)",
                    len(system_prompt_with_context),
                )
                LOGGER.debug("System prompt:\n%s", system_prompt_with_context)
                chat.append(XAIGateway.system_msg(system_prompt_with_context))
            else:
                LOGGER.error(
                    "FIRST MESSAGE BUT NO STATIC PROMPT! This should never happen!"
                )
        else:
            LOGGER.debug(
                "SUBSEQUENT MESSAGE: Skipping system prompt (using previous_response_id)"
            )

        # Handle conversation history based on server-side memory availability
        store_messages = self._entity._get_option(
            CONF_STORE_MESSAGES, RECOMMENDED_STORE_MESSAGES
        )

        if prev_id or store_messages:
            # Server-side memory is active: send only the last user message
            last_user_message = get_last_user_message(chat_log)
            if last_user_message:
                send_user_name = self._entity._get_option(CONF_SEND_USER_NAME, RECOMMENDED_SEND_USER_NAME)

                # Format user message with metadata
                user_message_with_time = await format_user_message_with_metadata(
                    last_user_message, user_input, self._entity.hass, send_user_name
                )

                LOGGER.debug(
                    "Tools mode: sending last message only (server-side memory): %s",
                    last_user_message[:100],
                )

                chat.append(XAIGateway.user_msg(user_message_with_time))
        else:
            # NO server-side memory: send the last N turns manually
            await add_manual_history_to_chat(
                self._entity.hass,
                self._entity,
                chat,
                chat_log,
                user_input,
                RECOMMENDED_HISTORY_LIMIT_TURNS,
            )

        return chat

    async def _stream_final_answer_after_tools(
        self,
        tool_results: list[ToolOutput],
        previous_response_id: str,
        conv_key: str,
        user_input,
        chat_log,
        timer: LogTimeServices,
    ) -> None:
        """Stream the final answer after tool execution, passing the timer for metrics."""
        # If store_messages is disabled, don't use previous_response_id (server won't have it)
        store_messages = self._entity._get_option(
            CONF_STORE_MESSAGES, RECOMMENDED_STORE_MESSAGES
        )
        response_id = previous_response_id if store_messages else None

        chat = self._entity.gateway.create_chat(
            service_type="conversation",
            subentry_id=self._entity.subentry.subentry_id,
            previous_response_id=response_id,
            # No client tools needed for final answer generation (unless we want follow-up tools?)
            # Usually final answer is text.
            client_tools=None,
        )

        # When memory is disabled, rebuild the full context for this API call
        if not store_messages:
            LOGGER.debug(
                "_stream_final_answer: store_messages=False, rebuilding full context"
            )

            # 1. Add system prompt
            if self._cached_static_prompt:
                context_info = build_session_context_info(self._entity.hass)
                system_prompt_with_context = self._cached_static_prompt + context_info
                chat.append(XAIGateway.system_msg(system_prompt_with_context))
                LOGGER.debug(
                    "_stream_final_answer: added system prompt (%d chars)",
                    len(system_prompt_with_context),
                )
            else:
                LOGGER.warning(
                    "_stream_final_answer: no cached static prompt available!"
                )

            # 2. Add conversation history from chat_log
            await add_manual_history_to_chat(
                self._entity.hass,
                self._entity,
                chat,
                chat_log,
                user_input,
                RECOMMENDED_HISTORY_LIMIT_TURNS,
            )
        else:
            LOGGER.debug(
                "_stream_final_answer: store_messages=True, using previous_response_id=%s",
                response_id[:8] if response_id else None,
            )

        # 3. Append tool results as user messages
        for result in tool_results:
            tool_output_content = (
                result.tool_result
                if not result.is_error
                else f"Error: {result.tool_result}"
            )
            result_text = f"Tool '{result.tool_name}' result: {tool_output_content}"
            chat.append(XAIGateway.user_msg(result_text))
            LOGGER.debug(
                "Appending tool result for '%s' to get final answer.", result.tool_name
            )

        # Stream the final response
        response_holder = {"response": None}

        async for _ in chat_log.async_add_delta_content_stream(
            agent_id=self._entity.entity_id,
            stream=self._delta_generator(chat, response_holder, timer),
        ):
            await asyncio.sleep(0)

        # Extract final response
        final_response = response_holder.get("response")
        if not final_response:
            LOGGER.error("_stream_final_answer: no response from stream!")
            return

        # Save response metadata (store_messages already calculated at function start)
        await save_response_metadata(
            hass=self._entity.hass,
            entry_id=self._entity.entry.entry_id,
            usage=getattr(final_response, "usage", None),
            model=self._entity._model,  # Explicitly pass model
            service_type="conversation",
            mode="tools",
            is_fallback=False,
            store_messages=store_messages,
            conv_key=conv_key,
            response_id=getattr(final_response, "id", None),
            entity=self._entity,
        )

        content = getattr(final_response, "content", None)
        LOGGER.debug(
            "Streamed final answer from Grok (len=%d)", len(content) if content else 0
        )

    async def _delta_generator(
        self, chat, response_holder: dict, timer: LogTimeServices
    ):
        """Generate deltas for streaming, with API timing handled by timed_stream_generator."""
        new_message = True
        last_response = None

        try:
            # timed_stream_generator wrapper handles all API timing and reports to the timer
            async for response, chunk in timed_stream_generator(chat.stream(), timer):
                last_response = response

                # CRITICAL: Yield role BEFORE any content
                if new_message:
                    yield {"role": "assistant"}
                    new_message = False

                # Yield content chunk
                if chunk.content:
                    yield {"content": chunk.content}

            # Store the final response object, including citations and usage info
            response_holder["response"] = last_response
            response_holder["citations"] = getattr(last_response, "citations", [])

            # Extract num_sources_used from usage object
            usage_obj = getattr(last_response, "usage", None)
            num_sources = getattr(usage_obj, "num_sources_used", 0)
            response_holder["num_sources_used"] = num_sources

            # Extract server_side_tool_usage
            server_tools = getattr(last_response, "server_side_tool_usage", None)
            response_holder["server_side_tool_usage"] = server_tools

            # Debug logging
            if server_tools or num_sources > 0:
                LOGGER.debug(
                    "tools_stream: server_side_tool_usage=%s, num_sources_used=%d",
                    server_tools,
                    num_sources
                )

        except Exception as err:
            # Store error info in response_holder for caller to check
            response_holder["error"] = err
            response_holder["is_not_found"] = False

            # Check if it's a retryable NOT_FOUND error (use module-level imports)
            if (
                _InactiveRpcError is not None
                and StatusCode is not None
                and isinstance(err, _InactiveRpcError)
                and err.code() == StatusCode.NOT_FOUND
            ):
                response_holder["is_not_found"] = True
                LOGGER.warning("tools_stream: NOT_FOUND error detected, will retry")
                return

            # For other errors, yield error message to user
            LOGGER.error("tools_stream: error during native async streaming: %s", err)
            if new_message:
                yield {"role": "assistant"}
