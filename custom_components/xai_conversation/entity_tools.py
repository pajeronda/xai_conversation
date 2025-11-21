"""Tools processing logic for the xAI Conversation integration."""

from __future__ import annotations

# Standard library imports
import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

# Home Assistant and standard library imports (re-exported from __init__)
from .__init__ import (
    datetime,
    json,
    time,
    xai_system, xai_user, xai_assistant, xai_tool,
    ha_conversation,
    ha_llm,
)

# Local imports
from .const import (
    CONF_ALLOW_SMART_HOME_CONTROL,
    CONF_SEND_USER_NAME,
    CONF_STORE_MESSAGES,
    DOMAIN,
    LOGGER,
    RECOMMENDED_HISTORY_LIMIT_TURNS,
    RECOMMENDED_STORE_MESSAGES,
)
from .helpers import (
    convert_xai_to_ha_tool,
    extract_device_id,
    format_tools_for_xai,
    get_last_user_message,
    get_user_or_device_name,
    PromptManager,
)
from .exceptions import (
    raise_generic_error,
)

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
        self._cached_ha_tools_dict: dict[str, Any] | None = None  # Cache for O(1) tool lookup by name
        self._cached_llm_context: ha_llm.LLMContext | None = None  # Cache for llm_context
        # Instance of HA's AssistAPI
        self._assist_api: ha_llm.AssistAPI | None = None
        # Calculate and cache tools enabled/disabled state once
        allow_control = self._entity._get_option(CONF_ALLOW_SMART_HOME_CONTROL, True)
        self._use_tools = allow_control

    async def async_process_with_loop(
        self,
        user_input,
        chat_log,
        force_tools: bool = False
    ) -> None:
        """Process a conversation with tools using streaming."""
        start_time = time.time()
        LOGGER.debug("tools_loop_start (delegated) for user_input: %s", user_input.text[:50])

        # Override use_tools if force_tools is True (e.g., from fallback)
        use_tools = self._use_tools or force_tools

        # Get memory key and previous_response_id from ConversationMemory
        conv_key, prev_id_from_memory = await self._entity._conversation_memory.get_conv_key_and_prev_id(
            user_input,
            "tools",
            self._entity.subentry.data
        )

        max_iterations = 5  # Safety break to prevent infinite loops

        for i in range(max_iterations):
            LOGGER.debug("tools_loop: iteration %d/%d", i + 1, max_iterations)

            # Get the latest previous_response_id for this iteration
            if i == 0:
                current_prev_id = prev_id_from_memory
                LOGGER.debug("tools_loop: using memory prev_id=%s", current_prev_id[:8] if current_prev_id else None)
            else:
                current_prev_id = await self._entity._memory_get_prev_id(conv_key)

            # Prepare chat object (but don't call API yet - that happens in streaming)
            chat = await self._prepare_chat_for_tools(user_input, chat_log, current_prev_id, use_tools)

            # Stream the response while accumulating full response object
            response_holder = {"response": None}
            is_fallback = force_tools and i == 0

            async for _ in chat_log.async_add_delta_content_stream(
                agent_id=self._entity.entity_id,
                stream=self._delta_generator(chat, response_holder),
            ):
                # Yield control to event loop to allow UI updates
                await asyncio.sleep(0)

            # Extract final response from holder
            # The _delta_generator stores the final response in the holder
            response = response_holder.get("response")
            if not response:
                LOGGER.error("tools_loop: no response from stream!")
                break

            # Save response metadata
            if getattr(response, "id", None):
                await self._entity._memory_set_prev_id(conv_key, response.id)
                LOGGER.debug(
                    "memory_save: mode=tools conv_key=%s response_id=%s",
                    conv_key, response.id[:8]
                )

            # Update token sensors
            usage = getattr(response, "usage", None)
            model = getattr(response, "model", None)
            if usage:
                self._entity._update_token_sensors(
                    usage,
                    model=model,
                    service_type="conversation",
                    mode="tools",
                    is_fallback=is_fallback
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
                LOGGER.debug("tools_loop: unique search sources used: %d", num_sources_used)

            # Append citations to the chat log if available
            if citations:
                formatted_citations = "\n\nCitations:\n"
                for i, citation in enumerate(citations):
                    formatted_citations += f"  [{i+1}] {getattr(citation, 'title', 'No Title')} - {getattr(citation, 'url', 'No URL')}\n"
                async for _ in chat_log.async_add_assistant_content(
                    ha_conversation.AssistantContent(
                        agent_id=self._entity.entity_id,
                        content=formatted_citations,
                    )
                ):
                    pass

            # Check for tool calls
            has_tool_calls = hasattr(response, 'tool_calls') and response.tool_calls

            if not has_tool_calls:
                # No tool calls - streaming is complete, exit
                LOGGER.debug("tools_loop: no tool calls, final answer streamed.")
                break

            # Has tool calls - execute them
            llm_context = self._build_llm_context(user_input)
            tool_results = []

            for tool_call in response.tool_calls:
                tool_start = time.time()
                LOGGER.debug("tools_loop: executing tool '%s'", tool_call.function.name)

                # Security check
                if tool_call.function.name == "GetLiveContext" and not use_tools:
                    LOGGER.warning("Blocked attempt to call GetLiveContext while tools are disabled.")
                    tool_results.append(ToolOutput(
                        tool_name=tool_call.function.name,
                        tool_result="Tool execution blocked: home control is disabled in chat-only mode.",
                        is_error=True,
                    ))
                    continue

                try:
                    ha_tool_input = convert_xai_to_ha_tool(self._entity.hass, tool_call)
                    ha_tool = self._cached_ha_tools_dict.get(ha_tool_input.tool_name)
                    if not ha_tool:
                        raise_generic_error(f"Tool '{ha_tool_input.tool_name}' not found in cached tools")

                    result_data = await ha_tool.async_call(self._entity.hass, ha_tool_input, llm_context)
                    tool_results.append(ToolOutput(
                        tool_name=ha_tool_input.tool_name,
                        tool_result=result_data
                    ))
                    tool_time = time.time() - tool_start
                    LOGGER.info("tool_exec: name=%s duration=%.2fs success=True", tool_call.function.name, tool_time)

                except Exception as err:
                    tool_time = time.time() - tool_start
                    LOGGER.debug("tool_exec: name=%s duration=%.2fs success=False error=%s params=%s",
                                tool_call.function.name, tool_time, str(err)[:100],
                                str(tool_call.function.arguments)[:100])
                    tool_results.append(ToolOutput(
                        tool_name=tool_call.function.name,
                        tool_result=f"An error occurred while trying to execute the tool: {err}",
                        is_error=True,
                    ))

            # Stream the final answer after tool execution
            await self._stream_final_answer_after_tools(
                tool_results, response.id, conv_key, user_input, chat_log
            )
            break  # Tool cycle complete

        else:
            LOGGER.warning("tools_loop: reached max iterations (%d), aborting.", max_iterations)
            await self._entity._add_error_response_and_continue(
                user_input, chat_log, "I'm having trouble completing the request, it seems to be stuck in a loop."
            )

        total_time = time.time() - start_time
        LOGGER.debug("tools_loop_end: duration=%.2fs", total_time)

    async def _prepare_chat_for_tools(self, user_input, chat_log, previous_response_id: str | None = None, use_tools: bool = None):
        """Prepare xAI chat object for tools mode (without calling API).

        This method builds the chat object with system prompt, history, and tools,
        but doesn't execute the API call (that happens during streaming).

        Returns:
            chat: xAI chat object ready for streaming
        """
        # Use provided use_tools or fall back to instance default
        if use_tools is None:
            use_tools = self._use_tools

        client = self._entity.gateway.create_client()

        # Only use previous_response_id in server-side mode
        store_messages = self._entity._get_option(CONF_STORE_MESSAGES, RECOMMENDED_STORE_MESSAGES)
        prev_id = previous_response_id if store_messages else None

        # Handle static context (prompt and tools) with caching
        needs_cache_rebuild = (
            self._cached_xai_tools is None or
            (use_tools and (not self._cached_ha_tools or self._cached_xai_tools == []))
        )

        if needs_cache_rebuild:
            cache_start = time.time()
            rebuild_reason = "cache_empty" if self._cached_xai_tools is None else "force_tools_enabled"
            LOGGER.debug("cache_rebuild: reason=%s use_tools=%s", rebuild_reason, use_tools)

            # Build LLMContext independently from chat_log
            llm_context = self._build_llm_context(user_input)

            if use_tools:
                # --- TOOLS ENABLED ---
                self._assist_api = ha_llm._async_get_apis(self._entity.hass).get(ha_llm.LLM_API_ASSIST)
                if not self._assist_api:
                    raise_generic_error("AssistAPI is not available in Home Assistant.")

                # Get exposed entities
                exposed_entities_result = ha_llm._get_exposed_entities(
                    self._entity.hass, "conversation", include_state=False
                )

                # Get static list of tools
                ha_tools = self._assist_api._async_get_tools(llm_context, exposed_entities_result)
                self._cached_ha_tools = ha_tools
                self._cached_ha_tools_dict = {tool.name: tool for tool in ha_tools}
                self._cached_xai_tools = format_tools_for_xai(ha_tools, xai_tool)

                # Build textual description of tools
                tool_descriptions = []
                for ha_tool in ha_tools:
                    tool_desc = f"- **{ha_tool.name}**: {ha_tool.description or 'No description'}"
                    tool_descriptions.append(tool_desc)
                tool_definitions_text = "\n".join(tool_descriptions)

                # Format static context as compact JSON
                static_context_json = json.dumps(
                    list(exposed_entities_result["entities"].values()),
                    ensure_ascii=False,
                    separators=(',', ':')
                ) if exposed_entities_result["entities"] else "[]"

                # Build the full system prompt for tools mode
                tools_prompt_mgr = PromptManager(self._entity.subentry.data, "tools")
                self._cached_static_prompt = tools_prompt_mgr.build_base_prompt_with_user_instructions(
                    static_context=static_context_json,
                    tool_definitions=tool_definitions_text
                )
            else:
                # --- TOOLS DISABLED (SIMPLE CHAT) ---
                tools_prompt_mgr = PromptManager(self._entity.subentry.data, "tools")
                self._cached_xai_tools = []
                self._cached_static_prompt = tools_prompt_mgr.build_base_prompt_with_user_instructions()

            cache_time = time.time() - cache_start
            LOGGER.debug("cache_build: duration=%.2fs tools_count=%d prompt_length=%d",
                       cache_time, len(self._cached_xai_tools), len(self._cached_static_prompt))
        else:
            LOGGER.debug("cache_hit: Reusing cached static context (tools=%d)", len(self._cached_xai_tools))

        # Build the API call
        tools_for_call = self._cached_xai_tools

        LOGGER.debug("PREPARING API CALL: prev_id=%s, tools_count=%d, has_static_prompt=%s",
                   prev_id[:8] if prev_id else None,
                   len(tools_for_call) if tools_for_call else 0,
                   bool(self._cached_static_prompt))

        chat = self._entity.gateway.create_chat(client, tools=tools_for_call, previous_response_id=prev_id)

        # On first call, send the full static prompt
        if not prev_id:
            if self._cached_static_prompt:
                # Add temporal and geographic context to system prompt
                session_start = datetime.now()
                context_info = (
                    f"\n\nSession Context:"
                    f"\n- Started at: {session_start.strftime('%Y-%m-%d %H:%M:%S %Z')}"
                    f"\n- Timezone: {self._entity.hass.config.time_zone}"
                    f"\n- Country: {self._entity.hass.config.country}"
                )
                system_prompt_with_context = self._cached_static_prompt + context_info

                LOGGER.debug("FIRST MESSAGE: Adding system prompt (length=%d chars)", len(system_prompt_with_context))
                LOGGER.debug("System prompt:\n%s", system_prompt_with_context)
                chat.append(xai_system(system_prompt_with_context))
            else:
                LOGGER.error("FIRST MESSAGE BUT NO STATIC PROMPT! This should never happen!")
        else:
            LOGGER.debug("SUBSEQUENT MESSAGE: Skipping system prompt (using previous_response_id)")

        # Handle conversation history based on server-side memory availability
        store_messages = self._entity._get_option(CONF_STORE_MESSAGES, RECOMMENDED_STORE_MESSAGES)

        if prev_id or store_messages:
            # Server-side memory is active: send only the last user message
            last_user_message = get_last_user_message(chat_log)
            if last_user_message:
                send_user_name = self._entity._get_option(CONF_SEND_USER_NAME, False)

                # Build message prefix based on configuration
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if send_user_name:
                    name, source_type = await get_user_or_device_name(user_input, self._entity.hass)
                    if name and source_type == "user":
                        prefix = f"[User: {name}] [Time: {timestamp}] "
                    elif name and source_type == "device":
                        prefix = f"[Device: {name}] [Time: {timestamp}] "
                    else:
                        prefix = f"[Time: {timestamp}] "
                    user_message_with_time = f"{prefix}{last_user_message}"
                    LOGGER.debug("Tools mode: sending last message (user/device=%s): %s",
                                name or "unknown", last_user_message[:100])
                else:
                    user_message_with_time = f"({timestamp}) {last_user_message}"
                    LOGGER.debug("Tools mode: sending last message only (server-side memory): %s",
                               last_user_message[:100])

                chat.append(xai_user(user_message_with_time))
        else:
            # NO server-side memory: send the last N turns manually
            await self._add_manual_history_to_chat(chat, chat_log, user_input)

        return chat

    async def _stream_final_answer_after_tools(
        self,
        tool_results: list[ToolOutput],
        previous_response_id: str,
        conv_key: str,
        user_input,
        chat_log
    ) -> None:
        """Stream the final answer after tool execution.

        Similar to _get_final_answer_after_tools but uses streaming instead of returning a string.
        """
        # If store_messages is disabled, don't use previous_response_id (server won't have it)
        store_messages = self._entity._get_option(CONF_STORE_MESSAGES, True)
        response_id = previous_response_id if store_messages else None

        client = self._entity.gateway.create_client()
        chat = self._entity.gateway.create_chat(client, tools=None, previous_response_id=response_id)

        # When memory is disabled, rebuild the full context for this API call
        if not store_messages:
            LOGGER.debug("_stream_final_answer: store_messages=False, rebuilding full context")

            # 1. Add system prompt
            if self._cached_static_prompt:
                session_start = datetime.now()
                context_info = (
                    f"\n\nSession Context:"
                    f"\n- Started at: {session_start.strftime('%Y-%m-%d %H:%M:%S %Z')}"
                    f"\n- Timezone: {self._entity.hass.config.time_zone}"
                    f"\n- Country: {self._entity.hass.config.country}"
                )
                system_prompt_with_context = self._cached_static_prompt + context_info
                chat.append(xai_system(system_prompt_with_context))
                LOGGER.debug("_stream_final_answer: added system prompt (%d chars)", len(system_prompt_with_context))
            else:
                LOGGER.warning("_stream_final_answer: no cached static prompt available!")

            # 2. Add conversation history from chat_log
            await self._add_manual_history_to_chat(chat, chat_log, user_input)
        else:
            LOGGER.debug("_stream_final_answer: store_messages=True, using previous_response_id=%s",
                        response_id[:8] if response_id else None)

        # 3. Append tool results as user messages
        for result in tool_results:
            tool_output_content = result.tool_result if not result.is_error else f"Error: {result.tool_result}"
            result_text = f"Tool '{result.tool_name}' result: {tool_output_content}"
            chat.append(xai_user(result_text))
            LOGGER.debug("Appending tool result for '%s' to get final answer.", result.tool_name)

        # Stream the final response
        response_holder = {"response": None}

        async for _ in chat_log.async_add_delta_content_stream(
            agent_id=self._entity.entity_id,
            stream=self._delta_generator(chat, response_holder),
        ):
            await asyncio.sleep(0)

        # Extract final response
        final_response = response_holder.get("response")
        if not final_response:
            LOGGER.error("_stream_final_answer: no response from stream!")
            return

        # Save the final response ID to continue the conversation
        if getattr(final_response, "id", None):
            await self._entity._memory_set_prev_id(conv_key, final_response.id)
            LOGGER.debug(
                "memory_save: mode=tools conv_key=%s response_id=%s",
                conv_key, final_response.id[:8]
            )

        # Update token sensors with usage data
        usage = getattr(final_response, "usage", None)
        model = getattr(final_response, "model", None)
        if usage:
            self._entity._update_token_sensors(
                usage,
                model=model,
                service_type="conversation",
                mode="tools",
                is_fallback=False
            )

        content = getattr(final_response, "content", None)
        LOGGER.debug("Streamed final answer from Grok (len=%d)", len(content) if content else 0)

    async def _add_manual_history_to_chat(self, chat, chat_log, user_input):
        """Add manual conversation history to the chat when server-side memory is disabled."""
        LOGGER.debug("Adding manual history to chat (server-side memory disabled).")
        limit = RECOMMENDED_HISTORY_LIMIT_TURNS * 2

        send_user_name = self._entity._get_option(CONF_SEND_USER_NAME, False)
        user_prefix = ""
        name = None  # Ensure name is defined

        if send_user_name:
            name, source_type = await get_user_or_device_name(user_input, self._entity.hass)
            if name and source_type == "user":
                user_prefix = f"[User: {name}] "
            elif name and source_type == "device":
                user_prefix = f"[Device: {name}] "

        # Get last N turns from chat_log
        content_list = list(chat_log.content)
        history_content = content_list[-limit:] if len(content_list) > limit else content_list

        if not history_content:
            LOGGER.warning("Tools mode: no messages in chat_log, aborting")
            return

        LOGGER.debug("Tools mode: sending manual history: %d messages (last %d turns)",
                   len(history_content), RECOMMENDED_HISTORY_LIMIT_TURNS)

        for content in history_content:
            if isinstance(content, ha_conversation.UserContent):
                prefixed_content = f"{user_prefix}{content.content}" if user_prefix else content.content
                chat.append(xai_user(prefixed_content or ""))
            elif isinstance(content, ha_conversation.AssistantContent):
                tool_calls = getattr(content, 'tool_calls', None)
                if tool_calls:
                    chat.append(xai_assistant(tool_calls=tool_calls))
                else:
                    chat.append(xai_assistant(content.content or ""))

        last_msg = get_last_user_message(chat_log)
        LOGGER.debug("Tools mode: last user message in history: %s",
                   last_msg[:100] if last_msg else "None")

    async def _delta_generator(self, chat, response_holder: dict):
        """Generate deltas for streaming response from xAI using native async.

        Yields deltas in the format expected by async_add_delta_content_stream.

        Args:
            chat: xAI chat object
            response_holder: dict to populate with final response
        """
        new_message = True
        last_response = None
        
        try:
            # Natively iterate over the async stream
            async for response, chunk in chat.stream():
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
            response_holder["num_sources_used"] = getattr(getattr(last_response, "usage", None), "num_sources_used", 0)

        except Exception as err:
            LOGGER.error("tools_stream: error during native async streaming: %s", err)
            if new_message:
                yield {"role": "assistant"}
            yield {"content": f"\n\nAn error occurred during streaming: {err}"}

    # _xai_stream_async has been removed as part of the migration to native async streaming.
    # The logic is now integrated directly into _delta_generator.

    def _build_llm_context(self, user_input) -> ha_llm.LLMContext:
        """Build LLMContext independently from chat_log. NOT cached - rebuilt each time."""
        # Use "conversation" domain to match exposed entities
        # This matches how Home Assistant's conversation system checks entity exposure
        assistant_id = "conversation"

        return ha_llm.LLMContext(
            platform=DOMAIN,
            context=user_input.context,
            language=user_input.language,
            assistant=assistant_id,
            device_id=extract_device_id(user_input)
        )
