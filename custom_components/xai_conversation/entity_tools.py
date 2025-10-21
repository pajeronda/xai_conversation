"""Tools processing logic for the xAI Conversation integration."""

from __future__ import annotations

# Standard library imports
from dataclasses import dataclass
from datetime import datetime
import json
import time
from typing import TYPE_CHECKING, Any

# Home Assistant imports (re-exported from __init__)
from .__init__ import (
    xai_system, xai_user, xai_assistant, xai_tool,
    ha_conversation,
    ha_llm,
)

# Local imports
from .const import (
    CONF_ALLOW_SMART_HOME_CONTROL,
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
    PromptManager,
)
from .exceptions import (
    raise_generic_error,
    handle_api_error,
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

    async def async_process_with_loop(
        self,
        user_input,
        chat_log,
        previous_response_id: str | None = None,
        force_tools: bool = False
    ) -> None:
        """Process a conversation with tools using the standard HA loop pattern."""
        start_time = time.time()
        LOGGER.debug("tools_loop_start (delegated) for user_input: %s", user_input.text[:50])

        # Override use_tools if force_tools is True (e.g., from fallback)
        use_tools = self._use_tools or force_tools

        # Create PromptManager for tools mode
        tools_prompt_mgr = PromptManager(self._entity.subentry.data, "tools")

        # Get memory key and previous_response_id using PromptManager (centralizes memory_scope logic)
        conv_key, prev_id_from_memory = await tools_prompt_mgr.get_conv_key_and_prev_id(self._entity, user_input)

        max_iterations = 5  # Safety break to prevent infinite loops

        for i in range(max_iterations):
            LOGGER.debug("tools_loop: iteration %d/%d", i + 1, max_iterations)

            # Get the latest previous_response_id for this iteration
            # First iteration: use provided previous_response_id if available, else use prev_id_from_memory
            # Subsequent iterations: always fetch from memory (updated after each API call)
            if i == 0 and previous_response_id is not None:
                current_prev_id = previous_response_id
                LOGGER.debug("tools_loop: using provided prev_id=%s", current_prev_id[:8] if current_prev_id else None)
            elif i == 0:
                current_prev_id = prev_id_from_memory
                LOGGER.debug("tools_loop: using memory prev_id=%s", current_prev_id[:8] if current_prev_id else None)
            else:
                current_prev_id = await self._entity._memory_get_prev_id(conv_key)

            # Call the LLM with the current state of the chat log
            response = await self.call_xai_api_with_tools(user_input, chat_log, current_prev_id, use_tools)

            # Immediately save the response ID to maintain the conversation chain
            if getattr(response, "id", None):
                await self._entity._save_response_chain(conv_key, response.id, "tools")

            # Update token sensors with usage data
            usage = getattr(response, "usage", None)
            model = getattr(response, "model", None)
            if usage:
                self._entity._update_token_sensors(
                    usage,
                    model=model,
                    service_type="conversation",
                    mode="tools",
                    is_fallback=False
                )

            # Check if the LLM wants to use tools
            has_tool_calls = hasattr(response, 'tool_calls') and response.tool_calls

            if not has_tool_calls:
                # No more tools to call, this is the final answer
                LOGGER.debug("tools_loop: no tool calls, providing final answer.")
                async for _ in chat_log.async_add_assistant_content(
                    ha_conversation.AssistantContent(
                        agent_id=self._entity.entity_id,
                        content=response.content,
                    )
                ):
                    pass
                break  # Exit the loop

            # Convert xAI tool calls to HA format
            llm_context = self._build_llm_context(user_input)
            ha_tool_calls = []
            for tool_call in response.tool_calls:
                ha_tool_input = convert_xai_to_ha_tool(self._entity.hass, tool_call)
                ha_tool_calls.append(ha_tool_input)

            # Only add tool_calls to chat_log if not in force_tools mode
            # (pipeline chat_log doesn't support tool_calls - no LLM API configured)
            if not force_tools:
                async for _ in chat_log.async_add_assistant_content(
                    ha_conversation.AssistantContent(
                        agent_id=self._entity.entity_id,
                        content=response.content or "",
                        tool_calls=ha_tool_calls,
                    )
                ):
                    pass

            # Execute the tools
            tool_results = []

            for tool_call in response.tool_calls:
                tool_start = time.time()
                LOGGER.debug("tools_loop: executing tool '%s'", tool_call.function.name)

                # Security check: Prevent GetLiveContext call when tools are disabled.
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

                    # Execute tool directly using cached HA tools dict for O(1) lookup
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
                                str(tool_call.function.arguments)[:200])
                    # Use ToolOutput for error reporting
                    tool_results.append(ToolOutput(
                        tool_name=tool_call.function.name,
                        tool_result=f"An error occurred while trying to execute the tool: {err}",
                        is_error=True,
                    ))

            # Get the final answer from Grok by sending only the tool results
            final_answer = await self._get_final_answer_after_tools(tool_results, response.id, conv_key)
            async for _ in chat_log.async_add_assistant_content(
                ha_conversation.AssistantContent(
                    agent_id=self._entity.entity_id,
                    content=final_answer,
                )
            ):
                pass
            break # Tool cycle is complete, exit loop

        else:  # This 'else' belongs to the 'for' loop
            LOGGER.warning("tools_loop: reached max iterations (%d), aborting.", max_iterations)
            await self._entity._add_error_response_and_continue(
                user_input, chat_log, "I'm having trouble completing the request, it seems to be stuck in a loop."
            )
    
        total_time = time.time() - start_time
        LOGGER.debug("tools_loop_end: duration=%.2fs", total_time)

    async def _get_final_answer_after_tools(self, tool_results: list[ToolOutput], previous_response_id: str, conv_key: str) -> str:
        """Call Grok with tool results to get the final natural language answer."""
        # If store_messages is disabled, don't use previous_response_id (server won't have it)
        store_messages = self._entity._get_option(CONF_STORE_MESSAGES, True)
        response_id = previous_response_id if store_messages else None

        client = self._entity._create_client()
        chat = self._entity._create_chat(client, tools=None, previous_response_id=response_id)

        # Append tool results as user messages containing the tool output
        # xAI SDK expects tool results to be sent as regular messages, not special tool objects
        for result in tool_results:
            tool_output_content = result.tool_result if not result.is_error else f"Error: {result.tool_result}"
            # Format tool result as a user message for xAI
            result_text = f"Tool '{result.tool_name}' result: {tool_output_content}"
            chat.append(xai_user(result_text))
            LOGGER.debug("Appending tool result for '%s' to get final answer.", result.tool_name)

        # Get the final natural language response
        def _sample_sync():
            return chat.sample()
        final_response = await self._entity.hass.async_add_executor_job(_sample_sync)

        # Save the final response ID to continue the conversation
        if getattr(final_response, "id", None):
            await self._entity._save_response_chain(conv_key, final_response.id, "tools")

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

        LOGGER.debug("Received final answer from Grok: '%s'", final_response.content[:100])
        return final_response.content



    async def call_xai_api_with_tools(self, user_input, chat_log, previous_response_id: str | None = None, use_tools: bool = None):
        """Call xAI API with tools enabled, using context separation and caching."""
        start_time = time.time()

        # Use provided use_tools or fall back to instance default
        if use_tools is None:
            use_tools = self._use_tools

        try:
            client = self._entity._create_client()
            prev_id = previous_response_id

            # 2. Handle static context (prompt and tools) with caching.
            # This block runs if the cache is empty, regardless of whether it's the first call.
            # CRITICAL: Also rebuild cache if force_tools=True but cache was built without tools
            # Check for both None and empty list [] (tools disabled sets it to [])
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
                    # Get an instance of the AssistAPI to call its internal methods.
                    self._assist_api = ha_llm._async_get_apis(self._entity.hass).get(ha_llm.LLM_API_ASSIST)
                    if not self._assist_api:
                        raise_generic_error("AssistAPI is not available in Home Assistant.")

                    # Get exposed entities first (needed for GetLiveContext tool inclusion)
                    exposed_entities_result = ha_llm._get_exposed_entities(
                        self._entity.hass, "conversation", include_state=False
                    )

                    # Get static list of tools using the correct internal method
                    # Pass exposed_entities to enable GetLiveContext tool
                    ha_tools = self._assist_api._async_get_tools(llm_context, exposed_entities_result)
                    self._cached_ha_tools = ha_tools  # Cache HA tools for execution
                    self._cached_ha_tools_dict = {tool.name: tool for tool in ha_tools}  # Create dict for O(1) lookup
                    self._cached_xai_tools = format_tools_for_xai(ha_tools, xai_tool)

                    # Build textual description of tools for the system prompt
                    # The actual tool definitions are passed separately via SDK
                    tool_descriptions = []
                    for ha_tool in ha_tools:
                        tool_desc = f"- **{ha_tool.name}**: {ha_tool.description or 'No description'}"
                        tool_descriptions.append(tool_desc)
                    tool_definitions_text = "\n".join(tool_descriptions)

                    # 2. Format static context as compact JSON (token-optimized)
                    static_context_json = json.dumps(
                        list(exposed_entities_result["entities"].values()),
                        ensure_ascii=False,
                        separators=(',', ':')
                    ) if exposed_entities_result["entities"] else "[]"

                    # 3. Build the full system prompt for tools mode
                    tools_prompt_mgr = PromptManager(self._entity.subentry.data, "tools")

                    # Build complete prompt with tools and context using modular system
                    self._cached_static_prompt = tools_prompt_mgr.build_base_prompt_with_user_instructions(
                        static_context=static_context_json,
                        tool_definitions=tool_definitions_text
                    )
                else:
                    # --- TOOLS DISABLED (SIMPLE CHAT) ---
                    tools_prompt_mgr = PromptManager(self._entity.subentry.data, "tools")
                    self._cached_xai_tools = []
                    # Chat-only mode: just base + user instructions (no tools/context)
                    self._cached_static_prompt = tools_prompt_mgr.build_base_prompt_with_user_instructions()

                cache_time = time.time() - cache_start
                LOGGER.info("cache_build: duration=%.2fs tools_count=%d prompt_length=%d",
                           cache_time, len(self._cached_xai_tools), len(self._cached_static_prompt))
                # Prompt details are logged when actually sent to API (see call_xai_api_with_tools)
            else:
                LOGGER.debug("cache_hit: Reusing cached static context (tools=%d)", len(self._cached_xai_tools))

            # 3. Dynamic context is now handled by GetLiveContext tool, as per the prompt's logic.
            # We no longer send the live state of devices with every user message.

            # 4. Build the API call
            # Always pass tools since xAI may not persist them across conversation turns
            tools_for_call = self._cached_xai_tools

            LOGGER.info("PREPARING API CALL: prev_id=%s, tools_count=%d, has_static_prompt=%s",
                       prev_id[:8] if prev_id else None,
                       len(tools_for_call) if tools_for_call else 0,
                       bool(self._cached_static_prompt))

            chat = self._entity._create_chat(client, tools=tools_for_call, previous_response_id=prev_id)

            # On first call, send the full static prompt.
            # This now includes our manual instructions and is logged completely.
            if not prev_id:
                if self._cached_static_prompt:
                    # Add temporal and geographic context to system prompt (only on first message)
                    # This provides Grok with timezone and location info without breaking cache on subsequent messages
                    session_start = datetime.now()
                    context_info = (
                        f"\n\nSession Context:"
                        f"\n- Started at: {session_start.strftime('%Y-%m-%d %H:%M:%S %Z')}"
                        f"\n- Timezone: {self._entity.hass.config.time_zone}"
                        f"\n- Country: {self._entity.hass.config.country}"
                    )
                    system_prompt_with_context = self._cached_static_prompt + context_info

                    LOGGER.info("FIRST MESSAGE: Adding system prompt (length=%d chars)", len(system_prompt_with_context))
                    LOGGER.debug("System prompt:\n%s", system_prompt_with_context)
                    chat.append(xai_system(system_prompt_with_context))
                else:
                    LOGGER.error("FIRST MESSAGE BUT NO STATIC PROMPT! This should never happen!")
            else:
                LOGGER.info("SUBSEQUENT MESSAGE: Skipping system prompt (using previous_response_id)")

            # Handle conversation history based on server-side memory availability
            store_messages = self._entity._get_option(CONF_STORE_MESSAGES, RECOMMENDED_STORE_MESSAGES)

            if prev_id or store_messages:
                # Server-side memory is active (or will be for first message):
                # send only the last user message
                # The server reconstructs the full history using previous_response_id
                last_user_message = get_last_user_message(chat_log)
                if last_user_message:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    user_message_with_time = f"({timestamp}) {last_user_message}"
                    chat.append(xai_user(user_message_with_time))
                LOGGER.debug("Tools mode: sending last message only (server-side memory): %s",
                           last_user_message[:100] if last_user_message else "None")
            else:
                # NO server-side memory: send the last N turns manually
                # This ensures Grok has conversation context for continuity
                limit = RECOMMENDED_HISTORY_LIMIT_TURNS * 2  # turns = pairs of (user + assistant)

                # Convert ChatLog.content to list of messages
                all_messages = []
                for content in chat_log.content:
                    if isinstance(content, ha_conversation.UserContent):
                        all_messages.append({"role": "user", "content": content.content or ""})
                    elif isinstance(content, ha_conversation.AssistantContent):
                        all_messages.append({"role": "assistant", "content": content.content or ""})

                # Get last N turns
                history_messages = all_messages[-limit:] if len(all_messages) > limit else all_messages

                # Validate that we have at least one message
                if not history_messages:
                    LOGGER.warning("Tools mode: no messages in chat_log, aborting")
                    return None

                LOGGER.info("Tools mode: sending manual history: %d messages (last %d turns)",
                           len(history_messages), RECOMMENDED_HISTORY_LIMIT_TURNS)

                for msg in history_messages:
                    if msg["role"] == "user":
                        chat.append(xai_user(msg["content"]))
                    elif msg["role"] == "assistant":
                        chat.append(xai_assistant(msg["content"]))

                # Log the final user message for debugging
                last_msg = get_last_user_message(chat_log)
                LOGGER.debug("Tools mode: last user message in history: %s",
                           last_msg[:100] if last_msg else "None")

            api_start = time.time()
            # Use the correct streaming pattern to get the full response object,
            # which includes tool_calls. chat.sample() returns the full response object for non-streaming calls.
            def _sample_sync():
                return chat.sample()
            response = await self._entity.hass.async_add_executor_job(_sample_sync)
            api_time = time.time() - api_start

            # Debug log for the raw response object
            LOGGER.debug("RESPONSE from Grok (Raw Object): %s", response)

            has_tool_calls = hasattr(response, 'tool_calls') and response.tool_calls
            LOGGER.info("tools_api_core: duration=%.2fs tool_calls=%s", api_time, bool(has_tool_calls))
            if has_tool_calls:
                LOGGER.debug("tool_calls: %s", [tc.function.name for tc in response.tool_calls])
            return response

        except Exception as err:
            handle_api_error(err, start_time, "tools API call")
