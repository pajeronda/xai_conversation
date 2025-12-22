from __future__ import annotations

# Standard library imports
import asyncio

# Home Assistant imports
from homeassistant.components.conversation import ConversationInput
from homeassistant.core import Context

# Home Assistant imports (excluding xAI SDK wrappers)
from homeassistant.components import conversation as ha_conversation

# Local application imports
from .const import (
    CONF_SEND_USER_NAME,
    CONF_SHOW_CITATIONS,
    HA_LOCAL_TAG_PATTERN,
    LOGGER,
    RECOMMENDED_SEND_USER_NAME,
    RECOMMENDED_SHOW_CITATIONS,
)
from .helpers import (
    format_user_message_with_metadata,
    get_last_user_message,
    parse_ha_local_payload,
    extract_scope_and_identifier,
    extract_device_id,
    LogTimeServices,
    timed_stream_generator,
    format_citations,
    add_manual_history_to_chat,
    RECOMMENDED_HISTORY_LIMIT_TURNS,
)
from .exceptions import handle_response_not_found_error
from .xai_gateway import XAIGateway


class IntelligentPipeline:
    """Intelligent routing pipeline streaming support.

    Logic:
    - Call xAI without tools using modular prompt system to let the model decide.
    - Inspect the beginning of the streamed content:
      - Starts with "[[ " => treat as local command payload [[HA_LOCAL: {...}]]
      - Otherwise => conversational text; stream directly to the user
    - For commands, extract payload.text and call Home Assistant conversation.process.
    """

    def __init__(self, hass, entity, user_input) -> None:
        self.hass = hass
        self.entity = entity
        self.user_input = user_input

    async def run(self, chat_log, timer: LogTimeServices) -> None:
        """Execute pipeline, passing the timer down for API measurement."""
        # Identify scope (User or Device) for memory key generation
        scope, identifier = extract_scope_and_identifier(self.user_input)

        # The gateway now handles conv_key and previous_id retrieval.
        # We still need a retry loop for handling expired response_ids that the gateway might use.
        for attempt in range(2):  # Max 2 attempts (0 and 1)
            # Create chat object via the unified gateway method
            chat, conv_key = await self.entity.gateway.create_chat(
                service_type="conversation",
                subentry_id=self.entity.subentry.subentry_id,
                client_tools=None,
                entity=self.entity,
                mode_override="pipeline",
                scope=scope,
                identifier=identifier,
            )

            if conv_key:
                last_user_message = get_last_user_message(chat_log)
                if not last_user_message:
                    return

                send_user_name = self.entity._get_option(
                    CONF_SEND_USER_NAME, RECOMMENDED_SEND_USER_NAME
                )
                user_message_with_time = await format_user_message_with_metadata(
                    last_user_message, self.user_input, self.hass, send_user_name
                )

                chat.append(XAIGateway.user_msg(user_message_with_time))
            else:
                # Full history hydration for client-side memory or first message
                await add_manual_history_to_chat(
                    self.hass,
                    self.entity,
                    chat,
                    chat_log,
                    self.user_input,
                    RECOMMENDED_HISTORY_LIMIT_TURNS,
                )

            try:
                await self._stream_and_route(chat, chat_log, conv_key, timer)
                break  # Success, exit loop

            except Exception as err:
                # Use centralized handler for NOT_FOUND errors
                should_retry = await handle_response_not_found_error(
                    err=err,
                    attempt=attempt,
                    memory=self.entity._conversation_memory,
                    conv_key=conv_key,
                    context_id=self.user_input.conversation_id,
                )

                if should_retry:
                    continue  # Retry

                # Not a retryable error, use fallback
                LOGGER.warning(
                    "pipeline_stream: streaming failed, using non-streaming fallback | error=%s",
                    err,
                )
                # Fallback: single-shot non-streaming response
                async with timer.record_api_call():
                    response = await chat.sample()

                response_text = getattr(response, "content", "")

                # Add response directly
                if response_text:
                    # Check for HA_LOCAL command tag in the response
                    match = HA_LOCAL_TAG_PATTERN.search(response_text)
                    if match:
                        LOGGER.debug(
                            "pipeline_fallback: detected command tag in response: %s",
                            match.group(0),
                        )

                        # Extract pre-command text (conversational part)
                        pre_text = response_text[: match.start()].strip()

                        # Extract and parse payload
                        payload_json = match.group(1)
                        payload_data = parse_ha_local_payload(payload_json)

                        # CRITICAL: Send pre-text FIRST to maintain correct order if commands stream output
                        if pre_text:
                            async for _ in chat_log.async_add_assistant_content(
                                ha_conversation.AssistantContent(
                                    agent_id=self.entity.entity_id,
                                    content=pre_text + "\n",
                                )
                            ):
                                pass

                        if payload_data:
                            # Use shared execution logic (iterates and yields strings)
                            final_results = []
                            async for result in self._execute_payload_actions(
                                payload_data, chat_log, timer
                            ):
                                final_results.append(result)

                            # Send any string results (conversation.process output)
                            if final_results:
                                async for _ in chat_log.async_add_assistant_content(
                                    ha_conversation.AssistantContent(
                                        agent_id=self.entity.entity_id,
                                        content="".join(final_results),
                                    )
                                ):
                                    pass

                        else:
                            # Parsing failed
                            async for _ in chat_log.async_add_assistant_content(
                                ha_conversation.AssistantContent(
                                    agent_id=self.entity.entity_id,
                                    content="Error: Could not parse command.",
                                )
                            ):
                                pass

                    else:
                        # No command, just standard text response
                        async for _ in chat_log.async_add_assistant_content(
                            ha_conversation.AssistantContent(
                                agent_id=self.entity.entity_id,
                                content=response_text,
                            )
                        ):
                            pass

                await self.entity.gateway.async_log_completion(
                    response=response,
                    service_type="conversation",
                    subentry_id=self.entity.subentry.subentry_id,
                    conv_key=conv_key,
                    mode="pipeline",
                    is_fallback=True,  # Mark as fallback (method) not model fallback
                    entity=self.entity,
                )
                break  # Exit loop after fallback

    async def _delta_generator(
        self,
        chat,
        chat_log,
        conv_key: str,
        response_holder: dict,
        timer: LogTimeServices,
    ):
        """
        Generator that yields deltas progressively, with API timing handled by timed_stream_generator.
        """
        buffer = ""
        streaming_active = True
        suspicious_buffer = ""  # Holds a trailing "[" while waiting for next chunk
        new_message = True
        last_response = None

        try:
            # The timed_stream_generator wrapper handles all timing and reports it to the timer
            async for response, chunk in timed_stream_generator(chat.stream(), timer):
                last_response = response
                content_chunk = getattr(chunk, "content", None)
                if content_chunk is None:
                    content_chunk = getattr(chunk, "delta", None)

                if not content_chunk:
                    continue

                if new_message:
                    yield {"role": "assistant"}
                    new_message = False

                if streaming_active:
                    # Check if we have a pending suspicious "[" from previous chunk
                    if suspicious_buffer:
                        if content_chunk.startswith("["):
                            # Found "[[": stop streaming, switch to buffering command
                            streaming_active = False
                            buffer = suspicious_buffer + content_chunk
                            suspicious_buffer = ""
                            LOGGER.debug(
                                "pipeline_stream: suspended at split '[[', buffer: %s",
                                buffer[:20],
                            )
                            continue  # Skip normal processing for this chunk
                        else:
                            # False alarm: was just a single "[", stream it now
                            yield {"content": suspicious_buffer}
                            suspicious_buffer = ""
                            # Continue to process current chunk normally

                    # Check for "[[HA_LOCAL" pattern in current chunk
                    if "[[" in content_chunk:
                        idx = content_chunk.index("[[")
                        if idx > 0:
                            yield {"content": content_chunk[:idx]}

                        streaming_active = False
                        buffer = content_chunk[idx:]
                        LOGGER.debug(
                            "pipeline_stream: suspended at '[[', buffer: %s",
                            buffer[:20],
                        )

                    elif content_chunk.endswith("["):
                        # Possible start of command split across chunks
                        # Stream everything except the last "["
                        if len(content_chunk) > 1:
                            yield {"content": content_chunk[:-1]}
                        suspicious_buffer = "["

                    else:
                        # Normal content, just stream it
                        yield {"content": content_chunk}
                else:
                    # Buffering mode: accumulate everything
                    buffer += content_chunk

                    # Fail-fast check: if buffer no longer matches "[[HA_LOCAL" prefix, flush it
                    # The tag is "[[HA_LOCAL" (10 chars).
                    # We check if the start of buffer matches the expected prefix up to its current length.
                    expected_prefix = "[[HA_LOCAL"
                    check_len = min(len(buffer), len(expected_prefix))

                    if buffer[:check_len] != expected_prefix[:check_len]:
                        # Mismatch detected! Flush buffer and resume streaming
                        LOGGER.debug(
                            "pipeline_stream: false positive tag detected, flushing buffer: '%s'",
                            buffer[:20],
                        )
                        yield {"content": buffer}
                        buffer = ""
                        streaming_active = True

            # End of stream handling
            if suspicious_buffer:
                # Flush any pending "[" if stream ended
                yield {"content": suspicious_buffer}

            # Store final response metadata
            if last_response:
                response_holder["id"] = getattr(last_response, "id", None)
                response_holder["usage"] = getattr(last_response, "usage", None)
                # Ensure model is set, fallback to configured model if missing in response
                response_model = getattr(last_response, "model", None)
                if (
                    not response_model
                    and self.entity
                    and hasattr(self.entity, "_model")
                ):
                    response_model = self.entity._model
                response_holder["model"] = response_model

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
                        "pipeline_stream: server_side_tool_usage=%s, num_sources_used=%d",
                        server_tools,
                        num_sources,
                    )

                response_holder["command_buffer"] = buffer

        except Exception as err:
            LOGGER.error(
                "pipeline_stream: error during native async streaming: %s", err
            )
            # The timer's __aexit__ will handle error logging. We just add content for the user.
            if new_message:
                yield {"role": "assistant"}
            yield {"content": f"\n\nAn error occurred during streaming: {err}"}
            return

        if buffer and not HA_LOCAL_TAG_PATTERN.search(buffer):
            LOGGER.debug(
                "pipeline_stream: no command tag in buffer, treating as dialogue"
            )
            yield {"content": buffer}

        # Yield citations if enabled (useful for UI, noisy for voice)
        citations = response_holder.get("citations")
        show_citations = self.entity._get_option(
            CONF_SHOW_CITATIONS, RECOMMENDED_SHOW_CITATIONS
        )

        if citations and show_citations:
            formatted_citations = format_citations(citations)
            yield {"content": formatted_citations}
        elif citations and not show_citations:
            LOGGER.debug(
                "pipeline_stream: citations available (%d) but show_citations=False, skipping yield",
                len(citations),
            )

    async def _stream_and_route(
        self, chat, chat_log, conv_key: str, timer: LogTimeServices
    ) -> None:
        """
        Stream using async_add_delta_content_stream and handle routing.
        The timer is passed down to the generator for measurement.
        """
        response_holder = {
            "id": None,
            "usage": None,
            "model": None,
            "command_buffer": "",
        }

        async for _ in chat_log.async_add_delta_content_stream(
            agent_id=self.entity.entity_id,
            stream=self._delta_generator(
                chat, chat_log, conv_key, response_holder, timer
            ),
        ):
            await asyncio.sleep(0)

        await self.entity.gateway.async_log_completion(
            response=response_holder,
            service_type="conversation",
            subentry_id=self.entity.subentry.subentry_id,
            conv_key=conv_key,
            mode="pipeline",
            is_fallback=False,
            entity=self.entity,
            citations=response_holder.get("citations"),
            num_sources_used=response_holder.get("num_sources_used", 0),
        )

        command_buffer = response_holder.get("command_buffer", "")
        if command_buffer and HA_LOCAL_TAG_PATTERN.search(command_buffer):
            LOGGER.info(
                "pipeline_stream: starting LOCAL command processing (Home Assistant Assist)"
            )

            match = HA_LOCAL_TAG_PATTERN.search(command_buffer)
            payload_json = match.group(1) if match else None

            if payload_json:
                payload_data = parse_ha_local_payload(payload_json)
                if payload_data:
                    # Execute all actions and for each resulting string, create a NEW bubble
                    async for result in self._execute_payload_actions(
                        payload_data, chat_log, timer
                    ):
                        if result:
                            async for _ in chat_log.async_add_delta_content_stream(
                                agent_id=self.entity.entity_id,
                                stream=self._single_bubble_stream(result),
                            ):
                                await asyncio.sleep(0)
                else:
                    LOGGER.error("pipeline_stream: failed to parse JSON payload")
            else:
                LOGGER.error("pipeline_stream: no valid JSON payload found in buffer")

    async def _execute_payload_actions(
        self, payload_data: dict, chat_log, timer: LogTimeServices
    ):
        """
        Execute commands from parsed payload (sequential or parallel).
        Yields result strings.
        """
        # Handle multiple commands
        if "commands" in payload_data:
            commands = payload_data.get("commands", [])
            sequential = payload_data.get("sequential", False)

            if not commands:
                LOGGER.warning(
                    "pipeline_stream: multi-command payload has empty commands array"
                )
                return

            if not isinstance(commands, list):
                LOGGER.error(
                    "pipeline_stream: 'commands' field is not a list. Got %s",
                    type(commands).__name__,
                )
                return

            total = len(commands)
            LOGGER.info(
                "pipeline_stream: processing %d commands | sequential=%s",
                total,
                sequential,
            )

            if sequential:
                for idx, cmd in enumerate(commands, 1):
                    command_text = str(cmd.get("text", "")).strip()
                    if command_text:
                        LOGGER.debug(
                            "pipeline_stream: [%d/%d] executing command: '%s'",
                            idx,
                            total,
                            command_text[:50],
                        )
                        result = await self._execute_single_command_streaming(
                            command_text, chat_log, timer, idx, total
                        )
                        if result:
                            yield result
            else:
                # Parallel execution
                # We use a small stagger delay (100ms) between starting parallel tasks
                # to avoid race conditions in Home Assistant script execution (Already Running warnings).
                tasks = []

                async def _staggered_exec(c_text, i, t, delay):
                    if delay > 0:
                        await asyncio.sleep(delay)
                    return await self._execute_single_command_streaming(
                        c_text, chat_log, timer, i, t
                    )

                for idx, cmd in enumerate(commands, 1):
                    command_text = str(cmd.get("text", "")).strip()
                    if command_text:
                        LOGGER.debug(
                            "pipeline_stream: [%d/%d] queueing command: '%s' (stagger=%.1fs)",
                            idx,
                            total,
                            command_text[:50],
                            (idx - 1) * 0.1,
                        )
                        tasks.append(
                            _staggered_exec(command_text, idx, total, (idx - 1) * 0.1)
                        )

                if tasks:
                    for coro in asyncio.as_completed(tasks):
                        try:
                            result = await coro
                            if result:
                                yield result
                        except Exception as err:
                            LOGGER.warning(
                                "pipeline_stream: parallel command failed: %s", err
                            )

        # Handle single command
        elif "text" in payload_data:
            cmd_text = str(payload_data.get("text", "")).strip()
            if cmd_text:
                result = await self._execute_single_command_streaming(
                    cmd_text, chat_log, timer
                )
                if result:
                    yield result
        else:
            LOGGER.error("pipeline_stream: payload missing 'text' or 'commands' field")
            yield "Invalid command format."

    async def _single_bubble_stream(self, content: str):
        """Helper generator to create a single assistant bubble for a pre-calculated result."""
        yield {"role": "assistant"}
        yield {"content": content}

    async def _command_results_generator(self, *args, **kwargs):
        """Deprecated: Logic moved into _stream_and_route for multi-bubble support."""
        # This is kept only to avoid potential import/attribute errors during hot-reload
        # if other components were referencing it (unlikely but safe).
        yield {"role": "assistant"}
        yield {"content": "Error: Deprecated stream called."}

    async def _execute_single_command_streaming(
        self,
        command_text: str,
        chat_log,
        timer: LogTimeServices,
        command_index: int | None = None,
        total_commands: int | None = None,
    ) -> str | None:
        """
        Execute a single command via conversation.process or fallback to tools.
        Returns a result string for non-fallback cases.
        In fallback cases, streams directly to the chat_log and returns None.
        """
        prefix = f"[{command_index}/{total_commands}] " if command_index else ""
        LOGGER.debug(
            "pipeline_stream: %sprocessing command='%s'", prefix, command_text[:50]
        )

        # Try conversation.process first
        trigger_tools_fallback = False
        error_code = None
        try:
            language = self.hass.config.language or "en"
            conv_id = self.user_input.conversation_id
            svc_context = getattr(chat_log, "context", None) or Context()
            data = {
                "text": command_text,
                "language": language,
                "agent_id": "conversation.home_assistant",
            }
            if conv_id:
                data["conversation_id"] = conv_id

            LOGGER.debug(
                "pipeline_stream: %s→ conversation.process | text='%s' language=%s agent=%s conv_id=%s",
                prefix,
                command_text,
                language,
                data.get("agent_id"),
                conv_id or "None",
            )

            result = await self.hass.services.async_call(
                "conversation",
                "process",
                data,
                blocking=True,
                return_response=True,
                context=svc_context,
            )

            # Check if result is an error
            response_type = (result or {}).get("response", {}).get("response_type")
            error_code = (result or {}).get("response", {}).get("data", {}).get("code")
            speech_text = (
                (result or {})
                .get("response", {})
                .get("speech", {})
                .get("plain", {})
                .get("speech", "")
            )

            LOGGER.info(
                "pipeline_stream: %s← LOCAL response received | response_type=%s speech='%s' error_code=%s",
                prefix,
                response_type or "unknown",
                speech_text[:100] if speech_text else "none",
                error_code or "none",
            )

            if response_type == "error":
                LOGGER.debug(
                    "pipeline_stream: %sconversation.process returned error, triggering fallback",
                    prefix,
                )
                trigger_tools_fallback = True

            # Return response if NOT triggering fallback
            if not trigger_tools_fallback:
                speech = (
                    (result or {})
                    .get("response", {})
                    .get("speech", {})
                    .get("plain", {})
                    .get("speech")
                )
                return speech or "Done."

        except Exception as err:
            LOGGER.debug(
                "pipeline_stream: %sconversation.process fallback triggered: error_type=%s error=%s",
                prefix,
                type(err).__name__,
                str(err)[:100],
            )
            trigger_tools_fallback = True

        # Fallback to tools mode
        if trigger_tools_fallback:
            LOGGER.debug(
                "pipeline_stream: %sFALLBACK→tools | reason=%s command='%s' error_code=%s",
                prefix,
                "error" if error_code else "no_intent_match",
                command_text[:50],
                error_code or "none",
            )

            # Create fallback input (preserving original user context)
            fallback_input = ConversationInput(
                text=command_text,
                context=self.user_input.context,
                conversation_id=self.user_input.conversation_id,
                device_id=extract_device_id(self.user_input),
                language=self.user_input.language,
                agent_id=self.user_input.agent_id,
                satellite_id=self.user_input.satellite_id,
            )

            # Execute tools processor directly on the main chat_log.
            # This will stream the response directly to the user interface.
            await self.entity._tools_processor.async_process_with_loop(
                fallback_input, chat_log, timer, force_tools=True
            )

            # Return None because the response has already been streamed.
            return None
