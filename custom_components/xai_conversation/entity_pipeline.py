from __future__ import annotations

# Standard library imports
import asyncio
import re
import time

# Home Assistant imports
from homeassistant.components.conversation import ConversationInput
from homeassistant.core import Context

# Home Assistant imports (excluding xAI SDK wrappers)
from .__init__ import ha_conversation

# Local application imports
from .const import (
    CONF_SEND_USER_NAME,
    CONF_STORE_MESSAGES,
    LOGGER,
    RECOMMENDED_STORE_MESSAGES,
)
from .helpers import (
    build_session_context_info,
    format_user_message_with_metadata,
    get_last_user_message,
    parse_ha_local_payload,
    PromptManager,
    save_response_metadata,
    XAIGateway,
    extract_device_id,
    LogTimeServices,
    timed_stream_generator,
)

# tag pattern for command parsing
HA_LOCAL_TAG_PATTERN = re.compile(r"\[\[HA_LOCAL\s*:\s*({.*?})\]\]", re.DOTALL)


class IntelligentPipeline:
    """Intelligent routing pipeline with optional streaming support.

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
        # Cache for base system prompt (without dynamic timestamp context)
        self._cached_base_prompt: str | None = None

    def _get_base_prompt(self) -> str:
        """Get cached base system prompt, building it if not already cached.

        Returns:
            Base system prompt without dynamic timestamp context
        """
        if self._cached_base_prompt is None:
            pipeline_prompt_mgr = PromptManager(self.entity.subentry.data, "pipeline")
            self._cached_base_prompt = (
                pipeline_prompt_mgr.build_base_prompt_with_user_instructions()
            )
        return self._cached_base_prompt

    async def run(self, chat_log, timer: LogTimeServices) -> None:
        """Execute pipeline, passing the timer down for API measurement."""
        client = self.entity.gateway.create_client()

        # Get memory key and previous response ID from ConversationMemory
        (
            conv_key,
            previous_response_id,
        ) = await self.entity._conversation_memory.get_conv_key_and_prev_id(
            self.user_input, "pipeline", self.entity.subentry.data
        )

        # Only use previous_response_id in server-side mode
        store_messages = self.entity._get_option(
            CONF_STORE_MESSAGES, RECOMMENDED_STORE_MESSAGES
        )
        previous_response_id = previous_response_id if store_messages else None

        # Create chat with filtered previous_response_id
        chat = self.entity.gateway.create_chat(
            client, tools=None, previous_response_id=previous_response_id
        )

        # Build system prompt using cached base prompt + dynamic context
        if not previous_response_id:
            # Get cached base prompt (without timestamp)
            base_prompt = self._get_base_prompt()

            # Add temporal and geographic context to system prompt (only on first message)
            context_info = build_session_context_info(self.hass)
            system_prompt = base_prompt + context_info

            LOGGER.debug(
                "PIPELINE FIRST MESSAGE: Adding system prompt (length=%d chars)",
                len(system_prompt),
            )
            LOGGER.debug("=" * 80)
            LOGGER.debug(
                "FULL PIPELINE SYSTEM PROMPT SENT TO GROK (COMPLETE - NOT TRUNCATED)"
            )
            LOGGER.debug("=" * 80)
            LOGGER.debug("%s", system_prompt)
            LOGGER.debug("=" * 80)
            LOGGER.debug("END PIPELINE SYSTEM PROMPT")
            LOGGER.debug("=" * 80)
            chat.append(XAIGateway.system_msg(system_prompt))
        else:
            LOGGER.debug(
                "PIPELINE SUBSEQUENT MESSAGE: Skipping system prompt (using previous_response_id)"
            )

        last_user_message = get_last_user_message(chat_log)
        if not last_user_message:
            return

        send_user_name = self.entity._get_option(CONF_SEND_USER_NAME, False)
        user_message_with_time = await format_user_message_with_metadata(
            last_user_message, self.user_input, self.hass, send_user_name
        )

        chat.append(XAIGateway.user_msg(user_message_with_time))

        try:
            await self._stream_and_route(chat, chat_log, conv_key, timer)
        except Exception as err:
            LOGGER.warning(
                "pipeline_stream: streaming failed, using non-streaming fallback | error=%s",
                err,
            )
            # Fallback: single-shot non-streaming response
            async with timer.record_api_call():
                response = await chat.sample()

            content = getattr(response, "content", "")

            # Add response directly
            if content:
                async for _ in chat_log.async_add_assistant_content(
                    ha_conversation.AssistantContent(
                        agent_id=self.entity.entity_id,
                        content=content,
                    )
                ):
                    pass

            await save_response_metadata(
                hass=self.hass,
                entry_id=self.entity.entry.entry_id,
                usage=getattr(response, "usage", None),
                model=getattr(response, "model", None),
                service_type="conversation",
                mode="pipeline",
                is_fallback=True,  # Mark as fallback
                store_messages=store_messages,
                conv_key=conv_key,
                response_id=getattr(response, "id", None),
                entity=self.entity,
            )

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
                            LOGGER.debug("pipeline_stream: suspended at split '[[', buffer: %s", buffer[:20])
                            continue # Skip normal processing for this chunk
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
                        LOGGER.debug("pipeline_stream: suspended at '[[', buffer: %s", buffer[:20])
                    
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

            # End of stream handling
            if suspicious_buffer:
                # Flush any pending "[" if stream ended
                yield {"content": suspicious_buffer}

            # Store final response metadata
            if last_response:
                response_holder["id"] = getattr(last_response, "id", None)
                response_holder["usage"] = getattr(last_response, "usage", None)

                model_value = getattr(last_response, "model", None)
                if hasattr(last_response, "model") and last_response.model:
                    response_holder["model"] = last_response.model

                response_holder["citations"] = getattr(last_response, "citations", [])
                response_holder["num_sources_used"] = getattr(
                    getattr(last_response, "usage", None), "num_sources_used", 0
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

        if buffer and "[[HA_LOCAL:" not in buffer:
            LOGGER.debug(
                "pipeline_stream: no command tag in buffer, treating as dialogue"
            )
            yield {"content": buffer}

        citations = response_holder.get("citations")
        if citations:
            formatted_citations = "\n\nCitations:\n"
            for i, citation in enumerate(citations):
                formatted_citations += f"  [{i + 1}] {getattr(citation, 'title', 'No Title')} - {getattr(citation, 'url', 'No URL')}\n"
            yield {"content": formatted_citations}

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

        store_messages = self.entity._get_option(
            CONF_STORE_MESSAGES, RECOMMENDED_STORE_MESSAGES
        )

        await save_response_metadata(
            hass=self.hass,
            entry_id=self.entity.entry.entry_id,
            usage=response_holder.get("usage"),
            model=response_holder.get("model"),
            service_type="conversation",
            mode="pipeline",
            is_fallback=False,
            store_messages=store_messages,
            conv_key=conv_key,
            response_id=response_holder.get("id"),
            entity=self.entity,
            citations=response_holder.get("citations"),
            num_sources_used=response_holder.get("num_sources_used", 0),
        )

        command_buffer = response_holder.get("command_buffer", "")
        if command_buffer and "[[HA_LOCAL:" in command_buffer:
            LOGGER.debug("pipeline_stream: executing commands in SEPARATE stream")
            async for _ in chat_log.async_add_delta_content_stream(
                agent_id=self.entity.entity_id,
                stream=self._command_results_generator(command_buffer, chat_log, timer),
            ):
                await asyncio.sleep(0)

    async def _command_results_generator(
        self, command_buffer: str, chat_log, timer: LogTimeServices
    ):
        """
        Generator that yields command results as a new separate stream.
        """
        # CRITICAL: New message = new role
        yield {"role": "assistant"}

        payload_json = self._extract_payload_json(command_buffer)
        if not payload_json:
            LOGGER.debug("pipeline_stream: no valid JSON payload found")
            yield {"content": "Invalid command format."}
            return

        LOGGER.debug(
            "pipeline_stream: extracted JSON payload: '%s'", payload_json[:200]
        )

        payload_data = parse_ha_local_payload(payload_json)
        if not payload_data:
            LOGGER.error(
                "pipeline_stream: failed to parse JSON payload | raw='%s'",
                payload_json[:200],
            )
            yield {"content": "Could not parse the command."}
            return

        # Check if multi-command format
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
                            yield {"content": f"{result}\n"}
            else:
                # Parallel execution
                tasks = []
                for idx, cmd in enumerate(commands, 1):
                    command_text = str(cmd.get("text", "")).strip()
                    if command_text:
                        LOGGER.debug(
                            "pipeline_stream: [%d/%d] queueing command: '%s'",
                            idx,
                            total,
                            command_text[:50],
                        )
                        tasks.append(
                            self._execute_single_command_streaming(
                                command_text, chat_log, timer, idx, total
                            )
                        )

                if tasks:
                    for coro in asyncio.as_completed(tasks):
                        try:
                            result = await coro
                            if result:
                                yield {"content": f"{result}\n"}
                        except Exception as err:
                            LOGGER.warning(
                                "pipeline_stream: parallel command failed: %s", err
                            )

        elif "text" in payload_data:
            # Single command
            command_text = str(payload_data.get("text", "")).strip()
            if not command_text:
                yield {"content": "Could not parse the command."}
                return

            result = await self._execute_single_command_streaming(
                command_text, chat_log, timer
            )
            if result:
                yield {"content": result}
                LOGGER.debug(
                    "pipeline_stream: yielded command result (len=%d)", len(result)
                )
        else:
            LOGGER.error("pipeline_stream: payload missing 'text' or 'commands' field")
            yield {"content": "Invalid command format."}

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
        Returns the result string (not yielded as delta - caller handles that).
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

            LOGGER.debug(
                "pipeline_stream: %s← conversation.process | response_type=%s speech='%s' error_code=%s",
                prefix,
                response_type or "unknown",
                speech_text[:100] if speech_text else "none",
                error_code or "none",
            )

            if response_type == "error":
                LOGGER.info(
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

            # Create minimal chat_log for tools processor
            fallback_chat_log = MinimalChatLog(
                content=[
                    ha_conversation.UserContent(
                        content=command_text,
                    )
                ]
            )

            # Execute tools processor (writes to MinimalChatLog)
            await self.entity._tools_processor.async_process_with_loop(
                fallback_input, fallback_chat_log, timer, force_tools=True
            )

            # Extract the assistant response from MinimalChatLog
            for item in fallback_chat_log.content:
                if isinstance(item, ha_conversation.AssistantContent):
                    return item.content or "Done."

            return "Done."

    def _extract_payload_json(self, buffer: str) -> str | None:
        """Extract JSON payload from [[HA_LOCAL: {...}]] tag."""
        match = HA_LOCAL_TAG_PATTERN.search(buffer or "")
        if match:
            return match.group(1)
        return None


class MinimalChatLog:
    """Minimal chat_log object for fallback mode.

    Simulates ChatLog with only the methods needed by async_process_with_loop:
    - .content: list of messages
    - async_add_assistant_content(): to receive the response from tools mode
    - async_add_delta_content_stream(): for streaming support in fallback mode
    """

    def __init__(self, content):
        self.content = content
        self._accumulated_content = ""

    async def async_add_assistant_content(self, assistant_content):
        """Add assistant content to the chat log (simulates HA ChatLog behavior)."""
        self.content.append(assistant_content)
        # HA's async_add_assistant_content is an async generator that yields once
        yield None

    async def async_add_delta_content_stream(self, agent_id: str, stream):
        """
        Consume a delta stream and add the final content as a single
        AssistantContent message. This is NOT a generator, it fully consumes
        the stream. This is to avoid race conditions when waiting for the result.
        """
        self._accumulated_content = ""

        async for delta in stream:
            # Consume the stream but do nothing with the deltas,
            # as the caller in the fallback doesn't need real-time updates.
            if "content" in delta:
                self._accumulated_content += delta["content"]

        # After stream ends, add the single accumulated content to the chat log
        if self._accumulated_content:
            self.content.append(
                ha_conversation.AssistantContent(
                    agent_id=agent_id,
                    content=self._accumulated_content,
                )
            )

        # This function is called in an `async for` loop, so it must be a generator.
        # We yield once at the end after the stream is fully processed.
        yield
