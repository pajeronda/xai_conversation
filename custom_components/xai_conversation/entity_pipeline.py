from __future__ import annotations

# Standard library imports
import asyncio
import re
from typing import Optional

# Home Assistant imports
from homeassistant.components.conversation import ConversationInput
from homeassistant.core import Context

# Home Assistant and standard library imports (re-exported from __init__)
from .__init__ import datetime, ha_conversation, xai_system, xai_user

# Local application imports
from .const import (
    CONF_SEND_USER_NAME,
    CONF_STORE_MESSAGES,
    LOGGER,
    RECOMMENDED_STORE_MESSAGES,
)
from .helpers import (
    extract_device_id,
    get_last_user_message,
    get_user_or_device_name,
    parse_ha_local_payload,
    PromptManager,
)
# tag pattern for command parsing
HA_LOCAL_TAG_PATTERN = re.compile(r"\[\[HA_LOCAL\s*:\s*({.*?})\]\]", re.DOTALL)

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
        """Add delta content stream to the chat log (simulates HA ChatLog streaming behavior).

        This method accumulates deltas and builds the final AssistantContent,
        mimicking what the real ChatLog does during streaming.
        """
        self._accumulated_content = ""

        async for delta in stream:
            # Accumulate content from deltas
            if "content" in delta:
                self._accumulated_content += delta["content"]

            # Yield the delta for consumer to process
            yield delta

        # After stream ends, add accumulated content to chat log
        if self._accumulated_content:
            self.content.append(
                ha_conversation.AssistantContent(
                    agent_id=agent_id,
                    content=self._accumulated_content,
                )
            )


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
            self._cached_base_prompt = pipeline_prompt_mgr.build_base_prompt_with_user_instructions()
        return self._cached_base_prompt

    async def run(self, chat_log) -> None:
        """Execute pipeline using chat_log prepared by conversation.py"""
        client = self.entity.gateway.create_client()

        # Get memory key and previous response ID from ConversationMemory
        conv_key, previous_response_id = await self.entity._conversation_memory.get_conv_key_and_prev_id(
            self.user_input,
            "pipeline",
            self.entity.subentry.data
        )

        # Only use previous_response_id in server-side mode
        store_messages = self.entity._get_option(CONF_STORE_MESSAGES, RECOMMENDED_STORE_MESSAGES)
        previous_response_id = previous_response_id if store_messages else None

        # Create chat with filtered previous_response_id
        chat = self.entity.gateway.create_chat(client, tools=None, previous_response_id=previous_response_id)

        # Build system prompt using cached base prompt + dynamic context
        if not previous_response_id:
            # Get cached base prompt (without timestamp)
            base_prompt = self._get_base_prompt()

            # Add temporal and geographic context to system prompt (only on first message)
            # This provides Grok with timezone and location info without breaking cache on subsequent messages
            session_start = datetime.now()
            context_info = (
                f"\n\nSession Context:"
                f"\n- Started at: {session_start.strftime('%Y-%m-%d %H:%M:%S %Z')}"
                f"\n- Timezone: {self.hass.config.time_zone}"
                f"\n- Country: {self.hass.config.country}"
            )
            system_prompt = base_prompt + context_info

            LOGGER.debug("PIPELINE FIRST MESSAGE: Adding system prompt (length=%d chars)", len(system_prompt))
            LOGGER.debug("=" * 80)
            LOGGER.debug("FULL PIPELINE SYSTEM PROMPT SENT TO GROK (COMPLETE - NOT TRUNCATED)")
            LOGGER.debug("=" * 80)
            LOGGER.debug("%s", system_prompt)
            LOGGER.debug("=" * 80)
            LOGGER.debug("END PIPELINE SYSTEM PROMPT")
            LOGGER.debug("=" * 80)
            chat.append(xai_system(system_prompt))
        else:
            LOGGER.debug("PIPELINE SUBSEQUENT MESSAGE: Skipping system prompt (using previous_response_id)")

        # Pipeline mode is stateless by design - always send only the last user message
        # The [[HA_LOCAL]] system doesn't require conversation history
        last_user_message = get_last_user_message(chat_log)
        if not last_user_message:
            return

        # Check if user wants to include user/device name
        send_user_name = self.entity._get_option(CONF_SEND_USER_NAME, False)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if send_user_name:
            name, source_type = await get_user_or_device_name(self.user_input, self.hass)
            if name and source_type == "user":
                prefix = f"[User: {name}] [Time: {timestamp}] "
            elif name and source_type == "device":
                prefix = f"[Device: {name}] [Time: {timestamp}] "
            else:
                # No name available, just timestamp
                prefix = f"[Time: {timestamp}] "
            user_message_with_time = f"{prefix}{last_user_message}"
        else:
            # Default behavior: only timestamp
            user_message_with_time = f"({timestamp}) {last_user_message}"

        chat.append(xai_user(user_message_with_time))
        
        # Use native xAI SDK streaming with early detection
        try:
            await self._stream_and_route(chat, chat_log, conv_key)
        except Exception as err:
            LOGGER.warning("pipeline_stream: streaming failed, using non-streaming fallback | error=%s", err)
            # Fallback: single-shot non-streaming response
            response = await chat.sample()
            content = getattr(response, "content", "")

            # Add response directly (no streaming, no command routing)
            if content:
                async for _ in chat_log.async_add_assistant_content(
                    ha_conversation.AssistantContent(
                        agent_id=self.entity.entity_id,
                        content=content,
                    )
                ):
                    pass

            # Save response metadata using extracted method
            await self._save_response_metadata(
                conv_key,
                getattr(response, "id", None),
                getattr(response, "usage", None),
                getattr(response, "model", None)
            )

    # _xai_stream_async has been removed as part of the migration to native async streaming.
    # The logic is now integrated directly into _delta_generator.

    async def _delta_generator(self, chat, chat_log, conv_key: str, response_id_holder: dict):
        """
        Generator that yields deltas progressively for streaming using native async.
        Stream until '[' character, then suspend streaming and buffer everything for command execution.
        """
        buffer = ""
        streaming_active = True
        new_message = True
        last_response = None

        try:
            # Natively iterate over the async stream
            async for response, chunk in chat.stream():
                last_response = response
                content_chunk = getattr(chunk, "content", None)
                if content_chunk is None:
                    content_chunk = getattr(chunk, "delta", None)
                
                if not content_chunk:
                    continue

                # CRITICAL: Yield role BEFORE any content
                if new_message:
                    yield {"role": "assistant"}
                    new_message = False
                
                if streaming_active:
                    if '[' in content_chunk:
                        idx = content_chunk.index('[')
                        if idx > 0:
                            yield {"content": content_chunk[:idx]}
                            LOGGER.debug("pipeline_stream: streamed %d chars before '[', suspending", idx)
                        
                        streaming_active = False
                        buffer = content_chunk[idx:]
                        LOGGER.debug("pipeline_stream: streaming suspended at '[', buffer starts with: %s", buffer[:20])
                    else:
                        yield {"content": content_chunk}
                else:
                    buffer += content_chunk
            
            # Store final response metadata
            if last_response:
                response_id_holder["id"] = getattr(last_response, "id", None)
                response_id_holder["usage"] = getattr(last_response, "usage", None)
                response_id_holder["model"] = getattr(last_response, "model", None)
                # Capture search citations and number of sources used
                response_id_holder["citations"] = getattr(last_response, "citations", [])
                response_id_holder["num_sources_used"] = getattr(getattr(last_response, "usage", None), "num_sources_used", 0)

        except Exception as err:
            LOGGER.error("pipeline_stream: error during native async streaming: %s", err)
            # Yield an error message to the user
            if new_message: # Ensure role is set if error happens on first chunk
                yield {"role": "assistant"}
            yield {"content": f"\n\nAn error occurred during streaming: {err}"}
            return

        # End of stream - process buffer if it contains commands
        if buffer:
            LOGGER.debug("pipeline_stream: stream ended, processing buffer (len=%d)", len(buffer))

            # Check if buffer contains command tag
            if "[[HA_LOCAL:" in buffer:
                # Execute commands and stream results as deltas
                LOGGER.debug("pipeline_stream: executing command(s) to add to streamed dialogue")
                async for delta in self._handle_command_with_streaming(buffer, chat_log):
                    yield delta
            else:
                # No command tag - stream it as dialogue
                LOGGER.debug("pipeline_stream: no command tag in buffer, treating as dialogue")
                yield {"content": buffer}
        
        # Append citations to the chat log if available
        citations = response_id_holder.get("citations")
        if citations:
            formatted_citations = "\n\nCitations:\n"
            for i, citation in enumerate(citations):
                formatted_citations += f"  [{i+1}] {getattr(citation, 'title', 'No Title')} - {getattr(citation, 'url', 'No URL')}\n"
            yield {"content": formatted_citations}


    async def _save_response_metadata(self, conv_key: str, response_id: str | None, usage, model: str | None = None, citations: list | None = None, num_sources_used: int = 0) -> None:
        """Save response ID to memory and update token sensors.

        Args:
            conv_key: Conversation key for memory storage
            response_id: xAI response ID to save
            usage: Usage statistics from xAI response
            model: Model name (extracted from response.model or usage)
            citations: List of citations from xAI response
            num_sources_used: Number of unique search sources used
        """
        # Store response ID
        if response_id:
            try:
                await self.entity._memory_set_prev_id(conv_key, response_id)
                LOGGER.debug(
                    "memory_save: mode=pipeline conv_key=%s response_id=%s",
                    conv_key, response_id[:8]
                )
            except Exception as err:
                LOGGER.error("MEMORY_DEBUG: entity_pipeline.py - failed to store response_id: %s", err)

        # Update token sensors
        if usage:
            self.entity._update_token_sensors(
                usage,
                model=model,
                service_type="conversation",
                mode="pipeline",
                is_fallback=False
            )
        
        # Log search details if available
        if citations:
            LOGGER.debug("pipeline_stream: citations found: %d", len(citations))
            for citation in citations:
                LOGGER.debug("Citation: %s", citation)
        if num_sources_used > 0:
            LOGGER.debug("pipeline_stream: unique search sources used: %d", num_sources_used)

    async def _stream_and_route(self, chat, chat_log, conv_key: str) -> None:
        """
        Stream using async_add_delta_content_stream for progressive output.
        Handles dialogue streaming, command execution, and fallback to tools mode.
        """
        response_id_holder = {"id": None, "usage": None, "model": None}

        # Use delta generator with async_add_delta_content_stream
        # CRITICAL: We must consume the generator to allow delta_listener to be called
        async for _ in chat_log.async_add_delta_content_stream(
            agent_id=self.entity.entity_id,
            stream=self._delta_generator(chat, chat_log, conv_key, response_id_holder),
        ):
            # Yield control to event loop to allow UI updates
            await asyncio.sleep(0)

        # Save response metadata (ID, usage, model, citations, num_sources_used)
        await self._save_response_metadata(
            conv_key,
            response_id_holder.get("id"),
            response_id_holder.get("usage"),
            response_id_holder.get("model"),
            response_id_holder.get("citations"),
            response_id_holder.get("num_sources_used", 0)
        )

    async def _handle_command_with_streaming(self, command_tag: str, chat_log):
        """
        Handle command execution and yield result as delta for streaming.
        Supports single commands, multi-commands (sequential/parallel), and fallback to tools.
        """
        payload_json = self._extract_payload_json(command_tag)
        if not payload_json:
            LOGGER.debug("pipeline_stream: no valid JSON payload found")
            yield {"content": "\nInvalid command format."}
            yield {"role": "assistant"}  # Flush error
            return

        LOGGER.debug("pipeline_stream: extracted JSON payload: '%s'", payload_json[:200])

        # Parse payload to detect single vs multi-command
        payload_data = parse_ha_local_payload(payload_json)
        if not payload_data:
            LOGGER.error("pipeline_stream: failed to parse JSON payload | raw='%s'", payload_json[:200])
            yield {"content": "\nCould not parse the command."}
            yield {"role": "assistant"}  # Flush error
            return

        # Check if multi-command format
        if "commands" in payload_data:
            # Multi-command: process and yield results
            commands = payload_data.get("commands", [])
            sequential = payload_data.get("sequential", False)

            if not commands:
                LOGGER.warning("pipeline_stream: multi-command payload has empty commands array")
                return

            if not isinstance(commands, list):
                LOGGER.error("pipeline_stream: 'commands' field is not a list. Got %s", type(commands).__name__)
                return

            total = len(commands)
            LOGGER.info("pipeline_stream: processing %d commands | sequential=%s", total, sequential)

            if sequential:
                # Sequential execution - yield each result as it completes
                for idx, cmd in enumerate(commands, 1):
                    command_text = str(cmd.get("text", "")).strip()
                    if command_text:
                        LOGGER.debug("pipeline_stream: [%d/%d] executing command: '%s'", idx, total, command_text[:50])
                        result = await self._execute_single_command_streaming(command_text, chat_log, idx, total)
                        if result:
                            # Stream result as part of the same message
                            yield {"content": f"\n{result}"}
            else:
                # Parallel execution - yield results as they complete (streaming via as_completed)
                tasks = []
                for idx, cmd in enumerate(commands, 1):
                    command_text = str(cmd.get("text", "")).strip()
                    if command_text:
                        LOGGER.debug("pipeline_stream: [%d/%d] queueing command: '%s'", idx, total, command_text[:50])
                        tasks.append(self._execute_single_command_streaming(command_text, chat_log, idx, total))

                if tasks:
                    # Use as_completed to yield results as they finish (not waiting for all)
                    for coro in asyncio.as_completed(tasks):
                        try:
                            result = await coro
                            if result:
                                # Stream each result as it completes (as part of same message)
                                yield {"content": f"\n{result}"}
                        except Exception as err:
                            LOGGER.warning("pipeline_stream: parallel command failed: %s", err)

        elif "text" in payload_data:
            # Single command
            command_text = str(payload_data.get("text", "")).strip()
            if not command_text:
                yield {"content": "\nCould not parse the command."}
                yield {"role": "assistant"}  # Flush error message
                return

            result = await self._execute_single_command_streaming(command_text, chat_log)
            if result:
                # CRITICAL: Il flush del dialogo è già stato fatto PRIMA del comando
                # Qui aggiungiamo SOLO il risultato come continuazione dello stesso messaggio
                yield {"content": "\n" + result}
                LOGGER.debug("pipeline_stream: yielded command result (len=%d)", len(result))
        else:
            LOGGER.error("pipeline_stream: payload missing 'text' or 'commands' field")
            yield {"content": "\nInvalid command format."}
            yield {"role": "assistant"}  # Flush error message

    async def _execute_single_command_streaming(self, command_text: str, chat_log, command_index: Optional[int] = None, total_commands: Optional[int] = None) -> Optional[str]:
        """
        Execute a single command via conversation.process or fallback to tools.
        Returns the result string (not yielded as delta - caller handles that).
        """
        prefix = f"[{command_index}/{total_commands}] " if command_index else ""
        LOGGER.debug("pipeline_stream: %sprocessing command='%s'", prefix, command_text[:50])

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

            LOGGER.debug("pipeline_stream: %s→ conversation.process | text='%s' language=%s agent=%s conv_id=%s",
                        prefix, command_text, language, data.get("agent_id"), conv_id or "None")

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
            speech_text = (result or {}).get("response", {}).get("speech", {}).get("plain", {}).get("speech", "")

            LOGGER.debug("pipeline_stream: %s← conversation.process | response_type=%s speech='%s' error_code=%s",
                        prefix, response_type or "unknown", speech_text[:100] if speech_text else "none", error_code or "none")

            if response_type == "error":
                LOGGER.info("pipeline_stream: %sconversation.process returned error, triggering fallback", prefix)
                trigger_tools_fallback = True

            # Return response if NOT triggering fallback
            if not trigger_tools_fallback:
                speech = (result or {}).get("response", {}).get("speech", {}).get("plain", {}).get("speech")
                return speech or "Done."

        except Exception as err:
            LOGGER.debug("pipeline_stream: %sconversation.process fallback triggered: error_type=%s error=%s",
                        prefix, type(err).__name__, str(err)[:100])
            trigger_tools_fallback = True

        # Fallback to tools mode
        if trigger_tools_fallback:
            LOGGER.debug("pipeline_stream: %sFALLBACK→tools | reason=%s command='%s' error_code=%s",
                        prefix, "error" if error_code else "no_intent_match", command_text[:50], error_code or "none")

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
                fallback_input,
                fallback_chat_log,
                force_tools=True
            )

            # Extract the assistant response from MinimalChatLog
            for item in fallback_chat_log.content:
                if isinstance(item, ha_conversation.AssistantContent):
                    return item.content or "Done."

            return "Done."

    def _extract_payload_json(self, buffer: str) -> Optional[str]:
        """Extract JSON payload from [[HA_LOCAL: {...}]] tag."""
        match = HA_LOCAL_TAG_PATTERN.search(buffer or "")
        if match:
            return match.group(1)
        return None
