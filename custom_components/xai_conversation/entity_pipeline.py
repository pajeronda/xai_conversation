from __future__ import annotations

# Standard library imports
import asyncio
import re
from datetime import datetime
import time
from typing import Optional

# Home Assistant imports
from homeassistant.components.conversation import ConversationInput
from homeassistant.core import Context

# Home Assistant imports (re-exported from __init__)
from .__init__ import ha_conversation, ha_json_loads, ha_llm, xai_system, xai_user

# Local application imports
from .const import (
    CONF_ALLOW_SMART_HOME_CONTROL,
    CONF_PROMPT,
    CONF_PROMPT_PIPELINE,
    CONF_STORE_MESSAGES,
    DOMAIN,
    LOGGER,
    RECOMMENDED_HISTORY_LIMIT_TURNS,
    RECOMMENDED_STORE_MESSAGES,
)
from .exceptions import raise_generic_error
from .helpers import (
    extract_device_id,
    get_last_user_message,
    parse_ha_local_payload,
    prompt_hash,
    PromptManager,
)


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
        # Lock to protect chat_log during fallback operations in multi-command scenarios
        self._chat_log_lock = asyncio.Lock()

    async def run(self, chat_log, previous_response_id: str | None) -> None:
        """Execute pipeline using chat_log prepared by conversation.py"""
        client = self.entity._create_client()

        # Create PromptManager for pipeline mode
        pipeline_prompt_mgr = PromptManager(self.entity.subentry.data, "pipeline")

        # Get memory key and previous response ID
        conv_key, retrieved_prev_id = await pipeline_prompt_mgr.get_conv_key_and_prev_id(self.entity, self.user_input)

        # Store conv_key for later use in saving response_id
        self._conv_key = conv_key

        # Use the previous_response_id passed from conversation.py, or the retrieved one
        # This ensures we respect any ID passed from a fallback scenario
        if previous_response_id is None:
            previous_response_id = retrieved_prev_id

        # Create chat with retrieved previous_response_id
        chat = self.entity._create_chat(client, tools=None, previous_response_id=previous_response_id)

        # Store conv_key for later use in saving response_id
        self._conv_key = conv_key

        # Build system prompt using PromptManager
        if not previous_response_id:
            system_prompt = pipeline_prompt_mgr.build_base_prompt_with_user_instructions()

            # Add temporal and geographic context to system prompt (only on first message)
            # This provides Grok with timezone and location info without breaking cache on subsequent messages
            session_start = datetime.now()
            context_info = (
                f"\n\nSession Context:"
                f"\n- Started at: {session_start.strftime('%Y-%m-%d %H:%M:%S %Z')}"
                f"\n- Timezone: {self.hass.config.time_zone}"
                f"\n- Country: {self.hass.config.country}"
            )
            system_prompt += context_info

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
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        user_message_with_time = f"({timestamp}) {last_user_message}"
        chat.append(xai_user(user_message_with_time))
        
        # Use native xAI SDK streaming with early detection
        try:
            await self._stream_and_route(chat, chat_log, conv_key)
        except Exception as err:
            # Fallback: single-shot response
            def _sample_sync():
                return chat.sample()

            response = await self.hass.async_add_executor_job(_sample_sync)
            content = getattr(response, "content", "")
            await self._route_final_text(content, chat_log)

            # Update token sensors for fallback non-streaming response
            usage = getattr(response, "usage", None)
            model = getattr(response, "model", None)
            if usage:
                self.entity._update_token_sensors(
                    usage,
                    model=model,
                    service_type="conversation",
                    mode="pipeline",
                    is_fallback=False
                )

    async def _stream_and_route(self, chat, chat_log, conv_key: str) -> None:
        """Stream from xAI SDK with native streaming and early detection.
        Uses official SDK pattern: for response, chunk in chat.stream()
        Implements real-time early detection of [[HA_LOCAL]] commands during streaming.
        """
        buffer: str = ""
        decided = False
        is_command = False
        queue: asyncio.Queue = asyncio.Queue()
        response_id_holder = {"id": None, "usage": None}

        def _producer(loop) -> None:
            try:
                # Official SDK pattern: for response, chunk in chat.stream():
                for response, chunk in chat.stream():
                    part = getattr(chunk, "content", None)
                    if part is None:
                        part = getattr(chunk, "delta", None) or getattr(chunk, "text", None)
                    if part:
                        try:
                            loop.call_soon_threadsafe(queue.put_nowait, str(part))
                        except Exception:
                            pass
                # capture final accumulated response id and usage
                try:
                    response_id_holder["id"] = getattr(response, "id", None)
                    response_id_holder["usage"] = getattr(response, "usage", None)
                except Exception:
                    response_id_holder["id"] = None
                    response_id_holder["usage"] = None
            except Exception:
                pass
            finally:
                try:
                    loop.call_soon_threadsafe(queue.put_nowait, "__END__")
                except Exception:
                    pass

        loop = asyncio.get_running_loop()
        # Start background producer in a thread managed by HA
        producer_task = self.hass.async_create_background_task(
            asyncio.to_thread(_producer, loop),
            "xai_pipeline_stream_producer"
        )

        # Create async generator that processes the queue and yields conversation content
        async def _conversation_stream():
            nonlocal buffer, decided, is_command
            chunk_count = 0
            while True:
                piece = await queue.get()
                if piece == "__END__":
                    break

                if not decided:
                    buffer += piece
                    stripped = buffer.lstrip()
                    if stripped.startswith("[["):
                        is_command = True
                        decided = True
                        # Continue consuming queue for command buffer, but don't yield
                        while True:
                            piece = await queue.get()
                            if piece == "__END__":
                                break
                            buffer += piece
                        break  # Exit generator
                    elif stripped and not stripped.startswith("["):
                        decided = True
                        # Yield buffered content immediately in HA format
                        yield {"content": buffer}
                    # else: keep buffering until decision
                else:
                    # Must be conversation since we handle commands above
                    chunk_count += 1
                    yield {"content": piece}

        # Stream conversation content using HA streaming API
        # The generator handles both decision making and queue consumption
        async for content in chat_log.async_add_delta_content_stream(
            self.entity.entity_id, _conversation_stream()
        ):
            pass  # Content already processed by generator

        # If it was a command, handle it now (buffer is already filled by generator)
        if is_command:
            await self._handle_command_buffer(buffer, chat_log)

        try:  # Wait for producer task completion to ensure proper cleanup
            await producer_task
        except Exception:
            pass

        # Store response id for server-side chaining if available
        if response_id_holder["id"]:
            try:
                await self.entity._save_response_chain(conv_key, response_id_holder["id"], "pipeline")
            except Exception as err:
                LOGGER.error("MEMORY_DEBUG: entity_pipeline.py - failed to store response_id: %s", err)

        # Update token sensors with usage data
        if response_id_holder["usage"]:
            self.entity._update_token_sensors(
                response_id_holder["usage"],
                model=response_id_holder.get("model"),
                service_type="conversation",
                mode="pipeline",
                is_fallback=False
            )

    async def _route_final_text(self, content: str, chat_log) -> None:
        """Route a complete non-streamed response."""
        stripped = content.lstrip()
        if stripped.startswith("[["):
            await self._handle_command_buffer(content, chat_log)
        else:
            async for _ in chat_log.async_add_assistant_content(
                ha_conversation.AssistantContent(
                    agent_id=self.entity.entity_id,
                    content=content,
                )
            ):
                pass

    async def _handle_command_buffer(self, buffer: str, chat_log) -> None:
        """Extract HA_LOCAL payload and route to single/multi command handlers."""
        LOGGER.debug("pipeline: [[HA_LOCAL]] detected | buffer_length=%d first_100_chars='%s'",
                    len(buffer), buffer[:100])
        payload_json = self._extract_payload_json(buffer)
        if not payload_json:
            LOGGER.debug("pipeline: no valid JSON payload found, emitting raw buffer")
            async for _ in chat_log.async_add_assistant_content(
                ha_conversation.AssistantContent(
                    agent_id=self.entity.entity_id,
                    content=buffer,
                )
            ):
                pass
            return

        LOGGER.debug("pipeline: extracted JSON payload: '%s'", payload_json[:100])

        # Parse payload to detect single vs multi-command (with fallback for malformed JSON)
        payload_data = parse_ha_local_payload(payload_json)
        if not payload_data:
            LOGGER.error("pipeline: failed to parse JSON payload | raw='%s'", payload_json[:200])
            async for _ in chat_log.async_add_assistant_content(
                ha_conversation.AssistantContent(
                    agent_id=self.entity.entity_id,
                    content="I could not parse the command.",
                )
            ):
                pass
            return

        # Check if multi-command format
        if "commands" in payload_data:
            # Multi-command: route to handler
            await self._handle_multi_commands(payload_data, chat_log)
        elif "text" in payload_data:
            # Single command: extract text and process
            command_text = str(payload_data.get("text", "")).strip()
            if not command_text:
                async for _ in chat_log.async_add_assistant_content(
                    ha_conversation.AssistantContent(
                        agent_id=self.entity.entity_id,
                        content="I could not parse the command.",
                    )
                ):
                    pass
                return
            await self._handle_single_command(command_text, chat_log)
        else:
            LOGGER.error("pipeline: payload missing 'text' or 'commands' field")
            async for _ in chat_log.async_add_assistant_content(
                ha_conversation.AssistantContent(
                    agent_id=self.entity.entity_id,
                    content="Invalid command format.",
                )
            ):
                pass

    async def _handle_multi_commands(self, payload_data: dict, chat_log) -> None:
        """Process multiple commands from payload, respecting sequential flag."""
        commands = payload_data.get("commands", [])
        sequential = payload_data.get("sequential", False)

        if not commands:
            LOGGER.warning("pipeline: multi-command payload has empty commands array")
            return

        if not isinstance(commands, list):
            LOGGER.error("pipeline: 'commands' field is not a list. Got %s", type(commands).__name__)
            return

        total = len(commands)
        LOGGER.info("pipeline: processing %d commands | sequential=%s", total, sequential)

        if sequential:
            # Sequential execution: await each command one by one
            for idx, cmd in enumerate(commands, 1):
                command_text = str(cmd.get("text", "")).strip()
                if command_text:
                    LOGGER.debug("pipeline: [%d/%d] executing command: '%s'", idx, total, command_text[:50])
                    await self._handle_single_command(command_text, chat_log, command_index=idx, total_commands=total)
                else:
                    LOGGER.warning("pipeline: [%d/%d] skipping empty command", idx, total)
        else:
            # Parallel execution: gather all commands
            tasks = []
            for idx, cmd in enumerate(commands, 1):
                command_text = str(cmd.get("text", "")).strip()
                if command_text:
                    LOGGER.debug("pipeline: [%d/%d] queueing command: '%s'", idx, total, command_text[:50])
                    tasks.append(self._handle_single_command(command_text, chat_log, command_index=idx, total_commands=total))
                else:
                    LOGGER.warning("pipeline: [%d/%d] skipping empty command", idx, total)

            # Execute all in parallel
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    async def _handle_single_command(self, command_text: str, chat_log, command_index: Optional[int] = None, total_commands: Optional[int] = None) -> None:
        """Process a single command: call conversation.process, handle fallback, emit response."""
        # Logging prefix for multi-command context
        prefix = f"[{command_index}/{total_commands}] " if command_index else ""
        LOGGER.debug("pipeline: %sprocessing command='%s'", prefix, command_text[:50])

        # Call HA conversation.process with exact syntax required
        trigger_tools_fallback = False
        error_code = None
        try:
            # Get language from HA system configuration
            language = self.hass.config.language or "en"
            LOGGER.debug("pipeline: %susing HA system language: %s", prefix, language)
            # Provide more context to improve target resolution and avoid HA assertions
            conv_id = self.user_input.conversation_id
            svc_context = getattr(chat_log, "context", None) or Context()
            data = {
                "text": command_text,
                "language": language,
                "agent_id": "conversation.home_assistant",
            }
            if conv_id:
                data["conversation_id"] = conv_id

            # Detailed log before calling conversation.process
            LOGGER.debug("pipeline: %s→ conversation.process | text='%s' language=%s agent=%s conv_id=%s",
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

            # Detailed log after conversation.process response
            LOGGER.debug("pipeline: %s← conversation.process | response_type=%s speech='%s' error_code=%s",
                        prefix, response_type or "unknown", speech_text[:100] if speech_text else "none", error_code or "none")

            if response_type == "error":
                LOGGER.info("pipeline: %sconversation.process returned error, triggering fallback", prefix)
                trigger_tools_fallback = True

            # Only add response to chat_log if NOT triggering fallback
            if not trigger_tools_fallback:
                speech = (result or {}).get("response", {}).get("speech", {}).get("plain", {}).get("speech")

                # Add the response to the chat log
                async for _ in chat_log.async_add_assistant_content(
                    ha_conversation.AssistantContent(
                        agent_id=self.entity.entity_id,
                        content=speech or "Done.",
                    )
                ):
                    pass

        except Exception as err:
            LOGGER.debug("pipeline: %sconversation.process fallback triggered: error_type=%s error=%s",
                        prefix, type(err).__name__, str(err)[:100])
            trigger_tools_fallback = True

        if trigger_tools_fallback:
            LOGGER.debug("pipeline: %sFALLBACK→tools | reason=%s command='%s' error_code=%s",
                        prefix, "error" if error_code else "no_intent_match", command_text[:50], error_code or "none")

            # Fallback to tools mode using Grok-processed command_text (with ASR/NLU corrections)
            fallback_text = command_text or get_last_user_message(chat_log) or ""

            # Create a user_input with the fallback text (preserving original user context)
            fallback_input = ConversationInput(
                text=fallback_text,
                context=self.user_input.context,
                conversation_id=self.user_input.conversation_id,
                device_id=extract_device_id(self.user_input),
                language=self.user_input.language,
                agent_id=self.user_input.agent_id,
                satellite_id=self.user_input.satellite_id,
            )

            # Create PromptManager in tools mode for fallback
            fallback_prompt_mgr = PromptManager(self.entity.subentry.data, "tools")

            # Get the tools mode previous_response_id using PromptManager
            tools_prev_id = await fallback_prompt_mgr.get_prev_id(self.entity, fallback_input)

            LOGGER.debug("Fallback: %susing tools mode prev_id=%s", prefix, tools_prev_id[:8] if tools_prev_id else None)

            # Protect chat_log manipulation with lock to prevent race conditions in parallel execution
            async with self._chat_log_lock:
                # Create a new, clean chat_log for the fallback to avoid context contamination.
                # The tools processor needs a chat_log object to add the final response to.
                # We will temporarily replace the content of the existing chat_log.
                original_content = chat_log.content

                chat_log.content = [
                    ha_conversation.UserContent(
                        content=fallback_text,
                    )
                ]

                # Execute tools processor with the clean chat_log.
                # The tools processor will use this chat_log for its loop and for the final output.
                # Pass None for prev_id to ensure the system prompt with tool definitions is sent.
                await self.entity._tools_processor.async_process_with_loop(
                    fallback_input, chat_log, None, force_tools=True
                )

                # Restore the original content so that the full history is logged
                chat_log.content = original_content + chat_log.content

    def _extract_payload_json(self, buffer: str) -> Optional[str]:
        """Extract JSON payload from [[HA_LOCAL: {...}]] tag."""
        match = HA_LOCAL_TAG_PATTERN.search(buffer or "")
        if match:
            return match.group(1)
        return None
