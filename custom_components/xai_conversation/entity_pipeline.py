from __future__ import annotations

# Standard library imports
import asyncio
import re
from datetime import datetime
from typing import Optional

# Home Assistant imports
from homeassistant.components.conversation import ConversationInput
from homeassistant.core import Context

# Home Assistant imports (re-exported from __init__)
from .__init__ import ha_conversation, xai_system, xai_user

# Local application imports
from .const import (
    CONF_SEND_USER_NAME,
    LOGGER
)
from .helpers import (
    extract_device_id,
    get_last_user_message,
    get_user_or_device_name,
    parse_ha_local_payload,
    PromptManager,
)


HA_LOCAL_TAG_PATTERN = re.compile(r"\[\[HA_LOCAL\s*:\s*({.*?})\]\]", re.DOTALL)


class MinimalChatLog:
    """Minimal chat_log object for fallback mode.

    Simulates ChatLog with only the methods needed by async_process_with_loop:
    - .content: list of messages
    - async_add_assistant_content(): to receive the response from tools mode
    """
    def __init__(self, content):
        self.content = content

    async def async_add_assistant_content(self, assistant_content):
        """Add assistant content to the chat log (simulates HA ChatLog behavior)."""
        self.content.append(assistant_content)
        # HA's async_add_assistant_content is an async generator that yields once
        yield None


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
        """
        Stream from xAI SDK, buffer the complete response, and then route it.
        This approach robustly handles responses that mix dialogue and commands.
        """
        buffer: str = ""
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
            except Exception:
                pass
            finally:
                try:
                    response_id_holder["id"] = response.id
                    response_id_holder["usage"] = response.usage
                except Exception:
                    response_id_holder["id"] = None
                    response_id_holder["usage"] = None
                try:
                    loop.call_soon_threadsafe(queue.put_nowait, "__END__")
                except Exception:
                    pass

        loop = asyncio.get_running_loop()
        producer_task = self.hass.async_create_background_task(
            asyncio.to_thread(_producer, loop),
            "xai_pipeline_stream_producer"
        )

        # Step 1: Consume the entire stream into a single buffer.
        while True:
            piece = await queue.get()
            if piece == "__END__":
                break
            buffer += piece

        # Wait for producer task to finish to ensure response_id is captured
        try:
            await producer_task
        except Exception:
            pass

        # Step 2: Decide how to route the complete response.
        match = HA_LOCAL_TAG_PATTERN.search(buffer)

        if match:
            # Response contains a command (and maybe dialogue).
            await self._handle_mixed_response(buffer, chat_log)
        else:
            # Response is purely conversational.
            if buffer:
                async for _ in chat_log.async_add_assistant_content(
                    ha_conversation.AssistantContent(
                        agent_id=self.entity.entity_id,
                        content=buffer,
                    )
                ):
                    pass

        # Step 3: Store response ID and update token sensors.
        if response_id_holder["id"]:
            try:
                await self.entity._save_response_chain(conv_key, response_id_holder["id"], "pipeline")
            except Exception as err:
                LOGGER.error("MEMORY_DEBUG: entity_pipeline.py - failed to store response_id: %s", err)

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
        match = HA_LOCAL_TAG_PATTERN.search(content)
        if match:
            # Response contains a command (and maybe dialogue).
            await self._handle_mixed_response(content, chat_log)
        else:
            # Response is purely conversational.
            if content:
                async for _ in chat_log.async_add_assistant_content(
                    ha_conversation.AssistantContent(
                        agent_id=self.entity.entity_id,
                        content=content,
                    )
                ):
                    pass

    async def _handle_mixed_response(self, buffer: str, chat_log) -> None:
        """
        Handles a response containing both dialogue and a command.
        Combines the dialogue and command result into a single response to prevent
        dialogue from being lost during command execution.
        """
        match = HA_LOCAL_TAG_PATTERN.search(buffer)
        if not match:
            # Fallback for safety, should not be reached
            async for _ in chat_log.async_add_assistant_content(
                ha_conversation.AssistantContent(
                    agent_id=self.entity.entity_id,
                    content=buffer,
                )
            ):
                pass
            return

        command_tag = match.group(0)
        dialogue = buffer.replace(command_tag, "").strip()
        LOGGER.debug("pipeline: [[HA_LOCAL]] detected | dialogue=%d chars, dialogue='%s'", len(dialogue), dialogue)

        # Execute the command and get result
        command_result = await self._handle_command_buffer(command_tag, chat_log)

        # Combine dialogue + command result
        combined_parts = []
        if dialogue:
            combined_parts.append(dialogue)
        if command_result:
            combined_parts.append(command_result)

        combined_response = "\n".join(combined_parts) if combined_parts else "Done."

        # Add combined response to chat_log in single call
        async for _ in chat_log.async_add_assistant_content(
            ha_conversation.AssistantContent(
                agent_id=self.entity.entity_id,
                content=combined_response,
            )
        ):
            pass

    async def _handle_command_buffer(self, buffer: str, chat_log) -> Optional[str]:
        """Extract HA_LOCAL payload and route to single/multi command handlers.

        Returns the command result instead of adding to chat_log.
        """
        LOGGER.debug("pipeline: [[HA_LOCAL]] detected | buffer_length=%d, buffer='%s'", len(buffer), buffer)
        payload_json = self._extract_payload_json(buffer)
        if not payload_json:
            LOGGER.debug("pipeline: no valid JSON payload found, returning raw buffer")
            return buffer

        LOGGER.debug("pipeline: extracted JSON payload: '%s'", payload_json[:200])

        # Parse payload to detect single vs multi-command (with fallback for malformed JSON)
        payload_data = parse_ha_local_payload(payload_json)
        if not payload_data:
            LOGGER.error("pipeline: failed to parse JSON payload | raw='%s'", payload_json[:200])
            return "I could not parse the command."

        # Check if multi-command format
        if "commands" in payload_data:
            # Multi-command: route to handler
            return await self._handle_multi_commands(payload_data, chat_log)
        elif "text" in payload_data:
            # Single command: extract text and process
            command_text = str(payload_data.get("text", "")).strip()
            if not command_text:
                return "I could not parse the command."
            # Get result from single command
            return await self._handle_single_command(command_text, chat_log)
        else:
            LOGGER.error("pipeline: payload missing 'text' or 'commands' field")
            return "Invalid command format."

    async def _handle_multi_commands(self, payload_data: dict, chat_log) -> Optional[str]:
        """Process multiple commands from payload, respecting sequential flag.

        Collects all results and returns combined response.
        """
        commands = payload_data.get("commands", [])
        # Default to parallel execution (False)
        sequential = payload_data.get("sequential", False)

        if not commands:
            LOGGER.warning("pipeline: multi-command payload has empty commands array")
            return None

        if not isinstance(commands, list):
            LOGGER.error("pipeline: 'commands' field is not a list. Got %s", type(commands).__name__)
            return None

        total = len(commands)
        LOGGER.info("pipeline: processing %d commands | sequential=%s", total, sequential)

        results = []

        if sequential:
            # Sequential execution: await each command one by one
            for idx, cmd in enumerate(commands, 1):
                command_text = str(cmd.get("text", "")).strip()
                if command_text:
                    LOGGER.debug("pipeline: [%d/%d] executing command: '%s'", idx, total, command_text[:50])
                    result = await self._handle_single_command(command_text, chat_log, command_index=idx, total_commands=total)
                    if result:
                        results.append(result)
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
                task_results = await asyncio.gather(*tasks, return_exceptions=True)
                # Filter out exceptions and None values
                for result in task_results:
                    if result and not isinstance(result, Exception):
                        results.append(result)

        # Return combined results
        if results:
            return "\n".join(results)
        return None

    async def _handle_single_command(self, command_text: str, chat_log, command_index: Optional[int] = None, total_commands: Optional[int] = None) -> Optional[str]:
        """Process a single command: call conversation.process, handle fallback, return response."""
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

            # Return response if NOT triggering fallback
            if not trigger_tools_fallback:
                speech = (result or {}).get("response", {}).get("speech", {}).get("plain", {}).get("speech")
                return speech or "Done."

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

            # Create a minimal chat_log containing ONLY the fallback command
            # This is used by async_process_with_loop to extract the user message
            fallback_chat_log = MinimalChatLog(
                content=[
                    ha_conversation.UserContent(
                        content=fallback_text,
                    )
                ]
            )

            # Execute tools processor with MinimalChatLog
            # This executes the full tool loop and adds the response to fallback_chat_log
            # Pass None for prev_id to ensure system prompt with tools is sent
            await self.entity._tools_processor.async_process_with_loop(
                fallback_input,
                fallback_chat_log,
                previous_response_id=None,
                force_tools=True
            )

            # Extract the assistant response from the MinimalChatLog
            fallback_response = ""
            for item in fallback_chat_log.content:
                if hasattr(item, '__class__') and 'AssistantContent' in item.__class__.__name__:
                    if hasattr(item, 'content') and isinstance(item.content, str):
                        fallback_response = item.content
                        break

            # Return the response (caller will add it to the REAL chat_log)
            # TRUE parallelism - no locks, no manipulation of original chat_log!
            return fallback_response or "Done."

    def _extract_payload_json(self, buffer: str) -> Optional[str]:
        """Extract JSON payload from [[HA_LOCAL: {...}]] tag."""
        match = HA_LOCAL_TAG_PATTERN.search(buffer or "")
        if match:
            return match.group(1)
        return None
