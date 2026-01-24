from __future__ import annotations

# Standard library imports
import asyncio
import time
from dataclasses import replace

# Home Assistant imports
from homeassistant.components.conversation import ConversationInput
from homeassistant.core import Context

# Home Assistant imports (excluding xAI SDK wrappers)
from homeassistant.components import conversation as ha_conversation

# Local application imports
from .const import (
    CHAT_MODE_PIPELINE,
    CHAT_MODE_TOOLS,
    HA_LOCAL_TAG_PATTERN,
    LOGGER,
)
from .helpers import (
    parse_ha_local_payload,
    extract_device_id,
    LogTimeServices,
    StreamParser,
    ChatOptions,
    async_log_completion,
    BaseConversationProcessor,
)


class IntelligentPipeline(BaseConversationProcessor):
    """Manages the intelligent routing and streaming pipeline.

    Orchestrates the flow between xAI model responses and Home Assistant local commands.
    Parses streamed content to detect and execute local command payloads or route text to the user.
    """

    def __init__(self, hass, entity, user_input) -> None:
        super().__init__(
            hass,
            entity.gateway,
            entity.entity_id,
            entity._conversation_memory,
            entity.get_config_dict(),
        )
        # Keep reference for local pipeline helpers if needed, though better to use members
        self._entity = entity
        self.user_input = user_input

    async def run(
        self,
        chat_log: ha_conversation.ChatLog,
        timer: LogTimeServices,
        options: ChatOptions | None = None,
    ) -> None:
        """Execute pipeline, passing the timer down for API measurement."""
        parser = StreamParser()

        # Execute chat loop via base processor
        resolved_options = options or ChatOptions(user_input=self.user_input)
        self._active_options = resolved_options

        try:
            results = await self._async_run_chat_loop(
                chat_log,
                timer,
                resolved_options,
                mode_override=CHAT_MODE_PIPELINE,
                parser=parser,
            )
            # If stream failed, results contains chat object for fallback
            if "error" in results:
                await self._execute_fallback(
                    results["chat"],
                    chat_log,
                    results["conv_key"],
                    timer,
                    results["error"],
                )
                return
        except Exception as err:
            # Gateway creation failure (no chat object available)
            LOGGER.error("[pipeline] failed: %s", err)
            return

        command_buffer = parser.command_buffer
        if command_buffer:
            await self._handle_ha_local_commands(
                command_buffer, chat_log, timer, send_pre_text=False
            )
            LOGGER.debug("[pipeline] complete: with LOCAL commands")
        else:
            LOGGER.debug("[pipeline] complete: text only")

    # ==========================================================================
    # ACTION EXECUTION (Routing)
    # ==========================================================================

    async def _execute_payload_actions(
        self, payload_data: dict, chat_log, timer: LogTimeServices
    ):
        """Execute payload actions with fast path for single commands."""
        # Fast path: single command (no orchestration overhead)
        if "text" in payload_data and "commands" not in payload_data:
            command_text = str(payload_data["text"]).strip()
            if command_text:
                result = await self._execute_single_command_streaming(
                    command_text, chat_log, timer, 1, 1
                )
                if result:
                    yield result
            return

        # Multi-command path
        commands = payload_data.get("commands", [])
        if not commands:
            LOGGER.error("[pipeline] invalid payload: missing 'text' or 'commands'")
            yield "Invalid command format."
            return

        sequential = payload_data.get("sequential", False)
        async for result in self._execute_command_orchestrator(
            commands, sequential, chat_log, timer
        ):
            yield result

    async def _execute_command_orchestrator(
        self, commands: list, sequential: bool, chat_log, timer: LogTimeServices
    ):
        """Orchestrate execution of multiple commands (sequential or parallel)."""
        if not isinstance(commands, list):
            LOGGER.error("[pipeline] invalid commands format")
            return

        total = len(commands)
        LOGGER.debug("[pipeline] commands: %d (seq=%s)", total, sequential)

        if sequential:
            # Sequential execution: await each command in order
            for idx, cmd in enumerate(commands, 1):
                command_text = str(cmd.get("text", "")).strip()
                if command_text:
                    result = await self._execute_single_command_streaming(
                        command_text, chat_log, timer, idx, total
                    )
                    if result:
                        yield result
            return

        # Parallel execution with staggered start (inline)
        async def _exec_with_stagger(cmd, idx):
            """Execute command with staggered delay."""
            if idx > 1:
                await asyncio.sleep((idx - 1) * 0.1)
            command_text = str(cmd.get("text", "")).strip()
            if command_text:
                return await self._execute_single_command_streaming(
                    command_text, chat_log, timer, idx, total
                )

        tasks = [
            _exec_with_stagger(cmd, idx)
            for idx, cmd in enumerate(commands, 1)
            if cmd.get("text")
        ]

        for coro in asyncio.as_completed(tasks):
            try:
                if result := await coro:
                    yield result
            except Exception as err:
                LOGGER.warning("[pipeline] parallel command failed: %s", err)

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
        exec_start = time.time()
        idx_info = f" {command_index}/{total_commands}" if command_index else ""

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
                # We intentionally DO NOT pass the conversation_id to the internal process
                # to avoid polluting the outer conversation's ChatLog with inner commands.
                # data["conversation_id"] = conv_id
                pass

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

            exec_time = time.time() - exec_start
            if response_type == "error":
                LOGGER.debug(
                    "[pipeline] exec: '%s'%s failed (%.0fms) - %s",
                    command_text[:40],
                    idx_info,
                    exec_time * 1000,
                    error_code or "unknown",
                )
                trigger_tools_fallback = True
            else:
                LOGGER.debug(
                    "[pipeline] exec: '%s'%s ok (%.0fms)",
                    command_text[:40],
                    idx_info,
                    exec_time * 1000,
                )

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
            exec_time = time.time() - exec_start
            LOGGER.debug(
                "[pipeline] exec: '%s'%s exception (%.2fs) - %s",
                command_text[:40],
                idx_info,
                exec_time,
                type(err).__name__,
            )
            trigger_tools_fallback = True

        # Fallback to tools mode
        if trigger_tools_fallback:
            LOGGER.debug("[pipeline] fallback→tools: '%s'", command_text[:50])

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
            # We pass is_fallback=True to clean history and forced_last_message to inject intent.
            # Create fallback options (cleaning history and forcing tools mode)
            # We use replace() to ensure we inherit user_input, timer, and other context
            fallback_options = replace(
                self._active_options,
                is_fallback=True,
                forced_last_message=command_text,
                mode_override=CHAT_MODE_TOOLS,
                is_resolved=False,  # Force re-resolution to use mode_override
            )
            await self._entity._tools_processor.async_process_with_loop(
                fallback_input,
                chat_log,
                timer,
                options=fallback_options,
                force_tools=True,
            )

            # Return None because the response has already been streamed.
            return None

    # ==========================================================================
    # FALLBACK & SECONDARY LOGIC
    # ==========================================================================

    async def _execute_fallback(
        self, chat, chat_log, conv_key: str, timer: LogTimeServices, err: Exception
    ) -> None:
        """Execute a non-streaming fallback request when streaming fails."""
        LOGGER.debug("[pipeline] stream failed, using sample fallback: %s", err)

        async with timer.record_api_call():
            response = await chat.sample()

        response_text = getattr(response, "content", "")
        if not response_text:
            return

        # Process potential commands (Fallback handles pre-text and results)
        await self._handle_ha_local_commands(
            response_text, chat_log, timer, send_pre_text=True
        )

        await async_log_completion(
            response=response,
            service_type="conversation",
            conv_key=conv_key,
            is_fallback=True,
            entity=self._entity,
            model_name=getattr(response, "model", None),
        )

    async def _handle_ha_local_commands(
        self, text: str, chat_log, timer: LogTimeServices, send_pre_text: bool
    ) -> None:
        """Centralized detection and execution of [[HA_LOCAL]] command tags."""
        match = HA_LOCAL_TAG_PATTERN.search(text)
        if not match:
            if send_pre_text:
                # No command, just standard text response (fallback mode)
                async for _ in chat_log.async_add_assistant_content(
                    ha_conversation.AssistantContent(
                        agent_id=self._entity.entity_id,
                        content=text,
                    )
                ):
                    pass
            return

        # 1. Command Detected

        # 2. Extract parts
        pre_text = text[: match.start()].strip()
        payload_json = match.group(1)
        payload_data = parse_ha_local_payload(payload_json)

        # 3. Handle Conversational Pre-text (if requested)
        # Optimization: only send pre_text if it's substantial and not already in chat
        if send_pre_text and pre_text and len(pre_text) > 3:
            async for _ in chat_log.async_add_assistant_content(
                ha_conversation.AssistantContent(
                    agent_id=self._entity.entity_id,
                    content=pre_text.strip(),
                )
            ):
                pass

        # 4. Execute Actions
        if not payload_data:
            err_msg = "Error: Could not parse command."
            LOGGER.error("[pipeline] parse error: %s", payload_json[:100])
            async for _ in chat_log.async_add_assistant_content(
                ha_conversation.AssistantContent(
                    agent_id=self._entity.entity_id,
                    content=err_msg,
                )
            ):
                pass
            return

        # Execute all actions
        if send_pre_text:
            # Fallback mode: collect and send in one/few bubbles (usually one)
            final_results = []
            async for result in self._execute_payload_actions(
                payload_data, chat_log, timer
            ):
                final_results.append(result)

            if final_results:
                async for _ in chat_log.async_add_assistant_content(
                    ha_conversation.AssistantContent(
                        agent_id=self._entity.entity_id,
                        content="".join(final_results),
                    )
                ):
                    pass
        else:
            # Happy path (streaming): create NEW bubbles for each result for better UX
            async for result in self._execute_payload_actions(
                payload_data, chat_log, timer
            ):
                if result:
                    async for _ in chat_log.async_add_delta_content_stream(
                        agent_id=self._entity.entity_id,
                        stream=self._single_bubble_stream(result),
                    ):
                        await asyncio.sleep(0)

    async def _single_bubble_stream(self, content: str):
        """Helper generator to create a single assistant bubble for a pre-calculated result."""
        yield {"role": "assistant"}
        yield {"content": content}
