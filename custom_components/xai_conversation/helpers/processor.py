"""Base processor for xAI conversation flows."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from homeassistant.components import conversation as ha_conversation

from ..const import (
    LOGGER,
    CONF_ZDR,
    RECOMMENDED_HISTORY_LIMIT_TURNS,
    HA_LOCAL_TAG_PATTERN,
)
from ..exceptions import handle_response_not_found_error
from .log_time_services import LogTimeServices, timed_stream_generator
from .response import (
    capture_zdr_content,
    extract_response_metadata,
    format_citations,
    restore_zdr_content,
)
from .xaigateway_functions import ChatOptions, async_log_completion
from .utils import prepare_history_payload, should_show_citations

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant


class BaseConversationProcessor:
    """Base class for conversation processors (Pipeline, Tools).
    Centralizes the conversation loop, streaming logic, and error handling.
    """

    def __init__(
        self,
        hass: HomeAssistant,
        gateway: Any,
        entity_id: str,
        conversation_memory: Any,
        config: dict,
    ) -> None:
        """Initialize the processor."""
        self.hass = hass
        self._gateway = gateway
        self._entity_id = entity_id
        self._memory = conversation_memory
        self._config = config

    async def _async_run_chat_loop(
        self,
        chat_log: ha_conversation.ChatLog,
        timer: LogTimeServices,
        params: ChatOptions,
        mode_override: str,
        parser: Any | None = None,
        is_fallback: bool = False,
        chat_override: Any | None = None,
        conv_key_override: str | None = None,
    ) -> dict[str, Any]:
        """Execute a single turn of the chat loop with streaming.
        Returns a response_holder dictionary with results and metadata.
        """
        # 1. Create or reuse chat object
        if chat_override:
            chat = chat_override
            conv_key = conv_key_override
        else:
            # Prepare the SDK-neutral history payload (Layer 2)
            params.messages = await prepare_history_payload(
                chat_log,
                params,
                history_limit=RECOMMENDED_HISTORY_LIMIT_TURNS,
                last_only=params.store_messages or params.use_encrypted_content,
                skip_history_system=params.store_messages
                or params.use_encrypted_content,
            )

        for attempt in range(2):
            try:
                # Execute chat creation via Gateway (Layer 3)
                # Pass entity object if available (for prompt_manager access)
                if not chat_override:
                    entity_ref = getattr(self, "_entity", None) or self._entity_id
                    chat, conv_key, _, params = await self._gateway.create_chat(
                        service_type="conversation",
                        options=params,
                        entity=entity_ref,
                    )
                else:
                    chat = chat_override
                    conv_key = conv_key_override

                response_holder = {"response": None}
                is_zdr = self._config.get(CONF_ZDR, False)

                # 2. Stream response
                async for _ in chat_log.async_add_delta_content_stream(
                    agent_id=self._entity_id,
                    stream=unified_delta_generator(
                        chat.stream(),
                        timer,
                        response_holder,
                        config=self._config,
                        fallback_model=params.model,
                        is_zdr=is_zdr,
                        parser=parser,
                    ),
                ):
                    await asyncio.sleep(0)

                # 3. Log completion
                await async_log_completion(
                    response=response_holder.get("response"),
                    service_type="conversation",
                    conv_key=conv_key,
                    options=params,
                    is_fallback=is_fallback,
                    hass=self.hass,
                    citations=response_holder.get("citations"),
                    num_sources_used=response_holder.get("num_sources_used", 0),
                    model_name=response_holder.get("model"),
                    encrypted_content=response_holder.get("encrypted_content"),
                )

                # Success, return results
                return {
                    "chat": chat,
                    "conv_key": conv_key,
                    "response_holder": response_holder,
                }

            except Exception as err:
                # Handle NOT_FOUND errors (expired response_ids)
                should_retry = await handle_response_not_found_error(
                    err=err,
                    attempt=attempt,
                    memory=self._memory,
                    conv_key=conv_key if "conv_key" in locals() else None,
                    context_id=params.user_input.conversation_id
                    if params.user_input
                    else None,
                )
                if should_retry:
                    # Clear params.messages for retry if they were modified by Layer 3 (unlikely but safe)
                    continue

                # Return partial results to allow fallback if chat was created
                LOGGER.error("[%s] Stream error: %s", mode_override, err)
                return {
                    "chat": chat if "chat" in locals() else None,
                    "conv_key": conv_key if "conv_key" in locals() else None,
                    "error": err,
                }

        # Fallback return (should be covered by retry/error logic)
        return {"error": "Retry limit reached"}


async def unified_delta_generator(
    chat_stream,
    timer: Any,
    response_holder: dict,
    config: dict,
    fallback_model: str | None = None,
    is_zdr: bool = False,
    parser: Any | None = None,
):
    """Unified generator for streaming chat responses.

    Handles:
    1. Initial role yielding.
    2. API timing via timed_stream_generator.
    3. Content chunk yielding (optionally filtered by a parser).
    4. ZDR content capture.
    5. Final metadata extraction and citation appending.
    """
    new_message = True
    last_response = None

    try:
        async for response, chunk in timed_stream_generator(chat_stream, timer):
            last_response = response
            content_chunk = getattr(chunk, "content", "") or getattr(chunk, "delta", "")

            if new_message:
                yield {"role": "assistant"}
                new_message = False

            if is_zdr:
                capture_zdr_content(response, chunk, response_holder)

            if not content_chunk:
                continue

            if parser:
                # Delegate buffering and tag detection to parser (e.g. StreamParser)
                safe_chunks = parser.process_chunk(content_chunk)
                for safe_content in safe_chunks:
                    yield {"content": safe_content}
            else:
                yield {"content": content_chunk}

        # Flush parser if provided
        if parser:
            if flushed := parser.flush():
                yield {"content": flushed}

            # Check for partial buffer that wasn't a tag (pipeline fallback)
            buffer = parser.command_buffer
            if buffer:
                if not HA_LOCAL_TAG_PATTERN.search(buffer):
                    yield {"content": buffer}
                    parser.command_buffer = ""

            response_holder["command_buffer"] = parser.command_buffer

        # After stream, ensure we have the ZDR blob
        if is_zdr and not response_holder.get("encrypted_content"):
            if blob := getattr(last_response, "encrypted_content", None):
                response_holder["encrypted_content"] = blob

        if last_response:
            extract_response_metadata(
                last_response, response_holder, fallback_model=fallback_model
            )
            response_holder["response"] = last_response
            if is_zdr:
                restore_zdr_content(last_response, response_holder)

            # Extract reasoning_tokens for timer logging (reasoning models)
            if usage := response_holder.get("usage"):
                reasoning = getattr(usage, "reasoning_tokens", 0) or 0
                if reasoning == 0:
                    details = getattr(usage, "completion_tokens_details", None)
                    if details:
                        reasoning = getattr(details, "reasoning_tokens", 0) or 0
                if reasoning > 0:
                    timer.reasoning_tokens = reasoning

            # Citations
            if should_show_citations(config):
                citations = response_holder.get("citations")
                if citations:
                    yield {"content": format_citations(citations)}

    except Exception as err:
        from ..exceptions import is_not_found_error

        if is_not_found_error(err):
            LOGGER.debug("Stream session expired (NOT_FOUND): %s", err)
        else:
            LOGGER.error("Stream error in unified_delta_generator: %s", err)

        if new_message:
            yield {"role": "assistant"}
        # Do not yield error content directly to chat_log here to allow clean retry at processor level.
        # chat_log.async_add_delta_content_stream handles its own UI signaling.
        raise
