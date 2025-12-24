"""Tools processing logic for the xAI Conversation integration."""

from __future__ import annotations

# Standard library imports
import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

# Home Assistant imports
from homeassistant.components import conversation as ha_conversation

# Local imports
from .const import (
    CONF_SEND_USER_NAME,
    CONF_SHOW_CITATIONS,
    CONF_STORE_MESSAGES,
    LOGGER,
    RECOMMENDED_SHOW_CITATIONS,
    RECOMMENDED_SEND_USER_NAME,
    RECOMMENDED_STORE_MESSAGES,
)
from .helpers import (
    add_manual_history_to_chat,
    format_user_message_with_metadata,
    get_last_user_message,
    LogTimeServices,
    timed_stream_generator,
    ToolOrchestrator,
    ToolExecutionResult,
    format_citations,
    extract_scope_and_identifier,
    RECOMMENDED_HISTORY_LIMIT_TURNS,
)
from .exceptions import handle_response_not_found_error
from .xai_gateway import XAIGateway

# Import helper to distinguish tool types
try:
    from xai_sdk.tools import get_tool_call_type
except ImportError:
    get_tool_call_type = None

# gRPC imports for error handling (conditional)
try:
    from grpc import StatusCode
    from grpc._channel import _InactiveRpcError

    try:
        from grpc.aio import AioRpcError
    except ImportError:
        AioRpcError = _InactiveRpcError  # Fallback
except Exception:
    StatusCode = None
    _InactiveRpcError = None
    AioRpcError = None


if TYPE_CHECKING:
    from .entity import XAIBaseLLMEntity


@dataclass
class ToolOutput:
    """Result from a tool execution."""

    tool_name: str
    tool_result: Any
    tool_call_id: str
    is_error: bool = False


class XAIToolsProcessor:
    """
    Handles the processing of conversations in 'tools' mode for a XAIBaseLLMEntity.
    Delegates tool management to ToolOrchestrator.
    """

    def __init__(self, entity: XAIBaseLLMEntity):
        """Initialize the XAIToolsProcessor."""
        self._entity = entity
        self._orchestrator = ToolOrchestrator(entity.hass, entity)

    async def async_process_with_loop(
        self,
        user_input,
        chat_log,
        timer: LogTimeServices,
        force_tools: bool = False,
    ) -> None:
        """Process a conversation with tools, using the provided timer for metrics."""
        LOGGER.debug(
            "tools_loop_start (delegated) for user_input: %s", user_input.text[:50]
        )

        # Identify scope (User or Device) for memory key generation
        scope, identifier = extract_scope_and_identifier(user_input)

        # The gateway will now handle conv_key and previous_id retrieval
        conv_key = None
        max_iterations = 5

        for i in range(max_iterations):
            LOGGER.debug("tools_loop: iteration %d/%d", i + 1, max_iterations)

            # --- PREPARE CHAT ---
            # The gateway now handles previous_id lookup internally
            chat, conv_key = await self._prepare_chat_for_tools(
                user_input, chat_log, scope, identifier
            )

            # --- STREAM RESPONSE ---
            response_holder = {"response": None}
            is_fallback = force_tools and i == 0

            try:
                async for _ in chat_log.async_add_delta_content_stream(
                    agent_id=self._entity.entity_id,
                    stream=self._delta_generator(chat, response_holder, timer),
                ):
                    await asyncio.sleep(0)

            except Exception as err:
                is_grpc_error = _InactiveRpcError is not None and isinstance(
                    err, (_InactiveRpcError, AioRpcError)
                )
                if is_grpc_error:
                    should_retry = await handle_response_not_found_error(
                        err=err,
                        attempt=i,
                        memory=self._entity._conversation_memory,
                        conv_key=conv_key,
                        context_id=user_input.conversation_id,
                    )

                    if should_retry:
                        continue
                    else:
                        LOGGER.error(
                            "tools_loop: NOT_FOUND error but retry not possible"
                        )
                        break
                raise

            # --- PROCESS FINAL RESPONSE ---
            response = response_holder.get("response")
            if not response:
                LOGGER.error("tools_loop: no response from stream!")
                break

            await self._entity.gateway.async_log_completion(
                response=response,
                service_type="conversation",
                subentry_id=self._entity.subentry.subentry_id,
                conv_key=conv_key,
                mode="tools",
                is_fallback=is_fallback,
                entity=self._entity,
                citations=response_holder.get("citations"),
                num_sources_used=response_holder.get("num_sources_used", 0),
                await_save=True,
            )

            # Handle citations display
            citations = response_holder.get("citations")
            if citations:
                show_citations = self._entity._get_option(
                    CONF_SHOW_CITATIONS, RECOMMENDED_SHOW_CITATIONS
                )
                if show_citations:
                    formatted_citations = format_citations(citations)
                    async for _ in chat_log.async_add_assistant_content(
                        ha_conversation.AssistantContent(
                            agent_id=self._entity.entity_id,
                            content=formatted_citations,
                        )
                    ):
                        pass

            # --- CHECK FOR TOOL CALLS ---
            has_tool_calls = hasattr(response, "tool_calls") and response.tool_calls

            if not has_tool_calls:
                LOGGER.debug("tools_loop: no tool calls, final answer streamed.")
                break

            # --- EXECUTE TOOLS ---
            tool_results = []

            for tool_call in response.tool_calls:
                if get_tool_call_type:
                    tool_type = get_tool_call_type(tool_call)
                    if tool_type != "client_side_tool":
                        continue

                tool_start = time.time()
                tool_name = tool_call.function.name
                LOGGER.debug("tools_loop: executing tool '%s'", tool_name)

                # Delegate execution AND parsing to orchestrator
                # We pass the raw arguments (string or dict)
                result: ToolExecutionResult = (
                    await self._orchestrator.async_execute_tool(
                        tool_name, tool_call.function.arguments, user_input
                    )
                )

                tool_time = time.time() - tool_start
                if result.is_error:
                    LOGGER.debug(
                        "tool_exec: name=%s duration=%.2fs success=False error=%s",
                        tool_name,
                        tool_time,
                        str(result.result)[:500],
                    )
                else:
                    LOGGER.info(
                        "tool_exec: name=%s duration=%.2fs success=True",
                        tool_name,
                        tool_time,
                    )
                    LOGGER.debug(
                        "tool_exec_result: name=%s result='%s'",
                        tool_name,
                        str(result.result)[:500],
                    )

                tool_results.append(
                    ToolOutput(
                        tool_name=tool_name,
                        tool_result=result.result,
                        tool_call_id=tool_call.id,
                        is_error=result.is_error,
                    )
                )

            # --- STREAM FINAL ANSWER AFTER TOOLS ---
            if tool_results:
                await self._stream_final_answer_after_tools(
                    tool_results, conv_key, user_input, chat_log, timer
                )
            else:
                LOGGER.debug(
                    "tools_loop: no client-side tool results to report (server-side tools may have run)."
                )
            break

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
        scope: str,
        identifier: str,
    ) -> tuple[Any, str | None]:
        """Prepare xAI chat object for tools mode using the unified gateway method."""
        # 1. Refresh Tools Cache via Orchestrator
        await self._orchestrator.async_refresh_tools_if_needed(user_input)

        # 2. Get client tools from Orchestrator
        tools_for_call = self._orchestrator.get_xai_tools()

        # 3. Create Chat via Gateway
        # The gateway now handles system prompt, conv_key, and previous_id internally.
        chat, conv_key = await self._entity.gateway.create_chat(
            service_type="conversation",
            subentry_id=self._entity.subentry.subentry_id,
            client_tools=tools_for_call,
            entity=self._entity,
            mode_override="tools",
            scope=scope,
            identifier=identifier,
        )

        # 4. Handle History (append user messages)
        # We rely on 'conv_key' to determine if we need to hydrate history.
        # If we have a conv_key, the server has context, so we just add the new user message.
        # If we DON'T have a conv_key (e.g. first message, or store_messages=False),
        # we must send full history if available.
        if conv_key:
            last_user_message = get_last_user_message(chat_log)
            if last_user_message:
                send_user_name = self._entity._get_option(
                    CONF_SEND_USER_NAME, RECOMMENDED_SEND_USER_NAME
                )
                user_msg_formatted = await format_user_message_with_metadata(
                    last_user_message, user_input, self._entity.hass, send_user_name
                )
                chat.append(XAIGateway.user_msg(user_msg_formatted))
        else:
            # Full history hydration (Client-Side Memory or First Turn)
            await add_manual_history_to_chat(
                self._entity.hass,
                self._entity,
                chat,
                chat_log,
                user_input,
                RECOMMENDED_HISTORY_LIMIT_TURNS,
            )

        return chat, conv_key

    async def _stream_final_answer_after_tools(
        self,
        tool_results: list[ToolOutput],
        conv_key: str,
        user_input,
        chat_log,
        timer: LogTimeServices,
    ) -> None:
        """Stream the final answer after tool execution."""
        # Identify scope and identifier for continuity
        scope, identifier = extract_scope_and_identifier(user_input)

        chat, _ = await self._entity.gateway.create_chat(
            service_type="conversation",
            subentry_id=self._entity.subentry.subentry_id,
            scope=scope,
            identifier=identifier,
            client_tools=None,
            mode_override="tools",
            entity=self._entity,
        )

        # Retrieve store_messages dynamically for local logic (history append)
        # We can ask the entity helper, but since we just want to know if we should add history...
        store_messages = self._entity._get_option(
            CONF_STORE_MESSAGES, RECOMMENDED_STORE_MESSAGES
        )

        if not store_messages:
            # History is added here after the chat object is created,
            # which now contains the system prompt if created by the gateway.
            await add_manual_history_to_chat(
                self._entity.hass,
                self._entity,
                chat,
                chat_log,
                user_input,
                RECOMMENDED_HISTORY_LIMIT_TURNS,
            )

        for result in tool_results:
            tool_output_content = (
                str(result.tool_result)
                if not result.is_error
                else f"Error: {result.tool_result}"
            )
            LOGGER.debug(
                "Appending tool result to chat history: name=%s content='%s'",
                result.tool_name,
                tool_output_content[:500],
            )
            chat.append(XAIGateway.tool_msg(tool_output_content))

        response_holder = {"response": None}
        async for _ in chat_log.async_add_delta_content_stream(
            agent_id=self._entity.entity_id,
            stream=self._delta_generator(chat, response_holder, timer),
        ):
            await asyncio.sleep(0)

        final_response = response_holder.get("response")
        if final_response:
            await self._entity.gateway.async_log_completion(
                response=final_response,
                service_type="conversation",
                subentry_id=self._entity.subentry.subentry_id,
                conv_key=conv_key,
                mode="tools",
                is_fallback=False,
                entity=self._entity,
            )

    async def _delta_generator(
        self, chat, response_holder: dict, timer: LogTimeServices
    ):
        """Generate deltas for streaming."""
        new_message = True
        last_response = None

        try:
            async for response, chunk in timed_stream_generator(chat.stream(), timer):
                last_response = response

                if new_message:
                    yield {"role": "assistant"}
                    new_message = False

                if chunk.content:
                    yield {"content": chunk.content}

            response_holder["response"] = last_response
            response_holder["citations"] = getattr(last_response, "citations", [])

            usage_obj = getattr(last_response, "usage", None)
            response_holder["num_sources_used"] = getattr(
                usage_obj, "num_sources_used", 0
            )
            response_holder["server_side_tool_usage"] = getattr(
                last_response, "server_side_tool_usage", None
            )

        except Exception as err:
            response_holder["error"] = err

            if (
                _InactiveRpcError is not None
                and StatusCode is not None
                and isinstance(err, (_InactiveRpcError, AioRpcError))
                and err.code() == StatusCode.NOT_FOUND
            ):
                # Re-raise NOT_FOUND to trigger retry logic in async_process_with_loop
                raise

            LOGGER.error("tools_stream: error: %s", err)
            if new_message:
                yield {"role": "assistant"}
            yield {
                "content": f"An error occurred while processing your request with tools: {err}"
            }
