"""Tools processing logic for the xAI Conversation integration."""

from __future__ import annotations

# Standard library imports
import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

# Home Assistant imports

# Local imports
from .const import (
    CHAT_MODE_TOOLS,
    LOGGER,
)
from .helpers import (
    LogTimeServices,
    ToolOrchestrator,
    ToolSessionConfig,
    ToolExecutionResult,
    ChatOptions,
    BaseConversationProcessor,
    resolve_tool_session_config,
    get_tool_call_type,
    translate_tool_message,
)
from .exceptions import is_not_found_error

if TYPE_CHECKING:
    from .entity import XAIBaseLLMEntity


@dataclass
class ToolOutput:
    """Result from a tool execution."""

    tool_name: str
    tool_result: Any
    tool_call_id: str = ""
    is_error: bool = False


class XAIToolsProcessor(BaseConversationProcessor):
    """Manages tool execution and conversation processing in tools-mode.

    Delegates specific tool lifecycle and execution to the ToolOrchestrator,
    while managing the chat loop, streaming, and error handling.
    """

    def __init__(self, entity: XAIBaseLLMEntity):
        """Initialize the XAIToolsProcessor."""
        super().__init__(
            entity.hass,
            entity.gateway,
            entity.entity_id,
            entity._conversation_memory,
            entity.get_config_dict(),
        )
        self._entity = entity
        self._orchestrator = ToolOrchestrator(self.hass)

    async def async_process_with_loop(
        self,
        user_input,
        chat_log,
        timer: LogTimeServices,
        options: ChatOptions | None = None,
        force_tools: bool = False,
    ) -> None:
        """Execute the conversation loop, managing multiple turns of tool usage."""
        # 1. Prepare tools and chat once before the loop
        session_config = resolve_tool_session_config(
            config=self._entity.get_config_dict(),
            entry_data=self._entity.entry.data,
            registry=self._entity._extended_tools_registry,
        )
        await self._orchestrator.async_refresh_tools_if_needed(
            user_input, session_config
        )

        # Use pre-resolved options if available, otherwise fallback to local creation
        params = options or ChatOptions(
            user_input=user_input, mode_override=CHAT_MODE_TOOLS
        )
        params.client_tools = self._orchestrator.get_xai_tools()

        max_iterations = params.max_turns

        last_chat = None
        last_conv_key = None
        for i in range(max_iterations):
            is_primary = i == 0
            LOGGER.debug("[tools] turn %d/%d", i + 1, max_iterations)

            is_fallback = force_tools and is_primary

            try:
                # Execute chat loop via base processor
                # In turns > 1, reuse the chat object to maintain local turn state
                results = await self._async_run_chat_loop(
                    chat_log,
                    timer,
                    params,
                    mode_override=CHAT_MODE_TOOLS,
                    is_fallback=is_fallback,
                    chat_override=last_chat,
                    conv_key_override=last_conv_key,
                )
                chat = results["chat"]
                last_chat = chat
                last_conv_key = results.get("conv_key")
                response_holder = results.get("response_holder", {})

                if "error" in results:
                    LOGGER.error("[tools] stream failed: %s", results["error"])
                    break
            except Exception as err:
                # Retry handled by base processor
                if is_not_found_error(err):
                    LOGGER.error("[tools] NOT_FOUND error, retry exhausted")
                    break
                raise

            response = response_holder.get("response")
            if not response:
                break

            # --- CHECK FOR TOOL CALLS ---
            has_tool_calls = hasattr(response, "tool_calls") and response.tool_calls

            if not has_tool_calls:
                LOGGER.debug("[tools] complete: turns=%d", i + 1)
                break

            # --- ITERATION LIMIT CHECK ---
            if i + 1 >= max_iterations:
                LOGGER.warning("[tools] max iterations reached (%d)", max_iterations)
                await self._entity._add_error_response_and_continue(
                    user_input,
                    chat_log,
                    "I'm having trouble completing the request, it seems to be stuck in a loop.",
                )
                break

            # --- PREPARE FOR NEXT ITERATION ---
            chat.append(response)

            # --- EXECUTE TOOLS ---
            tool_results = await self._execute_tool_calls_batch(
                response.tool_calls, user_input
            )

            if tool_results:
                for result in tool_results:
                    tool_output_content = (
                        str(result.tool_result)
                        if not result.is_error
                        else f"Error: {result.tool_result}"
                    )
                    chat.append(
                        translate_tool_message(
                            tool_output_content, tool_call_id=result.tool_call_id
                        )
                    )
            else:
                LOGGER.debug("[tools] complete: server-side tools only")
                break

    async def _execute_tool_calls_batch(
        self, tool_calls: list, user_input
    ) -> list[ToolOutput]:
        """Execute a batch of tool calls with parallel execution.

        Uses staggered start (0.1s delay between tools) to avoid race conditions
        while still benefiting from parallel I/O. Uses asyncio.gather to preserve
        original order (important for SDK tool matching).
        """
        if not tool_calls:
            return []

        # Filter for client-side tools
        client_tool_calls = []
        for tc in tool_calls:
            # Use helper to determine tool type
            tool_type = get_tool_call_type(tc)

            if tool_type == "server_side_tool":
                # Debug logging for skipped server tools
                LOGGER.debug("[tools] skipping server-side tool: %s", tc.function.name)
                continue

            if tool_type == "client_side_tool":
                client_tool_calls.append(tc)
            # If function name is NOT in server tools (fallback logic if getting type fails or old SDK)
            elif not tool_type:
                # Default to client side if unknown
                client_tool_calls.append(tc)

        if not client_tool_calls:
            return []

        # Build config once for the batch
        session_config = resolve_tool_session_config(
            config=self._entity.get_config_dict(),
            entry_data=self._entity.entry.data,
            registry=self._entity._extended_tools_registry,
        )

        # Single tool: fast path (no parallelization overhead)
        if len(client_tool_calls) == 1:
            result = await self._execute_single_tool(
                client_tool_calls[0], user_input, session_config
            )
            return [result]

        # Multiple tools: parallel execution with staggered start
        # Uses asyncio.gather to preserve original order (important for SDK tool matching)
        LOGGER.debug("[tools] parallel exec: %d tools", len(client_tool_calls))

        async def _exec_with_stagger(tc, idx: int) -> ToolOutput:
            """Execute tool with staggered delay to avoid race conditions."""
            if idx > 0:
                await asyncio.sleep(idx * 0.1)
            return await self._execute_single_tool(tc, user_input, session_config)

        tasks = [
            _exec_with_stagger(tc, idx)
            for idx, tc in enumerate(client_tool_calls)
        ]

        # Gather preserves order (unlike as_completed) - important for SDK tool matching
        results = await asyncio.gather(*tasks, return_exceptions=True)

        tool_results = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                LOGGER.warning(
                    "[tools] parallel exec failed for tool %d: %s",
                    idx,
                    result,
                )
                # Create error result to maintain tool count consistency
                tool_results.append(
                    ToolOutput(
                        tool_name=client_tool_calls[idx].function.name,
                        tool_result=f"Execution error: {result}",
                        tool_call_id=getattr(client_tool_calls[idx], "id", ""),
                        is_error=True,
                    )
                )
            else:
                tool_results.append(result)

        return tool_results

    async def _execute_single_tool(
        self, tool_call, user_input, session_config: ToolSessionConfig
    ) -> ToolOutput:
        """Atomic execution of a single tool call."""
        tool_start = time.time()
        tool_name = tool_call.function.name
        LOGGER.debug("[tools] exec: '%s'", tool_name)

        # Delegate execution AND parsing to orchestrator
        # Pass the config explicitly
        result: ToolExecutionResult = await self._orchestrator.async_execute_tool(
            tool_name, tool_call.function.arguments, user_input, session_config
        )

        tool_time = time.time() - tool_start
        if result.is_error:
            LOGGER.debug(
                "[tools] exec: '%s' failed (%.0fms) - %s",
                tool_name,
                tool_time * 1000,
                str(result.result)[:100],
            )
        else:
            LOGGER.debug("[tools] exec: '%s' ok (%.0fms)", tool_name, tool_time * 1000)

        return ToolOutput(
            tool_name=tool_name,
            tool_result=result.result,
            tool_call_id=getattr(tool_call, "id", ""),
            is_error=result.is_error,
        )
