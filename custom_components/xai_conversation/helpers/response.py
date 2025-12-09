"""Response parsing utilities for xAI conversation.

This module provides a modular parsing system with composable strategies
for handling various JSON response formats from Grok API.

Architecture:
1. Base parsing strategies (strict JSON, fenced blocks, balanced braces, regex)
2. Strategy composer that applies strategies in order
3. Use-case specific parsers that configure appropriate strategies
"""

from __future__ import annotations

from collections.abc import Callable
import json
import re

# Import LOGGER and constants
from ..const import (
    LOGGER,
    DOMAIN,
    CONF_CHAT_MODEL,
    RECOMMENDED_CHAT_MODEL,
)



# ==============================================================================
# COMPILED REGEX PATTERNS (Module-level for performance)
# ==============================================================================
_JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
_CODE_BLOCK_PATTERN = re.compile(r"```(?:\w+)?\s*\n?(.*?)```", re.DOTALL)

# ==============================================================================
# BASE PARSING STRATEGIES (Building Blocks)
# ==============================================================================


def parse_strict_json(text: str) -> dict | None:
    """Strategy 1: Strict JSON parsing.

    Args:
        text: Raw text to parse

    Returns:
        Parsed dict or None if parsing fails
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def parse_json_fenced(text: str) -> dict | None:
    """Strategy 2: Extract JSON from markdown code fence (```json ... ```).

    Args:
        text: Text potentially containing fenced JSON

    Returns:
        Parsed dict from first fenced block or None
    """
    fence = _JSON_FENCE_PATTERN.search(text)
    if fence:
        inner = fence.group(1).strip()
        return parse_strict_json(inner)
    return None


def parse_balanced_braces(text: str) -> dict | None:
    """Strategy 3: Extract first balanced {...} or [...] and parse as JSON.

    Args:
        text: Text potentially containing balanced braces

    Returns:
        Parsed dict from first balanced structure or None
    """

    def _find_balanced(s: str) -> str | None:
        opens = "{["
        closes = "}]"
        stack = []
        start = -1
        for i, ch in enumerate(s):
            if ch in opens:
                if not stack:
                    start = i
                stack.append(ch)
            elif ch in closes and stack:
                # Ensure matching pair
                if (stack[-1] == "{" and ch == "}") or (stack[-1] == "[" and ch == "]"):
                    stack.pop()
                    if not stack and start != -1:
                        return s[start : i + 1]
                else:
                    # Mismatch, reset
                    stack.clear()
                    start = -1
        return None

    candidate = _find_balanced(text)
    if candidate:
        return parse_strict_json(candidate)
    return None


def parse_regex_fields(text: str, field_patterns: dict[str, str]) -> dict | None:
    r"""Strategy 4: Extract specific fields using regex patterns.

    Useful for malformed JSON where you know which fields to extract.

    Args:
        text: Text to extract fields from
        field_patterns: Dict mapping field names to regex patterns
                       Pattern should have one capture group for the value

    Returns:
        Dict with extracted fields or None if no fields matched

    Example:
        >>> parse_regex_fields(text, {
        ...     "text": r'"text"\s*:\s*"([^"]*)"',
        ...     "commands": r'"commands"\s*:\s*(\[.*?\])'
        ... })
    """
    result = {}
    for field, pattern in field_patterns.items():
        match = re.search(pattern, text, re.DOTALL)
        if match:
            value = match.group(1)
            # Try to parse as JSON if it looks like a structure
            if value.startswith(("{", "[")):
                parsed = parse_strict_json(value)
                result[field] = parsed if parsed is not None else value
            else:
                result[field] = value.strip()

    return result if result else None


# ==============================================================================
# STRATEGY COMPOSER
# ==============================================================================


def parse_with_strategies(
    text: str,
    strategies: list[Callable[[str], dict | None]],
    validator: Callable[[dict], bool] | None = None,
) -> dict | None:
    """Apply parsing strategies in order until one succeeds.

    Args:
        text: Text to parse
        strategies: List of parsing functions to try in order
        validator: Optional function to validate result (return True if valid)

    Returns:
        First successful parse result that passes validation, or None
    """
    for strategy in strategies:
        try:
            result = strategy(text)
            if result is not None:
                if validator is None or validator(result):
                    return result
        except Exception:
            continue
    return None


# ==============================================================================
# GROK CODE FAST PARSER - Uses modular strategy system
# ==============================================================================


def parse_grok_code_response(response_data: str) -> tuple[str, str]:
    """Parse Grok Code Fast response to extract text and code components.

    Uses modular parsing strategies to handle multiple response formats:
    1. Direct JSON: {"response_text": "...", "response_code": "..."}
    2. JSON in code fence: ```json\n{"response_text": ...}\n```
    3. Regex extraction (tolerant of malformed JSON/multiline strings)
    4. Plain text with code blocks
    5. Plain text only

    Args:
        response_data: Raw response string from Grok Code Fast

    Returns:
        Tuple of (response_text, response_code)

    Used by:
        - grok_code_fast service: Parses AI-generated code responses
    """

    # Validator: Check if parsed JSON has required structure
    def _validate_code_structure(data: dict) -> bool:
        """Validate that dict has response_text or response_code fields."""
        return isinstance(data, dict) and (
            "response_text" in data or "response_code" in data
        )

    # Strategy: Regex for tolerant extraction (handles malformed JSON/multiline strings)
    def _code_regex_strategy(text: str) -> dict | None:
        # Extract response_text (capture until response_code key or end)
        # This pattern is tolerant of unescaped newlines
        text_match = re.search(
            r'"response_text"\s*:\s*"(.*?)"(?=\s*,\s*"response_code"|\s*})',
            text,
            re.DOTALL,
        )
        code_match = re.search(r'"response_code"\s*:\s*"(.*?)"\s*}', text, re.DOTALL)

        if text_match or code_match:
            return {
                "response_text": text_match.group(1) if text_match else "",
                "response_code": code_match.group(1) if code_match else "",
            }
        return None

    # Try parsing using modular strategies
    parsed = parse_with_strategies(
        response_data,
        strategies=[
            parse_strict_json,
            parse_json_fenced,
            parse_balanced_braces,
            _code_regex_strategy,
        ],
        validator=_validate_code_structure,
    )

    if parsed:
        # Extract fields from parsed JSON
        response_text = parsed.get("response_text", "")
        response_code = parsed.get("response_code", "")

        # Handle double-nested JSON (Grok sometimes wraps JSON inside response_text)
        if isinstance(response_text, str) and response_text.strip().startswith("{"):
            try:
                inner_parsed = json.loads(response_text)
                if _validate_code_structure(inner_parsed):
                    response_text = inner_parsed.get("response_text", "")
                    response_code = inner_parsed.get("response_code", response_code)
            except (json.JSONDecodeError, ValueError):
                pass  # Keep original values

        # Cleanup: Unescape escaped newlines/quotes if regex extraction was used (json.loads does this automatically)
        # Simple cleanup for regex-extracted content
        if not isinstance(response_text, str):
            response_text = str(response_text)
        if not isinstance(response_code, str):
            response_code = str(response_code)

        # If strict JSON failed, we might have raw escaped chars from regex
        if "\\" in response_code and not isinstance(parsed, dict):  # Heuristic check
            try:
                response_code = response_code.encode("utf-8").decode("unicode_escape")
            except Exception:
                pass

        return response_text, response_code

    # Fallback: Try to extract code blocks from plain text
    code_blocks = _CODE_BLOCK_PATTERN.findall(response_data)

    if code_blocks:
        # Use first code block as code
        response_code = code_blocks[0].strip()
        # Remove code blocks from text
        response_text = _CODE_BLOCK_PATTERN.sub("", response_data).strip()

        # CLEANUP: Remove leftover JSON artifacts from text if fallback triggered on broken JSON
        # Remove leading {"response_text": " and trailing "}
        response_text = re.sub(r'^\s*{"response_text"\s*:\s*"', "", response_text)
        response_text = re.sub(
            r'"\s*,?\s*"response_code".*$', "", response_text, flags=re.DOTALL
        )
        response_text = re.sub(r'"\s*}?\s*$', "", response_text)

        return response_text, response_code

    # Final fallback: Treat everything as text
    return response_data, ""


def parse_ha_local_payload(payload_json: str) -> dict | None:
    """Parse [[HA_LOCAL: {...}]] JSON payload with robust fallback strategies.

    Uses modular strategy system:
    1. Strict JSON parsing
    2. Regex extraction for malformed JSON (text, commands, sequential fields)

    Handles:
    - Single command format: {"text": "command"}
    - Multi-command format: {"commands": [...], "sequential": bool}

    Args:
        payload_json: Raw JSON string extracted from [[HA_LOCAL: {...}]]

    Returns:
        Parsed dict with "text" or "commands" key, or None if parsing fails
    """

    def _ha_local_regex_strategy(text: str) -> dict | None:
        """Custom regex strategy for HA_LOCAL payloads."""
        # Multi-command format detection
        if '"commands"' in text:
            fields = parse_regex_fields(
                text,
                {
                    "commands": r'"commands"\s*:\s*(\[.*?\])',
                    "sequential": r'"sequential"\s*:\s*(true|false)',
                },
            )
            if fields and "commands" in fields:
                result = {"commands": fields["commands"]}
                if "sequential" in fields:
                    result["sequential"] = fields["sequential"] == "true"
                return result

        # Single command format
        fields = parse_regex_fields(text, {"text": r'"text"\s*:\s*"([^"]*)"'})
        return fields if fields else None

    def _validator(result: dict) -> bool:
        """Ensure result has required HA_LOCAL fields."""
        return "text" in result or "commands" in result

    return parse_with_strategies(
        payload_json,
        strategies=[
            parse_strict_json,
            _ha_local_regex_strategy,
        ],
        validator=_validator,
    )


# ==============================================================================
# RESPONSE METADATA MANAGEMENT
# ==============================================================================


async def save_response_metadata(
    hass,
    entry_id: str,
    usage,
    model: str | None,
    service_type: str,
    mode: str = "unknown",
    is_fallback: bool = False,
    store_messages: bool = True,
    conv_key: str | None = None,
    response_id: str | None = None,
    entity=None,
    citations: list | None = None,
    num_sources_used: int = 0,
    server_side_tool_usage: dict | None = None,
) -> None:
    """Save response metadata to memory and update token sensors.

    IMPORTANT: This function is fully non-blocking. All I/O operations
    (memory save, token stats update) are dispatched as background tasks
    to avoid delaying the conversation response to the user.

    Args:
        hass: Home Assistant instance
        entry_id: Config entry ID
        usage: Usage statistics from xAI response
        model: Model name (extracted from response.model or usage)
        service_type: Service type ("conversation", "ai_task", "code_fast")
        mode: Mode string ("pipeline" or "tools") - only used for conversation
        is_fallback: Whether this is a fallback response - only used for conversation
        store_messages: Server-side (True) or client-side (False) memory - only used for conversation
        conv_key: Conversation key for memory storage (None for ai_task/code_fast)
        response_id: xAI response ID to save (None if not using server-side memory)
        entity: Optional entity instance for graceful shutdown tracking (conversation only)
        citations: List of citations from xAI response (conversation only)
        num_sources_used: Number of unique search sources used (conversation only)
        server_side_tool_usage: Dictionary of server-side tool invocations (e.g. {"web_search": 1})
    """
    # Resolve model synchronously (no I/O needed)
    resolved_model = model
    fallback_source = None

    if resolved_model is None and entity is not None:
        resolved_model = entity._get_option(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)
        if resolved_model:
            fallback_source = "entity_config"

    if resolved_model is None:
        for entry in hass.config_entries.async_entries(DOMAIN):
            if entry.entry_id == entry_id:
                resolved_model = entry.data.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)
                if resolved_model:
                    fallback_source = "entry_data"
                break

    # Single unified log message
    if fallback_source:
        LOGGER.debug(
            "Model not in response, using %s fallback: %s",
            fallback_source,
            resolved_model
        )

    # Build list of async tasks to run in parallel
    tasks = []

    # Task 1: Save response ID to conversation memory
    if response_id and conv_key and service_type == "conversation":
        async def _save_response_id():
            try:
                memory = hass.data[DOMAIN]["conversation_memory"]
                await memory.save_response_id_by_key(conv_key, response_id)
                LOGGER.debug(
                    "memory_save: service=%s mode=%s conv_key=%s response_id=%s",
                    service_type,
                    mode,
                    conv_key,
                    response_id[:8],
                )
            except Exception as err:
                LOGGER.error("Failed to store response_id for %s: %s", service_type, err)

        tasks.append(_save_response_id())

    # Task 2: Update token stats
    if usage:
        async def _update_token_stats():
            storage = hass.data.get(DOMAIN, {}).get("token_stats")
            if storage:
                await storage.async_update_usage(
                    service_type=service_type,
                    model=resolved_model,
                    usage=usage,
                    mode=mode,
                    is_fallback=is_fallback,
                    store_messages=store_messages,
                    server_side_tool_usage=server_side_tool_usage,
                )

        tasks.append(_update_token_stats())

    # FIRE-AND-FORGET: Execute I/O in background without blocking the response
    if tasks:
        import asyncio

        async def _background_save():
            """Background task that executes I/O operations without blocking caller."""
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except Exception as err:
                LOGGER.error("Background save task failed for %s: %s", service_type, err)

        # Create background task and track for graceful shutdown
        task = asyncio.create_task(_background_save())

        # Store task reference in hass.data for graceful shutdown tracking
        if "pending_save_tasks" not in hass.data[DOMAIN]:
            hass.data[DOMAIN]["pending_save_tasks"] = set()

        hass.data[DOMAIN]["pending_save_tasks"].add(task)
        task.add_done_callback(
            lambda t: hass.data[DOMAIN]["pending_save_tasks"].discard(t)
        )

    # Log search details (non-blocking, informational only)
    if service_type == "conversation":
        if citations:
            LOGGER.debug("conversation: citations found: %d", len(citations))
        if num_sources_used > 0:
            LOGGER.debug("conversation: unique search sources used: %d", num_sources_used)


async def handle_response_not_found_error(
    err: Exception,
    attempt: int,
    memory,
    conv_key: str | None,
    mode: str,
    context_id: str = "",
) -> bool:
    """Handle NOT_FOUND error for expired response_id on xAI server.

    When xAI returns NOT_FOUND for a previous_response_id, it means the
    conversation context has expired server-side. This function clears
    the local memory and signals to retry with a fresh conversation.

    Args:
        err: The exception that was raised
        attempt: Current attempt number (0-indexed)
        memory: ConversationMemory or CodeMemory instance
        conv_key: Conversation key for memory cleanup
        mode: Conversation mode ("pipeline", "tools", "code")
        context_id: Optional context identifier for logging

    Returns:
        True if should retry (cleared memory), False if should re-raise error
    """
    # Check if it's a gRPC NOT_FOUND error
    try:
        from grpc import StatusCode
        from grpc._channel import _InactiveRpcError

        if not isinstance(err, _InactiveRpcError):
            return False

        if err.code() != StatusCode.NOT_FOUND:
            return False

    except ImportError:
        return False

    # Only retry on first attempt
    if attempt != 0:
        return False

    # Log the retry
    context_prefix = f"[Context: {context_id}] " if context_id else ""
    LOGGER.warning(
        "%sConversation context not found on xAI server (expired response_id). "
        "Clearing local memory and retrying with fresh conversation.",
        context_prefix,
    )

    # Clear memory by key if available
    if conv_key and hasattr(memory, "clear_memory_by_key"):
        try:
            await memory.clear_memory_by_key(conv_key)
            LOGGER.debug("Cleared expired memory for conv_key=%s", conv_key)
        except Exception as clear_err:
            LOGGER.warning("Failed to clear memory by key: %s", clear_err)
    # Fallback to clear_memory if clear_memory_by_key not available
    elif hasattr(memory, "clear_memory"):
        try:
            # Extract user_id from conv_key if possible
            # Format: "user:{user_id}:mode:{mode}:..."
            if conv_key and conv_key.startswith("user:"):
                parts = conv_key.split(":")
                if len(parts) >= 2:
                    user_id = parts[1]
                    await memory.clear_memory(user_id, mode)
                    LOGGER.debug(
                        "Cleared expired memory for user_id=%s mode=%s", user_id, mode
                    )
        except Exception as clear_err:
            LOGGER.warning("Failed to clear memory: %s", clear_err)

    return True  # Signal to retry
