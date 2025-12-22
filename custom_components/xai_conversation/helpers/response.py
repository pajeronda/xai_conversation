"""Response parsing utilities for xAI conversation.

This module provides a modular parsing system with composable strategies
for handling various JSON response formats from Grok API.

Architecture:
1. Base parsing strategies (strict JSON, fenced blocks, balanced braces, regex)
2. Strategy composer that applies strategies in order
3. Use-case specific parsers that configure appropriate strategies
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import json
import re
from urllib.parse import urlparse

# Import LOGGER and constants
from ..const import (
    LOGGER,
    DOMAIN,
)


# ==============================================================================
# COMPILED REGEX PATTERNS (Module-level for performance)
# ==============================================================================
_JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
_CODE_BLOCK_PATTERN = re.compile(r"```(?:\w+)?\s*\n?(.*?)```", re.DOTALL)

# Grok Code Fast Patterns
_GROK_TEXT_PATTERN = re.compile(
    r'"response_text"\s*:\s*"(.*?)"(?=\s*,\s*"response_code"|\s*})', re.DOTALL
)
_GROK_CODE_PATTERN = re.compile(r'"response_code"\s*:\s*"(.*?)"\s*}', re.DOTALL)
_GROK_CLEANUP_PREFIX = re.compile(r'^\s*{"response_text"\s*:\s*"')
_GROK_CLEANUP_MIDDLE = re.compile(r'"\s*,?\s*"response_code".*$', re.DOTALL)
_GROK_CLEANUP_SUFFIX = re.compile(r'"\s*}?\s*$')

# HA Local Patterns
_HA_LOCAL_COMMANDS_PATTERN = re.compile(r'"commands"\s*:\s*(\[.*?\])', re.DOTALL)
_HA_LOCAL_SEQUENTIAL_PATTERN = re.compile(r'"sequential"\s*:\s*(true|false)', re.DOTALL)
_HA_LOCAL_TEXT_PATTERN = re.compile(r'"text"\s*:\s*"([^"]*)"', re.DOTALL)


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
    """Strategy 3: Extract first balanced {{...}} or [...] and parse as JSON.

    Args:
        text: Text potentially containing balanced braces

    Returns:
        Parsed dict from first balanced structure or None
    """

    def _find_balanced(s: str) -> str | None:
        opens = "{[ "
        closes = "}] "
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


def parse_regex_fields(
    text: str, field_patterns: dict[str, str | re.Pattern]
) -> dict | None:
    r"""Strategy 4: Extract specific fields using regex patterns.

    Useful for malformed JSON where you know which fields to extract.

    Args:
        text: Text to extract fields from
        field_patterns: Dict mapping field names to regex patterns (str or compiled)
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
        # If pattern is compiled, re.DOTALL is ignored here (uses pattern's flags)
        # If pattern is str, re.DOTALL is applied
        match = re.search(pattern, text, re.DOTALL)
        if match:
            value = match.group(1)
            # Try to parse as JSON if it looks like a structure
            if value.startswith(("{ ", "[ ")):
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
        text_match = _GROK_TEXT_PATTERN.search(text)
        code_match = _GROK_CODE_PATTERN.search(text)

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
        # Remove leading {"response_text": " and trailing "}"
        response_text = _GROK_CLEANUP_PREFIX.sub("", response_text)
        response_text = _GROK_CLEANUP_MIDDLE.sub("", response_text)
        response_text = _GROK_CLEANUP_SUFFIX.sub("", response_text)

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
                    "commands": _HA_LOCAL_COMMANDS_PATTERN,
                    "sequential": _HA_LOCAL_SEQUENTIAL_PATTERN,
                },
            )
            if fields and "commands" in fields:
                result = {"commands": fields["commands"]}
                if "sequential" in fields:
                    result["sequential"] = fields["sequential"] == "true"
                return result

        # Single command format
        fields = parse_regex_fields(text, {"text": _HA_LOCAL_TEXT_PATTERN})
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
# CITATION MANAGEMENT
# ==============================================================================


def extract_citation_info(citation) -> tuple[str, str]:
    """Extract title and URL from citation (string, dict, or object).

    Args:
        citation: Citation object (string URL, dict with title/url, or object)

    Returns:
        Tuple of (title, url)
    """
    if isinstance(citation, str):
        url = citation
        if "x.com" in url or "twitter.com" in url:
            title = "X/Twitter Post"
        elif "github.com" in url:
            title = "GitHub"
        else:
            title = urlparse(url).netloc or "Web Source"
    elif isinstance(citation, dict):
        title = citation.get("title", "No Title")
        url = citation.get("url", "No URL")
    else:
        title = getattr(citation, "title", "No Title")
        url = getattr(citation, "url", "No URL")
    return title, url


def format_citations(citations: list) -> str:
    """Format citations as a numbered list string.

    Args:
        citations: List of citation objects

    Returns:
        Formatted string starting with "\n\nCitations:\n"
    """
    if not citations:
        return ""

    lines = [
        f"[{i + 1}] {t} - {u}"
        for i, c in enumerate(citations)
        for t, u in [extract_citation_info(c)]
    ]
    return "\n\nCitations:\n" + "\n".join(lines)


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
    await_save: bool = False,
) -> None:
    """Save response metadata to memory and update token sensors.

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
        await_save: If True, waits for I/O operations to complete before returning.
                    Set to True for intermediate tool calls to avoid race conditions.
    """
    # Build list of async tasks to run in parallel
    tasks = []

    # Task 1: Save response ID to conversation memory
    if response_id and conv_key and service_type in ["conversation", "code_fast"]:

        async def _save_response_id():
            try:
                memory = hass.data[DOMAIN]["conversation_memory"]
                await memory.async_save_response(conv_key, response_id, store_messages)
                LOGGER.debug(
                    "memory_save: service=%s mode=%s conv_key=%s response_id=%s",
                    service_type,
                    mode,
                    conv_key,
                    response_id[:8],
                )
            except Exception as err:
                LOGGER.error(
                    "Failed to store response_id for %s: %s", service_type, err
                )

        tasks.append(_save_response_id())

    # Task 2: Update token stats
    if usage:

        async def _update_token_stats():
            storage = hass.data.get(DOMAIN, {}).get("token_stats")
            if storage:
                await storage.async_update_usage(
                    service_type=service_type,
                    model=model,
                    usage=usage,
                    mode=mode,
                    is_fallback=is_fallback,
                    store_messages=store_messages,
                    server_side_tool_usage=server_side_tool_usage,
                    num_sources_used=num_sources_used,
                    response_id=response_id,
                )

        tasks.append(_update_token_stats())

    # EXECUTE: Either await or dispatch to background
    if tasks:
        async def _background_save():
            """Task that executes I/O operations."""
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except Exception as err:
                LOGGER.error(
                    "Save task failed for %s: %s", service_type, err
                )

        if await_save:
            # Sequential execution for critical paths (e.g. tool loop memory)
            await _background_save()
        else:
            # FIRE-AND-FORGET: Dispatch to background to avoid delaying user response
            task = asyncio.create_task(_background_save())

            # Store task reference in hass.data for graceful shutdown tracking
            if "pending_save_tasks" not in hass.data[DOMAIN]:
                hass.data[DOMAIN]["pending_save_tasks"] = set()

            hass.data[DOMAIN]["pending_save_tasks"].add(task)
            task.add_done_callback(
                lambda t: hass.data[DOMAIN]["pending_save_tasks"].discard(t)
            )

            # ALSO track in the specific entity if provided (for entity-level shutdown)
            if entity is not None and hasattr(entity, "_pending_save_tasks"):
                entity._pending_save_tasks.add(task)
                task.add_done_callback(lambda t: entity._pending_save_tasks.discard(t))

    # Log search details (non-blocking, informational only)
    if service_type == "conversation":
        if citations:
            LOGGER.debug("conversation: citations found: %d", len(citations))
        if num_sources_used > 0:
            LOGGER.debug(
                "conversation: unique search sources used: %d", num_sources_used
            )
