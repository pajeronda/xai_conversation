"""Response parsing utilities for xAI conversation.

This module provides a modular parsing system with composable strategies
for handling various JSON response formats from Grok API.

Architecture:
1. Base parsing strategies (strict JSON, fenced blocks, balanced braces, regex)
2. Strategy composer that applies strategies in order
3. Use-case specific parsers that configure appropriate strategies
"""
from __future__ import annotations

import json
import re
from typing import Optional, Callable, Any


# ==============================================================================
# BASE PARSING STRATEGIES (Building Blocks)
# ==============================================================================

def parse_strict_json(text: str) -> Optional[dict]:
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


def parse_json_fenced(text: str) -> Optional[dict]:
    """Strategy 2: Extract JSON from markdown code fence (```json ... ```).

    Args:
        text: Text potentially containing fenced JSON

    Returns:
        Parsed dict from first fenced block or None
    """
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence:
        inner = fence.group(1).strip()
        return parse_strict_json(inner)
    return None


def parse_balanced_braces(text: str) -> Optional[dict]:
    """Strategy 3: Extract first balanced {...} or [...] and parse as JSON.

    Args:
        text: Text potentially containing balanced braces

    Returns:
        Parsed dict from first balanced structure or None
    """
    def _find_balanced(s: str) -> Optional[str]:
        opens = '{['
        closes = '}]'
        stack = []
        start = -1
        for i, ch in enumerate(s):
            if ch in opens:
                if not stack:
                    start = i
                stack.append(ch)
            elif ch in closes and stack:
                # Ensure matching pair
                if (stack[-1] == '{' and ch == '}') or (stack[-1] == '[' and ch == ']'):
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


def parse_regex_fields(text: str, field_patterns: dict[str, str]) -> Optional[dict]:
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
            if value.startswith(('{', '[')):
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
    strategies: list[Callable[[str], Optional[dict]]],
    validator: Optional[Callable[[dict], bool]] = None
) -> Optional[dict]:
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
    3. Plain text with code blocks
    4. Plain text only

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
        return isinstance(data, dict) and ("response_text" in data or "response_code" in data)

    # Try parsing using modular strategies
    parsed = parse_with_strategies(
        response_data,
        strategies=[parse_strict_json, parse_json_fenced, parse_balanced_braces],
        validator=_validate_code_structure
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

        return response_text, response_code

    # Fallback: Try to extract code blocks from plain text
    code_block_pattern = r"```(?:\w+)?\n(.*?)```"
    code_blocks = re.findall(code_block_pattern, response_data, re.DOTALL)

    if code_blocks:
        # Use first code block as code
        response_code = code_blocks[0].strip()
        # Remove code blocks from text
        response_text = re.sub(code_block_pattern, "", response_data, flags=re.DOTALL).strip()
        return response_text, response_code

    # Final fallback: Treat everything as text
    return response_data, ""


def parse_ha_local_payload(payload_json: str) -> Optional[dict]:
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
    def _ha_local_regex_strategy(text: str) -> Optional[dict]:
        """Custom regex strategy for HA_LOCAL payloads."""
        # Multi-command format detection
        if '"commands"' in text:
            fields = parse_regex_fields(text, {
                "commands": r'"commands"\s*:\s*(\[.*?\])',
                "sequential": r'"sequential"\s*:\s*(true|false)'
            })
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
        validator=_validator
    )
