"""Response parsing utilities for xAI conversation.

Provides a modular strategy system for parsing various JSON response formats
(Strict, Fenced, Regex, etc.) from the xAI API.
"""

from __future__ import annotations

import datetime
import json
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

from ..const import (
    HA_LOCAL_COMMANDS_PATTERN,
    HA_LOCAL_SEQUENTIAL_PATTERN,
    HA_LOCAL_TAG_PREFIX,
    HA_LOCAL_TEXT_PATTERN,
    JSON_FENCE_PATTERN,
)


# ==============================================================================
# SHARED BUILDING BLOCKS
# ==============================================================================


def parse_strict_json(text: str) -> dict | None:
    """Fast-path strict JSON parsing."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


# ==============================================================================
# AI TASK PARSING (Extraction & Refinement)
# ==============================================================================
# These functions are used by XAITaskEntity to extract structured data
# from models that might include conversational filler or markdown.


def parse_json_fenced(text: str) -> dict | None:
    """Strategy: Extract JSON from markdown code fence (```json ... ```)."""
    fence = JSON_FENCE_PATTERN.search(text)
    if fence:
        inner = fence.group(1).strip()
        return parse_strict_json(inner)
    return None


def parse_balanced_braces(text: str) -> dict | None:
    """Strategy: Extract first balanced {...} or [...] and parse as JSON."""

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
                if (stack[-1] == "{" and ch == "}") or (stack[-1] == "[" and ch == "]"):
                    stack.pop()
                    if not stack and start != -1:
                        return s[start : i + 1]
                else:
                    stack.clear()
                    start = -1
        return None

    candidate = _find_balanced(text)
    if candidate:
        return parse_strict_json(candidate)
    return None


def parse_with_strategies(
    text: str,
    strategies: list[Callable[[str], dict | None]],
    validator: Callable[[dict], bool] | None = None,
) -> dict | None:
    """Apply multiple parsing strategies in order until one succeeds."""
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
# CITATION MANAGEMENT
# ==============================================================================


def _extract_citation_info(citation) -> tuple[str, str]:
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
    # Handle xAI SDK InlineCitation objects
    elif hasattr(citation, "web_citation") and citation.web_citation.url:
        url = citation.web_citation.url
        title = urlparse(url).netloc or "Web Source"
    elif hasattr(citation, "x_citation") and citation.x_citation.url:
        url = citation.x_citation.url
        title = "X/Twitter Post"
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
        for t, u in [_extract_citation_info(c)]
    ]
    return "\n\nCitations:\n" + "\n".join(lines)


# ==============================================================================
# RESPONSE METADATA EXTRACTION
# ==============================================================================


def extract_response_metadata(
    response: Any, holder: dict, fallback_model: str | None = None
) -> None:
    """Extract all metadata from the final response object into a holder dictionary.

    Args:
        response: The final response object from xAI SDK
        holder: Dictionary to populate with metadata
        fallback_model: Optional model name to use if response.model is missing
    """
    holder["id"] = getattr(response, "id", None)
    holder["usage"] = getattr(response, "usage", None)
    holder["citations"] = getattr(response, "citations", [])
    holder["inline_citations"] = getattr(response, "inline_citations", [])

    # Handle created timestamp (can be int or datetime)
    holder["created"] = None
    if hasattr(response, "created") and response.created:
        if isinstance(response.created, datetime.datetime):
            holder["created"] = response.created.isoformat()
        else:
            try:
                holder["created"] = datetime.datetime.fromtimestamp(
                    response.created
                ).isoformat()
            except (ValueError, TypeError):
                holder["created"] = datetime.datetime.now().isoformat()

    # Model resolution
    model = getattr(response, "model", None)
    if not model:
        model = fallback_model
    holder["model"] = model

    # Usage details
    usage_obj = holder.get("usage")
    holder["num_sources_used"] = getattr(usage_obj, "num_sources_used", 0)
    holder["server_side_tool_usage"] = getattr(response, "server_side_tool_usage", None)

    # Stats logging moved to sensors.py to avoid duplication


# ==============================================================================
# ZDR CONTENT MANAGEMENT (Zero Data Retention)
# ==============================================================================


def capture_zdr_content(response, chunk, holder: dict) -> None:
    """Capture encrypted content (ZDR blob) from a stream response or chunk.

    Looks into all common nested structures, including underlying protos.

    Args:
        response: The stream response object (accumulated)
        chunk: The current chunk object (delta)
        holder: Dictionary to store the encrypted content
    """

    def _get_blob(obj, label: str) -> str | None:
        if obj is None:
            return None

        # 1. Direct property check (on decorator or dict)
        if blob := getattr(obj, "encrypted_content", None):
            return blob

        # 2. Check underlying proto if it's a decorator
        proto = getattr(obj, "proto", obj)
        if blob := getattr(proto, "encrypted_content", None):
            return blob

        # 3. Check choices/outputs
        choices = getattr(obj, "choices", []) or getattr(proto, "outputs", [])
        for choice in choices:
            # Check choice property (e.g. choice.delta.encrypted_content)
            if blob := getattr(choice, "encrypted_content", None):
                return blob

            # Check choice proto
            c_proto = getattr(choice, "proto", choice)
            if blob := getattr(c_proto, "encrypted_content", None):
                return blob

            # Deep dive into delta/message
            for sub_name in ["delta", "message"]:
                if (sub := getattr(c_proto, sub_name, None)) is not None:
                    if blob := getattr(sub, "encrypted_content", None):
                        return blob

        return None

    # Check chunk (typically contains partial blob in stream)
    if blob := _get_blob(chunk, "chunk"):
        # If it's a new or partial blob, we should keep it.
        # SDK accumulated Response will also have the full one.
        holder["encrypted_content"] = (holder.get("encrypted_content") or "") + blob

    # Check response (accumulated total - usually more reliable at the end)
    if blob := _get_blob(response, "response"):
        # Overwrite with the total if available
        holder["encrypted_content"] = blob


def restore_zdr_content(last_response, holder: dict) -> None:
    """Restore captured ZDR content to the final response object.

    Ensures that if we captured encrypted content during the stream but it wasn't
    present in the very last chunk, it gets restored to the final response object.

    Args:
        last_response: The final response object
        holder: Dictionary containing potentially captured encrypted content
    """
    if holder.get("encrypted_content") and not getattr(
        last_response, "encrypted_content", None
    ):
        last_response.encrypted_content = holder["encrypted_content"]


# ==============================================================================
# PIPELINE STREAM PARSER (HA_LOCAL Tags)
# ==============================================================================


class StreamParser:
    """Parses streaming content to detect and extract tags like [[HA_LOCAL...]]."""

    def __init__(self):
        """Initialize the parser."""
        self.buffer = ""
        self.suspicious_buffer = ""
        self.streaming_active = True

    def process_chunk(self, content_chunk: str) -> list[str]:
        """Process a text chunk and return a list of safe-to-yield content.

        Args:
            content_chunk: The new text fragment from the stream.

        Returns:
            List of strings that are safe to yield to the user.
        """
        if not content_chunk:
            return []

        results = []

        if not self.streaming_active:
            # We are currently buffering potentially tagged content
            self.buffer += content_chunk

            # Fail-fast: if buffer doesn't look like a tag anymore, flush it
            if not self.buffer.startswith(HA_LOCAL_TAG_PREFIX):
                results.append(self.buffer)
                self.buffer = ""
                self.streaming_active = True
            return results

        # Handle suspicious buffer from previous chunk (e.g. ended with '[')
        if self.suspicious_buffer:
            if content_chunk.startswith(HA_LOCAL_TAG_PREFIX[1:]):
                # Found the full prefix across chunks (e.g. "[" + "[")
                self.streaming_active = False
                self.buffer = self.suspicious_buffer + content_chunk
                self.suspicious_buffer = ""
                return results  # Nothing to yield yet
            else:
                # False alarm
                results.append(self.suspicious_buffer)
                self.suspicious_buffer = ""

        # Main tag detection
        if HA_LOCAL_TAG_PREFIX in content_chunk:
            idx = content_chunk.index(HA_LOCAL_TAG_PREFIX)
            if idx > 0:
                results.append(content_chunk[:idx])
            self.streaming_active = False
            self.buffer = content_chunk[idx:]
            return results

        # Check for partial tag start at end of chunk (e.g. ends with "[")
        if content_chunk.endswith(HA_LOCAL_TAG_PREFIX[0]):
            if len(content_chunk) > 1:
                results.append(content_chunk[:-1])
            self.suspicious_buffer = HA_LOCAL_TAG_PREFIX[0]
            return results

        # Normal content
        results.append(content_chunk)
        return results

    def flush(self) -> str:
        """Flush any remaining buffered content."""
        flushed = ""
        if self.suspicious_buffer:
            flushed += self.suspicious_buffer
            self.suspicious_buffer = ""
        return flushed

    @property
    def command_buffer(self) -> str:
        """Return the current accumulated command buffer."""
        return self.buffer


def _parse_regex_fields(text: str, field_patterns: dict[str, Any]) -> dict | None:
    """Extract specific fields using regex patterns (fallback)."""
    result = {}
    for field, pattern in field_patterns.items():
        match = pattern.search(text)
        if match:
            value = match.group(1)
            # Try to parse as JSON if it looks like a structure
            if value.startswith(("{", "[")):
                parsed = parse_strict_json(value)
                result[field] = parsed if parsed is not None else value
            else:
                result[field] = value.strip()
    return result if result else None


def parse_ha_local_payload(payload_json: str) -> dict | None:
    """Parse `[[HA_LOCAL: {...}]]` JSON payloads with robust fallback strategies.
    Handles single command formats and multi-command/sequential structures.
    Args: payload_json: Raw JSON string extracted from the tag.
    Returns: Parsed dict with "text" or "commands" key, or None if parsing fails."""
    # Fast path: try strict JSON first
    result = parse_strict_json(payload_json)
    if result and ("text" in result or "commands" in result):
        return result

    # Fallback: regex extraction for malformed JSON
    # Multi-command format detection
    if '"commands"' in payload_json:
        fields = _parse_regex_fields(
            payload_json,
            {
                "commands": HA_LOCAL_COMMANDS_PATTERN,
                "sequential": HA_LOCAL_SEQUENTIAL_PATTERN,
            },
        )
        if fields and "commands" in fields:
            res = {"commands": fields["commands"]}
            if "sequential" in fields:
                res["sequential"] = fields["sequential"] == "true"
            return res

    # Single command format
    fields = _parse_regex_fields(payload_json, {"text": HA_LOCAL_TEXT_PATTERN})
    if fields and "text" in fields:
        return fields

    return None
