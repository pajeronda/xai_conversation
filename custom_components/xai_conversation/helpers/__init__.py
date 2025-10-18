"""Helper utilities for xAI conversation - modular package."""
from __future__ import annotations

# Configuration validation
from .config import validate_xai_configuration

# Response parsing (legacy + modular system)
from .response import (
    # Legacy parser
    parse_grok_response,
    # Modular parsing system
    parse_strict_json,
    parse_json_fenced,
    parse_balanced_braces,
    parse_regex_fields,
    parse_with_strategies,
    # Pre-configured parser for HA_LOCAL payloads
    parse_ha_local_payload,
)

# xAI tool formatting and schema conversion
from .tools_xai import (
    format_tools_for_xai,
)

# Tool conversion HA ↔ xAI
from .tools_convert import convert_xai_to_ha_tool

# Conversation utilities
from .conversation import (
    get_last_user_message,
    extract_user_id,
    extract_device_id,
    is_device_request,
    prompt_hash,
)

# Prompt management
from .prompt_manager import PromptManager

# Memory management
from .memory import ConversationMemory

# Chat history management
from .chat_history import ChatHistoryService

__all__ = [
    # Config
    "validate_xai_configuration",
    # Response - Legacy
    "parse_grok_response",
    # Response - Modular parsing system
    "parse_strict_json",
    "parse_json_fenced",
    "parse_balanced_braces",
    "parse_regex_fields",
    "parse_with_strategies",
    "parse_ha_local_payload",
    # Tools xAI
    "format_tools_for_xai",
    # Tools conversion
    "convert_xai_to_ha_tool",
    # Conversation
    "get_last_user_message",
    "extract_user_id",
    "extract_device_id",
    "is_device_request",
    "prompt_hash",
    # Classes
    "PromptManager",
    "ConversationMemory",
    "ChatHistoryService",
]
