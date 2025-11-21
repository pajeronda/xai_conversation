"""Helper utilities for xAI conversation - modular package."""
from __future__ import annotations

# Configuration validation
from .config import async_validate_api_key, validate_xai_configuration, async_get_xai_models_data

# Response parsing (modular system + specialized parsers)
from .response import (
    # Modular parsing system (base strategies)
    parse_strict_json,
    parse_json_fenced,
    parse_balanced_braces,
    parse_regex_fields,
    parse_with_strategies,
    # Pre-configured parsers
    parse_ha_local_payload,  # For [[HA_LOCAL: {...}]] payloads (pipeline mode)
    parse_grok_code_response,  # For Grok Code Fast responses
)

# Tool conversion: HA → xAI (outbound)
from .tools_ha_to_xai import (
    format_tools_for_xai,
)

# Tool conversion: xAI → HA (inbound)
from .tools_xai_to_ha import convert_xai_to_ha_tool

# Conversation utilities
from .conversation import (
    get_last_user_message,
    extract_user_id,
    extract_device_id,
    get_user_or_device_name,
    parse_id_list,
)

# Prompt management
from .prompt_manager import PromptManager

# Memory management
from .memory import ConversationMemory

# Chat history management
from .chat_history import ChatHistoryService

# xAI SDK Gateway
from .xai_gateway import XAIGateway

__all__ = [
    # Config
    "async_validate_api_key",
    "validate_xai_configuration",
    "async_get_xai_models_data",
    # Response - Modular parsing system
    "parse_strict_json",
    "parse_json_fenced",
    "parse_balanced_braces",
    "parse_regex_fields",
    "parse_with_strategies",
    # Response - Pre-configured parsers
    "parse_ha_local_payload",
    "parse_grok_code_response",
    # Tools xAI
    "format_tools_for_xai",
    # Tools conversion
    "convert_xai_to_ha_tool",
    # Conversation
    "get_last_user_message",
    "extract_user_id",
    "extract_device_id",
    "get_user_or_device_name",
    "parse_id_list",
    # Classes
    "PromptManager",
    "ConversationMemory",
    "ChatHistoryService",
    "XAIGateway",
]
