"""Helper utilities for xAI conversation - modular package."""

from __future__ import annotations

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
    # Response metadata management
    save_response_metadata,
)

# Log time services (needed by gateway)
from .log_time_services import LogTimeServices, timed_stream_generator

# xAI SDK Gateway (must be before model_manager to avoid circular import)
from .xai_gateway import (
    XAIGateway,
)

# Model Manager (uses XAIGateway)
from .model_manager import XAIModelManager

# Tool conversion: HA → xAI (outbound)
from .tools_ha_to_xai import (
    format_tools_for_xai,
    filter_tools_by_exposed_domains,
)

# Tool conversion: xAI → HA (inbound)
from .tools_xai_to_ha import convert_xai_to_ha_tool

# Custom tools
from .custom_tools import CUSTOM_TOOLS

# Conversation utilities
from .conversation import (
    get_last_user_message,
    extract_user_id,
    extract_device_id,
    get_user_or_device_name,
    parse_id_list,
    format_user_message_with_metadata,
    build_session_context_info,
    get_exposed_entities_with_aliases,
    build_llm_context,
    add_manual_history_to_chat,
    MinimalChatLog,
)

# Prompt management
from .prompt_manager import PromptManager

# Memory management
from .memory import ConversationMemory

# Chat history management
from .chat_history import ChatHistoryService

# Token statistics manager (V2 - simplified)
from .sensors import TokenStats

# Integration setup helpers
from .integration_setup import (
    migrate_subentry_types,
    ensure_memory_params_in_entry_data,
    add_subentries_if_needed,
    async_migrate_entry,
)

__all__ = [
    # Model Manager
    "XAIModelManager",
    # Response - Modular parsing system
    "parse_strict_json",
    "parse_json_fenced",
    "parse_balanced_braces",
    "parse_regex_fields",
    "parse_with_strategies",
    # Response - Pre-configured parsers
    "parse_ha_local_payload",
    "parse_grok_code_response",
    # Response - Metadata management
    "save_response_metadata",
    # Tools xAI
    "format_tools_for_xai",
    "filter_tools_by_exposed_domains",
    # Tools conversion
    "convert_xai_to_ha_tool",
    "CUSTOM_TOOLS",
    # Conversation
    "get_last_user_message",
    "extract_user_id",
    "extract_device_id",
    "get_user_or_device_name",
    "parse_id_list",
    "format_user_message_with_metadata",
    "build_session_context_info",
    "get_exposed_entities_with_aliases",
    "build_llm_context",
    "add_manual_history_to_chat",
    "MinimalChatLog",
    # Classes
    "PromptManager",
    "ConversationMemory",
    "ChatHistoryService",
    "TokenStats",
    "XAIGateway",
    "LogTimeServices",
    "timed_stream_generator",
    # Integration setup
    "migrate_subentry_types",
    "ensure_memory_params_in_entry_data",
    "add_subentries_if_needed",
    "async_migrate_entry",
]
