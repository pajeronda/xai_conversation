"""Helper utilities for xAI conversation - modular package."""

from __future__ import annotations

# Response parsing (modular system + specialized parsers)
from .response import (
    # Modular parsing system (base strategies)
    parse_strict_json,
    parse_json_fenced,
    parse_balanced_braces,
    parse_with_strategies,
    # Pre-configured parsers
    parse_ha_local_payload,  # For [[HA_LOCAL: {...}]] payloads (pipeline mode)
    parse_grok_code_response,  # For Grok Code Fast responses
    # Response metadata management
    save_response_metadata,
    # Citation management
    format_citations,
)

# Log time services (needed by gateway)
from .log_time_services import LogTimeServices, timed_stream_generator

# Model Manager (uses XAIGateway)
from .model_manager import XAIModelManager

# Tool conversion: HA → xAI (outbound)
from .tools_ha_to_xai import format_tools_for_xai

# Tool conversion: xAI → HA (inbound)
from .tools_xai_to_ha import convert_xai_to_ha_tool

# Custom tools
from .tools_custom import CUSTOM_TOOLS

# Tool Orchestrator
from .tool_orchestrator import (
    ToolOrchestrator,
    ToolExecutionResult,
)

# Extended tools
from .tools_extended import ExtendedToolsRegistry, ExtendedToolError

# Conversation utilities
from .conversation import (
    get_last_user_message,
    format_user_message_with_metadata,
    add_manual_history_to_chat,
)

# Prompt management
from .prompt_manager import PromptManager, build_system_prompt

# Memory management
from .memory_manager import MemoryManager

# Chat history management
from .chat_history import ChatHistoryService

# Token statistics manager (V2 - simplified)
from .sensors import TokenStats


# Shared utilities (public API only)
from .utils import (
    parse_id_list,
    get_xai_entity,
    extract_user_id,
    extract_device_id,
    extract_scope_and_identifier,
)

# Constants re-export for convenience
from ..const import RECOMMENDED_HISTORY_LIMIT_TURNS


__all__ = [
    # Utilities
    "get_xai_entity",
    # Model Manager
    "XAIModelManager",
    # Response - Modular parsing system
    "parse_strict_json",
    "parse_json_fenced",
    "parse_balanced_braces",
    "parse_with_strategies",
    # Response - Pre-configured parsers
    "parse_ha_local_payload",
    "parse_grok_code_response",
    # Response - Metadata management
    "save_response_metadata",
    # Citation management
    "format_citations",
    # Tools xAI
    "format_tools_for_xai",
    # Tools conversion
    "convert_xai_to_ha_tool",
    "CUSTOM_TOOLS",
    # Tool Orchestrator
    "ToolOrchestrator",
    "ToolExecutionResult",
    # Conversation
    "get_last_user_message",
    "extract_user_id",
    "extract_device_id",
    "extract_scope_and_identifier",
    "parse_id_list",
    "format_user_message_with_metadata",
    "add_manual_history_to_chat",
    # Classes
    "PromptManager",
    "build_system_prompt",
    "MemoryManager",
    "ChatHistoryService",
    "TokenStats",
    "LogTimeServices",
    "timed_stream_generator",
    "ExtendedToolsRegistry",
    "ExtendedToolError",
    "RECOMMENDED_HISTORY_LIMIT_TURNS",
]
