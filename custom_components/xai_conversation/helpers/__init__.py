"""Helper utilities for xAI conversation - modular package."""

from __future__ import annotations

# Response parsing (modular system + specialized parsers)
from .response import (
    # Modular parsing system (base strategies)
    parse_strict_json,
    parse_json_fenced,
    parse_balanced_braces,
    parse_with_strategies,
    # Citation management
    format_citations,
    # Response metadata management
    extract_response_metadata,
    capture_zdr_content,
    restore_zdr_content,
)

# Pipeline mode stream parsing (HA_LOCAL tag detection + payload parsing)
from .response import StreamParser, parse_ha_local_payload

# Log time services (needed by gateway)
from .log_time_services import LogTimeServices, timed_stream_generator

# Conversation Processors
from .processor import BaseConversationProcessor

# Gateway functions (Chat options, Logging)
from .xaigateway_functions import (
    ChatOptions,
    prepare_sdk_payload,
    async_log_completion,
    resolve_chat_parameters,
    resolve_memory_context,
    translate_messages_to_sdk,
    assemble_chat_args,
    log_api_request,
    get_tool_call_type,
    translate_system_message,
    translate_assistant_message,
    translate_tool_message,
    translate_image_message,
    build_session_context_info,
)

# Model Manager (uses XAIGateway)
from .model_manager import XAIModelManager

# Tool conversion: HA → xAI (outbound)
from .tools_ha_to_xai import (
    format_tools_for_xai,
    convert_ha_to_xai_tool_call,
    convert_ha_schema_to_xai,
)


# Tool conversion: xAI → HA (inbound)
from .tools_xai_to_ha import convert_xai_to_ha_tool

# Custom tools
from .tools_custom import CUSTOM_TOOLS

# Tool Orchestrator
from .tool_orchestrator import (
    ToolOrchestrator,
    ToolExecutionResult,
    ToolSessionConfig,
    resolve_tool_session_config,
)

# Extended tools
from .tools_extended import ExtendedToolsRegistry, ExtendedToolError

# Prompt management
from .prompt_manager import PromptManager

# Memory management
from .memory_manager import MemoryManager


# Token statistics manager (V2 - simplified)
from .sensors import (
    TokenStats,
    get_pricing_conversion_factor,
    get_tokens_per_million,
)

# Attachments management
from .attachments import async_prepare_attachments, async_parse_image_input

# Shared utilities (public API only)
from .utils import (
    parse_id_list,
    get_xai_entity,
    extract_user_id,
    extract_device_id,
    extract_scope_and_identifier,
    format_user_message_with_metadata,
    should_show_citations,
    enrich_last_user_message,
    prepare_history_payload,
    hash_text,
)


__all__ = [
    # Utilities
    "get_xai_entity",
    "hash_text",
    # Model Manager
    "XAIModelManager",
    # Response - Modular parsing system
    "parse_strict_json",
    "parse_json_fenced",
    "parse_balanced_braces",
    "parse_with_strategies",
    # Response - Pre-configured parsers
    "parse_ha_local_payload",
    # Response - Metadata management
    "extract_response_metadata",
    # Citation management
    "format_citations",
    "should_show_citations",
    "capture_zdr_content",
    "restore_zdr_content",
    # Stream Parser
    "StreamParser",
    # Tools xAI
    "format_tools_for_xai",
    "convert_ha_to_xai_tool_call",
    "convert_ha_schema_to_xai",
    # Tools conversion
    "convert_xai_to_ha_tool",
    "CUSTOM_TOOLS",
    # Tool Orchestrator
    "ToolOrchestrator",
    "ToolExecutionResult",
    "ToolSessionConfig",
    "resolve_tool_session_config",
    # Conversation
    "extract_user_id",
    "extract_device_id",
    "extract_scope_and_identifier",
    "parse_id_list",
    "format_user_message_with_metadata",
    "build_session_context_info",
    # Classes
    "PromptManager",
    "MemoryManager",
    "TokenStats",
    "get_pricing_conversion_factor",
    "get_tokens_per_million",
    "LogTimeServices",
    "timed_stream_generator",
    "BaseConversationProcessor",
    "ChatOptions",
    "prepare_sdk_payload",
    "async_log_completion",
    "resolve_chat_parameters",
    "resolve_memory_context",
    "translate_messages_to_sdk",
    "assemble_chat_args",
    "log_api_request",
    "get_tool_call_type",
    "translate_system_message",
    "translate_assistant_message",
    "translate_tool_message",
    "translate_image_message",
    "async_prepare_attachments",
    "async_parse_image_input",
    "enrich_last_user_message",
    "prepare_history_payload",
    "ExtendedToolsRegistry",
    "ExtendedToolError",
]
