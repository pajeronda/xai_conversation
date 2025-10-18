"""Helper utilities for xAI conversation - proxy module.

This module serves as a backward-compatible proxy to the modular helpers package.
All functionality has been reorganized into the helpers/ directory for better maintainability.

Directory structure:
    helpers/
    ├── __init__.py          - Re-exports all public functions
    ├── config.py            - Configuration validation
    ├── response.py          - Response parsing
    ├── tools_xai.py         - xAI tool formatting and schema conversion
    ├── tools_convert.py     - Tool conversion between HA and xAI formats
    ├── conversation.py      - Conversation utilities (user/device ID extraction)
    ├── prompt_manager.py    - PromptManager class for prompt composition
    └── memory.py            - ConversationMemory class for persistent memory
"""
from __future__ import annotations

# Re-export everything from the helpers package for backward compatibility
from .helpers import (
    # Configuration
    validate_xai_configuration,
    # Response parsing
    parse_grok_response,
    # xAI tool formatting
    format_tools_for_xai,
    # Tool conversion
    convert_xai_to_ha_tool,
    # Conversation utilities
    get_last_user_message,
    extract_user_id,
    extract_device_id,
    is_device_request,
    prompt_hash,
    # Classes
    PromptManager,
    ConversationMemory,
)

__all__ = [
    # Config
    "validate_xai_configuration",
    # Response
    "parse_grok_response",
    # Tools
    "format_tools_for_xai",
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
]
