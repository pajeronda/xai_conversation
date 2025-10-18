"""Conversation utilities for xAI conversation."""
from __future__ import annotations

import hashlib

from homeassistant.components import conversation as ha_conversation


def get_last_user_message(chat_log: ha_conversation.ChatLog) -> str | None:
    """Extract the last user message from the chat log."""
    for content in reversed(chat_log.content):
        if isinstance(content, ha_conversation.UserContent):
            return content.content
    return None


def extract_user_id(user_input) -> str | None:
    """Extract user_id from the conversation input context."""
    if user_input.context and user_input.context.user_id:
        return user_input.context.user_id
    return None


def extract_device_id(user_input) -> str | None:
    """Extract device_id from the conversation input."""
    if hasattr(user_input, "device_id") and user_input.device_id:
        return user_input.device_id
    return None


def is_device_request(user_input) -> bool:
    """Determine if the request comes from a device (voice satellite) or user.

    Args:
        user_input: ConversationInput object

    Returns:
        True if request is from a device (voice satellite), False if from user (smartphone/PC/tablet)
    """
    device_id = extract_device_id(user_input)
    return device_id is not None and device_id != "unknown"


def prompt_hash(prompt: str) -> str:
    """Generate a short SHA256 hash of the prompt."""
    return hashlib.sha256(prompt.encode()).hexdigest()[:8]
