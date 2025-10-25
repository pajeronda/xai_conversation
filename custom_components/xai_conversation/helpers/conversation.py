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


async def get_user_or_device_name(user_input, hass) -> tuple[str | None, str]:
    """Get user name or device name from the conversation input.

    Args:
        user_input: ConversationInput object
        hass: Home Assistant instance

    Returns:
        tuple: (name, source_type) where:
            - name: str or None - The user or device name
            - source_type: str - "user", "device", or "unknown"
    """
    # Try to get user first
    user_id = extract_user_id(user_input)
    if user_id:
        try:
            user = await hass.auth.async_get_user(user_id)
            if user and user.name:
                return (user.name, "user")
        except Exception:
            # Silent failure, try device fallback
            pass

    # Fallback: try to get device info
    device_id = extract_device_id(user_input)
    if device_id:
        try:
            from homeassistant.helpers import device_registry as dr
            device_registry = dr.async_get(hass)
            device = device_registry.async_get(device_id)
            if device:
                # Prefer device name_by_user, fallback to name
                device_name = device.name_by_user or device.name
                if device_name:
                    return (device_name, "device")
        except Exception:
            # Silent failure
            pass

    return (None, "unknown")


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
