"""Conversation utilities for xAI conversation."""

from __future__ import annotations

from datetime import datetime

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
    return getattr(user_input, "device_id", None) or None


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


def parse_id_list(value: str | list) -> list[str]:
    """Parse comma-separated string or list into list of IDs.

    Args:
        value: String (comma-separated) or list of IDs

    Returns:
        List of trimmed, non-empty ID strings
    """
    if isinstance(value, str):
        return [item for item in (s.strip() for s in value.split(",")) if item]
    return list(value) if value else []


async def format_user_message_with_metadata(
    message: str, user_input, hass, send_user_name: bool, include_timestamp: bool = True
) -> str:
    """Format user message with optional timestamp and user/device name prefix.

    This is a shared helper to avoid code duplication across pipeline and tools modes.

    Args:
        message: The raw user message
        user_input: ConversationInput object
        hass: Home Assistant instance
        send_user_name: Whether to include user/device name
        include_timestamp: Whether to include timestamp

    Returns:
        Formatted message with metadata
    """
    timestamp = (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S") if include_timestamp else None
    )

    if send_user_name:
        name, source_type = await get_user_or_device_name(user_input, hass)
        if name and source_type == "user":
            prefix = f"[User: {name}]"
        elif name and source_type == "device":
            prefix = f"[Device: {name}]"
        else:
            prefix = None

        # Build final prefix
        if prefix and timestamp:
            full_prefix = f"{prefix} [Time: {timestamp}] "
        elif prefix:
            full_prefix = f"{prefix} "
        elif timestamp:
            full_prefix = f"[Time: {timestamp}] "
        else:
            full_prefix = ""

        return f"{full_prefix}{message}"
    else:
        # Default behavior: only timestamp in parentheses
        if timestamp:
            return f"({timestamp}) {message}"
        return message


def build_session_context_info(hass) -> str:
    """Build session context information (timestamp, timezone, country).

    This is added to system prompts to provide temporal and geographic context
    without breaking cache on subsequent messages.

    Args:
        hass: Home Assistant instance

    Returns:
        Formatted session context string
    """
    session_start = datetime.now()
    return (
        f"\n\nSession Context:"
        f"\n- Started at: {session_start.strftime('%Y-%m-%d %H:%M:%S %Z')}"
        f"\n- Timezone: {hass.config.time_zone}"
        f"\n- Country: {hass.config.country}"
    )
