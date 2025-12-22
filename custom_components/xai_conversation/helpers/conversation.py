"""Conversation utilities for xAI conversation."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from homeassistant.components import conversation as ha_conversation
from homeassistant.helpers import llm as ha_llm

from ..const import (
    DOMAIN,
    CONF_SEND_USER_NAME,
    LOGGER,
    RECOMMENDED_HISTORY_LIMIT_TURNS,
    RECOMMENDED_SEND_USER_NAME,
)
from .utils import (
    extract_device_id,
    get_user_or_device_name,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant as HA_HomeAssistant
    from ..entity import XAIBaseLLMEntity


def get_last_user_message(chat_log: ha_conversation.ChatLog) -> str | None:
    """Extract the last user message from the chat log."""
    for content in reversed(chat_log.content):
        if isinstance(content, ha_conversation.UserContent):
            return content.content
    return None


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


def build_llm_context(hass, user_input) -> ha_llm.LLMContext:
    """Build LLMContext independently from chat_log. NOT cached - rebuilt each time."""
    # Use "conversation" domain to match exposed entities
    # This matches how Home Assistant's conversation system checks entity exposure
    assistant_id = "conversation"

    return ha_llm.LLMContext(
        platform=DOMAIN,
        context=user_input.context,
        language=user_input.language,
        assistant=assistant_id,
        device_id=extract_device_id(user_input),
    )


async def add_manual_history_to_chat(
    hass: HA_HomeAssistant,
    entity: XAIBaseLLMEntity,
    chat: Any,
    chat_log: ha_conversation.ChatLog,
    user_input: ha_conversation.ConversationInput,
    history_limit_turns: int = RECOMMENDED_HISTORY_LIMIT_TURNS,
) -> None:
    """Add manual conversation history to the chat when server-side memory is disabled.

    Optimization: Only the last user message gets full metadata (user/device lookup, timestamp).
    Historical messages are sent as plain text to avoid expensive async lookups on every API call.
    """
    from ..xai_gateway import XAIGateway

    LOGGER.debug("Adding manual history to chat (server-side memory disabled).")

    # Use the passed limit or calculate based on config if needed
    limit = history_limit_turns * 2

    send_user_name = entity._get_option(CONF_SEND_USER_NAME, RECOMMENDED_SEND_USER_NAME)

    # Get last N turns from chat_log
    content_list = list(chat_log.content)

    # Filter out system messages or other non-chat content if necessary
    # For now, we take everything that is User or Assistant content

    history_content = (
        content_list[-limit:] if len(content_list) > limit else content_list
    )

    if not history_content:
        LOGGER.warning("Tools mode: no messages in chat_log to add history")
        return

    LOGGER.debug(
        "Tools mode: sending manual history: %d messages (last %d turns)",
        len(history_content),
        history_limit_turns,
    )

    # Optimization: identify the last user message index to apply full metadata only there
    last_user_msg_index = -1
    for i in range(len(history_content) - 1, -1, -1):
        if isinstance(history_content[i], ha_conversation.UserContent):
            last_user_msg_index = i
            break

    for i, content in enumerate(history_content):
        if isinstance(content, ha_conversation.UserContent):
            # Only the LAST user message gets full metadata (user/device lookup, timestamp)
            # Historical messages are plain text to avoid expensive async lookups
            is_last_user_msg = i == last_user_msg_index

            formatted_msg = await format_user_message_with_metadata(
                content.content or "",
                user_input,
                hass,
                send_user_name
                and is_last_user_msg,  # Only lookup user/device for last message
                include_timestamp=is_last_user_msg,  # Only timestamp on last message
            )
            chat.append(XAIGateway.user_msg(formatted_msg))
        elif isinstance(content, ha_conversation.AssistantContent):
            tool_calls = getattr(content, "tool_calls", None)
            if tool_calls:
                chat.append(
                    XAIGateway.assistant_msg(
                        content.content or "", tool_calls=tool_calls
                    )
                )
            else:
                chat.append(XAIGateway.assistant_msg(content.content or ""))
