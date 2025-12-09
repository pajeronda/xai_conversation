"""Conversation utilities for xAI conversation."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from homeassistant.components import conversation as ha_conversation
from homeassistant.helpers import llm as ha_llm
from homeassistant.helpers import entity_registry as er

from ..const import (
    DOMAIN,
    CONF_SEND_USER_NAME,
    LOGGER,
    RECOMMENDED_HISTORY_LIMIT_TURNS,
    RECOMMENDED_SEND_USER_NAME,
)
from .xai_gateway import XAIGateway

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant as HA_HomeAssistant
    from ..entity import XAIBaseLLMEntity


def get_exposed_entities_with_aliases(
    hass, assistant_id: str = "conversation"
) -> list[dict]:
    """Get exposed entities enriched with aliases from entity registry.

    Args:
        hass: Home Assistant instance
        assistant_id: Assistant ID to filter exposed entities (default: "conversation")

    Returns:
        List of entity data dictionaries including 'aliases' field if present.
    """
    # Get base exposed entities
    exposed_entities_result = ha_llm._get_exposed_entities(
        hass, assistant_id, include_state=False
    )

    if not exposed_entities_result or "entities" not in exposed_entities_result:
        return []

    ent_reg = er.async_get(hass)
    enriched_entities = []

    for entity_id, entity_data in exposed_entities_result["entities"].items():
        # Create a copy to avoid mutating the cached result from HA
        entity_data_copy = entity_data.copy()
        entity_data_copy["entity_id"] = (
            entity_id  # ADDED: Ensure entity_id is explicitly present
        )

        # Enrich with aliases from registry
        entry = ent_reg.async_get(entity_id)
        if entry and entry.aliases:
            entity_data_copy["aliases"] = list(entry.aliases)

        enriched_entities.append(entity_data_copy)

    return enriched_entities


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


class MinimalChatLog:
    """Minimal chat_log object for fallback mode.

    Simulates ChatLog with only the methods needed by async_process_with_loop:
    - .content: list of messages
    - async_add_assistant_content(): to receive the response from tools mode
    - async_add_delta_content_stream(): for streaming support in fallback mode
    """

    def __init__(self, content):
        self.content = content
        self._accumulated_content = ""

    async def async_add_assistant_content(self, assistant_content):
        """Add assistant content to the chat log (simulates HA ChatLog behavior)."""
        self.content.append(assistant_content)
        # HA's async_add_assistant_content is an async generator that yields once
        yield None

    async def async_add_delta_content_stream(self, agent_id: str, stream):
        """
        Consume a delta stream and add the final content as a single
        AssistantContent message. This is NOT a generator, it fully consumes
        the stream. This is to avoid race conditions when waiting for the result.
        """
        self._accumulated_content = ""

        async for delta in stream:
            # Consume the stream but do nothing with the deltas,
            # as the caller in the fallback doesn't need real-time updates.
            if "content" in delta:
                self._accumulated_content += delta["content"]

        # After stream ends, add the single accumulated content to the chat log
        if self._accumulated_content:
            self.content.append(
                ha_conversation.AssistantContent(
                    agent_id=agent_id,
                    content=self._accumulated_content,
                )
            )

        # This function is called in an `async for` loop, so it must be a generator.
        # We yield once at the end after the stream is fully processed.
        yield
