"""Shared utilities for xAI conversation."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import datetime
from typing import TYPE_CHECKING, Any

from homeassistant.components import conversation as ha_conversation
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as ha_entity_registry

from ..const import (
    CHAT_MODE_CHATONLY,
    CHAT_MODE_PIPELINE,
    CHAT_MODE_TOOLS,
    CONF_SHOW_CITATIONS,
    DOMAIN,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant


def hash_text(text: str) -> str:
    """Generate a short SHA256 hash of the text (8 chars)."""
    return hashlib.sha256(text.encode()).hexdigest()[:8]


async def format_user_message_with_metadata(
    message: str,
    user_input,
    hass: HomeAssistant,
    send_user_name: bool,
    include_timestamp: bool = True,
    mode: str = "conversation",
) -> str:
    """Format user message with optional timestamp and user/device name prefix.

    Args:
        message: The raw user message
        user_input: ConversationInput or ServiceCall object
        hass: Home Assistant instance
        send_user_name: Whether to include user/device name
        include_timestamp: Whether to include timestamp
        mode: Operational mode (e.g., "conversation", "ai_task")

    Returns:
        Formatted message with metadata
    """
    # Metadata (prefix/timestamp) for conversation modes (pipeline, tools, chatonly)
    # Excluded only for ai_task and vision which don't need user context
    conversation_modes = {CHAT_MODE_PIPELINE, CHAT_MODE_TOOLS, CHAT_MODE_CHATONLY}
    if mode not in conversation_modes:
        return message

    # Avoid double-formatting if message already contains a recognized prefix pattern
    # [User: ...] or [Device: ...] or (YYYY-MM-DD HH:MM)
    if (
        message.startswith("[User: ")
        or message.startswith("[Device: ")
        or message.startswith("(")
    ):
        return message

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M") if include_timestamp else None
    prefix = None

    # 1. Try User/Person Name (only if enabled via config)
    if send_user_name:
        uid = extract_user_id(user_input)
        if uid:
            try:
                # First try to get name from person entity (preferred)
                person_name = None
                for state in hass.states.async_all("person"):
                    if state.attributes.get("user_id") == uid:
                        person_name = state.name
                        break

                if person_name:
                    prefix = f"[User: {person_name}]"
                else:
                    # Fallback to auth user name
                    user = await hass.auth.async_get_user(uid)
                    if user and user.name:
                        prefix = f"[User: {user.name}]"
            except Exception:
                pass

    # 2. Try Device Name (always enabled by default for context)
    if not prefix:
        did = extract_device_id(user_input)
        if did:
            try:
                device_registry = dr.async_get(hass)
                device = device_registry.async_get(did)
                if device:
                    device_name = device.name_by_user or device.name
                    if device_name:
                        prefix = f"[Device: {device_name}]"
            except Exception:
                pass

    # Build final formatted message
    if prefix and timestamp:
        return f"{prefix} [Time: {timestamp}] {message}"
    if prefix:
        return f"{prefix} {message}"
    if timestamp:
        return f"({timestamp}) {message}"

    return message


async def enrich_last_user_message(
    chat_log: Any,
    user_input: Any,
    hass: HomeAssistant,
    send_user_name: bool,
    mode: str = "conversation",
) -> None:
    """Find the latest user message in ChatLog and enrich it with metadata.

    Layer 1: Enrichment. Modifies ChatLog in-place for UI and downstream layers.
    """
    content_list = list(chat_log.content)
    last_user_idx = -1
    for i in range(len(content_list) - 1, -1, -1):
        if isinstance(content_list[i], ha_conversation.UserContent):
            last_user_idx = i
            break

    if last_user_idx != -1:
        content = content_list[last_user_idx]
        text = await format_user_message_with_metadata(
            content.content or "",
            user_input,
            hass,
            send_user_name=send_user_name,
            include_timestamp=True,
            mode=mode,
        )
        chat_log.content[last_user_idx] = replace(content, content=text)


def parse_id_list(value: str | list) -> list[str]:
    """Parse comma-separated string or list into list of IDs."""
    if isinstance(value, str):
        return [item for item in (s.strip() for s in value.split(",")) if item]
    return list(value) if value else []


def get_xai_entity(hass: HomeAssistant, domain_type: str = "conversation"):
    """Find an xAI entity of a specific type efficiently."""
    ent_reg = ha_entity_registry.async_get(hass)
    for e in ent_reg.entities.values():
        if e.platform == DOMAIN and e.domain == domain_type:
            comp = hass.data.get("entity_components", {}).get(domain_type)
            if comp:
                entity = comp.get_entity(e.entity_id)
                if entity:
                    return entity
    return None


def extract_user_id(obj: Any) -> str | None:
    # Support ChatOptions wrapper via user_input attribute
    if hasattr(obj, "user_input") and obj.user_input:
        obj = obj.user_input

    if hasattr(obj, "data") and isinstance(obj.data, dict):
        uid = obj.data.get("user_id")
        if uid:
            return uid

    if hasattr(obj, "context") and obj.context and obj.context.user_id:
        return obj.context.user_id

    return None


def extract_device_id(obj: Any) -> str | None:
    # Support ChatOptions wrapper via user_input attribute
    if hasattr(obj, "user_input") and obj.user_input:
        obj = obj.user_input

    if hasattr(obj, "data") and isinstance(obj.data, dict):
        return obj.data.get("device_id") or obj.data.get("satellite_id")

    return getattr(obj, "device_id", None) or None


def extract_scope_and_identifier(user_input) -> tuple[str, str]:
    """Extract memory scope and identifier from user_input.

    Determines whether to use user-based or device-based memory scope.

    Args:
        user_input: ConversationInput or ServiceCall object

    Returns:
        tuple: (scope, identifier) where:
            - scope: "user" or "device"
            - identifier: user_id, device_id, or "unknown"
    """
    user_id = extract_user_id(user_input)
    if user_id:
        return ("user", user_id)

    device_id = extract_device_id(user_input)
    return ("device", device_id or "unknown")


async def prepare_history_payload(
    chat_log: Any,
    params: Any,
    history_limit: int | None = None,
    last_only: bool = False,
    skip_history_system: bool = False,
) -> list[dict[str, Any]]:
    """Slice ChatLog and transform it into an SDK-neutral message payload.

    Layer 2: Logic Layer. Shared by conversational and stateless flows.
    """
    messages = []
    content_list = list(chat_log.content)

    # 1. Selection Strategy (Slicing)
    if last_only:
        # Memory/ZDR: Keep only system prompts and the latest interaction block
        filtered = []
        if not skip_history_system:
            filtered.extend(
                [
                    c
                    for c in content_list
                    if isinstance(c, ha_conversation.SystemContent)
                    and (c.content or "").strip()
                ]
            )

        # Find the last user message to include it and anything follow (tool results)
        last_user_idx = -1
        for i in range(len(content_list) - 1, -1, -1):
            if isinstance(content_list[i], ha_conversation.UserContent):
                last_user_idx = i
                break

        if last_user_idx != -1:
            filtered.extend(content_list[last_user_idx:])
        content_list = filtered

    elif history_limit:
        # Robust Turn-Aware Slicing for Local Chatlog
        # We want to keep all System messages and exactly N latest User turns
        system_messages = [
            c for c in content_list if isinstance(c, ha_conversation.SystemContent)
        ]
        non_system_messages = [
            c for c in content_list if not isinstance(c, ha_conversation.SystemContent)
        ]

        # Find indices of UserContent in the non-system list
        user_indices = [
            i
            for i, c in enumerate(non_system_messages)
            if isinstance(c, ha_conversation.UserContent)
        ]

        if len(user_indices) > history_limit:
            # Slice from the start of the N-th last user message
            start_idx = user_indices[-history_limit]
            non_system_messages = non_system_messages[start_idx:]

        content_list = system_messages + non_system_messages

    # 1b. Pipeline fallback: strip assistant messages after the last user message
    # to prevent context pollution from the pipeline's streaming response
    if getattr(params, "forced_last_message", None):
        for i in range(len(content_list) - 1, -1, -1):
            if isinstance(content_list[i], ha_conversation.UserContent):
                content_list = content_list[: i + 1]
                break

    # 2. Transformation Strategy (Neutral Format)
    actual_last_user_idx = -1
    for i in range(len(content_list) - 1, -1, -1):
        if isinstance(content_list[i], ha_conversation.UserContent):
            actual_last_user_idx = i
            break

    for idx, content in enumerate(content_list):
        msg = {}
        if isinstance(content, ha_conversation.SystemContent):
            msg = {"role": "system", "content": content.content or ""}
        elif isinstance(content, ha_conversation.UserContent):
            # Override content if forced_last_message is set (e.g. pipeline fallback)
            text_content = content.content or ""
            if idx == actual_last_user_idx and getattr(
                params, "forced_last_message", None
            ):
                text_content = params.forced_last_message

            msg = {"role": "user", "content": text_content}

            # Inject attachments only into the actual last user message of the payload
            if idx == actual_last_user_idx:
                parts = [msg["content"]]
                if hasattr(params, "mixed_content") and params.mixed_content:
                    parts.extend(params.mixed_content)
                if hasattr(params, "extra_content") and params.extra_content:
                    parts.extend(params.extra_content)
                if len(parts) > 1:
                    msg["content"] = parts

        elif isinstance(content, ha_conversation.AssistantContent):
            msg = {
                "role": "assistant",
                "content": content.content or "",
                "tool_calls": getattr(content, "tool_calls", None),
            }
        elif isinstance(content, ha_conversation.ToolResultContent):
            tool_result = None
            for attr in ("content", "tool_result", "result", "output"):
                if hasattr(content, attr):
                    tool_result = getattr(content, attr)
                    break
            msg = {
                "role": "tool",
                "content": str(tool_result),
                "tool_call_id": content.tool_call_id,
            }

        if msg:
            messages.append(msg)

    return messages


def should_show_citations(source: Any) -> bool:
    """Check if citations should be displayed based on configuration.

    Args:
        source: Entity, Gateway or ConfigEntry instance

    Returns:
        True if citations should be shown
    """
    if not source:
        return False

    # 1. Direct config dict support (cleanest for helpers)
    if isinstance(source, dict):
        return bool(source.get(CONF_SHOW_CITATIONS))

    # 2. Try entity.get_config_dict (legacy support for core classes)
    if hasattr(source, "get_config_dict"):
        return bool(source.get_config_dict().get(CONF_SHOW_CITATIONS))

    # 3. Try gateway.entry or source as entry
    entry = getattr(source, "entry", source)

    # For stateless services (ask), we check the default conversation config
    if hasattr(entry, "subentries"):
        for subentry in entry.subentries.values():
            if subentry.subentry_type == "conversation":
                return bool(subentry.data.get(CONF_SHOW_CITATIONS))
    return False
