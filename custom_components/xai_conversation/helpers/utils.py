"""Shared utilities for xAI conversation."""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import TYPE_CHECKING, Any

from homeassistant.helpers import entity_registry as ha_entity_registry
from homeassistant.helpers import device_registry as dr

from ..const import DOMAIN

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant


def hash_text(text: str) -> str:
    """Generate a short SHA256 hash of the text."""
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def build_session_context_info(hass: HomeAssistant) -> str:
    """Build session context information (timestamp, timezone, country).

    This is added to system prompts to provide temporal and geographic context
    without breaking cache on subsequent messages.

    Args:
        hass: Home Assistant instance

    Returns:
        Formatted session context string
    """
    session_start = datetime.now()
    return f"\nContext: {session_start.strftime('%Y-%m-%d %H:%M:%S')} {hass.config.time_zone} ({hass.config.country})"


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
    """Extract user_id from ServiceCall or ConversationInput generically.

    Priority:
    1. obj.data['user_id'] (ServiceCall)
    2. obj.context.user_id (ServiceCall or ConversationInput)
    """
    if hasattr(obj, "data") and isinstance(obj.data, dict):
        uid = obj.data.get("user_id")
        if uid:
            return uid

    if hasattr(obj, "context") and obj.context and obj.context.user_id:
        return obj.context.user_id

    return None


def extract_device_id(obj: Any) -> str | None:
    """Extract device_id from ServiceCall or ConversationInput generically.

    Priority:
    1. obj.data['device_id'] or ['satellite_id'] (ServiceCall)
    2. obj.device_id (ConversationInput)
    """
    if hasattr(obj, "data") and isinstance(obj.data, dict):
        return obj.data.get("device_id") or obj.data.get("satellite_id")

    return getattr(obj, "device_id", None) or None


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
