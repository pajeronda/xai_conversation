"""Integration setup and migration helpers for xAI Conversation.

This module contains all setup-related logic that doesn't belong in __init__.py,
including subentry creation and config entry migrations.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant
    from homeassistant.config_entries import ConfigSubentry
    from .. import XAIConfigEntry

from ..const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_MEMORY_CLEANUP_INTERVAL_HOURS,
    CONF_MEMORY_DEVICE_MAX_TURNS,
    CONF_MEMORY_DEVICE_TTL_HOURS,
    CONF_MEMORY_USER_MAX_TURNS,
    CONF_MEMORY_USER_TTL_HOURS,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DEFAULT_AI_TASK_NAME,
    DEFAULT_CONVERSATION_NAME,
    DEFAULT_GROK_CODE_FAST_NAME,
    DEFAULT_SENSORS_NAME,
    DOMAIN,
    LOGGER,
    RECOMMENDED_AI_TASK_OPTIONS,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_GROK_CODE_FAST_OPTIONS,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_MEMORY_CLEANUP_INTERVAL_HOURS,
    RECOMMENDED_MEMORY_DEVICE_MAX_TURNS,
    RECOMMENDED_MEMORY_DEVICE_TTL_HOURS,
    RECOMMENDED_MEMORY_USER_MAX_TURNS,
    RECOMMENDED_MEMORY_USER_TTL_HOURS,
    RECOMMENDED_PIPELINE_OPTIONS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
    SUBENTRY_TYPE_AI_TASK,
    SUBENTRY_TYPE_CODE_TASK,
    SUBENTRY_TYPE_CONVERSATION,
    SUBENTRY_TYPE_SENSORS,
)


def migrate_subentry_types(hass: HomeAssistant, entry: XAIConfigEntry) -> None:
    """Migrate old subentry_type values to new format (one-time migration).

    Old format:
    - ai_task_data → ai_task
    - code_task → code_fast

    This migration runs once when old values are detected and updates the
    config entry storage automatically.

    Args:
        hass: Home Assistant instance
        entry: Config entry to migrate
    """
    from homeassistant.config_entries import ConfigSubentry

    # Check if migration is needed
    needs_migration = False
    for subentry in entry.subentries.values():
        if subentry.subentry_type in ("ai_task_data", "code_task"):
            needs_migration = True
            break

    # Perform migration if needed
    if needs_migration:
        LOGGER.info("Migrating old subentry_type values to new format")

        # Remove old subentries and add new ones with updated type
        for subentry_id, subentry in list(entry.subentries.items()):
            new_type = None
            if subentry.subentry_type == "ai_task_data":
                new_type = "ai_task"
                LOGGER.debug(
                    "Migrating subentry %s: ai_task_data → ai_task", subentry_id
                )
            elif subentry.subentry_type == "code_task":
                new_type = "code_fast"
                LOGGER.debug(
                    "Migrating subentry %s: code_task → code_fast", subentry_id
                )

            if new_type:
                # Remove old subentry
                hass.config_entries.async_remove_subentry(entry, subentry_id)

                # Add new subentry with updated type (preserving all original fields)
                hass.config_entries.async_add_subentry(
                    entry,
                    ConfigSubentry(
                        subentry_id=subentry_id,
                        subentry_type=new_type,
                        title=subentry.title,
                        data=dict(subentry.data),
                        unique_id=subentry.unique_id,
                    ),
                )

        LOGGER.info("Subentry migration completed successfully")


def ensure_memory_params_in_entry_data(
    hass: HomeAssistant, entry: XAIConfigEntry
) -> None:
    """Ensure memory parameters are in entry.data (migration for existing entries).

    Args:
        hass: Home Assistant instance
        entry: Config entry to update
    """
    data = dict(entry.data)
    updated = False

    # Check if memory parameters are missing
    if CONF_MEMORY_USER_TTL_HOURS not in data:
        data[CONF_MEMORY_USER_TTL_HOURS] = RECOMMENDED_MEMORY_USER_TTL_HOURS
        updated = True
    if CONF_MEMORY_USER_MAX_TURNS not in data:
        data[CONF_MEMORY_USER_MAX_TURNS] = RECOMMENDED_MEMORY_USER_MAX_TURNS
        updated = True
    if CONF_MEMORY_DEVICE_TTL_HOURS not in data:
        data[CONF_MEMORY_DEVICE_TTL_HOURS] = RECOMMENDED_MEMORY_DEVICE_TTL_HOURS
        updated = True
    if CONF_MEMORY_DEVICE_MAX_TURNS not in data:
        data[CONF_MEMORY_DEVICE_MAX_TURNS] = RECOMMENDED_MEMORY_DEVICE_MAX_TURNS
        updated = True
    if CONF_MEMORY_CLEANUP_INTERVAL_HOURS not in data:
        data[CONF_MEMORY_CLEANUP_INTERVAL_HOURS] = (
            RECOMMENDED_MEMORY_CLEANUP_INTERVAL_HOURS
        )
        updated = True

    if updated:
        hass.config_entries.async_update_entry(entry, data=data)
        LOGGER.info("Added memory parameters to integration configuration")


def add_subentries_if_needed(hass: HomeAssistant, entry: XAIConfigEntry) -> None:
    """Add subentries for conversation, AI Task, and Grok Code Fast if they don't exist.

    Args:
        hass: Home Assistant instance
        entry: Config entry to add subentries to
    """
    from homeassistant.config_entries import ConfigSubentry

    subentries = {se.subentry_type for se in entry.subentries.values()}

    if SUBENTRY_TYPE_CONVERSATION not in subentries:
        options = MappingProxyType(
            {
                "name": DEFAULT_CONVERSATION_NAME,
                CONF_CHAT_MODEL: RECOMMENDED_CHAT_MODEL,
                CONF_MAX_TOKENS: RECOMMENDED_MAX_TOKENS,
                CONF_TEMPERATURE: RECOMMENDED_TEMPERATURE,
                CONF_TOP_P: RECOMMENDED_TOP_P,
                **RECOMMENDED_PIPELINE_OPTIONS,
            }
        )
        hass.config_entries.async_add_subentry(
            entry,
            ConfigSubentry(
                data=options,
                subentry_type=SUBENTRY_TYPE_CONVERSATION,
                title=DEFAULT_CONVERSATION_NAME,
                unique_id=f"{DOMAIN}:conversation",
            ),
        )

    if SUBENTRY_TYPE_AI_TASK not in subentries:
        hass.config_entries.async_add_subentry(
            entry,
            ConfigSubentry(
                data=MappingProxyType(
                    {"name": DEFAULT_AI_TASK_NAME, **RECOMMENDED_AI_TASK_OPTIONS}
                ),
                subentry_type=SUBENTRY_TYPE_AI_TASK,
                title=DEFAULT_AI_TASK_NAME,
                unique_id=f"{DOMAIN}:ai_task",
            ),
        )

    if SUBENTRY_TYPE_CODE_TASK not in subentries:
        hass.config_entries.async_add_subentry(
            entry,
            ConfigSubentry(
                data=MappingProxyType(
                    {
                        "name": DEFAULT_GROK_CODE_FAST_NAME,
                        **RECOMMENDED_GROK_CODE_FAST_OPTIONS,
                    }
                ),
                subentry_type=SUBENTRY_TYPE_CODE_TASK,
                title=DEFAULT_GROK_CODE_FAST_NAME,
                unique_id=f"{DOMAIN}:grok_code_fast",
            ),
        )

    if SUBENTRY_TYPE_SENSORS not in subentries:
        hass.config_entries.async_add_subentry(
            entry,
            ConfigSubentry(
                data=MappingProxyType({"name": DEFAULT_SENSORS_NAME}),
                subentry_type=SUBENTRY_TYPE_SENSORS,
                title=DEFAULT_SENSORS_NAME,
                unique_id=f"{DOMAIN}:sensors",
            ),
        )


async def async_migrate_entry(hass: HomeAssistant, entry: XAIConfigEntry) -> bool:
    """Migrate old entry.

    Args:
        hass: Home Assistant instance
        entry: Config entry to migrate

    Returns:
        True if migration successful, False otherwise
    """
    from homeassistant.config_entries import ConfigSubentry

    LOGGER.debug("Migrating from version %s", entry.version)

    if entry.version > 1:
        # This means the user has downgraded from a future version
        return False

    if entry.version == 1:
        # Migration logic: rename grok_code_fast_data to code_task
        subentries_to_update = []
        for subentry in entry.subentries.values():
            if subentry.subentry_type == "grok_code_fast_data":
                # Create new subentry with updated type
                new_subentry = ConfigSubentry(
                    data=subentry.data,
                    subentry_type=SUBENTRY_TYPE_CODE_TASK,
                    title=subentry.title,
                    unique_id=subentry.unique_id,
                )
                subentries_to_update.append((subentry, new_subentry))

        # Apply updates
        for old_subentry, new_subentry in subentries_to_update:
            hass.config_entries.async_remove_subentry(entry, old_subentry)
            hass.config_entries.async_add_subentry(entry, new_subentry)
            LOGGER.info(
                "Migrated subentry from 'grok_code_fast_data' to '%s'",
                SUBENTRY_TYPE_CODE_TASK,
            )

    hass.config_entries.async_update_entry(entry, version=1)

    LOGGER.info("Migration to version %s successful", entry.version)

    return True
