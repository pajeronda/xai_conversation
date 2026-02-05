"""The xAI Conversation integration."""

from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.typing import ConfigType

import shutil
from pathlib import Path
from .const import DOMAIN, DEFAULT_CONVERSATION_NAME, LOGGER, XAIConfigEntry

from .init_manager import XaiInitManager
from .services import register_services

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)
PLATFORMS = (
    Platform.AI_TASK,
    Platform.CONVERSATION,
    Platform.SENSOR,
)


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the xAI Conversation integration."""
    hass.data.setdefault(DOMAIN, {})
    return True


async def async_migrate_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """Migrate old entry to new schema."""
    LOGGER.info(
        "Migration: from version %s.%s to 2.2",
        config_entry.version,
        config_entry.minor_version,
    )

    # Prevent downgrade
    if config_entry.version > 2:
        LOGGER.error(
            "Migration: cannot downgrade from version %s to version 2",
            config_entry.version,
        )
        return False

    # Create coordinator for migration helpers
    coordinator = XaiInitManager(hass, config_entry)

    # Migrate V1 → V2
    if config_entry.version == 1:
        # Clean up deprecated subentries and migrate types
        await coordinator.clean_deprecated_subentries()

        # Clean up deprecated entities from registry
        await coordinator.clean_deprecated_entities()

        # Clean up deprecated storage files
        await coordinator.clean_deprecated_storage()

        # Ensure valid config keys in entry.data and subentries
        await coordinator.ensure_valid_config_keys()

    # Migrate to V2.2 (subentry_id in memory keys)
    # Applies to: V1 (any), V2.0, V2.1 - anyone not already at V2.2+
    needs_memory_migration = config_entry.version < 2 or (
        config_entry.version == 2 and config_entry.minor_version < 2
    )
    if needs_memory_migration:
        await coordinator.migrate_memory_v2_2()

    # Update version
    hass.config_entries.async_update_entry(
        config_entry,
        version=2,
        minor_version=2,
    )

    LOGGER.info("Migration: completed successfully")
    return True


async def async_setup_entry(hass: HomeAssistant, entry: XAIConfigEntry) -> bool:
    """Set up xAI Conversation from a config entry."""
    coordinator = XaiInitManager(hass, entry)
    await coordinator.async_setup()

    hass.data[DOMAIN][entry.entry_id] = coordinator

    # Set up platforms
    # Refresh entry to ensure any subentry modifications during setup are reflected
    updated_entry = hass.config_entries.async_get_entry(entry.entry_id) or entry
    await hass.config_entries.async_forward_entry_setups(updated_entry, PLATFORMS)

    # Register services
    register_services(hass, entry)

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    coordinator: XaiInitManager = hass.data[DOMAIN].get(entry.entry_id)

    # First, unload platforms
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

    # Then, run coordinator cleanup
    if coordinator:
        await coordinator.async_unload()

    if unload_ok:
        hass.data[DOMAIN].pop(entry.entry_id, None)

    return unload_ok


async def async_remove_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle removal of an entry."""
    # Attempt to use coordinator if it still exists (unlikely if unloaded first)
    coordinator: XaiInitManager = hass.data[DOMAIN].get(entry.entry_id)
    if coordinator:
        await coordinator.async_remove()

    # Ensure storage folder is deleted even if coordinator is gone
    folder_name = DEFAULT_CONVERSATION_NAME.lower().replace(" ", "_")
    storage_base = Path(hass.config.path(".storage"))
    memory_folder = storage_base / folder_name

    if memory_folder.exists() and memory_folder.is_dir():
        try:
            await hass.async_add_executor_job(shutil.rmtree, memory_folder)
        except Exception as err:
            LOGGER.warning(
                "Removal: fallback cleanup failed for folder %s - %s",
                memory_folder,
                err,
            )
