"""The xAI Conversation integration."""

from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.typing import ConfigType

import shutil
from pathlib import Path
from .const import DOMAIN, DEFAULT_CONVERSATION_NAME, LOGGER

from .init_manager import XaiInitManager
from .services import register_services

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)
PLATFORMS = (
    Platform.AI_TASK,
    Platform.CONVERSATION,
    Platform.SENSOR,
)


type XAIConfigEntry = ConfigEntry


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the xAI Conversation integration."""
    hass.data.setdefault(DOMAIN, {})
    return True


async def async_setup_entry(hass: HomeAssistant, entry: XAIConfigEntry) -> bool:
    """Set up xAI Conversation from a config entry."""
    coordinator = XaiInitManager(hass, entry)
    await coordinator.async_setup()

    hass.data[DOMAIN][entry.entry_id] = coordinator

    # Set up platforms
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

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
            LOGGER.info(
                "Final cleanup: Removed memory storage folder: %s", memory_folder
            )
        except Exception as err:
            LOGGER.warning(
                "Final cleanup: Failed to remove memory folder %s: %s",
                memory_folder,
                err,
            )
