"""The xAI Conversation integration."""

from __future__ import annotations

# Standard library imports
import asyncio
import importlib
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING

# Third-party imports
import voluptuous as vol

# Home Assistant imports
from homeassistant.components import (
    ai_task as ha_ai_task,
    conversation as ha_conversation,
)
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigEntryState,
)
from homeassistant.const import CONF_API_KEY, Platform
from homeassistant.core import HomeAssistant as HA_HomeAssistant
from homeassistant.helpers import (
    config_validation as cv,
    entity_registry as ha_entity_registry,
    restore_state,
)
from homeassistant.helpers.typing import ConfigType

# Local application imports
from .exceptions import (
    HA_ConfigEntryNotReady,
    HA_HomeAssistantError,
    ServiceValidationError,
    raise_config_not_ready,
    raise_generic_error,
    raise_validation_error,
)
from .sensor import async_update_pricing_sensors_periodically

from .const import (
    CONF_MEMORY_CLEANUP_INTERVAL_HOURS,
    DEFAULT_CONVERSATION_NAME,
    DOMAIN,
    LOGGER,
    RECOMMENDED_MEMORY_CLEANUP_INTERVAL_HOURS,
)


CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)
PLATFORMS = (
    Platform.AI_TASK,
    Platform.CONVERSATION,
    Platform.SENSOR,
)


def _import_platforms() -> None:
    """Pre-import platforms to avoid blocking imports in event loop."""
    for platform in PLATFORMS:
        try:
            # Handle both Platform enums and string platform names
            platform_name = platform.value if hasattr(platform, "value") else platform
            importlib.import_module(f".{platform_name}", __name__)
            LOGGER.debug("Pre-imported platform: %s", platform_name)
        except ImportError as err:
            platform_name = platform.value if hasattr(platform, "value") else platform
            LOGGER.warning("Failed to pre-import platform %s: %s", platform_name, err)


type XAIConfigEntry = ConfigEntry[None]


# Public helper functions - re-exported from helpers module
from .helpers import (
    format_tools_for_xai,
    convert_xai_to_ha_tool,
    get_last_user_message,
    extract_user_id,
    extract_device_id,
    PromptManager,
    ConversationMemory,
    ChatHistoryService,
    TokenStatsStorage,
    XAIModelManager,
    XAIGateway,
    migrate_subentry_types,
    ensure_memory_params_in_entry_data,
    add_subentries_if_needed,
    async_migrate_entry,
)
from .services import register_services


async def async_setup(hass: HA_HomeAssistant, config: ConfigType) -> bool:
    """Set up the xAI Conversation integration."""
    return True


async def async_setup_entry(hass: HA_HomeAssistant, entry: XAIConfigEntry) -> bool:
    """Set up xAI Conversation from a config entry."""
    # Fetch dynamic model data using the centralized manager
    # Note: The manager automatically populates SUPPORTED_MODELS and REASONING_EFFORT_MODELS
    model_manager = XAIModelManager(hass)
    xai_models_data = await model_manager.async_get_models_data(entry.data[CONF_API_KEY])

    if not xai_models_data:
        LOGGER.error("Failed to fetch xAI model data. Aborting setup.")
        raise HA_ConfigEntryNotReady("Failed to fetch xAI model data.")

    # Initialize hass.data for global access
    if DOMAIN not in hass.data:
        hass.data[DOMAIN] = {}

    hass.data[DOMAIN]["xai_models_data"] = xai_models_data
    hass.data[DOMAIN]["xai_models_data_timestamp"] = time.time()
    LOGGER.debug(
        "Fetched and stored xAI model data for %d models.", len(xai_models_data)
    )

    # Migrate old subentry_type values to new format (one-time migration)
    # MUST run BEFORE add_subentries_if_needed to avoid conflicts
    migrate_subentry_types(hass, entry)

    # Add subentries for conversation and AI task BEFORE setting up platforms
    add_subentries_if_needed(hass, entry)

    # Ensure memory parameters are in entry.data (migration for existing entries)
    ensure_memory_params_in_entry_data(hass, entry)

    # Register the options update listener to reload on changes
    entry.async_on_unload(
        entry.add_update_listener(
            lambda hass, entry: hass.config_entries.async_reload(entry.entry_id)
        )
    )

    # Pre-import platforms to avoid blocking imports in event loop
    await hass.async_add_executor_job(_import_platforms)

    # Create global storage instances for shared access
    folder_name = DEFAULT_CONVERSATION_NAME.lower().replace(" ", "_")
    memory_path = f"{folder_name}/{DOMAIN}.memory"
    chat_history_path = f"{folder_name}/{DOMAIN}.chat_history"
    token_stats_path = f"{folder_name}/{DOMAIN}.token_stats"

    hass.data[DOMAIN]["conversation_memory"] = ConversationMemory(
        hass, memory_path, entry
    )
    hass.data[DOMAIN]["chat_history"] = ChatHistoryService(
        hass, chat_history_path, entry
    )
    hass.data[DOMAIN]["token_stats_storage"] = TokenStatsStorage(
        hass, token_stats_path, entry
    )
    LOGGER.debug("Created global ConversationMemory instance at %s", memory_path)
    LOGGER.debug("Created global ChatHistoryService instance at %s", chat_history_path)
    LOGGER.debug("Created global TokenStatsStorage instance at %s", token_stats_path)

    # Set up periodic cleanup task for ConversationMemory
    cleanup_interval_hours = entry.data.get(
        CONF_MEMORY_CLEANUP_INTERVAL_HOURS, RECOMMENDED_MEMORY_CLEANUP_INTERVAL_HOURS
    )
    memory = hass.data[DOMAIN]["conversation_memory"]
    cleanup_unsub = memory.setup_periodic_cleanup(cleanup_interval_hours)
    entry.async_on_unload(cleanup_unsub)

    # Set up platforms
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    # Update pricing sensors from API on startup
    try:
        await async_update_pricing_sensors_periodically(hass, entry)
        LOGGER.debug("Updated pricing sensors on startup")
    except Exception as err:
        LOGGER.warning("Failed to update pricing sensors on startup: %s", err)

    # Register services
    register_services(hass, entry)

    # Check if this is the first setup (installation) and reset token stats
    # This ensures cleanup of any residual data from previous installations
    if entry.state == ConfigEntryState.NOT_LOADED:
        try:
            await hass.services.async_call(
                DOMAIN,
                "reset_token_stats",
                blocking=True,
            )
            LOGGER.debug("Reset token stats on first setup")
        except Exception as err:
            LOGGER.warning("Failed to reset token stats on first setup: %s", err)

    return True


async def async_unload_entry(hass: HA_HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    # Save token stats to disk before unloading
    storage = hass.data[DOMAIN].get("token_stats_storage")
    if storage:
        try:
            await storage.async_save()
            LOGGER.debug("Successfully saved token stats during unload.")
        except Exception as err:
            LOGGER.error("Failed to save token stats during unload: %s", err)

    # Wait for all pending save tasks before unloading
    # This ensures token statistics are flushed to disk before Home Assistant closes
    pending_tasks = hass.data.get(DOMAIN, {}).get("pending_save_tasks", set())
    if pending_tasks:
        LOGGER.info(
            "Waiting for %d pending save tasks to complete before unload...",
            len(pending_tasks),
        )
        try:
            # Wait for all tasks with a timeout of 30 seconds
            await asyncio.wait_for(
                asyncio.gather(*pending_tasks, return_exceptions=True), timeout=30.0
            )
            LOGGER.debug("All pending save tasks completed")
        except asyncio.TimeoutError:
            LOGGER.warning(
                "Timeout waiting for pending save tasks (30s) - some data may not be saved"
            )
        except Exception as err:
            LOGGER.error("Error waiting for pending save tasks: %s", err)

    # Unregister entity services when unloading
    try:
        hass.services.async_remove(DOMAIN, "clear_memory")
        hass.services.async_remove(DOMAIN, "clear_code_memory")
        hass.services.async_remove(DOMAIN, "grok_code_fast")
        hass.services.async_remove(DOMAIN, "reset_token_stats")
    except Exception:
        pass
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def async_remove_entry(hass: HA_HomeAssistant, entry: ConfigEntry) -> None:
    """Handle removal of an entry.

    This is called when the user deletes the integration from the UI.
    We clean up the memory storage folder and remove all entity states.
    """
    # Remove all entities from entity registry to prevent state restoration
    ent_reg = ha_entity_registry.async_get(hass)

    # Collect all config entry IDs: main entry + all subentries
    config_entry_ids = {entry.entry_id}
    for subentry in entry.subentries.values():
        config_entry_ids.add(subentry.subentry_id)

    entities_to_remove = [
        entity.entity_id
        for entity in ent_reg.entities.values()
        if entity.config_entry_id in config_entry_ids
    ]

    # Get RestoreStateData to clean up saved states
    restore_data = restore_state.async_get(hass)

    for entity_id in entities_to_remove:
        # Remove entity state to prevent restoration
        hass.states.async_remove(entity_id)
        # Remove from restore state storage
        if entity_id in restore_data.last_states:
            del restore_data.last_states[entity_id]
            LOGGER.debug("Removed restore state for: %s", entity_id)
        # Remove entity from registry
        ent_reg.async_remove(entity_id)
        LOGGER.debug("Removed entity and state from registry: %s", entity_id)

    # Clean up sensor references and other data from hass.data
    if DOMAIN in hass.data:
        # Remove sensor references
        sensors_key = f"{entry.entry_id}_sensors"
        if sensors_key in hass.data[DOMAIN]:
            del hass.data[DOMAIN][sensors_key]
            LOGGER.debug("Cleaned up sensor references from memory: %s", sensors_key)

        # Remove other entry-specific data (chat_history, conversation_memory, etc.)
        # These are cleaned up by their respective cleanup methods, but we ensure
        # no dangling references remain
        for key in list(hass.data[DOMAIN].keys()):
            if key.startswith(entry.entry_id):
                del hass.data[DOMAIN][key]
                LOGGER.debug("Cleaned up data key from memory: %s", key)

    # Get the storage path for memory files
    folder_name = DEFAULT_CONVERSATION_NAME.lower().replace(" ", "_")
    storage_base = Path(hass.config.path(".storage"))
    memory_folder = storage_base / folder_name

    # Remove the entire memory folder if it exists
    if memory_folder.exists() and memory_folder.is_dir():
        try:
            # Use executor to avoid blocking the event loop
            await hass.async_add_executor_job(shutil.rmtree, memory_folder)
            LOGGER.info("Cleaned up memory storage folder: %s", memory_folder)
        except Exception as err:
            LOGGER.warning(
                "Failed to remove memory storage folder %s: %s", memory_folder, err
            )
