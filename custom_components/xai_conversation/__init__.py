"""The xAI Conversation integration."""

from __future__ import annotations

# Standard library imports
import asyncio
import shutil
from datetime import timedelta
from pathlib import Path

# Home Assistant imports
from homeassistant.config_entries import ConfigEntry

from homeassistant.const import CONF_API_KEY, EVENT_HOMEASSISTANT_STARTED, EVENT_HOMEASSISTANT_STOP, Platform
from homeassistant.core import HomeAssistant as HA_HomeAssistant, Event
from homeassistant.helpers import (
    config_validation as cv,
    entity_registry as ha_entity_registry,
    restore_state,
)
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.helpers.typing import ConfigType

# Local application imports
from .const import (
    CONF_MEMORY_CLEANUP_INTERVAL_HOURS,
    CONF_PRICING_UPDATE_INTERVAL_HOURS,
    DEFAULT_CONVERSATION_NAME,
    DOMAIN,
    LOGGER,
    MEMORY_FLUSH_INTERVAL_MINUTES,
    RECOMMENDED_MEMORY_CLEANUP_INTERVAL_HOURS,
    RECOMMENDED_PRICING_UPDATE_INTERVAL_HOURS,
)

# Public helper functions - re-exported from helpers module
from .helpers import (
    ConversationMemory,
    ChatHistoryService,
    TokenStats,
    XAIModelManager,
    migrate_subentry_types,
    ensure_memory_params_in_entry_data,
    add_subentries_if_needed,
)
from .services import register_services

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)
PLATFORMS = (
    Platform.AI_TASK,
    Platform.CONVERSATION,
    Platform.SENSOR,
)


type XAIConfigEntry = ConfigEntry


async def async_setup(hass: HA_HomeAssistant, config: ConfigType) -> bool:
    """Set up the xAI Conversation integration."""
    return True


async def async_setup_entry(hass: HA_HomeAssistant, entry: XAIConfigEntry) -> bool:
    """Set up xAI Conversation from a config entry."""
    # Initialize hass.data for global access early
    if DOMAIN not in hass.data:
        hass.data[DOMAIN] = {}

    # Initialize placeholder for models data
    if "xai_models_data" not in hass.data[DOMAIN]:
        hass.data[DOMAIN]["xai_models_data"] = {}
        hass.data[DOMAIN]["xai_models_data_timestamp"] = 0

    # Create global storage instances for shared access EARLY
    # TokenStats must exist before ModelManager.async_update_models is called
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
    # V2: Use new TokenStats class (simplified architecture)
    hass.data[DOMAIN]["token_stats"] = TokenStats(hass, token_stats_path, entry)
    LOGGER.debug("Created global ConversationMemory instance at %s", memory_path)
    LOGGER.debug("Created global ChatHistoryService instance at %s", chat_history_path)
    LOGGER.debug("Created global TokenStats instance at %s", token_stats_path)

    # Initialize Model Manager (after TokenStats is available)
    model_manager = XAIModelManager(hass)
    hass.data[DOMAIN]["model_manager"] = model_manager

    # Perform initial model fetch BEFORE platform setup
    # This ensures xai_models_data is populated when sensors are created
    try:
        await model_manager.async_update_models(entry.data[CONF_API_KEY])
        LOGGER.debug("Initial xAI model data fetch completed")
    except Exception as err:
        LOGGER.warning("Initial xAI model fetch failed (will retry after startup): %s", err)

    # Define startup handler for retry and periodic updates
    async def async_initial_update(event: Event) -> None:
        """Retry model data fetch when Home Assistant is fully started."""
        # Only retry if initial fetch failed (no models data)
        if not hass.data[DOMAIN].get("xai_models_data"):
            LOGGER.debug(
                "Home Assistant started, retrying xAI model data fetch..."
            )
            try:
                await model_manager.async_update_models(entry.data[CONF_API_KEY])
            except Exception as err:
                LOGGER.error("Failed to fetch xAI model data: %s", err)

    # Register startup listener for retry
    # Note: async_listen_once auto-removes the listener after firing, no manual cleanup needed
    hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STARTED, async_initial_update)

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

    # Set up periodic cleanup task for ConversationMemory
    cleanup_interval_hours = entry.data.get(
        CONF_MEMORY_CLEANUP_INTERVAL_HOURS, RECOMMENDED_MEMORY_CLEANUP_INTERVAL_HOURS
    )
    memory = hass.data[DOMAIN]["conversation_memory"]
    cleanup_unsub = memory.setup_periodic_cleanup(cleanup_interval_hours)
    entry.async_on_unload(cleanup_unsub)

    # Set up periodic memory flush (write-behind)
    token_stats = hass.data[DOMAIN]["token_stats"]

    async def async_flush_all():
        """Flush both memory and token stats."""
        await memory.async_flush()
        await token_stats.async_flush()

    flush_interval = timedelta(minutes=MEMORY_FLUSH_INTERVAL_MINUTES)
    flush_unsub = async_track_time_interval(
        hass, lambda now: hass.async_create_task(async_flush_all()), flush_interval
    )
    entry.async_on_unload(flush_unsub)

    # Ensure memory flush on Home Assistant stop
    async def _async_flush_on_stop(event):
        """Flush memory to disk on Home Assistant stop."""
        LOGGER.debug("Home Assistant stopping, flushing conversation memory...")
        await async_flush_all()

    entry.async_on_unload(
        hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STOP, _async_flush_on_stop)
    )

    # Set up periodic pricing update task
    # Get interval from sensors subentry config, fallback to default
    pricing_interval = RECOMMENDED_PRICING_UPDATE_INTERVAL_HOURS
    for subentry in entry.subentries.values():
        if subentry.subentry_type == "sensors":
            pricing_interval = subentry.data.get(
                CONF_PRICING_UPDATE_INTERVAL_HOURS,
                RECOMMENDED_PRICING_UPDATE_INTERVAL_HOURS,
            )
            break

    pricing_unsub = async_track_time_interval(
        hass,
        lambda now: hass.async_create_task(
            model_manager.async_update_models(entry.data[CONF_API_KEY])
        ),
        timedelta(hours=pricing_interval),
    )
    entry.async_on_unload(pricing_unsub)
    LOGGER.debug(
        "Set up periodic model update check (every %d hours)",
        pricing_interval,
    )

    # Set up platforms
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    # Register services
    register_services(hass, entry)

    return True


async def async_unload_entry(hass: HA_HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    # Explicitly flush memory before unloading to ensure pending changes are saved
    try:
        if DOMAIN in hass.data:
            if "conversation_memory" in hass.data[DOMAIN]:
                await hass.data[DOMAIN]["conversation_memory"].async_flush()
            if "token_stats" in hass.data[DOMAIN]:
                await hass.data[DOMAIN]["token_stats"].async_flush()
            LOGGER.debug("Memory and token stats flushed successfully before unload")
    except Exception as err:
        LOGGER.warning("Failed to flush memory during unload: %s", err)

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
