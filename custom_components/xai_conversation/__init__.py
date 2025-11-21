"""The xAI Conversation integration."""
from __future__ import annotations

# Standard library imports
import base64
import importlib
import json
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

# Third-party imports
import voluptuous as vol

# xAI SDK imports (conditional)
try:
    from xai_sdk import AsyncClient as XAI_CLIENT_CLASS
    from xai_sdk.chat import user as xai_user, system as xai_system, assistant as xai_assistant, tool as xai_tool, image as xai_image
    from xai_sdk.search import SearchParameters as xai_search_parameters
    XAI_SDK_AVAILABLE = True
except ImportError as err:
    XAI_CLIENT_CLASS = None
    xai_user = None
    xai_system = None
    xai_assistant = None
    xai_tool = None
    xai_image = None
    xai_search_parameters = None
    XAI_SDK_AVAILABLE = False

# Home Assistant imports
from homeassistant.components import ai_task as ha_ai_task, conversation as ha_conversation
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigEntryState,
    ConfigFlow,
    ConfigFlowResult,
    ConfigSubentry as HA_ConfigSubentry,
    ConfigSubentryFlow,
    SubentryFlowResult,
)
from homeassistant.const import CONF_API_KEY, Platform
from homeassistant.core import (
    HomeAssistant as HA_HomeAssistant,
    ServiceCall as HA_ServiceCall,
    ServiceResponse as HA_ServiceResponse,
    SupportsResponse,
    callback as ha_callback,
)
from homeassistant.helpers import (
    area_registry as ha_area_registry,
    config_validation as cv,
    device_registry as ha_device_registry,
    entity_registry as ha_entity_registry,
    selector,
)
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.helpers.entity import Entity as HA_Entity
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback as HA_AddConfigEntryEntitiesCallback
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.typing import ConfigType
from homeassistant.util.json import json_loads as ha_json_loads

# Local application imports
from .exceptions import (
    HA_ConfigEntryNotReady,
    HA_HomeAssistantError,
    ServiceValidationError,
    raise_config_not_ready,
    raise_generic_error,
    raise_validation_error,
)

from .const import (
    CONF_CHAT_MODEL,
    CONF_LLM_HASS_API,
    CONF_MAX_TOKENS,
    CONF_MEMORY_CLEANUP_INTERVAL_HOURS,
    CONF_MEMORY_DEVICE_MAX_TURNS,
    CONF_MEMORY_DEVICE_TTL_HOURS,
    CONF_MEMORY_USER_MAX_TURNS,
    CONF_MEMORY_USER_TTL_HOURS,
    CONF_PROMPT,
    CONF_REASONING_EFFORT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DEFAULT_AI_TASK_NAME,
    DEFAULT_CONVERSATION_NAME,
    DEFAULT_GROK_CODE_FAST_NAME,
    DEFAULT_SENSORS_NAME,
    DOMAIN,
    LOGGER,
    RECOMMENDED_AI_TASK_OPTIONS,
    RECOMMENDED_GROK_CODE_FAST_OPTIONS,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_HISTORY_LIMIT_TURNS,
    RECOMMENDED_MEMORY_CLEANUP_INTERVAL_HOURS,
    RECOMMENDED_MEMORY_DEVICE_MAX_TURNS,
    RECOMMENDED_MEMORY_DEVICE_TTL_HOURS,
    RECOMMENDED_MEMORY_USER_MAX_TURNS,
    RECOMMENDED_MEMORY_USER_TTL_HOURS,
    RECOMMENDED_TOOLS_OPTIONS,
    RECOMMENDED_PIPELINE_OPTIONS,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_REASONING_EFFORT,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TIMEOUT,
    RECOMMENDED_TOP_P,
    SUPPORTED_MODELS,
    SUBENTRY_TYPE_AI_TASK,
    SUBENTRY_TYPE_CODE_TASK,
    SUBENTRY_TYPE_CONVERSATION,
    SUBENTRY_TYPE_SENSORS,
    llm as ha_llm,
)




CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)
PLATFORMS = (
    Platform.AI_TASK,  # Handles both ai_task_data and code_task subentry types
    Platform.CONVERSATION,
    Platform.SENSOR,
)


def _import_platforms() -> None:
    """Pre-import platforms to avoid blocking imports in event loop."""
    for platform in PLATFORMS:
        try:
            # Handle both Platform enums and string platform names
            platform_name = platform.value if hasattr(platform, 'value') else platform
            importlib.import_module(f".{platform_name}", __name__)
            LOGGER.debug("Pre-imported platform: %s", platform_name)
        except ImportError as err:
            platform_name = platform.value if hasattr(platform, 'value') else platform
            LOGGER.warning("Failed to pre-import platform %s: %s", platform_name, err)


type XAIConfigEntry = ConfigEntry[None]


# Public helper functions - re-exported from helpers module
from .helpers import (
    async_validate_api_key,
    format_tools_for_xai,
    convert_xai_to_ha_tool,
    validate_xai_configuration,
    get_last_user_message,
    extract_user_id,
    extract_device_id,
    PromptManager,
    ConversationMemory,
    ChatHistoryService,
    async_get_xai_models_data,
)
from .services import register_services


async def async_setup(hass: HA_HomeAssistant, config: ConfigType) -> bool:
    """Set up the xAI Conversation integration."""
    return True


async def async_setup_entry(hass: HA_HomeAssistant, entry: XAIConfigEntry) -> bool:
    """Set up xAI Conversation from a config entry."""
    # Log xAI SDK import status once during integration setup
    if XAI_SDK_AVAILABLE:
        LOGGER.debug("xAI SDK imported successfully")
    else:
        LOGGER.warning("xAI SDK not available")

    try:
        # Validate API key
        await async_validate_api_key(hass, entry.data[CONF_API_KEY])
    except Exception as err:
        LOGGER.error("Failed to validate API key: %s", err)
        raise HA_ConfigEntryNotReady("Failed to setup xAI integration") from err

    # Fetch dynamic model data
    xai_models_data = await async_get_xai_models_data(hass, entry.data[CONF_API_KEY])
    if not xai_models_data:
        LOGGER.error("Failed to fetch xAI model data. Aborting setup.")
        raise HA_ConfigEntryNotReady("Failed to fetch xAI model data.")
    
    # Store in hass.data for global access
    if DOMAIN not in hass.data:
        hass.data[DOMAIN] = {}
    hass.data[DOMAIN]["xai_models_data"] = xai_models_data
    LOGGER.debug("Fetched and stored xAI model data for %d models.", len(xai_models_data))

    # Dynamically populate SUPPORTED_MODELS and REASONING_EFFORT_MODELS
    # This needs to be done AFTER xai_models_data is fetched and BEFORE _add_subentries_if_needed
    from .const import SUPPORTED_MODELS, REASONING_EFFORT_MODELS # Import the lists
    SUPPORTED_MODELS.clear()
    REASONING_EFFORT_MODELS.clear()
    
    # Use a set to avoid duplicates and preserve order if desired later
    dynamic_supported_models = set()
    dynamic_reasoning_models = set()

    for model_name, model_data in xai_models_data.items():
        # Only add primary model name, not aliases, to the main SUPPORTED_MODELS list
        # Check if model_name is not an alias (aliases point to existing model_data)
        if model_data["name"] == model_name: # This checks if it's the primary name
            if model_data["type"] == "language": # Only language models are currently 'supported' for chat
                dynamic_supported_models.add(model_name)
                # Keep specific models hardcoded for reasoning effort if the SDK doesn't provide a flag
                if model_name in ["grok-3", "grok-3-mini"]: # Re-applying the old hardcoded logic
                    dynamic_reasoning_models.add(model_name)

    # Sort the lists for consistency
    SUPPORTED_MODELS.extend(sorted(list(dynamic_supported_models)))
    REASONING_EFFORT_MODELS.extend(sorted(list(dynamic_reasoning_models)))

    LOGGER.debug("Dynamically populated SUPPORTED_MODELS: %s", SUPPORTED_MODELS)
    LOGGER.debug("Dynamically populated REASONING_EFFORT_MODELS: %s", REASONING_EFFORT_MODELS)

    # Add subentries for conversation and AI task BEFORE setting up platforms
    _add_subentries_if_needed(hass, entry)

    # Ensure memory parameters are in entry.data (migration for existing entries)
    _ensure_memory_params_in_entry_data(hass, entry)

    # Register the options update listener to reload on changes
    entry.async_on_unload(entry.add_update_listener(async_update_options))

    # Pre-import platforms to avoid blocking imports in event loop
    await hass.async_add_executor_job(_import_platforms)

    # Create global ConversationMemory instance for shared storage
    folder_name = DEFAULT_CONVERSATION_NAME.lower().replace(" ", "_")
    memory_path = f"{folder_name}/{DOMAIN}.memory"
    chat_history_path = f"{folder_name}/{DOMAIN}.chat_history"

    # Store in hass.data for access by entities and services
    if DOMAIN not in hass.data:
        hass.data[DOMAIN] = {}
    hass.data[DOMAIN]["conversation_memory"] = ConversationMemory(hass, memory_path, entry)
    hass.data[DOMAIN]["chat_history"] = ChatHistoryService(hass, chat_history_path, entry)
    LOGGER.debug("Created global ConversationMemory instance at %s", memory_path)
    LOGGER.debug("Created global ChatHistoryService instance at %s", chat_history_path)

    # Set up periodic cleanup task for ConversationMemory
    cleanup_interval_hours = entry.data.get(CONF_MEMORY_CLEANUP_INTERVAL_HOURS, RECOMMENDED_MEMORY_CLEANUP_INTERVAL_HOURS)
    cleanup_interval = timedelta(hours=cleanup_interval_hours)

    async def periodic_cleanup(_now):
        """Periodic cleanup task for conversation memory."""
        memory = hass.data[DOMAIN]["conversation_memory"]
        LOGGER.debug("Running periodic memory cleanup (interval: %s hours)", cleanup_interval_hours)
        stats = await memory.async_cleanup_expired()
        if stats["keys_removed"] > 0 or stats["keys_cleaned"] > 0:
            LOGGER.info(
                "Memory cleanup: cleaned %d keys, removed %d keys, deleted %d responses",
                stats["keys_cleaned"], stats["keys_removed"], stats["responses_removed"]
            )

    # Start periodic cleanup and register unload callback
    cleanup_unsub = async_track_time_interval(hass, periodic_cleanup, cleanup_interval)
    entry.async_on_unload(cleanup_unsub)
    LOGGER.debug("Started periodic memory cleanup task (interval: %s hours)", cleanup_interval_hours)

    # Set up platforms
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    # Register services
    await _register_services(hass, entry)

    return True


async def async_unload_entry(hass: HA_HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
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

    for entity_id in entities_to_remove:
        ent_reg.async_remove(entity_id)
        LOGGER.debug("Removed entity from registry: %s", entity_id)

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
            LOGGER.warning("Failed to remove memory storage folder %s: %s", memory_folder, err)


async def async_update_options(hass: HA_HomeAssistant, entry: XAIConfigEntry) -> None:
    """Update options."""
    await hass.config_entries.async_reload(entry.entry_id)


def _ensure_memory_params_in_entry_data(hass: HA_HomeAssistant, entry: XAIConfigEntry) -> None:
    """Ensure memory parameters are in entry.data (migration for existing entries)."""
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
        data[CONF_MEMORY_CLEANUP_INTERVAL_HOURS] = RECOMMENDED_MEMORY_CLEANUP_INTERVAL_HOURS
        updated = True

    if updated:
        hass.config_entries.async_update_entry(entry, data=data)
        LOGGER.info("Added memory parameters to integration configuration")


def _add_subentries_if_needed(hass: HA_HomeAssistant, entry: XAIConfigEntry) -> None:
    """Add subentries for conversation, AI Task, and Grok Code Fast if they don't exist."""
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
            HA_ConfigSubentry(
                data=options,
                subentry_type=SUBENTRY_TYPE_CONVERSATION,
                title=DEFAULT_CONVERSATION_NAME,
                unique_id=f"{DOMAIN}:conversation",
            )
        )

    if SUBENTRY_TYPE_AI_TASK not in subentries:
        hass.config_entries.async_add_subentry(
            entry,
            HA_ConfigSubentry(
                data=MappingProxyType({"name": DEFAULT_AI_TASK_NAME, **RECOMMENDED_AI_TASK_OPTIONS}),
                subentry_type=SUBENTRY_TYPE_AI_TASK,
                title=DEFAULT_AI_TASK_NAME,
                unique_id=f"{DOMAIN}:ai_task",
            )
        )

    if SUBENTRY_TYPE_CODE_TASK not in subentries:
        hass.config_entries.async_add_subentry(
            entry,
            HA_ConfigSubentry(
                data=MappingProxyType({"name": DEFAULT_GROK_CODE_FAST_NAME, **RECOMMENDED_GROK_CODE_FAST_OPTIONS}),
                subentry_type=SUBENTRY_TYPE_CODE_TASK,
                title=DEFAULT_GROK_CODE_FAST_NAME,
                unique_id=f"{DOMAIN}:grok_code_fast",
            )
        )

    if SUBENTRY_TYPE_SENSORS not in subentries:
        hass.config_entries.async_add_subentry(
            entry,
            HA_ConfigSubentry(
                data=MappingProxyType({"name": DEFAULT_SENSORS_NAME}),
                subentry_type=SUBENTRY_TYPE_SENSORS,
                title=DEFAULT_SENSORS_NAME,
                unique_id=f"{DOMAIN}:sensors",
            )
        )


async def async_migrate_entry(hass: HA_HomeAssistant, entry: XAIConfigEntry) -> bool:
    """Migrate old entry."""
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
                new_subentry = HA_ConfigSubentry(
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
            LOGGER.info("Migrated subentry from 'grok_code_fast_data' to '%s'", SUBENTRY_TYPE_CODE_TASK)

    hass.config_entries.async_update_entry(entry, version=1)

    LOGGER.info("Migration to version %s successful", entry.version)

    return True


def _get_entity_from_components(hass: HA_HomeAssistant, entity_id: str, components: list[str]):
    """Try to get entity from multiple platform components.

    Args:
        hass: Home Assistant instance
        entity_id: Entity ID to retrieve
        components: List of component names to search

    Returns:
        Entity object if found, None otherwise
    """
    for component_name in components:
        comp = hass.data.get("entity_components", {}).get(component_name)
        if comp:
            entity = comp.get_entity(entity_id)
            if entity:
                return entity
    return None


async def _register_services(hass: HA_HomeAssistant, entry: XAIConfigEntry) -> None:
    """Register xAI services.

    Service implementations are in services.py module for better organization.
    """
    register_services(hass, entry)
