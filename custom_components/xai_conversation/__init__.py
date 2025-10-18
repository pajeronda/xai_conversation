"""The xAI Conversation integration."""
from __future__ import annotations

# Standard library imports
import importlib
from pathlib import Path
from types import MappingProxyType

# Third-party imports
import voluptuous as vol

# xAI SDK imports (conditional)
try:
    from xai_sdk import Client as XAI_CLIENT_CLASS
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
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API, Platform
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
    llm as ha_llm,
    selector,
)
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
    format_tools_for_xai,
    convert_xai_to_ha_tool,
    validate_xai_configuration,
    get_last_user_message,
    extract_user_id,
    extract_device_id,
    prompt_hash,
    PromptManager,
)


async def async_validate_api_key(hass: HA_HomeAssistant, api_key: str) -> None:
    """Validate the API key by making a minimal chat request.
    Runs in executor to avoid blocking the event loop.
    """
    def _validate():
        if not XAI_SDK_AVAILABLE or XAI_CLIENT_CLASS is None:
            raise_generic_error("xAI SDK not available")

        client = XAI_CLIENT_CLASS(api_key=api_key, timeout=float(RECOMMENDED_TIMEOUT))
        chat = client.chat.create(model=RECOMMENDED_CHAT_MODEL, max_tokens=1, temperature=0.1)
        chat.append(xai_user("ok"))
        chat.sample()  # Will raise on invalid key/permissions

    await hass.async_add_executor_job(_validate)


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

    # Add subentries for conversation and AI task BEFORE setting up platforms
    _add_subentries_if_needed(hass, entry)

    # Ensure memory parameters are in entry.data (migration for existing entries)
    _ensure_memory_params_in_entry_data(hass, entry)

    # Register the options update listener to reload on changes
    entry.async_on_unload(entry.add_update_listener(async_update_options))

    # Pre-import platforms to avoid blocking imports in event loop
    await hass.async_add_executor_job(_import_platforms)

    # Create global ConversationMemory instance for shared storage
    from .helpers import ConversationMemory, ChatHistoryService
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
    import shutil
    from pathlib import Path

    # Remove all entities from entity registry to prevent state restoration
    ent_reg = ha_entity_registry.async_get(hass)
    entities_to_remove = [
        entity.entity_id
        for entity in ent_reg.entities.values()
        if entity.config_entry_id == entry.entry_id
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


def _parse_id_list(value: str | list) -> list[str]:
    """Parse comma-separated string or list into list of IDs.

    Args:
        value: String (comma-separated) or list of IDs

    Returns:
        List of trimmed, non-empty ID strings
    """
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(value) if value else []


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
    """Register xAI services."""
    # Get global ConversationMemory instance from hass.data
    code_memory = hass.data[DOMAIN]["conversation_memory"]

    async def handle_home_assistant_code_editor(call: HA_ServiceCall) -> HA_ServiceResponse:
        """Handle the Grok Code Fast service - direct proxy to xAI API."""
        import json
        import base64
        import time
        from datetime import datetime
        from .helpers import parse_grok_response
        from .const import (
            CONF_PROMPT, CONF_CHAT_MODEL, CONF_TEMPERATURE, CONF_MAX_TOKENS,
            CONF_STORE_MESSAGES, GROK_CODE_FAST_PROMPT, SUBENTRY_TYPE_CODE_TASK,
        )

        context_id = call.context.id
        LOGGER.debug(f"[Context: {context_id}] Service 'grok_code_fast' called.")

        try:
            if not XAI_SDK_AVAILABLE:
                raise_generic_error("xAI SDK not available")

            # 1. Parse instructions as JSON
            instructions_str = call.data.get("instructions", "{}")
            try:
                request_data = ha_json_loads(instructions_str)
            except Exception:
                # Fallback: treat as plain text prompt
                request_data = {"prompt": instructions_str}

            # Get prompt
            user_prompt = request_data.get("prompt", "")
            if not user_prompt:
                raise_generic_error("Prompt is required")

            # Get optional parameters
            previous_response_id = request_data.get("previous_response_id")
            current_code = request_data.get("code")
            attachments = request_data.get("attachments", [])

            # Get user_id from request_data first, fallback to context
            user_id = request_data.get("user_id")
            if not user_id and call.context.user_id:
                user_id = call.context.user_id
                LOGGER.debug(f"[Context: {context_id}] Using user_id from context: {user_id}")

            # If previous_response_id not provided but user_id is, try to recover from memory
            if not previous_response_id and user_id:
                previous_response_id = await code_memory.get_response_id(user_id, "code")
                if previous_response_id:
                    LOGGER.debug(f"[Context: {context_id}] Recovered previous_response_id from memory: {previous_response_id[:8]}")
            
            LOGGER.debug(
                f"[Context: {context_id}] Request data: "
                f"prompt_length={len(user_prompt)}, "
                f"previous_response_id={previous_response_id}, "
                f"has_code={bool(current_code)}, "
                f"attachments={len(attachments)}"
            )

            # 2. Find code_task subentry for configuration
            code_task_subentry = None
            for subentry in entry.subentries.values():
                if subentry.subentry_type == SUBENTRY_TYPE_CODE_TASK:
                    code_task_subentry = subentry
                    break

            if not code_task_subentry:
                raise_generic_error("No code_task configuration found. Please configure Grok Code Fast in xAI integration settings.")

            # Get configuration from subentry
            prompt = code_task_subentry.data.get(CONF_PROMPT, GROK_CODE_FAST_PROMPT)
            model = code_task_subentry.data.get(CONF_CHAT_MODEL, "grok-code-fast-1")
            temperature = float(code_task_subentry.data.get(CONF_TEMPERATURE, 0.1))
            max_tokens = int(code_task_subentry.data.get(CONF_MAX_TOKENS, 4000))
            store_messages = bool(code_task_subentry.data.get(CONF_STORE_MESSAGES, True))

            # 3. Create xAI client and chat
            start_time = time.time()

            # Get API configuration from main entry
            api_key = entry.data.get(CONF_API_KEY)
            if not api_key:
                raise_generic_error("API key not configured")

            client = XAI_CLIENT_CLASS(api_key=api_key)

            chat = client.chat.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                store_messages=store_messages,
                previous_response_id=previous_response_id  # Continue conversation chain
            )

            # 4. Add system prompt
            # Server-side memory: only first message (when previous_response_id is None)
            # Client-side memory: always (model needs context every time)
            if not store_messages:
                LOGGER.debug(f"[Context: {context_id}] Client-side mode: Adding system prompt")
                chat.append(xai_system(prompt))
            elif not previous_response_id:
                LOGGER.debug(f"[Context: {context_id}] Server-side mode - FIRST MESSAGE: Adding system prompt")
                chat.append(xai_system(prompt))
            else:
                LOGGER.debug(f"[Context: {context_id}] Server-side mode - SUBSEQUENT MESSAGE: Skipping system prompt (using previous_response_id)")

            # 5. Load conversation history for client-side mode (when store_messages=False)
            if not store_messages and user_id:
                # NO server-side memory: load and send recent history manually
                chat_history = hass.data.get(DOMAIN, {}).get("chat_history")
                if chat_history:
                    # Load last N messages from chat history (10 turns = 20 messages)
                    limit = 20
                    messages = await chat_history.load_history(user_id, "code", limit)

                    LOGGER.debug(
                        f"[Context: {context_id}] Client-side mode: loading {len(messages)} messages from history"
                    )

                    # Append historical messages to chat
                    for msg in messages:
                        if msg["role"] == "user":
                            chat.append(xai_user(msg["content"]))
                        elif msg["role"] == "assistant":
                            # Parse JSON if it's a stored code response
                            try:
                                parsed = json.loads(msg["content"])
                                # Reconstruct assistant message from stored response
                                response_text = parsed.get("response_text", "")
                                response_code = parsed.get("response_code", "")
                                # Combine text and code for context
                                assistant_content = response_text
                                if response_code:
                                    assistant_content += f"\n\n```\n{response_code}\n```"
                                chat.append(xai_assistant(assistant_content))
                            except json.JSONDecodeError:
                                # Fallback: use content as-is (shouldn't happen with code mode)
                                chat.append(xai_assistant(msg["content"]))
                                LOGGER.warning(
                                    f"[Context: {context_id}] Failed to parse assistant message as JSON, using raw content"
                                )
                else:
                    LOGGER.warning(f"[Context: {context_id}] Client-side mode but chat_history service not available")

            # 6. Build user message
            user_message = user_prompt

            # Include current code if modified
            if current_code:
                user_message += f"\n\nCurrent code:\n```\n{current_code}\n```"

            # Handle attachments (text files already read by frontend)
            if attachments:
                for att in attachments:
                    try:
                        # Frontend sends: {filename: "file.py", content: "text content"}
                        att_name = att.get("filename", att.get("name", "file"))
                        att_content = att.get("content", att.get("data", ""))

                        if not att_content:
                            LOGGER.warning(f"[Context: {context_id}] Attachment %s has no content", att_name)
                            continue

                        # Check if content is base64-encoded (for backward compatibility)
                        # Frontend sends plain text, but handle base64 if present
                        try:
                            if isinstance(att_content, str) and len(att_content) > 0:
                                # Try to detect if it's base64 by attempting decode
                                # Base64 strings are typically longer and have specific chars
                                if att_content.startswith("data:") or (len(att_content) % 4 == 0 and all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=" for c in att_content[:100])):
                                    content = base64.b64decode(att_content).decode('utf-8')
                                else:
                                    # Plain text content from frontend
                                    content = att_content
                            else:
                                content = att_content
                        except Exception:
                            # If decode fails, assume plain text
                            content = att_content

                        LOGGER.debug(f"[Context: {context_id}] Processing attachment: {att_name}, content_length={len(content)}")
                        user_message += f"\n\nFile: {att_name}\n```\n{content}\n```"
                    except Exception as e:
                        LOGGER.warning(f"[Context: {context_id}] Failed to process attachment %s: %s", att.get("filename", att.get("name")), e)

            chat.append(xai_user(user_message))

            # Save RAW user message to chat history (fire-and-forget, async)
            if user_id:
                chat_history = hass.data.get(DOMAIN, {}).get("chat_history")
                if chat_history:
                    # Save the complete user message as sent to xAI
                    chat_history.save_message_async(user_id, "code", "user", user_message)

            LOGGER.debug(f"[Context: {context_id}] Calling xAI API with model '{model}'.")

            # 6. Call xAI API
            response = await hass.async_add_executor_job(lambda: chat.sample())
            content = getattr(response, "content", "")
            response_id = getattr(response, "id", None)
            usage = getattr(response, "usage", None)

            # Convert content to string if it's a dict (xAI SDK may return dict or string)
            if isinstance(content, dict):
                content = json.dumps(content)

            elapsed = time.time() - start_time
            LOGGER.info(
                f"[Context: {context_id}] Grok Code Fast: response_id={response_id}, elapsed={elapsed:.2f}s"
            )

            # Update token usage sensors (only code_fast sensors)
            if usage:
                sensors = hass.data.get(DOMAIN, {}).get(f"{entry.entry_id}_sensors")
                if sensors:
                    LOGGER.debug(f"[Context: {context_id}] Updating code_fast token sensors with usage data for model={model}")
                    for sensor in sensors:
                        try:
                            # Filter: only update code_fast sensors
                            sensor_service = getattr(sensor, "_service_type", None)
                            if sensor_service != "code_fast":
                                continue

                            sensor.update_token_usage(usage, model)
                            LOGGER.debug(f"[Context: {context_id}] Updated sensor {getattr(sensor, 'entity_id', 'unknown')}")
                        except Exception as err:
                            LOGGER.error(f"[Context: {context_id}] Failed to update sensor: %s", err)
                else:
                    LOGGER.warning(f"[Context: {context_id}] No token sensors found for entry {entry.entry_id}")
            else:
                LOGGER.warning(f"[Context: {context_id}] No usage data in response")

            # 7. Save response_id using ConversationMemory for cache recovery
            # This allows frontend to recover conversation if localStorage is cleared
            if response_id and store_messages:
                # Get user_id from request_data (passed by frontend)
                user_id = request_data.get("user_id")
                if not user_id:
                    LOGGER.warning(f"[Context: {context_id}] No user_id provided, cannot save conversation")
                else:
                    await code_memory.save_response_id(user_id, "code", response_id)
                    LOGGER.debug(f"[Context: {context_id}] Saved response_id={response_id[:8]} for user={user_id[:8]} mode=code")

            # 8. Parse response
            LOGGER.debug(f"[Context: {context_id}] Raw xAI response content (full): %s", content)
            response_text, response_code = parse_grok_response(content)
            LOGGER.debug(
                f"[Context: {context_id}] Parsed - text_length={len(response_text)}, code_length={len(response_code)}"
            )
            if response_code:
                LOGGER.debug(f"[Context: {context_id}] Response code (first 200 chars): %s", response_code[:200])

            # Save PARSED assistant response to chat history (fire-and-forget, async)
            # Save as JSON with response_text and response_code for full recovery
            if user_id:
                chat_history = hass.data.get(DOMAIN, {}).get("chat_history")
                if chat_history:
                    assistant_json = json.dumps({
                        "response_text": response_text,
                        "response_code": response_code
                    })
                    chat_history.save_message_async(user_id, "code", "assistant", assistant_json)

            # 9. Fire event (for backwards compatibility if frontend listens)
            hass.bus.async_fire(
                "xai_conversation.grok_code_fast_response",
                {
                    "prompt": user_prompt,
                    "response_text": response_text,
                    "response_code": response_code,
                    "timestamp": datetime.now().isoformat(),
                },
            )

            # 10. Return structured response
            return {
                "response_text": response_text,
                "response_code": response_code,
                "previous_response_id": response_id,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as err:
            LOGGER.error(f"[Context: {context_id}] Error in Grok Code Fast service: %s", err, exc_info=True)
            return {"error": str(err)}

    # Register the service
    hass.services.async_register(
        DOMAIN,
        "grok_code_fast",
        handle_home_assistant_code_editor,
        supports_response=SupportsResponse.ONLY,
    )

    async def handle_clear_memory(call: HA_ServiceCall) -> HA_ServiceResponse:
        """Clear persistent conversation memory for the xAI conversation entity."""
        # Find the xAI conversation entity
        ent_reg = ha_entity_registry.async_get(hass)
        xai_entities = [
            e for e in ent_reg.entities.values()
            if e.platform == DOMAIN and e.domain == "conversation"
        ]

        if not xai_entities:
            raise_validation_error("No xAI conversation entity found")

        target_entity = hass.data.get("entity_components", {}).get("conversation").get_entity(xai_entities[0].entity_id)
        if not target_entity or not hasattr(target_entity, "async_clear_memory"):
            raise_generic_error("Target entity not found or does not support memory clearing")

        # Check if physical delete requested
        delete_storage_file = call.data.get("delete_storage_file", False)
        if delete_storage_file:
            await target_entity.async_clear_memory(clear_all=True, scope=None, target_id=None, physical_delete=True)
            return {"status": "ok", "message": "Memory storage file physically deleted"}

        # Get person entity IDs and convert to user IDs
        person_entity_ids = call.data.get("user_id", [])
        if isinstance(person_entity_ids, str):
            person_entity_ids = [person_entity_ids] if person_entity_ids else []

        user_ids = []
        user_names = []
        for person_entity_id in person_entity_ids:
            person_state = hass.states.get(person_entity_id)
            if person_state and person_state.attributes.get("user_id"):
                user_ids.append(person_state.attributes["user_id"])
                user_names.append(person_state.attributes.get("friendly_name") or person_state.name)

        # Get satellite entity IDs and extract device IDs
        satellite_entity_ids = _parse_id_list(call.data.get("satellite_id", []))
        satellite_device_ids = []
        satellite_names = []

        for satellite_entity_id in satellite_entity_ids:
            satellite_state = hass.states.get(satellite_entity_id)
            if satellite_state:
                # Get device_id from entity registry
                ent_reg = ha_entity_registry.async_get(hass)
                entity_entry = ent_reg.async_get(satellite_entity_id)
                if entity_entry and entity_entry.device_id:
                    satellite_device_ids.append(entity_entry.device_id)
                    satellite_names.append(satellite_state.attributes.get("friendly_name") or satellite_state.name)

        # Must specify either users, satellites, or physical delete
        if not user_ids and not satellite_device_ids:
            raise_validation_error("Must select users, satellites, or enable 'Delete Storage File'")

        # Clear specific users/satellites
        cleared_items = []
        for uid, name in zip(user_ids, user_names):
            await target_entity.async_clear_memory(clear_all=False, scope="user", target_id=uid)
            cleared_items.append(f"user:{name}")

        for did, name in zip(satellite_device_ids, satellite_names):
            await target_entity.async_clear_memory(clear_all=False, scope="device", target_id=did)
            cleared_items.append(f"satellite:{name}")

        return {"status": "ok", "message": f"Memory cleared for: {', '.join(cleared_items)}"}

    async def handle_clear_code_memory(call: HA_ServiceCall) -> HA_ServiceResponse:
        """Clear Grok Code Fast conversation memory for specified user.

        This service clears only the mode:code entries from the memory file.
        Used when user clicks "Clear chat" in the Grok Code Fast card.
        """
        # Get user_id from service call data first, fallback to context
        user_id = call.data.get("user_id")
        if not user_id and call.context.user_id:
            user_id = call.context.user_id
            LOGGER.debug(f"clear_code_memory: Using user_id from context: {user_id}")

        if not user_id:
            raise_validation_error("user_id is required (not found in data or context)")

        # Clear memory using ConversationMemory
        await code_memory.clear_memory(user_id, "code")

        # Clear chat history too
        chat_history = hass.data.get(DOMAIN, {}).get("chat_history")
        if chat_history:
            await chat_history.clear_history(user_id, "code")

        LOGGER.info(f"Cleared code conversation memory and chat history for user {user_id[:8]}")
        return {
            "status": "ok",
            "message": "Code conversation memory and chat history cleared"
        }

    async def handle_sync_chat_history(call: HA_ServiceCall) -> HA_ServiceResponse:
        """Sync chat history from server for the current user.

        Loads the chat history for the specified mode and returns it to frontend.
        Frontend can use this to restore chat UI state across devices.
        """
        # Get user_id from service call data first, fallback to context
        user_id = call.data.get("user_id")
        if not user_id and call.context.user_id:
            user_id = call.context.user_id
            LOGGER.debug(f"sync_chat_history: Using user_id from context: {user_id}")

        if not user_id:
            raise_validation_error("user_id is required (not found in data or context)")

        # Get mode (default: "code")
        mode = call.data.get("mode", "code")

        # Get optional limit
        limit = call.data.get("limit", 50)

        # Load chat history
        chat_history = hass.data.get(DOMAIN, {}).get("chat_history")
        if not chat_history:
            raise_generic_error("Chat history service not initialized")

        messages = await chat_history.load_history(user_id, mode, limit)

        LOGGER.info(f"Synced chat history for user {user_id[:8]}, mode={mode}, messages={len(messages)}")
        return {
            "status": "ok",
            "messages": messages,
            "count": len(messages)
        }

    async def handle_list_users(call: HA_ServiceCall) -> HA_ServiceResponse:
        """List all Home Assistant users with their IDs and names from Person entities."""
        users_list = []

        # Get all HA users
        for user in hass.auth.async_get_users():
            if not user.system_generated and user.is_active:
                # Try to get friendly name from Person entity
                person_name = None
                for person_entity_id in hass.states.async_entity_ids("person"):
                    person_state = hass.states.get(person_entity_id)
                    if person_state and person_state.attributes.get("user_id") == user.id:
                        person_name = person_state.attributes.get("friendly_name") or person_state.name
                        break

                # Build user info
                display_name = person_name or user.name or "Unknown"
                users_list.append({
                    "user_id": user.id,
                    "name": display_name,
                    "username": user.name,
                    "has_person": bool(person_name),
                })

        return {
            "users": users_list,
            "count": len(users_list),
        }

    # Register services
    hass.services.async_register(
        DOMAIN,
        "clear_memory",
        handle_clear_memory,
        supports_response=SupportsResponse.OPTIONAL,
    )

    hass.services.async_register(
        DOMAIN,
        "clear_code_memory",
        handle_clear_code_memory,
        supports_response=SupportsResponse.OPTIONAL,
    )

    hass.services.async_register(
        DOMAIN,
        "sync_chat_history",
        handle_sync_chat_history,
        supports_response=SupportsResponse.ONLY,
    )

    # Note: list_users function available internally but not registered as a public service

    async def handle_reset_token_stats(call: HA_ServiceCall) -> HA_ServiceResponse:
        """Reset token statistics for xAI conversation integration.
        Behavior:
        - Resets all 10 token sensors (6 per-service + 4 aggregated) for the integration.
        - Uses sensors stored in hass.data for direct access (includes code_fast service).
        """
        # Get all xAI config entries
        config_entries = hass.config_entries.async_entries(DOMAIN)

        if not config_entries:
            raise_validation_error("No xAI Conversation integration entries found")

        sensors_reset = 0

        # Reset sensors for each config entry
        for entry in config_entries:
            # Get sensors from hass.data (all 10 sensors including code_fast)
            sensors = hass.data.get(DOMAIN, {}).get(f"{entry.entry_id}_sensors")

            if not sensors:
                LOGGER.debug("reset_token_stats: no sensors found for entry %s", entry.entry_id)
                continue

            # Reset all sensors (per-service + aggregated)
            for sensor in sensors:
                try:
                    sensor.reset_statistics()
                    LOGGER.debug("reset_token_stats: reset sensor %s", sensor.entity_id)
                    sensors_reset += 1
                except Exception as err:
                    LOGGER.error("reset_token_stats: failed to reset sensor %s: %s",
                               getattr(sensor, "entity_id", "unknown"), err)

        if sensors_reset == 0:
            LOGGER.warning("reset_token_stats: no token sensors found to reset")
            return {"status": "ok", "sensors_reset": 0, "message": "No token sensors found"}

        return {
            "status": "ok",
            "sensors_reset": sensors_reset,
            "message": f"Successfully reset {sensors_reset} token sensors"
        }

    # Register reset_token_stats service
    hass.services.async_register(
        DOMAIN,
        "reset_token_stats",
        handle_reset_token_stats,
        supports_response=SupportsResponse.OPTIONAL,
    )