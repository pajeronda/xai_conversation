"""Integration initialization and lifecycle manager for xAI Conversation."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path
import shutil
from types import MappingProxyType
from typing import TYPE_CHECKING, Callable

from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import (
    EVENT_HOMEASSISTANT_STARTED,
    EVENT_HOMEASSISTANT_STOP,
)
from homeassistant.core import Event
from homeassistant.helpers import (
    device_registry as ha_device_registry,
    entity_registry as ha_entity_registry,
    restore_state,
)
from homeassistant.helpers.event import async_track_time_interval

from .const import (
    CONF_MEMORY_CLEANUP_INTERVAL_HOURS,
    CONF_MEMORY_REMOTE_DELETE,
    CONF_PRICING_UPDATE_INTERVAL_HOURS,
    DEFAULT_AI_TASK_NAME,
    DEFAULT_CONVERSATION_NAME,
    DEFAULT_SENSORS_NAME,
    DOMAIN,
    LOGGER,
    MEMORY_DEFAULTS,
    MEMORY_FLUSH_INTERVAL_MINUTES,
    RECOMMENDED_AI_TASK_OPTIONS,
    RECOMMENDED_MEMORY_CLEANUP_INTERVAL_HOURS,
    RECOMMENDED_PIPELINE_OPTIONS,
    RECOMMENDED_PRICING_UPDATE_INTERVAL_HOURS,
    RECOMMENDED_SENSORS_OPTIONS,
    XAIConfigEntry,
)
from .helpers import (
    MemoryManager,
    TokenStats,
    XAIModelManager,
)
from .xai_gateway import XAIGateway
from .services import unregister_services


if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant


class XaiInitManager:
    """Manages initialization and lifecycle of the xAI Conversation integration."""

    def __init__(self, hass: HomeAssistant, entry: XAIConfigEntry) -> None:
        """Initialize the coordinator."""
        self.hass = hass
        self.entry = entry
        self._unload_listeners = []
        self.model_manager: XAIModelManager | None = None
        self.gateway = XAIGateway(hass, entry)

    async def _async_update_models_with_retry(
        self,
        max_retries: int = 3,
        context: str = "initial",
    ) -> bool:
        """Update models from API with exponential backoff retry logic.

        Args:
            max_retries: Maximum number of retry attempts
            context: Context string for logging

        Returns:
            True if successful, False otherwise
        """
        for retry_count in range(max_retries):
            try:
                await self.gateway.async_update_models()
                await self._async_cleanup_orphaned_pricing_sensors()
                return True
            except Exception as err:
                retry_count_actual = retry_count + 1
                if retry_count_actual < max_retries:
                    LOGGER.debug(
                        "Coordinator: Model fetch retry %d/%d failed: %s",
                        retry_count_actual,
                        max_retries,
                        err,
                    )
                    await asyncio.sleep(2**retry_count_actual)  # Exponential backoff
                else:
                    LOGGER.error(
                        "Coordinator: Model fetch on %s failed after %d retries: %s",
                        context,
                        max_retries,
                        err,
                    )
                    return False

    async def async_setup(self) -> None:
        """Set up the integration."""
        # Initialize placeholder for models data
        if "xai_models_data" not in self.hass.data[DOMAIN]:
            self.hass.data[DOMAIN]["xai_models_data"] = {}
            self.hass.data[DOMAIN]["xai_models_data_timestamp"] = 0

        # Create global storage instances
        folder_name = DEFAULT_CONVERSATION_NAME.lower().replace(" ", "_")
        memory_path = f"{folder_name}/{DOMAIN}.memory"
        token_stats_path = f"{folder_name}/{DOMAIN}.token_stats"

        self.hass.data[DOMAIN]["conversation_memory"] = MemoryManager(
            self.hass, memory_path, self.entry
        )
        self.hass.data[DOMAIN]["token_stats"] = TokenStats(
            self.hass, token_stats_path, self.entry
        )
        LOGGER.info(
            "Setup: initialized storage (memory=%s, stats=%s)",
            memory_path,
            token_stats_path,
        )

        # Initialize Model Manager
        self.model_manager = XAIModelManager(self.hass)
        self.hass.data[DOMAIN]["model_manager"] = self.model_manager

        # Perform initial model fetch (non-blocking, will retry on startup if needed)
        await self._async_update_models_with_retry(context="initial")

        # Run setup tasks (idempotent, safe to run on every startup)
        await self._async_add_subentries_if_needed()

        # FINAL SYNC: Ensure self.entry is updated after all subentry modifications
        self.entry = self.hass.config_entries.async_get_entry(self.entry.entry_id)

        # Clean up orphaned turn sensor devices/entities before platforms load
        self._cleanup_orphaned_turn_sensor_devices()

        # Set up listeners and periodic tasks (will raise on failure)
        try:
            self._setup_listeners()
            LOGGER.info("Setup: completed successfully")
        except Exception as err:
            LOGGER.error("Setup: failed to register listeners - %s", err, exc_info=True)
            raise

    def _async_remove_registered_entities(
        self,
        filter_func: Callable[[ha_entity_registry.RegistryEntry], bool],
        label: str,
    ) -> int:
        """Helper to remove entities from registry based on a filter.

        Returns:
            Number of successfully removed entities.
        """
        ent_reg = ha_entity_registry.async_get(self.hass)
        to_remove = [
            entity.entity_id
            for entity in ent_reg.entities.values()
            if entity.config_entry_id == self.entry.entry_id and filter_func(entity)
        ]

        if not to_remove:
            return 0

        removed_count = 0
        for entity_id in to_remove:
            try:
                ent_reg.async_remove(entity_id)
                removed_count += 1
            except Exception as err:
                LOGGER.warning(
                    "Cleanup: failed to remove %s %s - %s", label, entity_id, err
                )

        if removed_count > 0:
            LOGGER.info("Cleanup: removed %d %s entities", removed_count, label)
        return removed_count

    async def _async_cleanup_orphaned_pricing_sensors(self) -> None:
        """Remove pricing sensors for discontinued models or zero-price types."""
        xai_models_data = self.hass.data[DOMAIN].get("xai_models_data", {})
        if not xai_models_data:
            return

        pricing_suffixes = (
            "input_price",
            "output_price",
            "cached_input_price",
            "input_image_price",
            "search_price",
        )

        def is_orphaned_pricing_sensor(
            entity: ha_entity_registry.RegistryEntry,
        ) -> bool:
            uid = entity.unique_id
            suffix = next((s for s in pricing_suffixes if uid.endswith(s)), None)
            if not suffix:
                return False

            prefix = f"{self.entry.entry_id}_"
            if not uid.startswith(prefix):
                return False

            model_name = uid[len(prefix) : -(len(suffix) + 1)]

            # Model gone or price type is zero
            model_data = xai_models_data.get(model_name)
            if not model_data:
                return True
            return model_data.get(suffix, 0.0) <= 0

        self._async_remove_registered_entities(
            is_orphaned_pricing_sensor, "orphaned pricing"
        )

    def _cleanup_orphaned_turn_sensor_devices(self) -> None:
        """Migrate turn sensor entities and remove orphaned per-user/per-device devices.

        Old code registered turn sensors with separate per-user/per-device devices
        and without a subentry. Now they share the xAI Notifications device.
        Must run BEFORE async_forward_entry_setups.
        """
        # Find the sensors subentry
        sensors_subentry_id: str | None = None
        for se in self.entry.subentries.values():
            if se.subentry_type == "sensors":
                sensors_subentry_id = se.subentry_id
                break

        if sensors_subentry_id is None:
            return

        # 1. Fix config_subentry_id for turn sensor entities without one
        ent_reg = ha_entity_registry.async_get(self.hass)
        for ent in ha_entity_registry.async_entries_for_config_entry(
            ent_reg, self.entry.entry_id
        ):
            if ent.config_subentry_id is not None:
                continue
            if ent.unique_id and "_turns_" in ent.unique_id:
                LOGGER.debug(
                    "Migrating turn sensor entity %s to subentry %s",
                    ent.entity_id,
                    sensors_subentry_id,
                )
                ent_reg.async_update_entity(
                    ent.entity_id,
                    config_subentry_id=sensors_subentry_id,
                )

        # 2. Remove old per-user/per-device devices (no longer needed)
        dev_reg = ha_device_registry.async_get(self.hass)
        for device in list(
            ha_device_registry.async_entries_for_config_entry(
                dev_reg, self.entry.entry_id
            )
        ):
            if not device.name:
                continue
            if device.name.startswith("User: ") or device.name.startswith("Device: "):
                LOGGER.debug(
                    "Removing orphaned turn sensor device: %s (%s)",
                    device.name,
                    device.id,
                )
                dev_reg.async_remove_device(device.id)

    async def clean_deprecated_subentries(self) -> None:
        """Clean up deprecated subentries and migrate valid types.

        This method:
        - Converts "ai_task_data" → "ai_task"
        - Removes subentries with invalid types (not in VALID_SUBENTRY_TYPES)
        - Can be called from migration or runtime cleanup
        """
        current_entry = self.hass.config_entries.async_get_entry(self.entry.entry_id)
        if not current_entry:
            return

        valid_types = (
            "conversation",
            "ai_task",
            "sensors",
        )

        for subentry_id, subentry in list(current_entry.subentries.items()):
            stype = subentry.subentry_type

            # Migrate ai_task_data → ai_task
            if stype == "ai_task_data":
                self.hass.config_entries.async_remove_subentry(
                    current_entry, subentry_id
                )
                current_entry = self.hass.config_entries.async_get_entry(
                    self.entry.entry_id
                )
                self.hass.config_entries.async_add_subentry(
                    current_entry,
                    ConfigSubentry(
                        subentry_id=subentry_id,
                        subentry_type="ai_task",
                        title=subentry.title,
                        data=dict(subentry.data),
                        unique_id=subentry.unique_id,
                    ),
                )
                current_entry = self.hass.config_entries.async_get_entry(
                    self.entry.entry_id
                )
                LOGGER.info(
                    "Cleaned subentry '%s': ai_task_data → ai_task", subentry.title
                )

            # Remove invalid types
            elif stype not in valid_types:
                self.hass.config_entries.async_remove_subentry(
                    current_entry, subentry_id
                )
                current_entry = self.hass.config_entries.async_get_entry(
                    self.entry.entry_id
                )
                LOGGER.info(
                    "Removed invalid subentry '%s' (type: %s)", subentry.title, stype
                )

    async def clean_deprecated_entities(self) -> None:
        """Remove entities with deprecated patterns from entity registry.

        Pattern-based cleanup using unique_id matching.
        Can be called from migration or runtime cleanup.
        """
        deprecated_patterns = ("code_fast", "code_task", "grok_code_fast")

        def is_deprecated_entity(entity: ha_entity_registry.RegistryEntry) -> bool:
            return any(pattern in entity.unique_id for pattern in deprecated_patterns)

        removed = self._async_remove_registered_entities(
            is_deprecated_entity, "deprecated entities"
        )
        if removed > 0:
            LOGGER.info("Cleaned %d deprecated entities from registry", removed)

    async def clean_deprecated_storage(self) -> None:
        """Remove deprecated storage files.

        Can be called from migration or runtime cleanup.
        """
        folder_name = DEFAULT_CONVERSATION_NAME.lower().replace(" ", "_")
        storage_base = Path(self.hass.config.path(".storage"))

        # List of deprecated storage files (filename only, without DOMAIN prefix)
        deprecated_files = ("chat_history", "memory")

        for filename in deprecated_files:
            file_path = storage_base / folder_name / f"{DOMAIN}.{filename}"
            if file_path.exists():
                try:
                    await self.hass.async_add_executor_job(file_path.unlink)
                    LOGGER.info("Removed deprecated storage file: %s", filename)
                except Exception as err:
                    LOGGER.warning("Failed to remove file %s: %s", filename, err)

    async def migrate_memory_v2_2(self) -> None:
        """Migrate memory to V2.2 format with subentry_id in keys.

        Deletes memory file to force new keys with subentry_id isolation.
        """
        folder_name = DEFAULT_CONVERSATION_NAME.lower().replace(" ", "_")
        storage_base = Path(self.hass.config.path(".storage"))
        memory_file = storage_base / folder_name / f"{DOMAIN}.memory"

        if memory_file.exists():
            try:
                await self.hass.async_add_executor_job(memory_file.unlink)
                LOGGER.info(
                    "Migration V2.2: reset conversation memory for subentry isolation"
                )
            except Exception as err:
                LOGGER.warning("Migration V2.2: failed to reset memory - %s", err)

    async def ensure_valid_config_keys(self) -> None:
        """Ensure entry.data and subentry.data contain only valid keys.

        Removes obsolete keys and adds missing defaults based on RECOMMENDED_*_OPTIONS.
        Can be called from migration or runtime cleanup.
        """
        # 1. Clean entry.data
        new_data = {}

        # Always keep API credentials
        for key in ("api_key", "api_host"):
            if key in self.entry.data:
                new_data[key] = self.entry.data[key]

        # Keep valid memory defaults
        for key, default in MEMORY_DEFAULTS.items():
            new_data[key] = self.entry.data.get(key, default)

        # Keep extended tools config
        for key in ("use_extended_tools", "extended_tools_yaml"):
            if key in self.entry.data:
                new_data[key] = self.entry.data[key]

        # Update if changed
        if new_data != self.entry.data:
            self.hass.config_entries.async_update_entry(self.entry, data=new_data)

        # 2. Clean subentry.data
        current_entry = self.hass.config_entries.async_get_entry(self.entry.entry_id)
        if not current_entry:
            return

        for subentry_id, subentry in current_entry.subentries.items():
            stype = subentry.subentry_type

            # Select appropriate defaults
            if stype == "conversation":
                defaults = RECOMMENDED_PIPELINE_OPTIONS
            elif stype == "ai_task":
                defaults = RECOMMENDED_AI_TASK_OPTIONS
            elif stype == "sensors":
                defaults = RECOMMENDED_SENSORS_OPTIONS
            else:
                continue

            # Keep only valid keys, add missing defaults
            new_subdata = {}
            for key in defaults:
                new_subdata[key] = subentry.data.get(key, defaults[key])

            # Update if changed
            if new_subdata != subentry.data:
                self.hass.config_entries.async_update_subentry(
                    current_entry, subentry, data=new_subdata
                )

    async def _async_add_subentries_if_needed(self) -> None:
        """Add subentries if they don't exist."""
        # Refresh current state from registry
        current_entry = self.hass.config_entries.async_get_entry(self.entry.entry_id)
        if not current_entry:
            return

        subentries = {se.subentry_type for se in current_entry.subentries.values()}

        if "conversation" not in subentries:
            LOGGER.info("Configuring default conversation subentry")
            self.hass.config_entries.async_add_subentry(
                current_entry,
                ConfigSubentry(
                    data=MappingProxyType(
                        {
                            "name": DEFAULT_CONVERSATION_NAME,
                            **RECOMMENDED_PIPELINE_OPTIONS,
                        }
                    ),
                    subentry_type="conversation",
                    title=DEFAULT_CONVERSATION_NAME,
                    unique_id=f"{DOMAIN}:conversation",
                ),
            )
            current_entry = self.hass.config_entries.async_get_entry(
                self.entry.entry_id
            )
            subentries = {se.subentry_type for se in current_entry.subentries.values()}

        if "ai_task" not in subentries:
            LOGGER.info("Configuring default AI Task subentry")
            self.hass.config_entries.async_add_subentry(
                current_entry,
                ConfigSubentry(
                    data=MappingProxyType(
                        {"name": DEFAULT_AI_TASK_NAME, **RECOMMENDED_AI_TASK_OPTIONS}
                    ),
                    subentry_type="ai_task",
                    title=DEFAULT_AI_TASK_NAME,
                    unique_id=f"{DOMAIN}:ai_task",
                ),
            )
            current_entry = self.hass.config_entries.async_get_entry(
                self.entry.entry_id
            )
            subentries = {se.subentry_type for se in current_entry.subentries.values()}

        if "sensors" not in subentries:
            LOGGER.info("Configuring default sensors subentry")
            self.hass.config_entries.async_add_subentry(
                current_entry,
                ConfigSubentry(
                    data=MappingProxyType(
                        {"name": DEFAULT_SENSORS_NAME, **RECOMMENDED_SENSORS_OPTIONS}
                    ),
                    subentry_type="sensors",
                    title=DEFAULT_SENSORS_NAME,
                    unique_id=f"{DOMAIN}:sensors",
                ),
            )
            # Last sync, current_entry is updated for callers if needed
            current_entry = self.hass.config_entries.async_get_entry(
                self.entry.entry_id
            )

    def _setup_listeners(self) -> None:
        """Set up listeners and periodic tasks.

        Registers all event listeners and periodic timers. Failure to register any task
        will raise an exception, preventing partial initialization.
        """
        listeners_to_register = []

        try:
            # 1. Options update listener (always active until unload)
            update_listener = self.entry.add_update_listener(
                lambda hass, entry: self.hass.config_entries.async_reload(
                    entry.entry_id
                )
            )
            listeners_to_register.append(("update_listener", update_listener))

            # 2. Startup listener for model fetch retry (onetime, auto-removes after firing)
            async def async_initial_update(event: Event) -> None:
                """Retry model data fetch when Home Assistant is fully started."""
                if not self.hass.data[DOMAIN].get("xai_models_data"):
                    await self._async_update_models_with_retry(context="startup")

            self.hass.bus.async_listen_once(
                EVENT_HOMEASSISTANT_STARTED, async_initial_update
            )
            # Note: onetime listeners auto-remove after firing, so we don't track them for cleanup

            # 3. Periodic memory cleanup
            cleanup_interval_hours = self.entry.data.get(
                CONF_MEMORY_CLEANUP_INTERVAL_HOURS,
                RECOMMENDED_MEMORY_CLEANUP_INTERVAL_HOURS,
            )
            memory = self.hass.data[DOMAIN]["conversation_memory"]
            remote_delete_enabled = self.entry.data.get(
                CONF_MEMORY_REMOTE_DELETE, False
            )

            async def _handle_remote_cleanup(deleted_ids: list[str]) -> None:
                # 3. Remote deletion from xAI servers (if enabled)
                if remote_delete_enabled and deleted_ids:
                    deleted = await self.gateway.async_delete_remote_completions(
                        deleted_ids
                    )
                    if deleted > 0:
                        LOGGER.info(
                            "Periodic cleanup: %d IDs removed from xAI server",
                            deleted,
                        )

            cleanup_unsub = memory.setup_periodic_cleanup(
                cleanup_interval_hours, on_cleanup=_handle_remote_cleanup
            )

            listeners_to_register.append(("memory_cleanup", cleanup_unsub))

            # 4. Periodic memory/stats flush (write-behind cache)
            token_stats = self.hass.data[DOMAIN]["token_stats"]

            async def async_flush_all(now=None):
                """Flush both memory and token stats to storage."""
                try:
                    await memory.async_flush()
                    await token_stats.async_flush()
                except Exception as err:
                    LOGGER.error("Periodic flush failed: %s", err, exc_info=True)

            flush_interval = timedelta(minutes=MEMORY_FLUSH_INTERVAL_MINUTES)
            flush_unsub = async_track_time_interval(
                self.hass, async_flush_all, flush_interval
            )
            listeners_to_register.append(("periodic_flush", flush_unsub))

            # 5. Final flush on Home Assistant stop (onetime, critical)
            async def _async_flush_on_stop(event):
                """Flush all data to storage before Home Assistant stops.

                This is critical to prevent data loss. Must succeed or log error loudly.
                """
                try:
                    await async_flush_all()
                except Exception as err:
                    LOGGER.critical(
                        "CRITICAL - Final flush on stop failed: %s", err, exc_info=True
                    )

            self.hass.bus.async_listen_once(
                EVENT_HOMEASSISTANT_STOP, _async_flush_on_stop
            )
            # Note: onetime listeners auto-remove after firing, so we don't track them for cleanup

            # 6. Periodic pricing/model update
            pricing_interval = RECOMMENDED_PRICING_UPDATE_INTERVAL_HOURS
            for subentry in self.entry.subentries.values():
                if subentry.subentry_type == "sensors":
                    pricing_interval = subentry.data.get(
                        CONF_PRICING_UPDATE_INTERVAL_HOURS,
                        RECOMMENDED_PRICING_UPDATE_INTERVAL_HOURS,
                    )
                    break

            async def _async_update_pricing(now):
                """Periodically refresh model and pricing data from API."""
                try:
                    await self._async_update_models_with_retry(
                        max_retries=1,
                        context="periodic",
                    )
                except Exception as err:
                    LOGGER.error("Periodic model update failed: %s", err)

            pricing_unsub = async_track_time_interval(
                self.hass, _async_update_pricing, timedelta(hours=pricing_interval)
            )
            listeners_to_register.append(("periodic_pricing", pricing_unsub))

            # All listeners registered successfully, store for cleanup
            self._unload_listeners.extend([unsub for _, unsub in listeners_to_register])

        except Exception as err:
            LOGGER.error(
                "Coordinator: Failed to register listener: %s", err, exc_info=True
            )
            # Cleanup any listeners that were registered before failure
            for _, unsub in listeners_to_register:
                try:
                    unsub()
                except Exception as cleanup_err:
                    LOGGER.warning(
                        "Coordinator: Error cleaning up listener after failure: %s",
                        cleanup_err,
                    )
            raise

    async def async_unload(self) -> bool:
        """Unload the integration.

        Ensures all listeners are unsubscribed, all data is flushed, and pending tasks complete.
        """
        # Step 1: Unsubscribe from all persistent listeners
        for unsub in self._unload_listeners:
            try:
                unsub()
            except (ValueError, RuntimeError):
                # Listener already removed or subscription error - ignore
                pass

        self._unload_listeners.clear()

        # Step 2: Explicitly flush all data before unload (critical)
        try:
            if "conversation_memory" in self.hass.data[DOMAIN]:
                await self.hass.data[DOMAIN]["conversation_memory"].async_flush()
            if "token_stats" in self.hass.data[DOMAIN]:
                await self.hass.data[DOMAIN]["token_stats"].async_flush()
        except Exception as err:
            LOGGER.error(
                "CRITICAL - Final flush on unload failed: %s", err, exc_info=True
            )

        # Step 3: Wait for any pending save tasks with timeout
        pending_tasks = self.hass.data.get(DOMAIN, {}).get("pending_save_tasks", set())
        if pending_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*pending_tasks, return_exceptions=True), timeout=30.0
                )
            except asyncio.TimeoutError:
                LOGGER.error(
                    "Unload: timeout waiting for %d pending save tasks",
                    len(pending_tasks),
                )
            except Exception as err:
                LOGGER.error(
                    "Unload: error waiting for pending save tasks - %s",
                    err,
                    exc_info=True,
                )

        # Step 4: Unregister services
        unregister_services(self.hass, self.entry.entry_id)

        # Step 5: Close Gateway (gRPC channels cleanup)
        await self.gateway.close()

        LOGGER.info("Unload: completed successfully")
        return True

    async def async_remove(self) -> None:
        """Remove the integration."""
        # Remove all entities from entity registry
        ent_reg = ha_entity_registry.async_get(self.hass)
        config_entry_ids = {self.entry.entry_id} | {
            sub.subentry_id for sub in self.entry.subentries.values()
        }
        entities_to_remove = [
            entity.entity_id
            for entity in ent_reg.entities.values()
            if entity.config_entry_id in config_entry_ids
        ]

        # Clean up restored state data
        restore_data = restore_state.async_get(self.hass)
        for entity_id in entities_to_remove:
            self.hass.states.async_remove(entity_id)
            if entity_id in restore_data.last_states:
                del restore_data.last_states[entity_id]
            ent_reg.async_remove(entity_id)

        if entities_to_remove:
            LOGGER.info("Removal: cleaned %d entities", len(entities_to_remove))

        # Clean up hass.data
        for key in list(self.hass.data[DOMAIN].keys()):
            if key.startswith(self.entry.entry_id):
                del self.hass.data[DOMAIN][key]

        # Remove storage folder
        folder_name = DEFAULT_CONVERSATION_NAME.lower().replace(" ", "_")
        storage_base = Path(self.hass.config.path(".storage"))
        memory_folder = storage_base / folder_name
        if memory_folder.exists() and memory_folder.is_dir():
            try:
                await self.hass.async_add_executor_job(shutil.rmtree, memory_folder)
                LOGGER.info("Removal: cleaned storage folder %s", memory_folder)
            except Exception as err:
                LOGGER.warning(
                    "Removal: failed to remove folder %s - %s", memory_folder, err
                )
