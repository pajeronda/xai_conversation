"""Integration initialization and lifecycle manager for xAI Conversation."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path
import shutil
from types import MappingProxyType
from typing import TYPE_CHECKING

from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import (
    EVENT_HOMEASSISTANT_STARTED,
    EVENT_HOMEASSISTANT_STOP,
)
from homeassistant.core import Event
from homeassistant.helpers import entity_registry as ha_entity_registry, restore_state
from homeassistant.helpers.event import async_track_time_interval

from .const import (
    CONF_MEMORY_CLEANUP_INTERVAL_HOURS,
    CONF_PRICING_UPDATE_INTERVAL_HOURS,
    DEFAULT_AI_TASK_NAME,
    DEFAULT_GROK_CODE_FAST_NAME,
    DEFAULT_CONVERSATION_NAME,
    DEFAULT_SENSORS_NAME,
    DOMAIN,
    LOGGER,
    MEMORY_DEFAULTS,
    MEMORY_FLUSH_INTERVAL_MINUTES,
    RECOMMENDED_AI_TASK_OPTIONS,
    RECOMMENDED_GROK_CODE_FAST_OPTIONS,
    RECOMMENDED_MEMORY_CLEANUP_INTERVAL_HOURS,
    RECOMMENDED_PIPELINE_OPTIONS,
    RECOMMENDED_PRICING_UPDATE_INTERVAL_HOURS,
    RECOMMENDED_SENSORS_OPTIONS,
    SUBENTRY_TYPE_AI_TASK,
    SUBENTRY_TYPE_CODE_TASK,
    SUBENTRY_TYPE_CONVERSATION,
    SUBENTRY_TYPE_SENSORS,
)
from .helpers import (
    MemoryManager,
    ChatHistoryService,
    TokenStats,
    XAIModelManager,
)
from .xai_gateway import XAIGateway
from .services import unregister_services


if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant
    from . import XAIConfigEntry


class XaiInitManager:
    """Manages initialization and lifecycle of the xAI Conversation integration."""

    def __init__(self, hass: HomeAssistant, entry: XAIConfigEntry) -> None:
        """Initialize the coordinator."""
        self.hass = hass
        self.entry = entry
        self._unload_listeners = []
        self.model_manager: XAIModelManager | None = None

    @staticmethod
    async def _async_update_models_with_retry(
        model_manager: XAIModelManager,
        gateway: XAIGateway,
        max_retries: int = 3,
        context: str = "initial",
    ) -> bool:
        """Update models from API with exponential backoff retry logic.

        Args:
            model_manager: XAIModelManager instance
            gateway: XAIGateway instance
            max_retries: Maximum number of retry attempts
            context: Context string for logging

        Returns:
            True if successful, False otherwise
        """
        for retry_count in range(max_retries):
            try:
                await model_manager.async_update_models(gateway)
                LOGGER.debug(f"Coordinator: Model data fetch on {context} succeeded")
                return True
            except Exception as err:
                retry_count_actual = retry_count + 1
                if retry_count_actual < max_retries:
                    LOGGER.debug(
                        f"Coordinator: Model fetch retry {retry_count_actual}/{max_retries} failed: %s",
                        err,
                    )
                    await asyncio.sleep(2**retry_count_actual)  # Exponential backoff
                else:
                    LOGGER.error(
                        f"Coordinator: Model fetch on {context} failed after {max_retries} retries: %s",
                        err,
                    )
                    return False

    async def async_setup(self) -> None:
        """Set up the integration."""
        LOGGER.debug("Coordinator: Starting setup")

        # Initialize placeholder for models data
        if "xai_models_data" not in self.hass.data[DOMAIN]:
            self.hass.data[DOMAIN]["xai_models_data"] = {}
            self.hass.data[DOMAIN]["xai_models_data_timestamp"] = 0

        # Create global storage instances
        folder_name = DEFAULT_CONVERSATION_NAME.lower().replace(" ", "_")
        memory_path = f"{folder_name}/{DOMAIN}.memory"
        chat_history_path = f"{folder_name}/{DOMAIN}.chat_history"
        token_stats_path = f"{folder_name}/{DOMAIN}.token_stats"

        self.hass.data[DOMAIN]["conversation_memory"] = MemoryManager(
            self.hass, memory_path, self.entry
        )
        self.hass.data[DOMAIN]["chat_history"] = ChatHistoryService(
            self.hass, chat_history_path, self.entry
        )
        self.hass.data[DOMAIN]["token_stats"] = TokenStats(
            self.hass, token_stats_path, self.entry
        )
        LOGGER.debug("Created global ConversationMemory instance at %s", memory_path)
        LOGGER.debug(
            "Created global ChatHistoryService instance at %s", chat_history_path
        )
        LOGGER.debug("Created global TokenStats instance at %s", token_stats_path)

        # Initialize Model Manager
        self.model_manager = XAIModelManager(self.hass)
        self.hass.data[DOMAIN]["model_manager"] = self.model_manager

        # Perform initial model fetch (non-blocking, will retry on startup if needed)
        # Create a temporary Gateway instance just for this operation
        gateway = XAIGateway(self.hass, self.entry)
        await self._async_update_models_with_retry(
            self.model_manager,
            gateway,
            context="initial",
        )

        # Run migration and setup tasks
        self._migrate_subentry_types()
        self._ensure_memory_params_in_entry_data()
        self._add_subentries_if_needed()

        # Set up listeners and periodic tasks (will raise on failure)
        try:
            self._setup_listeners()
            LOGGER.debug(
                "Coordinator: All listeners and periodic tasks registered successfully"
            )
        except Exception as err:
            LOGGER.error(
                "Coordinator: Failed to setup listeners: %s", err, exc_info=True
            )
            raise

    def _migrate_subentry_types(self) -> None:
        """Migrate old subentry_type values to new format."""
        needs_migration = any(
            sub.subentry_type in ("ai_task_data", "code_task")
            for sub in self.entry.subentries.values()
        )

        if needs_migration:
            LOGGER.info("Migrating old subentry_type values to new format")
            for subentry_id, subentry in list(self.entry.subentries.items()):
                new_type = None
                if subentry.subentry_type == "ai_task_data":
                    new_type = "ai_task"
                elif subentry.subentry_type == "code_task":
                    new_type = "code_fast"

                if new_type:
                    self.hass.config_entries.async_remove_subentry(
                        self.entry, subentry_id
                    )
                    self.hass.config_entries.async_add_subentry(
                        self.entry,
                        ConfigSubentry(
                            subentry_id=subentry_id,
                            subentry_type=new_type,
                            title=subentry.title,
                            data=dict(subentry.data),
                            unique_id=subentry.unique_id,
                        ),
                    )
            LOGGER.info("Subentry migration completed successfully")

    def _ensure_memory_params_in_entry_data(self) -> None:
        """Ensure memory parameters are in entry.data."""
        data = dict(self.entry.data)
        updated = False

        for key, default_value in MEMORY_DEFAULTS.items():
            if key not in data:
                data[key] = default_value
                updated = True

        if updated:
            self.hass.config_entries.async_update_entry(self.entry, data=data)
            LOGGER.info("Added memory parameters to integration configuration")

    def _add_subentries_if_needed(self) -> None:
        """Add subentries if they don't exist."""
        subentries = {se.subentry_type for se in self.entry.subentries.values()}

        if SUBENTRY_TYPE_CONVERSATION not in subentries:
            self.hass.config_entries.async_add_subentry(
                self.entry,
                ConfigSubentry(
                    data=MappingProxyType(
                        {
                            "name": DEFAULT_CONVERSATION_NAME,
                            **RECOMMENDED_PIPELINE_OPTIONS,
                        }
                    ),
                    subentry_type=SUBENTRY_TYPE_CONVERSATION,
                    title=DEFAULT_CONVERSATION_NAME,
                    unique_id=f"{DOMAIN}:conversation",
                ),
            )

        if SUBENTRY_TYPE_AI_TASK not in subentries:
            self.hass.config_entries.async_add_subentry(
                self.entry,
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
            self.hass.config_entries.async_add_subentry(
                self.entry,
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
            self.hass.config_entries.async_add_subentry(
                self.entry,
                ConfigSubentry(
                    data=MappingProxyType(
                        {"name": DEFAULT_SENSORS_NAME, **RECOMMENDED_SENSORS_OPTIONS}
                    ),
                    subentry_type=SUBENTRY_TYPE_SENSORS,
                    title=DEFAULT_SENSORS_NAME,
                    unique_id=f"{DOMAIN}:sensors",
                ),
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
            LOGGER.debug("Registered: options update listener")

            # 2. Startup listener for model fetch retry (onetime, auto-removes after firing)
            async def async_initial_update(event: Event) -> None:
                """Retry model data fetch when Home Assistant is fully started."""
                if not self.hass.data[DOMAIN].get("xai_models_data"):
                    LOGGER.debug(
                        "Coordinator: HA started, retrying xAI model data fetch..."
                    )
                    # Create temporary gateway for this operation
                    gateway = XAIGateway(self.hass, self.entry)
                    await self._async_update_models_with_retry(
                        self.model_manager,
                        gateway,
                        context="startup",
                    )

            self.hass.bus.async_listen_once(
                EVENT_HOMEASSISTANT_STARTED, async_initial_update
            )
            # Note: onetime listeners auto-remove after firing, so we don't track them for cleanup
            LOGGER.debug("Registered: startup model fetch retry (onetime)")

            # 3. Periodic memory cleanup
            cleanup_interval_hours = self.entry.data.get(
                CONF_MEMORY_CLEANUP_INTERVAL_HOURS,
                RECOMMENDED_MEMORY_CLEANUP_INTERVAL_HOURS,
            )
            memory = self.hass.data[DOMAIN]["conversation_memory"]
            cleanup_unsub = memory.setup_periodic_cleanup(cleanup_interval_hours)
            listeners_to_register.append(("memory_cleanup", cleanup_unsub))
            LOGGER.debug(
                "Registered: periodic memory cleanup (every %d hours)",
                cleanup_interval_hours,
            )

            # 4. Periodic memory/stats flush (write-behind cache)
            token_stats = self.hass.data[DOMAIN]["token_stats"]

            async def async_flush_all(now=None):
                """Flush both memory and token stats to storage."""
                try:
                    await memory.async_flush()
                    await token_stats.async_flush()
                    LOGGER.debug("Coordinator: Periodic flush completed successfully")
                except Exception as err:
                    LOGGER.error(
                        "Coordinator: Periodic flush failed: %s", err, exc_info=True
                    )

            flush_interval = timedelta(minutes=MEMORY_FLUSH_INTERVAL_MINUTES)
            flush_unsub = async_track_time_interval(
                self.hass, async_flush_all, flush_interval
            )
            listeners_to_register.append(("periodic_flush", flush_unsub))
            LOGGER.debug(
                "Registered: periodic flush (every %d minutes)",
                MEMORY_FLUSH_INTERVAL_MINUTES,
            )

            # 5. Final flush on Home Assistant stop (onetime, critical)
            async def _async_flush_on_stop(event):
                """Flush all data to storage before Home Assistant stops.

                This is critical to prevent data loss. Must succeed or log error loudly.
                """
                LOGGER.info("Coordinator: HA stopping, performing final data flush...")
                try:
                    await async_flush_all()
                    LOGGER.info(
                        "Coordinator: Final flush on stop completed successfully"
                    )
                except Exception as err:
                    LOGGER.critical(
                        "Coordinator: CRITICAL - Final flush on stop failed: %s",
                        err,
                        exc_info=True,
                    )

            self.hass.bus.async_listen_once(
                EVENT_HOMEASSISTANT_STOP, _async_flush_on_stop
            )
            # Note: onetime listeners auto-remove after firing, so we don't track them for cleanup
            LOGGER.debug("Registered: final flush on stop (onetime)")

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
                    # Create temporary gateway for this operation
                    gateway = XAIGateway(self.hass, self.entry)
                    await self._async_update_models_with_retry(
                        self.model_manager,
                        gateway,
                        max_retries=1,
                        context="periodic",
                    )
                except Exception as err:
                    LOGGER.error("Coordinator: Periodic model update failed: %s", err)

            pricing_unsub = async_track_time_interval(
                self.hass, _async_update_pricing, timedelta(hours=pricing_interval)
            )
            listeners_to_register.append(("periodic_pricing", pricing_unsub))
            LOGGER.debug(
                "Registered: periodic model update (every %d hours)", pricing_interval
            )

            # All listeners registered successfully, store for cleanup
            self._unload_listeners.extend([unsub for _, unsub in listeners_to_register])
            LOGGER.info(
                "Coordinator: All %d persistent listeners registered successfully",
                len(listeners_to_register),
            )

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
        LOGGER.debug(
            "Coordinator: Starting unload (unsubscribing %d listeners)",
            len(self._unload_listeners),
        )

        # Step 1: Unsubscribe from all persistent listeners
        unsubscribe_errors = []
        for idx, unsub in enumerate(self._unload_listeners):
            try:
                unsub()
            except (ValueError, RuntimeError) as err:
                # ValueError: Listener already removed (onetime listener that fired)
                # RuntimeError: Other subscription errors
                unsubscribe_errors.append((idx, str(err)))

        if unsubscribe_errors:
            LOGGER.debug(
                "Coordinator: %d listeners already removed or had errors",
                len(unsubscribe_errors),
            )

        self._unload_listeners.clear()

        # Step 2: Explicitly flush all data before unload (critical)
        try:
            LOGGER.debug("Coordinator: Performing final data flush before unload...")
            if "conversation_memory" in self.hass.data[DOMAIN]:
                await self.hass.data[DOMAIN]["conversation_memory"].async_flush()
            if "token_stats" in self.hass.data[DOMAIN]:
                await self.hass.data[DOMAIN]["token_stats"].async_flush()
            if "chat_history" in self.hass.data[DOMAIN]:
                await self.hass.data[DOMAIN]["chat_history"].async_flush()
            LOGGER.debug("Coordinator: Final flush completed successfully")
        except Exception as err:
            LOGGER.error(
                "Coordinator: CRITICAL - Final flush on unload failed: %s",
                err,
                exc_info=True,
            )

        # Step 3: Wait for any pending save tasks with timeout
        pending_tasks = self.hass.data.get(DOMAIN, {}).get("pending_save_tasks", set())
        if pending_tasks:
            LOGGER.info(
                "Coordinator: Waiting for %d pending save tasks before unload...",
                len(pending_tasks),
            )
            try:
                await asyncio.wait_for(
                    asyncio.gather(*pending_tasks, return_exceptions=True), timeout=30.0
                )
                LOGGER.debug("Coordinator: All pending save tasks completed")
            except asyncio.TimeoutError:
                LOGGER.error(
                    "Coordinator: TIMEOUT - %d pending save tasks did not complete in 30s",
                    len(pending_tasks),
                )
            except Exception as err:
                LOGGER.error(
                    "Coordinator: Error waiting for pending save tasks: %s",
                    err,
                    exc_info=True,
                )

        # Step 4: Unregister services
        unregister_services(self.hass)

        LOGGER.info("Coordinator: Unload completed successfully")
        return True

    async def async_remove(self) -> None:
        """Remove the integration."""
        LOGGER.debug("Coordinator: Starting removal")

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
            LOGGER.debug("Coordinator: Removed entity and state for %s", entity_id)

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
                LOGGER.info(
                    "Coordinator: Cleaned up memory storage folder: %s", memory_folder
                )
            except Exception as err:
                LOGGER.warning(
                    "Coordinator: Failed to remove memory folder %s: %s",
                    memory_folder,
                    err,
                )
