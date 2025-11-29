"""Base entity for xAI."""

from __future__ import annotations

# Standard library imports
import asyncio
from typing import TYPE_CHECKING

# Home Assistant imports
from homeassistant.config_entries import ConfigSubentry as HA_ConfigSubentry
from homeassistant.helpers.entity import Entity as HA_Entity
from homeassistant.components import conversation as ha_conversation
from homeassistant.helpers import device_registry as ha_device_registry

# Local imports
from .const import (
    CONF_CHAT_MODEL,
    CONF_STORE_MESSAGES,
    CONF_USE_INTELLIGENT_PIPELINE,
    DEFAULT_MANUFACTURER,
    DOMAIN,
    LOGGER,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_STORE_MESSAGES,
)
from .entity_pipeline import IntelligentPipeline
from .entity_tools import XAIToolsProcessor
from .helpers import XAIGateway, LogTimeServices
from .exceptions import handle_api_error

if TYPE_CHECKING:
    from . import XAIConfigEntry


class XAIBaseLLMEntity(HA_Entity):
    """xAI conversation agent."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(self, entry: XAIConfigEntry, subentry: HA_ConfigSubentry) -> None:
        """Initialize the entity."""
        self.entry = entry
        self.subentry = subentry
        self._attr_unique_id = subentry.subentry_id
        model = subentry.data.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)
        self._attr_device_info = ha_device_registry.DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=subentry.title,
            manufacturer=DEFAULT_MANUFACTURER,
            model=model,
            entry_type=ha_device_registry.DeviceEntryType.SERVICE,
        )
        # Gateway for centralized xAI SDK interactions
        self.gateway = XAIGateway(self)
        # Tools processor instance to maintain cache across calls
        self._tools_processor = XAIToolsProcessor(self)
        # Track pending I/O tasks for graceful shutdown
        self._pending_save_tasks = set()
        LOGGER.debug(
            "Initialized XAI entity: %s (model: %s, unique_id: %s)",
            subentry.title,
            model,
            subentry.subentry_id,
        )

    async def async_added_to_hass(self) -> None:
        """Link to shared sensors and get global ConversationMemory."""
        await super().async_added_to_hass()

        # Get shared token sensors from hass.data using entry.entry_id
        entry_id = self.entry.entry_id
        sensor_key = f"{entry_id}_sensors"
        if DOMAIN in self.hass.data and sensor_key in self.hass.data[DOMAIN]:
            self._token_sensors = self.hass.data[DOMAIN][sensor_key]
            LOGGER.debug(
                "Entity %s linked to %d shared sensors",
                self.entity_id,
                len(self._token_sensors),
            )
        else:
            # Sensors will be linked later when they are created
            self._token_sensors = []
            LOGGER.debug(
                "Sensors not yet available for entry %s, will be linked when ready",
                entry_id,
            )

        # Get global ConversationMemory instance
        self._conversation_memory = self.hass.data[DOMAIN]["conversation_memory"]
        LOGGER.debug("Entity %s linked to global ConversationMemory", self.entity_id)

        # Store a reference to this entity for easy access by other components, if not already present
        if entry_id not in self.hass.data[DOMAIN]:
            self.hass.data[DOMAIN][entry_id] = self
            LOGGER.debug(
                "Stored reference to entity %s for entry %s", self.entity_id, entry_id
            )

    async def async_will_remove_from_hass(self) -> None:
        """Cleanup when entity is being removed."""
        await super().async_will_remove_from_hass()

        # Wait for pending save tasks to complete (graceful shutdown)
        if self._pending_save_tasks:
            LOGGER.debug(
                "Waiting for %d pending save tasks to complete for entity %s",
                len(self._pending_save_tasks),
                self.entity_id,
            )
            try:
                # Use asyncio.wait with timeout to avoid indefinite blocking
                done, pending = await asyncio.wait(
                    self._pending_save_tasks,
                    timeout=5.0,  # Max 5 seconds wait
                )
                if pending:
                    LOGGER.warning(
                        "Shutdown timeout: %d save tasks still pending for entity %s",
                        len(pending),
                        self.entity_id,
                    )
                    # Cancel remaining tasks
                    for task in pending:
                        task.cancel()
                else:
                    LOGGER.debug(
                        "All save tasks completed for entity %s", self.entity_id
                    )
            except Exception as err:
                LOGGER.error("Error waiting for pending save tasks: %s", err)

        # Close xAI client and cleanup gRPC channels
        LOGGER.debug("Closing xAI gateway for entity %s", self.entity_id)
        await self.gateway.close()
        LOGGER.debug("xAI gateway closed for entity %s", self.entity_id)

    async def _async_handle_chat_log(
        self,
        user_input,
        chat_log,
    ) -> None:
        """
        Generate an answer for the chat log, with logging and timing handled by LogTimeServices.
        The timer context is created here and passed down to the processors.
        """
        use_pipeline_raw = self._get_option(CONF_USE_INTELLIGENT_PIPELINE, True)
        store_messages = self._get_option(
            CONF_STORE_MESSAGES, RECOMMENDED_STORE_MESSAGES
        )

        mode = "intelligent_pipeline" if use_pipeline_raw else "tools_mode"
        context = {"memory": "with_memory" if store_messages else "no_memory"}

        async with LogTimeServices(LOGGER, mode, context) as timer:
            try:
                if use_pipeline_raw:
                    LOGGER.debug("Using intelligent pipeline")
                    pipeline = IntelligentPipeline(self.hass, self, user_input)
                    # The timer instance is passed down to the processor
                    await pipeline.run(chat_log, timer)
                else:
                    LOGGER.debug("Using tools processor (pipeline disabled)")
                    # The timer instance is passed down to the processor
                    await self._tools_processor.async_process_with_loop(
                        user_input, chat_log, timer
                    )
            except Exception as err:
                # The timer's __aexit__ will log the error details
                handle_api_error(err, timer.start_time, "xAI API call")

    async def _add_error_response_and_continue(
        self, user_input, chat_log, message: str
    ) -> None:
        """Add an error response to the chat log and keep the conversation going."""
        # Add error as assistant content to keep conversation flowing
        async for _ in chat_log.async_add_assistant_content(
            ha_conversation.AssistantContent(
                agent_id=self.entity_id,
                content=message,
            )
        ):
            pass

    def _get_option(self, key: str, default=None):
        """Get an option from subentry data with a fallback to defaults."""
        # Try options first (for modified settings), then data (for defaults)
        if (
            hasattr(self.subentry, "options")
            and self.subentry.options
            and key in self.subentry.options
        ):
            return self.subentry.options[key]
        return self.subentry.data.get(key, default)

    def _get_int_option(self, key: str, default: int) -> int:
        """Get an integer option with type safety."""
        return int(self._get_option(key, default))

    def _get_float_option(self, key: str, default: float) -> float:
        """Get a float option with type safety and locale handling."""
        value = self._get_option(key, default)
        if isinstance(value, str):
            value = value.replace(",", ".")
        return float(value)
