"""Base entity for xAI."""

from __future__ import annotations

# Standard library imports
import asyncio
import contextlib
from dataclasses import replace

# Home Assistant imports
from homeassistant.config_entries import ConfigSubentry as HA_ConfigSubentry
from homeassistant.helpers.entity import Entity as HA_Entity
from homeassistant.components import conversation as ha_conversation
from homeassistant.helpers import device_registry as ha_device_registry

# Local imports
from .const import (
    CHAT_MODE_CHATONLY,
    CHAT_MODE_PIPELINE,
    CONF_CHAT_MODEL,
    CONF_EXTENDED_TOOLS_YAML,
    DEFAULT_MANUFACTURER,
    DOMAIN,
    LOGGER,
    MODEL_TARGET_CHAT,
    MODEL_TARGET_VISION,
    XAIConfigEntry,
)
from .entity_pipeline import IntelligentPipeline
from .entity_tools import XAIToolsProcessor
from .xai_gateway import XAIGateway
from .helpers import (
    LogTimeServices,
    ExtendedToolsRegistry,
    async_prepare_attachments,
    ChatOptions,
    resolve_chat_parameters,
    enrich_last_user_message,
    prepare_history_payload,
)
from .exceptions import handle_api_error


class XAIBaseLLMEntity(HA_Entity):
    """Base class for xAI conversation entities.

    Provides shared logic for tool processing, timer management, and chat handling.
    """

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(self, entry: XAIConfigEntry, subentry: HA_ConfigSubentry) -> None:
        """Initialize the entity."""
        self.entry = entry
        self.subentry = subentry
        self._attr_unique_id = subentry.subentry_id
        model = subentry.data.get(CONF_CHAT_MODEL)
        self._attr_device_info = ha_device_registry.DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=subentry.title,
            manufacturer=DEFAULT_MANUFACTURER,
            model=model,
            entry_type=ha_device_registry.DeviceEntryType.SERVICE,
        )
        # Shared Gateway will be accessed via property from coordinator
        # Tools processor (initialized in async_added_to_hass when hass is available)
        self._tools_processor: XAIToolsProcessor | None = None
        # Extended tools registry
        self._extended_tools_registry: ExtendedToolsRegistry | None = None
        # Track pending I/O tasks for graceful shutdown
        self._pending_save_tasks = set()
        self._pipeline: IntelligentPipeline | None = None
        LOGGER.debug(
            "Initialized XAI entity: %s (model: %s, unique_id: %s)",
            subentry.title,
            model,
            subentry.subentry_id,
        )

    async def async_added_to_hass(self) -> None:
        """Register the entity and initialize shared resources."""
        await super().async_added_to_hass()

        # Gateway is now shared via coordinator, no local init needed

        # Initialize Extended Tools Registry ONLY for conversation entities
        if self.subentry.subentry_type == "conversation":
            # We load it here to ensure hass is available and config is fresh
            yaml_config = self.entry.data.get(CONF_EXTENDED_TOOLS_YAML, "")
            self._extended_tools_registry = ExtendedToolsRegistry(
                self.hass, yaml_config
            )
            if not self._extended_tools_registry.is_empty:
                LOGGER.debug("Extended Tools Registry initialized with tools")

        # Get global ConversationMemory instance
        self._conversation_memory = self.hass.data[DOMAIN]["conversation_memory"]

        # Initialize ToolsProcessor (needs hass for ToolOrchestrator)
        self._tools_processor = XAIToolsProcessor(self)

        # Initialize IntelligentPipeline
        self._pipeline = IntelligentPipeline(self.hass, self, None)

        LOGGER.debug("Entity %s linked to global ConversationMemory", self.entity_id)

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
                _, pending = await asyncio.wait(
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

        # Cleanup: Gateway is shared and closed by coordinator

    async def _async_handle_chat_log(
        self,
        chat_log: ha_conversation.ChatLog,
        user_input: ha_conversation.ConversationInput,
        timer: LogTimeServices | None = None,
        options: ChatOptions | None = None,
    ) -> None:
        """Process a conversational interaction.

        Initializes the unified timer and routes execution to the appropriate handler.
        """
        # 1. Resolve parameters once (merging subentry config and overrides)
        # Ensure user_input is set for memory context resolution
        if not options:
            options = ChatOptions(user_input=user_input)
        elif not options.user_input:
            options.user_input = user_input

        params = resolve_chat_parameters(
            "conversation", self.entry, self.subentry.subentry_id, options
        )
        mode = params.mode
        model = params.model

        # 1.1 Message Enrichment: Format the latest user message in ChatLog for UI and downstream layers.
        # This is Layer 1 of the architecture: Entity is responsible for HA-specific formatting.
        await enrich_last_user_message(
            chat_log,
            user_input,
            self.hass,
            send_user_name=params.send_user_name,
            mode=mode,
        )

        timer_context = {
            "memory": "zdr"
            if params.use_encrypted_content
            else "with_memory"
            if params.store_messages
            else "no_memory",
            "mode": mode,
            "model": model,
        }

        if timer:
            timer_cm = contextlib.nullcontext(timer)
        elif options and options.timer:
            timer_cm = contextlib.nullcontext(options.timer)
        else:
            timer_cm = LogTimeServices(LOGGER, "conversation", timer_context)

        async with timer_cm as active_timer:
            try:
                if mode == CHAT_MODE_PIPELINE:
                    # Update pipeline's user_input before running
                    self._pipeline.user_input = user_input
                    await self._pipeline.run(chat_log, active_timer, params)
                elif mode == CHAT_MODE_CHATONLY:
                    # Direct chat without tools - no orchestrator overhead
                    await self._tools_processor._async_run_chat_loop(
                        chat_log, active_timer, params, mode_override=CHAT_MODE_CHATONLY
                    )
                else:
                    await self._tools_processor.async_process_with_loop(
                        user_input, chat_log, active_timer, params
                    )
            except Exception as err:
                handle_api_error(err, active_timer.start_time, "xAI API call")

    async def _async_handle_stateless_task(
        self,
        chat_log: ha_conversation.ChatLog,
        task_name: str | None = None,
        task_structure: dict | None = None,
        extra_attachments: list | None = None,
        extra_images: list[str] | None = None,
        timer: LogTimeServices | None = None,
        options: ChatOptions | None = None,
    ) -> None:
        """Process a stateless task (AI Task).

        Handles attachment preparation and routes execution to the gateway.
        """
        service_type = getattr(self, "service_type", "ai_task")

        # 1. Pre-process image/attachment parts to detect if vision is needed
        extra_content = await async_prepare_attachments(
            self.hass, extra_attachments, extra_images
        )

        # 2. Resolve parameters (pick model based on content)
        opts = options or ChatOptions()
        # Auto-switch to vision target if images are present and target is default
        if extra_content and opts.model_target == MODEL_TARGET_CHAT:
            opts.model_target = MODEL_TARGET_VISION

        params = resolve_chat_parameters(
            service_type, self.entry, self.subentry.subentry_id, opts
        )
        params.extra_content = extra_content
        model = params.model

        # Note: No enrichment (timestamps/user names) for AI tasks - not needed for one-shot

        timer_context = {
            "mode": params.mode or "stateless",
            "service": service_type,
            "model": model,
        }

        if timer:
            timer_cm = contextlib.nullcontext(timer)
        elif options and options.timer:
            timer_cm = contextlib.nullcontext(options.timer)
        else:
            timer_cm = LogTimeServices(LOGGER, "stateless", timer_context)

        async with timer_cm as active_timer:
            # Layer 2: Filtering (Turn selection)
            # For stateless tasks, we typically want the whole history passed so far if any,
            # but usually it's a one-shot turn.
            messages = await prepare_history_payload(
                chat_log,
                params,
                history_limit=10,  # Stateless usually doesn't need much, but let's be safe
            )

            # Layer 3: Transport
            content = await self.gateway.execute_stateless_chat(
                messages,
                service_type=service_type,
                options=replace(
                    params,
                    task_name=task_name or (options.task_name if options else None),
                    task_structure=task_structure
                    or (options.task_structure if options else None),
                    response_format=options.response_format if options else None,
                    timer=active_timer,
                ),
                entity=self,
            )

            # Layer 1: Persistence (State Management)
            if content:
                chat_log.content.append(
                    ha_conversation.AssistantContent(
                        agent_id=self.entity_id,
                        content=content,
                    )
                )

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

    @property
    def gateway(self) -> XAIGateway:
        """Get the shared gateway from the coordinator."""
        coordinator = self.hass.data[DOMAIN][self.entry.entry_id]
        return coordinator.gateway

    def get_config_dict(self) -> dict:
        """Extract configuration with global fallbacks."""
        params = resolve_chat_parameters(
            getattr(self, "service_type", "conversation"),
            self.entry,
            self.subentry.subentry_id,
        )
        return params.config
