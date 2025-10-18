"""Base entity for xAI."""

from __future__ import annotations

# Standard library imports
import grpc
import json
import time
from typing import TYPE_CHECKING

# Home Assistant imports
from homeassistant.helpers.storage import Store

# Home Assistant imports (re-exported from __init__)
from .__init__ import (
    XAI_CLIENT_CLASS, xai_user, xai_system, xai_assistant, xai_image, xai_search_parameters, XAI_SDK_AVAILABLE,
    HA_ConfigSubentry, HA_HomeAssistantError, HA_Entity,
    ha_conversation, ha_device_registry, ha_llm
)

# Local imports
from .const import (
    CONF_API_HOST,
    CONF_CHAT_MODEL,
    CONF_LIVE_SEARCH,
    CONF_MAX_TOKENS,
    CONF_MEMORY_CLEANUP_INTERVAL_HOURS,
    CONF_MEMORY_DEVICE_MAX_TURNS,
    CONF_MEMORY_DEVICE_TTL_HOURS,
    CONF_MEMORY_USER_MAX_TURNS,
    CONF_MEMORY_USER_TTL_HOURS,
    CONF_PROMPT,
    CONF_PROMPT_PIPELINE,
    CONF_REASONING_EFFORT,
    CONF_STORE_MESSAGES,
    CONF_TEMPERATURE,
    CONF_TIMEOUT,
    CONF_USE_INTELLIGENT_PIPELINE,
    DEFAULT_CONVERSATION_NAME,
    DOMAIN,
    LOGGER,
    REASONING_EFFORT_MODELS,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_LIVE_SEARCH,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_MEMORY_CLEANUP_INTERVAL_HOURS,
    RECOMMENDED_MEMORY_DEVICE_MAX_TURNS,
    RECOMMENDED_MEMORY_DEVICE_TTL_HOURS,
    RECOMMENDED_MEMORY_USER_MAX_TURNS,
    RECOMMENDED_MEMORY_USER_TTL_HOURS,
    RECOMMENDED_REASONING_EFFORT,
    RECOMMENDED_STORE_MESSAGES,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TIMEOUT,
)
from .entity_pipeline import IntelligentPipeline
from .entity_tools import XAIToolsProcessor
from .helpers import (
    validate_xai_configuration,
    get_last_user_message,
    prompt_hash,
    extract_user_id,
    extract_device_id,
    is_device_request,
)
from .exceptions import (
    XAIConnectionError,
    XAIToolConversionError,
    XAIConfigurationError,
    raise_generic_error,
    raise_validation_error,
    handle_api_error,
)

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
        # Get model with options fallback for Home Assistant 2024+ compatibility
        if hasattr(subentry, 'options') and subentry.options and CONF_CHAT_MODEL in subentry.options:
            model = subentry.options[CONF_CHAT_MODEL]
        else:
            model = subentry.data.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)
        self._attr_device_info = ha_device_registry.DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=subentry.title,
            manufacturer="xAI",
            model=model,
            entry_type=ha_device_registry.DeviceEntryType.SERVICE,
        )
        # Cached client to reuse gRPC channel and avoid reconnect/handshake on each call
        self._cached_client: Client | None = None
        # Tools processor instance to maintain cache across calls
        self._tools_processor = XAIToolsProcessor(self)
        LOGGER.debug("Initialized XAI entity: %s (model: %s, unique_id: %s)", subentry.title, model, subentry.subentry_id)

    async def async_added_to_hass(self) -> None:
        """Link to shared sensors and get global ConversationMemory."""
        await super().async_added_to_hass()

        # Get shared token sensors from hass.data using entry.entry_id
        entry_id = self.entry.entry_id
        sensor_key = f"{entry_id}_sensors"
        if DOMAIN in self.hass.data and sensor_key in self.hass.data[DOMAIN]:
            self._token_sensors = self.hass.data[DOMAIN][sensor_key]
            LOGGER.debug("Entity %s linked to %d shared sensors", self.entity_id, len(self._token_sensors))
        else:
            # Sensors will be linked later when they are created
            self._token_sensors = []
            LOGGER.debug("Sensors not yet available for entry %s, will be linked when ready", entry_id)

        # Get global ConversationMemory instance
        self._conversation_memory = self.hass.data[DOMAIN]["conversation_memory"]
        LOGGER.debug("Entity %s linked to global ConversationMemory", self.entity_id)

    async def _async_handle_chat_log(
        self,
        user_input,
        chat_log,
        previous_response_id: str | None = None,
        ) -> None:
        """Generate an answer for the chat log with optional persistent memory."""
        start_time = time.time()
        use_pipeline_raw = self._get_option(CONF_USE_INTELLIGENT_PIPELINE, True)
        store_messages = self._get_option(CONF_STORE_MESSAGES, RECOMMENDED_STORE_MESSAGES)

        mode = "pipeline" if use_pipeline_raw else "tools"
        memory_status = "with_memory" if store_messages else "no_memory"

        LOGGER.info("chat_start: entity=%s mode=%s memory=%s",
                   self.entity_id, mode, memory_status)

        try:
            # Route based on the configuration. The pipeline itself will handle fallbacks.
            if use_pipeline_raw:
                LOGGER.debug("Using intelligent pipeline")
                pipeline = IntelligentPipeline(self.hass, self, user_input)
                await pipeline.run(chat_log, previous_response_id=previous_response_id)
            else: # Handles "tools" mode (which includes simple chat if tools are off)
                LOGGER.debug("Using tools processor (pipeline disabled)")
                await self._tools_processor.async_process_with_loop(
                    user_input, chat_log, previous_response_id
                )

            # Persist latest prev_id if available (already handled in individual methods)
            # No additional persistence needed here as it's done in each API call

            processing_time = time.time() - start_time
            LOGGER.info("chat_end: entity=%s duration=%.2fs", self.entity_id, processing_time)
        except Exception as err:
            handle_api_error(err, start_time, "xAI API call")

    async def _async_process_chat_log_stateless(self, chat_log, attachments=None, service_type="ai_task") -> None:
        """Process chat_log without memory - for AI Task and Grok Code Fast.

        This method is designed for one-shot tasks where server-side memory is not used.
        It sends the entire chat_log as-is to the xAI API and appends the response.

        Args:
            chat_log: The conversation log to process
            attachments: Optional list of Attachment objects with images/media to include
            service_type: "ai_task" or "code_fast" for token sensor tracking
        """
        import time
        import base64

        start_time = time.time()
        client = self._create_client()

        # Create chat without previous_response_id (stateless)
        chat = self._create_chat(client, tools=None, previous_response_id=None)

        # Get system prompt from configuration
        system_prompt_text = self._get_option(CONF_PROMPT, "")
        if system_prompt_text:
            chat.append(xai_system(system_prompt_text))

        # Add all messages from chat_log
        # Track if we've already added attachments (only add once, to first user message)
        attachments_added = False

        for content in chat_log.content:
            if isinstance(content, ha_conversation.UserContent):
                # Build user message - may include text and attachments
                message_parts = []

                # Add text content if present
                if content.content:
                    message_parts.append(content.content)

                # Add attachments as images (for first user message only)
                # xAI vision API supports images interleaved with text in user messages
                if attachments and not attachments_added:
                    attachments_added = True
                    # Process attachments into base64 format for xAI
                    for attachment in attachments:
                        try:
                            # Read file from disk using executor to avoid blocking
                            def _read_file():
                                with open(attachment.path, "rb") as f:
                                    return f.read()

                            image_bytes = await self.hass.async_add_executor_job(_read_file)

                            # Convert to base64
                            base64_image = base64.b64encode(image_bytes).decode("utf-8")

                            # Create base64 data URI
                            data_uri = f"data:{attachment.mime_type};base64,{base64_image}"

                            # Add image to message using xai_image helper
                            # xai_image accepts URLs or base64 data URIs
                            message_parts.append(xai_image(data_uri))

                            LOGGER.debug(
                                "Added attachment to AI Task: mime=%s, size=%d bytes",
                                attachment.mime_type,
                                len(image_bytes),
                            )
                        except Exception as err:
                            LOGGER.warning(
                                "Failed to process attachment %s: %s",
                                attachment.path,
                                err,
                            )

                # Append user message with text and/or images
                if message_parts:
                    if len(message_parts) == 1 and isinstance(message_parts[0], str):
                        # Single text message
                        chat.append(xai_user(message_parts[0]))
                    else:
                        # Multiple parts (text + images)
                        chat.append(xai_user(*message_parts))
                else:
                    # Fallback empty message
                    chat.append(xai_user(""))

            elif isinstance(content, ha_conversation.AssistantContent):
                chat.append(xai_assistant(content.content or ""))

        # Call API (non-streaming for AI Task)
        try:
            def _sample_sync():
                return chat.sample()

            response = await self.hass.async_add_executor_job(_sample_sync)
            content_text = getattr(response, "content", "")

            # Update token sensors
            usage = getattr(response, "usage", None)
            model = getattr(response, "model", None)
            if usage:
                self._update_token_sensors(usage, model=model, service_type=service_type)

            # Append assistant response to chat_log
            chat_log.content.append(ha_conversation.AssistantContent(agent_id=self.entity_id, content=content_text))

            api_time = time.time() - start_time
            LOGGER.info("AI Task stateless call: duration=%.2fs tokens=%d",
                       api_time, usage.total_tokens if usage else 0)

        except Exception as err:
            handle_api_error(err, start_time, "AI Task API call")

    async def async_clear_memory(self, clear_all: bool = False, scope: str | None = None, target_id: str | None = None, physical_delete: bool = False) -> None:
        """Clear persistent memory using global ConversationMemory.

        Behavior:
        - If physical_delete = True: physically delete the entire storage file (irreversible).
        - If clear_all = True: remove all stored chains across scopes and modes.
        - Else: requires scope; for user/device, target_id is required; removes all :mode:* keys under the base.
        Additionally, when store_messages=True, attempt remote deletion of any stored completion IDs before clearing local entries.
        """
        try:
            # Get response IDs that will be deleted (for remote cleanup)
            response_ids = []

            if physical_delete:
                response_ids = await self._conversation_memory.physical_delete_storage()
            elif clear_all:
                response_ids = await self._conversation_memory.clear_all_memory()
            else:
                if not scope:
                    raise_validation_error("scope is required when not clearing all")
                if scope not in ("user", "device"):
                    raise_validation_error("invalid scope, must be 'user' or 'device'")
                if not target_id:
                    raise_validation_error("target_id is required when scope is user or device")

                response_ids = await self._conversation_memory.clear_memory_by_scope(scope, target_id)

            # Attempt remote deletion of stored completions
            if response_ids:
                context = "physical_delete" if physical_delete else "clear_all" if clear_all else "clear_memory"
                await self._async_delete_remote_completions(response_ids, context=context)

            LOGGER.info("Memory cleared successfully for scope: %s",
                       "physical_delete" if physical_delete else "all" if clear_all else f"{scope}:{target_id}")
        except (OSError, PermissionError) as err:
            LOGGER.warning("clear_memory: storage access denied: %s", err)
        except Exception as err:
            LOGGER.error("clear_memory failed: %s", err)
            raise

    async def _add_error_response_and_continue(self, user_input, chat_log, message: str) -> None:
        """Add an error response to the chat log and keep the conversation going."""
        # Add error as assistant content to keep conversation flowing
        async for _ in chat_log.async_add_assistant_content(
            ha_conversation.AssistantContent(
                agent_id=self.entity_id,
                content=message,
            )
        ):
            pass

    def _create_client(self) -> Client:
        """Create or return a cached xAI client for connection reuse."""
        # Reuse a cached client to avoid TLS handshake/channel setup on each call
        # On config change, the entry is reloaded, and a new entity is created,
        # so the client is recreated with new settings.
        # This check is purely for performance within a single entity's lifetime.
        # If options are updated, Home Assistant reloads the entry, creating a new
        # entity instance, which in turn will create a new client.

        if self._cached_client is not None:
            return self._cached_client

        LOGGER.debug("Creating xAI client with configuration validation")
        model = self._get_option(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)
        validate_xai_configuration(self.entry, self._get_option, model)

        timeout = self._get_option(CONF_TIMEOUT, RECOMMENDED_TIMEOUT)
        LOGGER.debug("Client configuration: timeout=%ss, API key=%s***", timeout, self.entry.data["api_key"][:8])
        # Downgrade to debug to reduce info-level noise on each call
        LOGGER.debug("Creating xAI client with timeout: %d seconds", timeout)

        # Configure retry/keepalive to keep channel warm
        retry_config = [
            ("grpc.keepalive_time_ms", 30000),
            ("grpc.keepalive_timeout_ms", 5000),
            ("grpc.keepalive_permit_without_calls", True),
            ("grpc.http2.max_pings_without_data", 0),
            ("grpc.http2.min_time_between_pings_ms", 10000),
            ("grpc.http2.min_ping_interval_without_data_ms", 300000)
        ]

        client_kwargs = {
            "api_key": self.entry.data["api_key"],
            "timeout": timeout,
            "channel_options": retry_config,
        }
        api_host = self._get_option(CONF_API_HOST, None)
        if api_host:
            client_kwargs["api_host"] = api_host

        client = XAI_CLIENT_CLASS(**client_kwargs)
        self._cached_client = client
        LOGGER.debug("xAI client created and cached successfully")
        return client

    def _create_chat(self, client: Client, tools: list | None = None, previous_response_id: str | None = None):
        """Create an xAI chat object with the correct model and parameters."""
        model = self._get_option(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)
        max_tokens = self._get_int_option(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS)
        temperature = self._get_float_option(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE)
        # Enable server-side memory when configured
        store_messages = bool(self._get_option(CONF_STORE_MESSAGES, RECOMMENDED_STORE_MESSAGES))

        # Configure Live Search if enabled and SDK available
        live_search_mode = self._get_option(CONF_LIVE_SEARCH, RECOMMENDED_LIVE_SEARCH)
        search_parameters = None
        if live_search_mode != "off" and XAI_SDK_AVAILABLE and xai_search_parameters:
            try:
                search_parameters = xai_search_parameters(mode=live_search_mode)
                LOGGER.debug("Live Search enabled with mode: %s", live_search_mode)
            except Exception as err:
                LOGGER.warning("Failed to configure Live Search: %s", err)

        chat_args = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "store_messages": store_messages,
        }

        # Add reasoning_effort only for supported models
        if model in REASONING_EFFORT_MODELS:
            reasoning_effort = self._get_option(CONF_REASONING_EFFORT, RECOMMENDED_REASONING_EFFORT)
            chat_args["reasoning_effort"] = reasoning_effort
            LOGGER.debug("Reasoning effort set to '%s' for model %s", reasoning_effort, model)

        if search_parameters:
            chat_args["search_parameters"] = search_parameters
        if previous_response_id:
            chat_args["previous_response_id"] = previous_response_id

        if tools:
            chat_args["tools"] = tools
            LOGGER.debug(
                "Creating chat with tools: model=%s, max_tokens=%d, temperature=%.2f, tools=%d, store_messages=%s, live_search=%s, prev_id=%s",
                model, max_tokens, temperature, len(tools), store_messages, live_search_mode, bool(previous_response_id),
            )
        else:
            LOGGER.debug(
                "Creating chat without tools: model=%s, max_tokens=%d, temperature=%.2f, store_messages=%s, live_search=%s, prev_id=%s",
                model, max_tokens, temperature, store_messages, live_search_mode, bool(previous_response_id),
            )

        return client.chat.create(**chat_args)

    def _get_conversation_key(self, user_input) -> str:
        """Generate the base key for storing conversation memory.

        Automatically detects if the request comes from:
        - A device (voice assistant satellite) → device:{device_id}
        - A user (smartphone/PC/tablet) → user:{user_id}

        This determines which memory parameters (TTL, max_turns) are used.
        """
        # Detect if request is from device or user
        if is_device_request(user_input):
            device_id = extract_device_id(user_input)
            return f"device:{device_id}"

        # User request (smartphone, PC, tablet)
        user_id = extract_user_id(user_input)
        return f"user:{user_id or 'unknown'}"

    def _get_xai_user_id(self, user_input) -> str | None:
        """Get the xAI user ID from Home Assistant user ID."""
        ha_user_id = extract_user_id(user_input)
        if not ha_user_id:
            return None
        return ha_user_id


    async def _save_response_chain(self, conv_key: str, response_id: str, mode: str) -> None:
        """Centralized function to save response_id with consistent logging.

        Args:
            conv_key: Conversation key for this chain
            response_id: Response ID to save
            mode: Mode string for logging ("pipeline" or "tools")
        """
        # Check if store_messages is enabled before attempting to save
        store_messages_enabled = self._get_option(CONF_STORE_MESSAGES, True)
        if not store_messages_enabled:
            LOGGER.debug("memory_save: skipped (store_messages=False) mode=%s", mode)
            return

        await self._memory_set_prev_id(conv_key, response_id)
        LOGGER.debug(
            "memory_save: mode=%s conv_key=%s response_id=%s",
            mode, conv_key, response_id[:8]
        )

    def _get_option(self, key: str, default=None):
        """Get an option from subentry data with a fallback to defaults."""
        # Try options first (for modified settings), then data (for defaults)
        if hasattr(self.subentry, 'options') and self.subentry.options and key in self.subentry.options:
            return self.subentry.options[key]
        return self.subentry.data.get(key, default)

    def _get_int_option(self, key: str, default: int) -> int:
        """Get an integer option with type safety."""
        return int(self._get_option(key, default))

    def _get_float_option(self, key: str, default: float) -> float:
        """Get a float option with type safety and locale handling."""
        value = self._get_option(key, default)
        if isinstance(value, str):
            value = value.replace(',', '.')
        return float(value)

    def _get_token_sensors(self) -> list:
        """Get token sensors with lazy loading if not yet available."""
        if not self._token_sensors:
            # Try to load sensors from hass.data if they were created after this entity
            entry_id = self.entry.entry_id
            sensor_key = f"{entry_id}_sensors"
            if DOMAIN in self.hass.data and sensor_key in self.hass.data[DOMAIN]:
                self._token_sensors = self.hass.data[DOMAIN][sensor_key]
                LOGGER.debug("Entity %s lazy-loaded %d shared sensors", self.entity_id, len(self._token_sensors))
        return self._token_sensors

    def _update_token_sensors(
        self,
        usage,
        model: str | None = None,
        service_type: str = "conversation",
        mode: str = "pipeline",
        is_fallback: bool = False
    ) -> None:
        """Update all registered token sensors with new usage data.

        Args:
            usage: xAI response.usage object containing token counts
            model: The model name from xAI response (e.g., "grok-4-fast")
                   If None, will be extracted from usage or fall back to config
            service_type: "conversation", "ai_task", or "code_fast"
            mode: "pipeline" or "tools" (only for conversation)
            is_fallback: True if fallback from pipeline to tools (only for conversation)
        """
        if not usage:
            LOGGER.debug("_update_token_sensors: no usage data provided")
            return

        # Extract model from response if available
        if model is None:
            model = getattr(usage, "model", None)

        # If still not available, get from config (fallback)
        if model is None:
            model = self._get_option(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)
            LOGGER.debug("_update_token_sensors: using model from config: %s", model)

        # Get store_messages setting
        store_messages = self._get_option("store_messages", True)

        # Log the usage data received
        LOGGER.debug("_update_token_sensors: service=%s mode=%s fallback=%s store=%s model=%s - completion=%s, prompt=%s, cached=%s",
                    service_type, mode, is_fallback, store_messages, model,
                    getattr(usage, "completion_tokens", None),
                    getattr(usage, "prompt_tokens", None),
                    getattr(usage, "cached_prompt_text_tokens", None))

        # Update all registered sensors (with lazy loading)
        sensors = self._get_token_sensors()
        LOGGER.debug("_update_token_sensors: found %d sensors to update", len(sensors))

        for sensor in sensors:
            try:
                # Filter sensors by service_type
                sensor_service = getattr(sensor, "_service_type", None)

                # Update per-service sensors only if they match
                if sensor_service is not None:
                    if sensor_service != service_type:
                        continue  # Skip this sensor, not for this service

                # Update sensor with full parameters
                sensor.update_token_usage(usage, model, mode, is_fallback, store_messages)
                LOGGER.debug("_update_token_sensors: updated sensor %s", getattr(sensor, "entity_id", "unknown"))
            except Exception as err:
                LOGGER.error("Failed to update sensor %s: %s", getattr(sensor, "entity_id", "unknown"), err)

    async def _async_delete_remote_completions(self, ids: list[str], context: str) -> int:
        """Delete stored completion IDs from the xAI server in the background."""
        if not ids or not self._get_option(CONF_STORE_MESSAGES, True):
            return 0

        def _delete_sync(id_list: list[str]) -> int:
            """Synchronous blocking function to delete completions."""
            deleted_count = 0
            try:
                client = self._create_client()
                for pid in id_list:
                    try:
                        client.chat.delete_stored_completion(pid)
                        deleted_count += 1
                        LOGGER.debug("%s: remote delete successful for %s", context, str(pid)[:8])
                    except Exception as derr:
                        LOGGER.debug("%s: remote delete failed for %s: %s", context, str(pid)[:8], derr)
            except Exception as cerr:
                LOGGER.warning("%s: remote deletion failed to start: %s", context, cerr)
            return deleted_count

        try:
            deleted_count = await self.hass.async_add_executor_job(_delete_sync, ids)
            if deleted_count > 0:
                LOGGER.info("%s: deleted %d/%d completion IDs from server", context, deleted_count, len(ids))
            return deleted_count
        except Exception as err:
            LOGGER.warning("%s: async executor for remote delete failed: %s", context, err)
            return 0

    # ---------------------------
    # Persistent memory helpers with TTL and max_turns
    # ---------------------------

    async def _memory_get_prev_id(self, conv_key: str) -> str | None:
        """Get last response_id using global ConversationMemory."""
        store_messages_enabled = self._get_option(CONF_STORE_MESSAGES, True)
        if not store_messages_enabled:
            return None

        # Delegate to global ConversationMemory which handles the full key format
        return await self._conversation_memory.get_response_id_by_key(conv_key)

    async def _memory_set_prev_id(self, conv_key: str, response_id: str) -> None:
        """Save response_id using global ConversationMemory."""
        store_messages_enabled = self._get_option(CONF_STORE_MESSAGES, True)
        if not store_messages_enabled:
            return

        # Delegate to global ConversationMemory which handles the full key format
        await self._conversation_memory.save_response_id_by_key(conv_key, response_id)
