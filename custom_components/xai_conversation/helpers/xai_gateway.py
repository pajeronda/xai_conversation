"""xAI Gateway - Single point of contact with xAI SDK.

This module provides XAIGateway class that centralizes ALL xAI SDK interactions
to eliminate code duplication across entity.py, services.py, and ai_task.py.

Architecture: One gateway instance per entity (not singleton).
Each XAIBaseLLMEntity creates its own gateway with access to entity's config.
"""

from __future__ import annotations

import json
import base64
import time
from typing import TYPE_CHECKING, Any

from ..__init__ import (
    XAI_CLIENT_CLASS,
    XAI_SDK_AVAILABLE,
    ha_conversation,
    xai_assistant,
    xai_image,
    xai_search_parameters,
    xai_system,
    xai_user,
)
from ..const import (
    CONF_API_HOST,
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_REASONING_EFFORT,
    CONF_LIVE_SEARCH,
    CONF_STORE_MESSAGES,
    CONF_TEMPERATURE,
    CONF_TIMEOUT,
    DOMAIN,
    LOGGER,
    REASONING_EFFORT_MODELS,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_IMAGE_MODEL,
    RECOMMENDED_LIVE_SEARCH,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_REASONING_EFFORT,
    RECOMMENDED_STORE_MESSAGES,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TIMEOUT,
)
from ..exceptions import handle_api_error, raise_generic_error, raise_validation_error
from .config import validate_xai_configuration

if TYPE_CHECKING:
    from typing import Literal
    from xai_sdk import AsyncClient as XAIClient

    from ..entity import XAIBaseLLMEntity


class XAIGateway:
    """Single point of contact with xAI SDK per entity.

    Centralizes all xAI SDK interactions to eliminate code duplication.
    Each entity creates its own gateway instance for isolated configuration.

    Responsibilities:
    - Create and cache xAI client with retry/keepalive configuration
    - Create xAI chat objects with model parameters
    - Execute stateless chats (AI Task, Code Fast)
    - Generate images using grok-2-image

    Does NOT handle:
    - Memory management (handled by ConversationMemory)
    - Prompt building (handled by PromptManager)
    - Response parsing (handled by helpers/response.py)
    - Token tracking logic (returns usage data only)
    """

    def __init__(self, entity: XAIBaseLLMEntity) -> None:
        """Initialize gateway with entity reference.

        Args:
            entity: The XAIBaseLLMEntity instance that owns this gateway
        """
        self.entity = entity
        self._cached_client: XAIClient | None = None

    async def async_update_models_data(self, hass: Any = None) -> None:
        """Fetch the list of models and their pricing from the xAI API.

        This method updates a central dictionary in hass.data which is then
        used by the XAIPricingSensor entities to update their states.
        This is called periodically, not on every conversation.

        Args:
            hass: HomeAssistant instance. If None, uses self.entity.hass
        """
        LOGGER.info("Attempting to fetch latest xAI model pricing information.")
        try:
            client = self.create_client()
            models_list = await client.models.list_language_models()

            processed_models = {}
            for model in models_list:
                model_name = getattr(model, "name", None)
                if not model_name:
                    continue

                # Log all available attributes for debugging
                model_attrs = {k: v for k, v in vars(model).items() if not k.startswith('_')}
                LOGGER.debug("Model %s attributes: %s", model_name, model_attrs)

                # Extract pricing using correct SDK attribute names
                # Based on xai-sdk-python/examples/aio/models.py
                # API returns values in 0.0001 USD per 1M tokens (divide by 10000 to get USD)
                raw_input = getattr(model, "prompt_text_token_price", 0.0)
                raw_output = getattr(model, "completion_text_token_price", 0.0)
                raw_cached = getattr(model, "cached_prompt_token_price", 0.0)

                LOGGER.debug(
                    "Raw pricing from API for %s: input=%s, output=%s, cached=%s",
                    model_name, raw_input, raw_output, raw_cached
                )

                # Convert from API units to USD per 1M tokens
                input_price = raw_input / 10000.0 if raw_input else 0.0
                output_price = raw_output / 10000.0 if raw_output else 0.0
                cached_price = raw_cached / 10000.0 if raw_cached else 0.0

                processed_models[model_name] = {
                    "name": model_name,
                    "input_price_per_million": input_price,
                    "output_price_per_million": output_price,
                    "cached_input_price_per_million": cached_price,
                    "context_window": getattr(model, "context_window", 0),
                    "description": getattr(model, "description", ""),
                }

            # Use provided hass or fall back to entity.hass
            hass_instance = hass if hass is not None else self.entity.hass
            if hass_instance is None:
                raise RuntimeError("HomeAssistant instance not available")

            # Ensure the domain data dictionary exists
            hass_instance.data.setdefault(DOMAIN, {})
            # Store the processed data
            hass_instance.data[DOMAIN]["xai_models_data"] = processed_models

            LOGGER.info("Successfully fetched and stored data for %d xAI models.", len(processed_models))

        except Exception as e:
            LOGGER.error("Failed to fetch or process xAI model data: %s", e)
            raise

    async def close(self) -> None:
        """Close the cached xAI client and cleanup gRPC channels asynchronously.

        This should be called when the entity is being removed to ensure
        proper cleanup of network resources (gRPC channels).
        """
        if self._cached_client is not None:
            try:
                # For grpc.aio based clients, close() is a coroutine
                if hasattr(self._cached_client, 'close'):
                    await self._cached_client.close()
                    LOGGER.debug("xAI async client closed successfully")
                # Fallback for other async patterns, though close() is standard
                elif hasattr(self._cached_client, '__aexit__'):
                    await self._cached_client.__aexit__(None, None, None)
                    LOGGER.debug("xAI async client closed via __aexit__")
                else:
                    LOGGER.debug("xAI async client does not have a recognized close method.")
            except Exception as err:
                LOGGER.warning("Error closing xAI async client: %s", err)
            finally:
                self._cached_client = None

    def create_client(self) -> XAIClient:
        """Create or return a cached xAI client for connection reuse.

        Client is cached per entity to avoid TLS handshake/channel setup on each call.
        On config change, the entry is reloaded and a new entity is created,
        so the client is recreated with new settings.

        Returns:
            XAIClient instance with retry/keepalive configuration

        Raises:
            XAIConfigurationError: If configuration is invalid
        """
        # Return cached client if available
        if self._cached_client is not None:
            return self._cached_client

        LOGGER.debug("Creating xAI client with configuration validation")
        model = self.entity._get_option(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)
        validate_xai_configuration(self.entity.entry, self.entity._get_option, model)

        timeout = self.entity._get_option(CONF_TIMEOUT, RECOMMENDED_TIMEOUT)
        LOGGER.debug(
            "Client configuration: timeout=%ss, API key=%s***",
            timeout,
            self.entity.entry.data["api_key"][:8],
        )
        LOGGER.debug("Creating xAI client with timeout: %d seconds", timeout)

        # Configure retry policy for transient failures
        retry_policy = json.dumps({
            "methodConfig": [{
                "name": [{}],
                "retryPolicy": {
                    "maxAttempts": 3,
                    "initialBackoff": "0.5s",
                    "maxBackoff": "5s",
                    "backoffMultiplier": 2,
                    "retryableStatusCodes": ["UNAVAILABLE"]
                }
            }]
        })

        # Configure keepalive and retry
        channel_options = [
            ("grpc.service_config", retry_policy),
            ("grpc.keepalive_time_ms", 30000),
            ("grpc.keepalive_timeout_ms", 5000),
            ("grpc.keepalive_permit_without_calls", True),
            ("grpc.http2.max_pings_without_data", 0),
            ("grpc.http2.min_time_between_pings_ms", 10000),
            ("grpc.http2.min_ping_interval_without_data_ms", 300000),
        ]

        client_kwargs = {
            "api_key": self.entity.entry.data["api_key"],
            "timeout": timeout,
            "channel_options": channel_options,
        }
        api_host = self.entity._get_option(CONF_API_HOST, None)
        if api_host:
            client_kwargs["api_host"] = api_host

        client = XAI_CLIENT_CLASS(**client_kwargs)
        self._cached_client = client
        LOGGER.debug("xAI client created and cached successfully")
        return client

    def create_chat(
        self,
        client: XAIClient,
        tools: list | None = None,
        previous_response_id: str | None = None,
    ) -> Any:
        """Create an xAI chat object with the correct model and parameters.

        Args:
            client: The xAI client instance
            tools: Optional list of tools for tools mode
            previous_response_id: Optional ID for conversation chaining

        Returns:
            xAI Chat object configured with entity's settings
        """
        model = self.entity._get_option(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)
        max_tokens = self.entity._get_int_option(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS)
        temperature = self.entity._get_float_option(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE)
        # Enable server-side memory when configured
        store_messages = bool(
            self.entity._get_option(CONF_STORE_MESSAGES, RECOMMENDED_STORE_MESSAGES)
        )

        # Configure Live Search if enabled and SDK available
        live_search_mode = self.entity._get_option(CONF_LIVE_SEARCH, RECOMMENDED_LIVE_SEARCH)
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
            reasoning_effort = self.entity._get_option(
                CONF_REASONING_EFFORT, RECOMMENDED_REASONING_EFFORT
            )
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
                model,
                max_tokens,
                temperature,
                len(tools),
                store_messages,
                live_search_mode,
                bool(previous_response_id),
            )
        else:
            LOGGER.debug(
                "Creating chat without tools: model=%s, max_tokens=%d, temperature=%.2f, store_messages=%s, live_search=%s, prev_id=%s",
                model,
                max_tokens,
                temperature,
                store_messages,
                live_search_mode,
                bool(previous_response_id),
            )

        return client.chat.create(**chat_args)

    async def execute_stateless_chat(
        self,
        chat_log: ha_conversation.ChatLog,
        attachments: list | None = None,
        service_type: Literal["ai_task", "code_fast"] = "ai_task",
    ) -> None:
        """Process chat_log without memory - for AI Task and Grok Code Fast.

        This method is designed for one-shot tasks where server-side memory is not used.
        It sends the entire chat_log as-is to the xAI API and appends the response.

        Args:
            chat_log: The conversation log to process (modified in place)
            attachments: Optional list of Attachment objects with images/media to include
            service_type: "ai_task" or "code_fast" for token sensor tracking

        Raises:
            Various exceptions via handle_api_error for API failures
        """
        start_time = time.time()

        LOGGER.info(
            "chat_start: service=%s mode=stateless memory=no_memory",
            service_type,
        )

        client = self.create_client()

        # Create chat without previous_response_id (stateless)
        chat = self.create_chat(client, tools=None, previous_response_id=None)

        # Get system prompt from configuration
        system_prompt_text = self.entity._get_option(CONF_PROMPT, "")
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

                            image_bytes = await self.entity.hass.async_add_executor_job(
                                _read_file
                            )

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
            response = await chat.sample()
            content_text = getattr(response, "content", "")

            # Update token sensors
            usage = getattr(response, "usage", None)
            model = getattr(response, "model", None)
            if usage:
                self.entity._update_token_sensors(
                    usage, model=model, service_type=service_type
                )

            # Append assistant response to chat_log
            chat_log.content.append(
                ha_conversation.AssistantContent(
                    agent_id=self.entity.entity_id, content=content_text
                )
            )

            api_time = time.time() - start_time
            LOGGER.info(
                "chat_end: service=%s duration=%.2fs tokens=%d",
                service_type,
                api_time,
                usage.total_tokens if usage else 0,
            )

        except Exception as err:
            handle_api_error(err, start_time, "AI Task API call")

    async def generate_image(
        self,
        prompt: str,
        chat_log: ha_conversation.ChatLog,
        model: str = RECOMMENDED_IMAGE_MODEL,
    ) -> dict[str, Any]:
        """Generate image using grok-2-image model.

        Args:
            prompt: Image generation prompt
            chat_log: Conversation context for tracking (not used for API call)
            model: Model name (default: RECOMMENDED_IMAGE_MODEL from const.py)

        Returns:
            Dictionary with:
                - image_data: bytes of generated image
                - mime_type: str (e.g. "image/jpeg")
                - model: str (model actually used)
                - revised_prompt: Optional[str] (xAI's prompt revision)

        Raises:
            XAIValidationError: If prompt is empty
            Various exceptions via handle_api_error for auth, network, API errors
        """
        if not XAI_SDK_AVAILABLE:
            raise_generic_error("xAI SDK not available")

        # Validate prompt is not empty
        if not prompt or not prompt.strip():
            raise_validation_error("Image generation prompt cannot be empty")

        start_time = time.time()
        client = self.create_client()

        try:
            # Use base64 format to get raw bytes directly from xAI
            # Note: grok-2-image does not support image-to-image (attachments ignored)
            response = await client.image.sample(
                model=model,
                prompt=prompt,
                image_format="base64",
            )

            # Validate response structure
            if not hasattr(response, "image") or not response.image:
                raise_generic_error("xAI API returned invalid image response")

            if not isinstance(response.image, bytes):
                raise_generic_error(
                    f"Expected bytes, got {type(response.image).__name__}"
                )

            # Extract image data and optional metadata
            image_bytes = response.image
            revised_prompt = getattr(response, "prompt", None)

            elapsed = time.time() - start_time
            LOGGER.info(
                "Image generated successfully: size=%d bytes, prompt_length=%d, elapsed=%.2fs",
                len(image_bytes),
                len(prompt),
                elapsed,
            )

            return {
                "image_data": image_bytes,
                "mime_type": "image/jpeg",  # xAI generates jpg format
                "model": model,
                "revised_prompt": revised_prompt,
            }

        except Exception as err:
            handle_api_error(err, start_time, "image generation")

    async def analyze_photo(
        self,
        prompt: str,
        images: list[str] | None = None,
        attachments: list | None = None,
        model: str | None = None,
    ) -> str:
        """Analyze photos using grok-2-vision model.

        Args:
            prompt: Question or instruction about the images
            images: Optional list of image paths or URLs
            attachments: Optional list of Attachment objects from HA
            model: Optional model override (default from config CONF_VISION_MODEL)

        Returns:
            str: Analysis text response from the model

        Raises:
            XAIValidationError: If prompt is empty or no images provided
            Various exceptions via handle_api_error for auth, network, API errors
        """
        if not XAI_SDK_AVAILABLE:
            raise_generic_error("xAI SDK not available")

        # Validate prompt
        if not prompt or not prompt.strip():
            raise_validation_error("Photo analysis prompt cannot be empty")

        # Validate at least one image source
        if not images and not attachments:
            raise_validation_error("At least one image (path/URL or attachment) is required")

        start_time = time.time()
        client = self.create_client()

        # Get model from entity config if not provided
        if model is None:
            from ..const import CONF_VISION_MODEL, RECOMMENDED_VISION_MODEL
            model = self.entity._get_option(CONF_VISION_MODEL, RECOMMENDED_VISION_MODEL)

        try:
            # Create chat with vision model
            chat = client.chat.create(
                model=model,
                max_tokens=self.entity._get_int_option(CONF_MAX_TOKENS, 4000),
                temperature=self.entity._get_float_option(CONF_TEMPERATURE, 0.1),
                store_messages=False,  # Stateless for photo analysis
            )

            # Add system prompt for vision analysis from config
            from ..const import CONF_VISION_PROMPT, VISION_ANALYSIS_PROMPT
            vision_prompt = self.entity._get_option(CONF_VISION_PROMPT, VISION_ANALYSIS_PROMPT)
            if vision_prompt:
                chat.append(xai_system(vision_prompt))

            # Build user message with text and images
            message_parts = [prompt]

            # Process images from paths/URLs
            if images:
                for img in images:
                    if isinstance(img, str):
                        # Check if it's a URL or local path
                        if img.startswith(("http://", "https://")):
                            # Direct URL - xAI SDK supports URLs
                            message_parts.append(xai_image(img))
                            LOGGER.debug("Added image URL to vision analysis: %s", img[:50])
                        else:
                            # Local file path - read and convert to base64
                            try:
                                def _read_file():
                                    with open(img, "rb") as f:
                                        return f.read()

                                image_bytes = await self.entity.hass.async_add_executor_job(_read_file)
                                base64_image = base64.b64encode(image_bytes).decode("utf-8")

                                # Detect MIME type from file extension
                                mime_type = "image/jpeg"  # default
                                if img.lower().endswith(".png"):
                                    mime_type = "image/png"
                                elif img.lower().endswith(".webp"):
                                    mime_type = "image/webp"

                                data_uri = f"data:{mime_type};base64,{base64_image}"
                                message_parts.append(xai_image(data_uri))

                                LOGGER.debug(
                                    "Added local image to vision analysis: path=%s, size=%d bytes",
                                    img,
                                    len(image_bytes),
                                )
                            except Exception as err:
                                LOGGER.warning("Failed to read image file %s: %s", img, err)

            # Process attachments from HA
            if attachments:
                for attachment in attachments:
                    try:
                        def _read_file():
                            with open(attachment.path, "rb") as f:
                                return f.read()

                        image_bytes = await self.entity.hass.async_add_executor_job(_read_file)
                        base64_image = base64.b64encode(image_bytes).decode("utf-8")
                        data_uri = f"data:{attachment.mime_type};base64,{base64_image}"
                        message_parts.append(xai_image(data_uri))

                        LOGGER.debug(
                            "Added attachment to vision analysis: mime=%s, size=%d bytes",
                            attachment.mime_type,
                            len(image_bytes),
                        )
                    except Exception as err:
                        LOGGER.warning(
                            "Failed to process attachment %s: %s",
                            attachment.path,
                            err,
                        )

            # Append user message with text and images
            chat.append(xai_user(*message_parts))

            # Call API (async method - no executor needed)
            response = await chat.sample()
            content_text = getattr(response, "content", "")

            # Update token sensors
            usage = getattr(response, "usage", None)
            if usage:
                self.entity._update_token_sensors(
                    usage, model=model, service_type="photo_analysis"
                )

            elapsed = time.time() - start_time
            LOGGER.info(
                "Photo analysis completed: model=%s, images=%d, elapsed=%.2fs, tokens=%d",
                model,
                len(images or []) + len(attachments or []),
                elapsed,
                usage.total_tokens if usage else 0,
            )

            return content_text

        except Exception as err:
            handle_api_error(err, start_time, "photo analysis")

    async def delete_remote_completions(
        self, response_ids: list[str], context: str = "cleanup"
    ) -> int:
        """Delete stored completion IDs from xAI server asynchronously.

        This method removes conversation chains from xAI's server-side storage.
        Should be called after clearing local memory to prevent orphaned data.

        Args:
            response_ids: List of response IDs to delete from xAI server
            context: Context string for logging (e.g., "clear_memory", "physical_delete")

        Returns:
            Number of successfully deleted completions
        """
        if not response_ids:
            return 0

        # Check if store_messages is enabled (no point deleting if feature is off)
        store_messages = self.entity._get_option(CONF_STORE_MESSAGES, RECOMMENDED_STORE_MESSAGES)
        if not store_messages:
            LOGGER.debug("%s: skipping remote deletion (store_messages=False)", context)
            return 0

        deleted_count = 0
        try:
            client = self.create_client()
            for pid in response_ids:
                try:
                    await client.chat.delete_stored_completion(pid)
                    deleted_count += 1
                    LOGGER.debug(
                        "%s: remote delete successful for %s", context, str(pid)[:8]
                    )
                except Exception as derr:
                    LOGGER.debug(
                        "%s: remote delete failed for %s: %s",
                        context,
                        str(pid)[:8],
                        derr,
                    )
            
            if deleted_count > 0:
                LOGGER.info(
                    "%s: deleted %d/%d completion IDs from server",
                    context,
                    deleted_count,
                    len(response_ids),
                )

        except Exception as cerr:
            LOGGER.warning("%s: remote deletion failed to start: %s", context, cerr)
        
        return deleted_count
