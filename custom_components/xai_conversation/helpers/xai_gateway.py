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

# Home Assistant imports
from homeassistant.components import conversation as ha_conversation

# xAI SDK imports (conditional) - DEFINED HERE TO BREAK CIRCULAR DEP
try:
    from xai_sdk import AsyncClient as XAI_CLIENT_CLASS
    from xai_sdk.chat import (
        user as xai_user,
        system as xai_system,
        assistant as xai_assistant,
        tool as xai_tool,
        image as xai_image,
    )
    from xai_sdk.search import SearchParameters as xai_search_parameters

    XAI_SDK_AVAILABLE = True
except ImportError:
    XAI_CLIENT_CLASS = None
    xai_user = None
    xai_system = None
    xai_assistant = None
    xai_tool = None
    xai_image = None
    xai_search_parameters = None
    XAI_SDK_AVAILABLE = False

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
    RECOMMENDED_LIVE_SEARCH,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_REASONING_EFFORT,
    RECOMMENDED_STORE_MESSAGES,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TIMEOUT,
    SUPPORTED_MODELS,
)
from ..exceptions import handle_api_error, raise_generic_error, raise_validation_error
from .response import save_response_metadata
from .log_time_services import LogTimeServices

if TYPE_CHECKING:
    from typing import Literal
    from xai_sdk import AsyncClient as XAIClient

    from ..entity import XAIBaseLLMEntity


class XAIGateway:
    """Single point of contact with xAI SDK.

    Centralizes ALL xAI SDK interactions to eliminate code duplication.
    Each entity creates its own gateway instance for isolated configuration.

    Responsibilities:
    - Validate API key
    - Create and cache xAI client with retry/keepalive configuration
    - Create xAI chat objects with model parameters
    - Execute stateless chats (AI Task)
    """

    def __init__(self, entity: XAIBaseLLMEntity) -> None:
        """Initialize gateway with entity reference.

        Args:
            entity: The XAIBaseLLMEntity instance that owns this gateway
        """
        self.entity = entity
        self._cached_client: XAIClient | None = None

    # ==========================================================================
    # FACTORY METHODS FOR MESSAGE ABSTRACTION
    # ==========================================================================

    @staticmethod
    def user_msg(content: str | list[str | Any]) -> Any:
        """Create a user message object."""
        if isinstance(content, list):
            return xai_user(*content)
        return xai_user(content)

    @staticmethod
    def system_msg(content: str) -> Any:
        """Create a system message object."""
        return xai_system(content)

    @staticmethod
    def assistant_msg(content: str, tool_calls: list | None = None) -> Any:
        """Create an assistant message object."""
        if tool_calls:
            # When tool_calls present, return dict format that SDK accepts
            return {
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls,
            }
        # When no tool_calls, use the SDK helper
        return xai_assistant(content)

    @staticmethod
    def tool_msg(content: str, tool_call_id: str, name: str) -> Any:
        """Create a tool output message object."""
        return xai_tool(content, tool_call_id=tool_call_id, name=name)

    @staticmethod
    def tool_def(name: str, description: str, parameters: dict) -> Any:
        """Create a tool definition object."""
        return xai_tool(name=name, description=description, parameters=parameters)

    @staticmethod
    def img_msg(data_uri_or_url: str) -> Any:
        """Create an image message object."""
        return xai_image(data_uri_or_url)

    async def close(self) -> None:
        """Close the cached xAI client and cleanup gRPC channels asynchronously.

        This should be called when the entity is being removed to ensure
        proper cleanup of network resources (gRPC channels).
        """
        if self._cached_client is not None:
            try:
                if hasattr(self._cached_client, "close"):
                    await self._cached_client.close()
                elif hasattr(self._cached_client, "__aexit__"):
                    await self._cached_client.__aexit__(None, None, None)
            except Exception as err:
                LOGGER.warning("Error closing xAI async client: %s", err)
            finally:
                self._cached_client = None

    @staticmethod
    async def async_validate_api_key(api_key: str) -> None:
        """Validate the API key by making a minimal chat request.

        Args:
            api_key: The xAI API key to validate

        Raises:
            ValueError: If SDK not available or API key invalid
        """
        from ..const import RECOMMENDED_CHAT_MODEL, RECOMMENDED_TIMEOUT
        from .. import xai_user

        try:
            client = XAIGateway.create_standalone_client(
                api_key=api_key, timeout=float(RECOMMENDED_TIMEOUT)
            )

            chat = client.chat.create(
                model=RECOMMENDED_CHAT_MODEL, max_tokens=1, temperature=0.1
            )
            chat.append(xai_user("ok"))
            await chat.sample()

        except ValueError as exc:
            raise ValueError(f"Failed to validate API credentials: {exc}") from exc
        except Exception as exc:
            raise ValueError(f"Failed to validate API credentials: {exc}") from exc

    @staticmethod
    def _get_channel_options() -> list:
        """Get gRPC channel options with retry policy and keepalive configuration.

        Returns:
            List of channel options tuples
        """
        retry_policy = json.dumps(
            {
                "methodConfig": [
                    {
                        "name": [{}],
                        "retryPolicy": {
                            "maxAttempts": 3,
                            "initialBackoff": "0.5s",
                            "maxBackoff": "5s",
                            "backoffMultiplier": 2,
                            "retryableStatusCodes": ["UNAVAILABLE"],
                        },
                    }
                ]
            }
        )

        return [
            ("grpc.service_config", retry_policy),
            ("grpc.keepalive_time_ms", 30000),
            ("grpc.keepalive_timeout_ms", 5000),
            ("grpc.keepalive_permit_without_calls", True),
            ("grpc.http2.max_pings_without_data", 0),
            ("grpc.http2.min_time_between_pings_ms", 10000),
            ("grpc.http2.min_ping_interval_without_data_ms", 300000),
        ]

    @staticmethod
    def create_standalone_client(api_key: str, timeout: float) -> XAIClient:
        """Create a standalone xAI client for services (not entity-bound).

        Args:
            api_key: xAI API key
            timeout: Request timeout in seconds

        Returns:
            XAIClient instance configured with retry and keepalive

        Raises:
            ValueError: If xAI SDK is not available
        """
        if not XAI_SDK_AVAILABLE or XAI_CLIENT_CLASS is None:
            raise ValueError("xAI SDK not available")

        return XAI_CLIENT_CLASS(
            api_key=api_key,
            timeout=timeout,
            channel_options=XAIGateway._get_channel_options(),
        )

    @staticmethod
    def create_standalone_chat(
        client: XAIClient,
        model: str,
        max_tokens: int,
        temperature: float,
        store_messages: bool,
        previous_response_id: str | None = None,
    ) -> Any:
        """Create a standalone xAI chat for services (not entity-bound).

        Args:
            client: xAI client instance
            model: Model name to use
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            store_messages: Whether to enable server-side memory
            previous_response_id: Optional ID for conversation chaining

        Returns:
            xAI Chat object configured with provided settings
        """
        chat_args = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "store_messages": store_messages,
        }

        if previous_response_id:
            chat_args["previous_response_id"] = previous_response_id

        LOGGER.debug(
            "Creating standalone chat: model=%s, max_tokens=%d, temperature=%.2f, store_messages=%s, prev_id=%s",
            model,
            max_tokens,
            temperature,
            store_messages,
            bool(previous_response_id),
        )

        return client.chat.create(**chat_args)

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

        if not XAI_SDK_AVAILABLE or XAI_CLIENT_CLASS is None:
            raise ValueError("xAI SDK not available")

        LOGGER.debug("Creating xAI client with configuration validation")
        model = self.entity._get_option(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)

        # Validate model (warning only, doesn't block)
        if model not in SUPPORTED_MODELS:
            LOGGER.warning(
                "Unknown model '%s', proceeding anyway (supported: %s)",
                model,
                SUPPORTED_MODELS,
            )

        timeout = self.entity._get_option(CONF_TIMEOUT, RECOMMENDED_TIMEOUT)

        # Validate timeout (warning only, doesn't block)
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            LOGGER.warning(
                "Invalid timeout value '%s' (type: %s), using default %s",
                timeout,
                type(timeout).__name__,
                RECOMMENDED_TIMEOUT,
            )
            timeout = RECOMMENDED_TIMEOUT

        LOGGER.debug(
            "Client configuration: timeout=%ss, API key=%s***",
            timeout,
            self.entity.entry.data["api_key"][:8],
        )

        client_kwargs = {
            "api_key": self.entity.entry.data["api_key"],
            "timeout": timeout,
            "channel_options": self._get_channel_options(),
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
        max_tokens = self.entity._get_int_option(
            CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS
        )
        temperature = self.entity._get_float_option(
            CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE
        )
        # Enable server-side memory when configured
        store_messages = bool(
            self.entity._get_option(CONF_STORE_MESSAGES, RECOMMENDED_STORE_MESSAGES)
        )

        # Configure Live Search if enabled and SDK available
        live_search_mode = self.entity._get_option(
            CONF_LIVE_SEARCH, RECOMMENDED_LIVE_SEARCH
        )
        search_parameters = None
        if live_search_mode != "off" and xai_search_parameters:
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
            LOGGER.debug(
                "Reasoning effort set to '%s' for model %s", reasoning_effort, model
            )

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
        extra_messages: list[Any] | None = None,
        service_type: str = "ai_task",
    ) -> None:
        """Process chat_log without memory - for AI Task only.

        This method is "dumb" and only sends pre-formatted messages to the API.
        Any I/O or message preparation (like reading images) must be done by the caller.

        Args:
            chat_log: The conversation chat log
            extra_messages: Optional list of pre-built xAI message objects (e.g. images) to add
            service_type: Service type for logging/metrics
        """
        context = {"mode": "stateless", "memory": "no_memory"}
        async with LogTimeServices(LOGGER, service_type, context) as timer:
            try:
                client = self.create_client()
                chat = self.create_chat(client, tools=None, previous_response_id=None)

                system_prompt_text = self.entity._get_option(CONF_PROMPT, "")
                if system_prompt_text:
                    chat.append(xai_system(system_prompt_text))

                # Add any pre-prepared extra messages (e.g. images from attachments)
                if extra_messages:
                    for msg in extra_messages:
                        chat.append(msg)

                for content in chat_log.content:
                    if isinstance(content, ha_conversation.UserContent):
                        # User content is just text in stateless mode
                        if content.content:
                            chat.append(xai_user(content.content))
                    elif isinstance(content, ha_conversation.AssistantContent):
                        chat.append(xai_assistant(content.content or ""))

                # Time the API call specifically
                async with timer.record_api_call():
                    response = await chat.sample()

                content_text = getattr(response, "content", "")
                usage = getattr(response, "usage", None)
                model = getattr(response, "model", None)
                store_messages = self.entity._get_option(
                    CONF_STORE_MESSAGES, RECOMMENDED_STORE_MESSAGES
                )

                await save_response_metadata(
                    hass=self.entity.hass,
                    entry_id=self.entity.entry.entry_id,
                    usage=usage,
                    model=model,
                    service_type=service_type,
                    store_messages=store_messages,
                )

                chat_log.content.append(
                    ha_conversation.AssistantContent(
                        agent_id=self.entity.entity_id, content=content_text
                    )
                )

            except Exception as err:
                handle_api_error(err, timer.start_time, "AI Task API call")

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
        store_messages = self.entity._get_option(
            CONF_STORE_MESSAGES, RECOMMENDED_STORE_MESSAGES
        )
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
