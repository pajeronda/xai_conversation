"""xAI Gateway - Single point of contact with xAI SDK.

This module provides XAIGateway class that centralizes ALL xAI SDK interactions
to eliminate code duplication across entity.py, services.py, and ai_task.py.

Architecture: Centralized Gateway initialized with ConfigEntry.
Acts as a factory for Clients and Chats, resolving configurations automatically
based on service type (conversation, code_fast, ai_task).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from homeassistant.components import conversation as ha_conversation

# xAI SDK imports (conditional)
try:
    from xai_sdk import AsyncClient as XAI_CLIENT_CLASS
    from xai_sdk import Client as XAI_CLIENT_CLASS_SYNC
    from xai_sdk.chat import (
        user as xai_user,
        system as xai_system,
        assistant as xai_assistant,
        tool as xai_tool,
        tool_result as xai_tool_result,
        image as xai_image,
    )
    from xai_sdk.tools import (
        web_search,
        x_search,
        code_execution,
    )

    XAI_SDK_AVAILABLE = True
except ImportError:
    XAI_CLIENT_CLASS = None
    XAI_CLIENT_CLASS_SYNC = None
    xai_user = None
    xai_system = None
    xai_assistant = None
    xai_tool = None
    xai_tool_result = None
    xai_image = None
    web_search = None
    x_search = None
    code_execution = None
    XAI_SDK_AVAILABLE = False

from .const import (
    CONF_API_HOST,
    CONF_CHAT_MODEL,
    CONF_IMAGE_MODEL,
    CONF_VISION_MODEL,
    CONF_MAX_TOKENS,
    CONF_REASONING_EFFORT,
    CONF_LIVE_SEARCH,
    CONF_STORE_MESSAGES,
    CONF_TEMPERATURE,
    DEFAULT_API_HOST,
    DOMAIN,
    LOGGER,
    REASONING_EFFORT_MODELS,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_IMAGE_MODEL,
    RECOMMENDED_VISION_MODEL,
    RECOMMENDED_LIVE_SEARCH,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_REASONING_EFFORT,
    RECOMMENDED_STORE_MESSAGES,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TIMEOUT,
    RECOMMENDED_GROK_CODE_FAST_MODEL,
    RECOMMENDED_GROK_CODE_FAST_OPTIONS,
    RECOMMENDED_AI_TASK_OPTIONS,
    RECOMMENDED_PIPELINE_OPTIONS,
    SUBENTRY_TYPE_AI_TASK,
    SUBENTRY_TYPE_CODE_TASK,
    SUBENTRY_TYPE_CONVERSATION,
)
from .exceptions import handle_api_error
from .helpers import (
    MemoryManager,
    PromptManager,
    build_system_prompt,
    save_response_metadata,
    LogTimeServices,
)


if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant
    from homeassistant.config_entries import ConfigEntry
    from xai_sdk import AsyncClient as XAIClient


from functools import wraps


def require_xai_sdk(func):
    """Decorator to ensure the xAI SDK is available before calling a method."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        """Wrapper that checks for SDK availability."""
        if not XAI_SDK_AVAILABLE:
            raise ValueError(
                "xAI SDK not available. Please install the xai-sdk package."
            )
        return func(*args, **kwargs)

    return wrapper


class XAIGateway:
    """Single point of contact with xAI SDK.

    Centralizes ALL xAI SDK interactions.
    Initialized with ConfigEntry to access global and sub-entry configurations.

    Responsibilities:
    - Validate API key
    - Create and cache xAI client with retry/keepalive configuration
    - Create xAI chat objects with automatic config resolution
    - Manage Server-Side Tools injection
    """

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize gateway with config entry.

        Args:
            hass: Home Assistant instance
            entry: Config entry containing all settings
        """
        self.hass = hass
        self.entry = entry
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
            return {
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls,
            }
        return xai_assistant(content)

    @staticmethod
    def tool_msg(content: str) -> Any:
        """Create a tool output message object."""
        return xai_tool_result(content)

    @staticmethod
    def tool_def(name: str, description: str, parameters: dict) -> Any:
        """Create a tool definition object."""
        return xai_tool(name=name, description=description, parameters=parameters)

    @staticmethod
    def img_msg(data_uri_or_url: str) -> Any:
        """Create an image message object."""
        return xai_image(data_uri_or_url)

    # ==========================================================================
    # API KEY VALIDATION (now part of create_client_from_api_key)
    # ==========================================================================

    async def close(self) -> None:
        """Close the cached xAI client and cleanup gRPC channels asynchronously."""
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
    def _get_channel_options() -> list:
        """Get gRPC channel options with retry policy and keepalive configuration."""
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
    @require_xai_sdk
    def create_client_from_api_key(
        api_key: str,
        timeout: float | None = None,
        api_host: str | None = None,
    ) -> XAIClient:
        """Create a standalone xAI client from an API key.

        Useful for tasks that don't have access to the ConfigEntry (e.g., ModelManager).
        """
        if not api_key or not api_key.startswith("xai-"):
            raise ValueError("Invalid API key format. Must start with 'xai-'")

        client_kwargs = {
            "api_key": api_key,
            "channel_options": XAIGateway._get_channel_options(),
        }
        if timeout:
            client_kwargs["timeout"] = timeout
        if api_host:
            client_kwargs["api_host"] = api_host

        return XAI_CLIENT_CLASS(**client_kwargs)

    @require_xai_sdk
    def create_client(self) -> XAIClient:
        """Create or return a cached xAI client.

        Returns:
            XAIClient instance with retry/keepalive configuration
        """
        if self._cached_client is not None:
            return self._cached_client

        api_key = self.entry.data.get("api_key")
        if not api_key:
            raise ValueError("API Key not found in config entry")

        # Use global timeout setting or default.
        # Check options first, then data.
        timeout = self.entry.options.get("timeout", RECOMMENDED_TIMEOUT)

        client_kwargs = {
            "api_key": api_key,
            "timeout": timeout,
            "channel_options": self._get_channel_options(),
        }

        # Check for API Host override in the main entry's data.
        # This makes the setting global and deterministic for the client instance.
        api_host = self.entry.data.get(CONF_API_HOST, DEFAULT_API_HOST)

        if api_host:
            client_kwargs["api_host"] = api_host

        LOGGER.debug(
            "Creating xAI client (timeout=%ss, host=%s)",
            timeout,
            api_host or "default",
        )
        client = XAI_CLIENT_CLASS(**client_kwargs)
        self._cached_client = client
        return client

    @require_xai_sdk
    def _build_server_side_tools(
        self, live_search_mode: str, model: str, model_target: str
    ) -> list | None:
        """Build list of server-side tools based on mode.

        Note: `model` and `model_target` are unused for now but kept for signature compatibility.
        """
        if live_search_mode == "off":
            return None
        elif live_search_mode == "web search":
            return [web_search()]
        elif live_search_mode == "x search":
            return [x_search()]
        elif live_search_mode in ["full", "auto", "on"]:
            return [web_search(), x_search(), code_execution()]

        return None

    def get_service_config(
        self,
        service_type: str,
        subentry_id: str | None = None,
    ) -> dict:
        """Resolve configuration source for a specific service and returns it.

        Merges static data with runtime options to ensure user settings are respected.

        Args:
            service_type: "conversation", "code_fast", "ai_task", "ask"
            subentry_id: Required for "conversation" (multi-entity), ignored for others

        Returns:
            Dictionary with merged parameters (data + options).
        """

        # Helper to merge data and options
        def _merge_config(subentry) -> dict:
            config = dict(subentry.data)
            if hasattr(subentry, "options") and subentry.options:
                config.update(subentry.options)
            return config

        # 1. Handle 'ask' and 'conversation' (which map to conversation subentries)
        if service_type in ("conversation", "ask"):
            # If subentry_id is provided, use it
            if subentry_id and subentry_id in self.entry.subentries:
                return _merge_config(self.entry.subentries[subentry_id])

            # For 'ask', if no ID, find the first conversation subentry
            for subentry in self.entry.subentries.values():
                if subentry.subentry_type == SUBENTRY_TYPE_CONVERSATION:
                    return _merge_config(subentry)

            # Fallback for 'ask' if no conversation subentry exists
            if service_type == "ask":
                return RECOMMENDED_PIPELINE_OPTIONS

            if not subentry_id:
                raise ValueError("subentry_id required for conversation service")

            LOGGER.warning("Subentry %s not found, using empty config", subentry_id)
            return {}

        elif service_type == "code_fast":
            # Find the singleton code_task subentry
            for subentry in self.entry.subentries.values():
                if subentry.subentry_type == SUBENTRY_TYPE_CODE_TASK:
                    return _merge_config(subentry)
            return RECOMMENDED_GROK_CODE_FAST_OPTIONS

        elif service_type == "ai_task":
            # Find the singleton ai_task subentry
            for subentry in self.entry.subentries.values():
                if subentry.subentry_type == SUBENTRY_TYPE_AI_TASK:
                    return _merge_config(subentry)
            return RECOMMENDED_AI_TASK_OPTIONS

        return {}

    def _resolve_chat_parameters(
        self,
        service_type: str,
        subentry_id: str | None = None,
        model_target: str = "chat",
        model_override: str | None = None,
        max_tokens_override: int | None = None,
        temperature_override: float | None = None,
        store_messages_override: bool | None = None,
        system_prompt_override: str | None = None,
        entity: Any | None = None,
    ) -> dict[str, Any]:
        """Resolve final chat parameters from config, defaults, and overrides.

        Centralizes the logic for determining model, temperature, etc.
        """
        # 1. Get raw config
        config = self.get_service_config(service_type, subentry_id)

        # 2. Resolve Model
        if model_override:
            model = model_override
        elif model_target == "vision":
            model = config.get(CONF_VISION_MODEL, RECOMMENDED_VISION_MODEL)
        elif model_target == "image":
            model = config.get(CONF_IMAGE_MODEL, RECOMMENDED_IMAGE_MODEL)
        elif service_type == "code_fast":
            model = config.get(CONF_CHAT_MODEL, RECOMMENDED_GROK_CODE_FAST_MODEL)
        else:  # "conversation", "ai_task", "ask"
            model = config.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)

        # Fallback if still None (e.g., entity specific)
        if model is None and entity:
            model = entity._get_option(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)

        # 3. Resolve Other Parameters
        temperature = (
            temperature_override
            if temperature_override is not None
            else float(config.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE))
        )
        max_tokens = (
            max_tokens_override
            if max_tokens_override is not None
            else int(config.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS))
        )

        if store_messages_override is not None:
            store_messages = store_messages_override
        else:
            store_messages = bool(
                config.get(CONF_STORE_MESSAGES, RECOMMENDED_STORE_MESSAGES)
            )

        reasoning_effort = config.get(
            CONF_REASONING_EFFORT, RECOMMENDED_REASONING_EFFORT
        )

        return {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "store_messages": store_messages,
            "reasoning_effort": reasoning_effort,
            "config": config,
        }

    async def create_chat(
        self,
        service_type: str,
        subentry_id: str | None = None,
        model_target: str = "chat",
        client_tools: list | None = None,
        system_prompt_override: str | None = None,
        entity: Any | None = None,
        mode_override: str | None = None,
        # Overrides
        model_override: str | None = None,
        max_tokens_override: int | None = None,
        temperature_override: float | None = None,
        store_messages_override: bool | None = None,
        # For conv_key generation
        scope: str | None = None,
        identifier: str | None = None,
    ) -> tuple[Any, str | None]:
        """Create an xAI chat object configured for the specific service.

        This is the UNIFIED factory method for all chat interactions.
        """
        # 1. Resolve Parameters
        params = self._resolve_chat_parameters(
            service_type=service_type,
            subentry_id=subentry_id,
            model_target=model_target,
            system_prompt_override=system_prompt_override,
            entity=entity,
            model_override=model_override,
            max_tokens_override=max_tokens_override,
            temperature_override=temperature_override,
            store_messages_override=store_messages_override,
        )
        config = params["config"]
        model = params["model"]

        # 2. Determine mode
        if mode_override:
            mode = mode_override
        elif service_type == "code_fast":
            mode = "code"
        elif service_type == "ai_task":
            # For ai_task, check model_target for vision-specific handling
            mode = "vision" if model_target == "vision" else "ai_task"
        else:
            mode = "tools"

        # 3. Build System Prompt if not resolved yet
        system_prompt = system_prompt_override or build_system_prompt(
            entity, mode, self.hass, config
        )

        # 4. Handle Conversation Key and Memory
        conv_key = None
        previous_response_id = None
        if params["store_messages"] and scope and identifier:
            prompt_hash = PromptManager(config=config, mode=mode).get_stable_hash()
            allow_control = config.get("allow_smart_home_control", True)
            conv_key = MemoryManager.generate_key(
                scope, identifier, mode, prompt_hash, allow_control
            )
            memory = self.hass.data[DOMAIN]["conversation_memory"]
            previous_response_id = await memory.async_get_last_response_id(conv_key)

        # 5. Build Server-Side Tools
        live_search_mode = config.get(CONF_LIVE_SEARCH, RECOMMENDED_LIVE_SEARCH)
        server_tools = self._build_server_side_tools(
            live_search_mode, model, model_target
        )
        all_tools = []
        if server_tools:
            all_tools.extend(server_tools)
        if client_tools:
            all_tools.extend(client_tools)

        # 6. Get Client and build Chat Arguments
        client = self.create_client()
        chat_args = {
            "model": model,
            "max_tokens": params["max_tokens"],
            "temperature": params["temperature"],
            "store_messages": params["store_messages"],
        }

        if system_prompt and not previous_response_id:
            LOGGER.debug("=" * 80)
            LOGGER.debug(
                "SYSTEM PROMPT | service=%s mode=%s | length=%d",
                service_type,
                mode,
                len(system_prompt) if system_prompt else 0,
            )
            LOGGER.debug("=" * 80)
            LOGGER.debug("%s", system_prompt)
            LOGGER.debug("=" * 80)
            chat_args["messages"] = [self.system_msg(system_prompt)]

        if model in REASONING_EFFORT_MODELS:
            chat_args["reasoning_effort"] = params["reasoning_effort"]

        if previous_response_id:
            chat_args["previous_response_id"] = previous_response_id

        if all_tools:
            chat_args["tools"] = all_tools

        LOGGER.debug(
            "Creating chat [%s:%s]: model=%s, store_msgs=%s, conv_key=%s",
            service_type,
            model_target,
            model,
            params["store_messages"],
            conv_key,
        )

        chat_object = client.chat.create(**chat_args)
        return chat_object, conv_key

    async def execute_stateless_chat(
        self,
        input_data: ha_conversation.ChatLog | str | None,
        service_type: str = "ai_task",
        model_target: str = "chat",
        system_prompt: str | None = None,
        extra_messages: list[Any] | None = None,
        model_override: str | None = None,
        max_tokens_override: int | None = None,
        temp_override: float | None = None,
        mixed_content: list[Any] | None = None,
        entity: Any | None = None,
    ) -> str | None:
        """Execute a stateless chat (no memory, one-shot).

        Unified method for:
        1. AI Task (structured data generation) -> passes ChatLog
        2. Ask Service (simple Q&A) -> passes input_data string + system_prompt
        3. Photo Analysis (mixed content) -> passes mixed_content list
        """
        # Create chat object reusing the factory logic
        # We explicitly disable memory (store_messages=False)
        chat, _ = await self.create_chat(
            service_type=service_type,
            model_target=model_target,
            system_prompt_override=system_prompt,
            model_override=model_override,
            max_tokens_override=max_tokens_override,
            temperature_override=temp_override,
            store_messages_override=False,
            entity=entity,
        )

        context = {"mode": "stateless", "service": service_type}
        async with LogTimeServices(LOGGER, service_type, context) as timer:
            try:
                # Handle mixed content first (e.g., text and images)
                if mixed_content:
                    chat.append(self.user_msg(mixed_content))
                else:
                    # Add extra messages (e.g. images) if provided
                    if extra_messages:
                        for msg in extra_messages:
                            chat.append(msg)

                    # Handle Input Data
                    if isinstance(input_data, ha_conversation.ChatLog):
                        for content in input_data.content:
                            if isinstance(content, ha_conversation.UserContent):
                                if content.content:
                                    chat.append(xai_user(content.content))
                            elif isinstance(content, ha_conversation.AssistantContent):
                                chat.append(xai_assistant(content.content or ""))
                    elif input_data:
                        # String input (Ask Service)
                        chat.append(self.user_msg(input_data))

                # Execute
                async with timer.record_api_call():
                    response = await chat.sample()

                content_text = getattr(response, "content", "")

                # Log completion
                await self.async_log_completion(
                    response=response,
                    service_type=service_type,
                    store_messages_override=False,
                    model=model_override,  # Pass override if we have it, else it auto-resolves
                    model_target=model_target,
                )

                # Update ChatLog if passed
                if isinstance(input_data, ha_conversation.ChatLog):
                    input_data.content.append(
                        ha_conversation.AssistantContent(
                            agent_id="xai_conversation",
                            content=content_text,
                        )
                    )

                return content_text

            except Exception as err:
                handle_api_error(err, timer.start_time, f"{service_type} API call")
                raise

    async def async_log_completion(
        self,
        response: Any,
        service_type: str,
        subentry_id: str | None = None,
        conv_key: str | None = None,
        store_messages_override: bool | None = None,
        mode: str | None = None,
        is_fallback: bool = False,
        entity: Any | None = None,
        citations: list | None = None,
        num_sources_used: int = 0,
        model_target: str = "chat",
        model: str | None = None,
        await_save: bool = False,
    ) -> None:
        """Centralized logging of chat completion metadata.

        Resolves configuration (to fix KeyError bugs) and delegates to helper.
        """
        # 1. Resolve Parameters (Model & Config) if model not explicitly provided
        if model is None:
            # We let the central resolver determine the correct model based on target/service
            params = self._resolve_chat_parameters(
                service_type=service_type,
                subentry_id=subentry_id,
                model_target=model_target,
                entity=entity,
            )
            model = params["model"]
            config = params["config"]
        else:
            # We still need config for store_messages default
            config = self.get_service_config(service_type, subentry_id)

        if store_messages_override is not None:
            store_messages = store_messages_override
        else:
            store_messages = config.get(CONF_STORE_MESSAGES, RECOMMENDED_STORE_MESSAGES)

        # 3. Extract data from SDK response
        if isinstance(response, dict):
            usage = response.get("usage")
            server_tool_usage = response.get("server_side_tool_usage")
            response_id = response.get("id")
        else:
            usage = getattr(response, "usage", None)
            server_tool_usage = getattr(response, "server_side_tool_usage", None)
            response_id = getattr(response, "id", None)

        # 4. Delegate to helper
        await save_response_metadata(
            hass=self.hass,
            entry_id=self.entry.entry_id,
            usage=usage,
            model=model,
            service_type=service_type,
            server_side_tool_usage=server_tool_usage,
            conv_key=conv_key,
            response_id=response_id,
            store_messages=store_messages,
            mode=mode,
            is_fallback=is_fallback,
            entity=entity,
            citations=citations,
            num_sources_used=num_sources_used,
            await_save=await_save,
        )

    async def delete_remote_completions(
        self, response_ids: list[str], context: str = "cleanup"
    ) -> int:
        """Delete stored completion IDs from xAI server asynchronously."""
        if not response_ids:
            return 0

        # Note: We assume store_messages=True if we are trying to delete something.
        # We don't check config here because if we have IDs, they exist and should be deleted.

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
