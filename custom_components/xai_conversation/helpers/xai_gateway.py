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

# xAI SDK imports (conditional)
try:
    from xai_sdk import AsyncClient as XAI_CLIENT_CLASS
    from xai_sdk.chat import (
        user as xai_user,
        system as xai_system,
        assistant as xai_assistant,
        tool as xai_tool,
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
    xai_user = None
    xai_system = None
    xai_assistant = None
    xai_tool = None
    xai_image = None
    web_search = None
    x_search = None
    code_execution = None
    XAI_SDK_AVAILABLE = False

from ..const import (
    CONF_API_HOST,
    CONF_CHAT_MODEL,
    CONF_IMAGE_MODEL,
    CONF_VISION_MODEL,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_REASONING_EFFORT,
    CONF_LIVE_SEARCH,
    CONF_STORE_MESSAGES,
    CONF_TEMPERATURE,
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
    RECOMMENDED_GROK_CODE_FAST_OPTIONS,
    RECOMMENDED_AI_TASK_OPTIONS,
    SUBENTRY_TYPE_AI_TASK,
    SUBENTRY_TYPE_CODE_TASK,
)
from ..exceptions import handle_api_error
from .response import save_response_metadata
from .log_time_services import LogTimeServices

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant
    from homeassistant.config_entries import ConfigEntry
    from xai_sdk import AsyncClient as XAIClient
    from homeassistant.components import conversation as ha_conversation


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

    # ==========================================================================
    # API KEY VALIDATION
    # ==========================================================================

    @staticmethod
    async def async_validate_api_key(api_key: str) -> None:
        """Validate xAI API key format.

        Args:
            api_key: The API key to validate

        Raises:
            ValueError: If API key is invalid or SDK is not installed
        """
        try:
            from xai_sdk import Client as XAIClient
        except ImportError as err:
            raise ValueError("xai_sdk not installed") from err

        # Basic format validation
        if not api_key or not api_key.startswith("xai-"):
            raise ValueError("Invalid API key format. Must start with 'xai-'")

        # Try to create a client to validate the key
        try:
            client = XAIClient(api_key=api_key)
            # The client creation validates the key format
            # We don't need to make an actual API call
            del client
        except Exception as err:
            raise ValueError(f"Invalid API key: {err}") from err

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
    def create_client_from_api_key(api_key: str, timeout: float | None = None) -> XAIClient:
        """Create a standalone xAI client from an API key.

        Useful for tasks that don't have access to the ConfigEntry (e.g., ModelManager).
        """
        if not XAI_SDK_AVAILABLE or XAI_CLIENT_CLASS is None:
            raise ValueError("xAI SDK not available")

        client_kwargs = {
            "api_key": api_key,
            "channel_options": XAIGateway._get_channel_options(),
        }
        if timeout:
            client_kwargs["timeout"] = timeout

        return XAI_CLIENT_CLASS(**client_kwargs)

    def create_client(self) -> XAIClient:
        """Create or return a cached xAI client.

        Returns:
            XAIClient instance with retry/keepalive configuration
        """
        if self._cached_client is not None:
            return self._cached_client

        if not XAI_SDK_AVAILABLE or XAI_CLIENT_CLASS is None:
            raise ValueError("xAI SDK not available")

        api_key = self.entry.data.get("api_key")
        if not api_key:
            raise ValueError("API Key not found in config entry")

        # Use global timeout setting or default
        timeout = RECOMMENDED_TIMEOUT  # Default fallback
        # Ideally, we could fetch a global timeout from options if it existed
        
        client_kwargs = {
            "api_key": api_key,
            "timeout": timeout,
            "channel_options": self._get_channel_options(),
        }
        
        # Check for API Host override
        api_host = None
        # Try to find api_host in options flow (global) or subentries
        # For simplicity, we check if ANY subentry has an override, or use default
        for subentry in self.entry.subentries.values():
             if CONF_API_HOST in subentry.data:
                 api_host = subentry.data[CONF_API_HOST]
                 break
        
        if api_host:
            client_kwargs["api_host"] = api_host

        LOGGER.debug("Creating xAI client (timeout=%ss)", timeout)
        client = XAI_CLIENT_CLASS(**client_kwargs)
        self._cached_client = client
        return client

    def get_service_config(
        self, service_type: str, subentry_id: str | None = None, model_target: str = "chat"
    ) -> dict:
        """Resolve configuration for a specific service and mode.

        Args:
            service_type: "conversation", "code_fast", "ai_task"
            subentry_id: Required for "conversation" (multi-entity), ignored for others
            model_target: "chat", "vision", or "image" (determines model key)

        Returns:
            Dictionary with resolved parameters (model, temp, tools, etc.)
        """
        config_data = {}
        
        # 1. Locate Subentry Data
        if service_type == "conversation":
            if not subentry_id:
                raise ValueError("subentry_id required for conversation service")
            if subentry_id in self.entry.subentries:
                config_data = self.entry.subentries[subentry_id].data
            else:
                LOGGER.warning("Subentry %s not found, using defaults", subentry_id)

        elif service_type == "code_fast":
            # Find the singleton code_task subentry
            for subentry in self.entry.subentries.values():
                if subentry.subentry_type == SUBENTRY_TYPE_CODE_TASK:
                    config_data = subentry.data
                    break
            if not config_data:
                config_data = RECOMMENDED_GROK_CODE_FAST_OPTIONS

        elif service_type == "ai_task":
             # Find the singleton ai_task subentry
            for subentry in self.entry.subentries.values():
                if subentry.subentry_type == SUBENTRY_TYPE_AI_TASK:
                    config_data = subentry.data
                    break
            if not config_data:
                config_data = RECOMMENDED_AI_TASK_OPTIONS
        
        # 2. Resolve Parameters with Defaults
        resolved = {}
        
        # Model
        if model_target == "vision":
            resolved["model"] = config_data.get(CONF_VISION_MODEL, RECOMMENDED_VISION_MODEL)
        elif model_target == "image":
            resolved["model"] = config_data.get(CONF_IMAGE_MODEL, RECOMMENDED_IMAGE_MODEL)
        else:
            resolved["model"] = config_data.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)

        resolved["temperature"] = float(config_data.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE))
        resolved["max_tokens"] = int(config_data.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS))
        resolved["store_messages"] = bool(config_data.get(CONF_STORE_MESSAGES, RECOMMENDED_STORE_MESSAGES))
        resolved["reasoning_effort"] = config_data.get(CONF_REASONING_EFFORT, RECOMMENDED_REASONING_EFFORT)
        
        # Tools (Server-side) configuration
        # Vision/Image modes typically don't use search tools by default, but we respect config
        live_search_mode = config_data.get(CONF_LIVE_SEARCH, RECOMMENDED_LIVE_SEARCH)
        
        # If model_target is 'image', tools are irrelevant for the image API endpoint
        if model_target == "image":
             resolved["server_tools"] = None
        else:
             resolved["server_tools"] = self._build_server_side_tools(live_search_mode)

        return resolved

    def _build_server_side_tools(self, live_search_mode: str) -> list | None:
        """Build list of server-side tools based on mode."""
        if not XAI_SDK_AVAILABLE or web_search is None:
            return None

        if live_search_mode == "off":
            return None
        elif live_search_mode == "web search":
            return [web_search()]
        elif live_search_mode == "x search":
            return [x_search()]
        elif live_search_mode in ["full", "auto", "on"]:
            return [web_search(), x_search(), code_execution()]
        
        return None

    def create_chat(
        self,
        service_type: str,
        subentry_id: str | None = None,
        model_target: str = "chat",
        previous_response_id: str | None = None,
        client_tools: list | None = None,
    ) -> Any:
        """Create an xAI chat object configured for the specific service.

        This is the unified factory method for all chat interactions.

        Args:
            service_type: "conversation", "code_fast", "ai_task"
            subentry_id: Required for "conversation"
            model_target: "chat" or "vision"
            previous_response_id: Optional ID for chaining
            client_tools: Optional list of Client-Side (HA) tools to merge

        Returns:
            Configured xAI Chat object
        """
        # 1. Resolve Configuration
        config = self.get_service_config(service_type, subentry_id, model_target)
        
        # 2. Get Client
        client = self.create_client()
        
        # 3. Merge Tools
        server_tools = config["server_tools"]
        all_tools = []
        if server_tools:
            all_tools.extend(server_tools)
        if client_tools:
            all_tools.extend(client_tools)
        
        # 4. Build Chat Arguments
        chat_args = {
            "model": config["model"],
            "max_tokens": config["max_tokens"],
            "temperature": config["temperature"],
            "store_messages": config["store_messages"],
        }

        # Add reasoning_effort only for supported models
        if config["model"] in REASONING_EFFORT_MODELS:
            chat_args["reasoning_effort"] = config["reasoning_effort"]

        if previous_response_id:
            chat_args["previous_response_id"] = previous_response_id

        if all_tools:
            chat_args["tools"] = all_tools

        LOGGER.debug(
            "Creating chat [%s:%s]: model=%s, server_tools=%d, client_tools=%d, store_msgs=%s",
            service_type, model_target, config["model"], 
            len(server_tools) if server_tools else 0,
            len(client_tools) if client_tools else 0,
            config["store_messages"]
        )

        return client.chat.create(**chat_args)

    async def execute_stateless_chat(
        self,
        chat_log: ha_conversation.ChatLog,
        extra_messages: list[Any] | None = None,
        service_type: str = "ai_task",
        model_target: str = "chat",
    ) -> None:
        """Process chat_log without memory (AI Task)."""
        # Only for AI Task logic
        
        context = {"mode": "stateless", "service": service_type}
        async with LogTimeServices(LOGGER, service_type, context) as timer:
            try:
                # Use unified create_chat (injects tools automatically!)
                chat = self.create_chat(
                    service_type=service_type,
                    model_target=model_target,
                    previous_response_id=None
                )

                # TODO: Retrieve system prompt from config using get_service_config logic? 
                # For now, we manually fetch prompt from config to match legacy logic, 
                # but cleanly via the same config dictionary.
                config = self.get_service_config(service_type, model_target=model_target)
                
                # We need to fetch the custom prompt text which is NOT in the standard config dict above
                # This part is a bit tricky as PROMPT is in the raw subentry data.
                # Re-fetch raw data for prompt text:
                raw_data = self._get_raw_subentry_data(service_type)
                
                # Determine prompt key based on model target
                prompt_key = CONF_PROMPT
                if model_target == "vision":
                    prompt_key = "vision_prompt" # Hardcoded key from const.py CONF_VISION_PROMPT

                system_prompt_text = raw_data.get(prompt_key, "")
                if system_prompt_text:
                    chat.append(xai_system(system_prompt_text))

                if extra_messages:
                    for msg in extra_messages:
                        chat.append(msg)

                for content in chat_log.content:
                    if isinstance(content, ha_conversation.UserContent):
                        if content.content:
                            chat.append(xai_user(content.content))
                    elif isinstance(content, ha_conversation.AssistantContent):
                        chat.append(xai_assistant(content.content or ""))

                async with timer.record_api_call():
                    response = await chat.sample()

                content_text = getattr(response, "content", "")
                usage = getattr(response, "usage", None)
                model = getattr(response, "model", None)
                server_tool_usage = getattr(response, "server_side_tool_usage", None)

                await save_response_metadata(
                    hass=self.hass,
                    entry_id=self.entry.entry_id,
                    usage=usage,
                    model=model,
                    service_type=service_type,
                    store_messages=config["store_messages"],
                    server_side_tool_usage=server_tool_usage,
                )

                chat_log.content.append(
                    ha_conversation.AssistantContent(
                        agent_id="xai_conversation", # Generic ID
                        content=content_text
                    )
                )

            except Exception as err:
                handle_api_error(err, timer.start_time, f"{service_type} API call")

    def _get_raw_subentry_data(self, service_type: str, subentry_id: str = None) -> dict:
        """Helper to get raw data for prompts (internal use)."""
        if service_type == "ai_task":
             for subentry in self.entry.subentries.values():
                if subentry.subentry_type == SUBENTRY_TYPE_AI_TASK:
                    return subentry.data
        return {} # Fallback

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
                    LOGGER.debug("%s: remote delete successful for %s", context, str(pid)[:8])
                except Exception as derr:
                    LOGGER.debug("%s: remote delete failed for %s: %s", context, str(pid)[:8], derr)

            if deleted_count > 0:
                LOGGER.info("%s: deleted %d/%d completion IDs from server", context, deleted_count, len(response_ids))

        except Exception as cerr:
            LOGGER.warning("%s: remote deletion failed to start: %s", context, cerr)

        return deleted_count