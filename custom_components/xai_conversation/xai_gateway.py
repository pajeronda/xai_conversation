"""xAI Gateway - Single point of contact with xAI SDK.

This module provides XAIGateway class that centralizes ALL xAI SDK interactions
to eliminate code duplication across entity.py, services.py, and ai_task.py.

Architecture: Centralized Gateway initialized with ConfigEntry.
Acts as a factory for Clients and Chats, resolving configurations automatically
based on service type (conversation, ai_task).
"""

from __future__ import annotations

import json
import contextlib
from dataclasses import replace
from typing import TYPE_CHECKING, Any

# xAI SDK imports (conditional)
try:
    from xai_sdk import AsyncClient as XAI_CLIENT_CLASS
    from xai_sdk.tools import (
        get_tool_call_type as get_tool_call_type_sdk,
    )
    from xai_sdk.proto import chat_pb2

    XAI_SDK_AVAILABLE = True
except ImportError:
    XAI_CLIENT_CLASS = None
    get_tool_call_type_sdk = None
    chat_pb2 = None
    XAI_SDK_AVAILABLE = False

from .const import (
    CONF_API_HOST,
    CONF_TIMEOUT,
    DEFAULT_API_HOST,
    LOGGER,
    DOMAIN,
    RECOMMENDED_TIMEOUT,
)
from .exceptions import handle_api_error, raise_config_error, raise_generic_error
from .helpers import (
    build_session_context_info,
    LogTimeServices,
    extract_response_metadata,
    format_citations,
    XAIModelManager,
    resolve_memory_context,
    async_log_completion,
    resolve_chat_parameters,
    prepare_sdk_payload,
    ChatOptions,
    assemble_chat_args,
    log_api_request,
    PromptManager,
)


if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant
    from homeassistant.config_entries import ConfigEntry
    from xai_sdk import AsyncClient as XAIClient


from functools import wraps


def require_xai_sdk(func):
    """Decorator to check if xAI SDK is installed."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not XAI_SDK_AVAILABLE:
            raise_generic_error(
                "xAI SDK not installed. Please install 'xai-sdk' package."
            )
        return func(*args, **kwargs)

    return wrapper


class XAIGateway:
    """Centralized gateway for xAI SDK interactions.

    Handles authentication, client lifecycle management, configuration resolution,
    and telemetry logging for all integration components.
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
        self._prompt_manager = PromptManager()

    @property
    def prompt_manager(self) -> PromptManager:
        """Get the shared prompt manager."""
        return self._prompt_manager

    # SDK authentication and client management

    # Authenticated client management

    # ==========================================================================
    # API KEY & CLIENT FACTORY
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
                LOGGER.warning("[gateway] close client error: %s", err)
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
        """Create a standalone xAI client.

        Creates a new client instance (and gRPC channel) on each call.
        Intended for occasional or one-off operations where a shared authentication scope is not available.
        """
        if not api_key or not api_key.startswith("xai-"):
            raise_config_error("api_key", "Must start with 'xai-'")

        client_kwargs = {
            "api_key": api_key,
            "channel_options": XAIGateway._get_channel_options(),
        }
        if timeout:
            client_kwargs["timeout"] = timeout
        if api_host:
            client_kwargs["api_host"] = api_host

        return XAI_CLIENT_CLASS(**client_kwargs)

    @staticmethod
    @require_xai_sdk
    async def async_validate_api_key(
        api_key: str,
        api_host: str | None = None,
    ) -> dict[str, Any]:
        """Validate API key and return account info.

        Args:
            api_key: The xAI API key to validate
            api_host: Optional API host override

        Returns:
            Dictionary with key metadata (name, team_id, etc.)

        Raises:
            ValueError: If key is invalid, disabled, or blocked.
            Exception: For network or other API errors.
        """
        if not api_key or not api_key.startswith("xai-"):
            # We use the key as the error code for the config flow to map to strings.json
            raise ValueError("invalid_api_key")

        client = XAIGateway.create_client_from_api_key(
            api_key=api_key,
            api_host=api_host or DEFAULT_API_HOST,
        )

        try:
            info = await client.auth.get_api_key_info()

            if info.disabled:
                raise ValueError("api_key_disabled")
            if info.api_key_blocked:
                raise ValueError("api_key_blocked")
            if info.team_blocked:
                raise ValueError("team_blocked")

            return {
                "name": info.name,
                "team_id": info.team_id,
                "user_id": info.user_id,
                "api_key_id": info.api_key_id,
            }
        finally:
            await client.close()

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
            raise_config_error("api_key", "Not found in config entry")

        # Use global timeout setting or default (provided by init_manager).
        timeout = self.entry.options.get(CONF_TIMEOUT, RECOMMENDED_TIMEOUT)

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
            "[gateway] create client: timeout=%ss host=%s",
            timeout,
            api_host or "default",
        )
        client = XAI_CLIENT_CLASS(**client_kwargs)
        self._cached_client = client
        return client

    async def async_update_models(self) -> None:
        """Fetch and sync available models from xAI."""
        # Gateway executes ModelManager logic as a helper
        client = self.create_client()
        await XAIModelManager(self.hass).async_update_models(client)

    # ==========================================================================
    # Chat Lifecycle Factory
    # ==========================================================================

    @require_xai_sdk
    async def create_chat(
        self,
        service_type: str,
        subentry_id: str | None = None,
        options: ChatOptions | None = None,
        entity: Any | None = None,
    ) -> tuple[Any, str | None, str, ChatOptions]:
        """Unified factory for all chat interactions."""
        params = resolve_chat_parameters(service_type, self.entry, subentry_id, options)
        model, mode = params.model, params.mode

        # 1. Prompt and Memory Context resolution
        system_prompt = None

        # Retrieve orchestrator first (needed for prompt hash calculation)
        orchestrator = None
        if entity and hasattr(entity, "_tools_processor"):
            orchestrator = getattr(entity._tools_processor, "_orchestrator", None)

        # Calculate prompt hash using gateway's prompt_manager with full context
        prompt_hash = self.prompt_manager.get_prompt_hash(mode, params.config, orchestrator)

        conv_key, prev_id, stored_hash = await resolve_memory_context(
            self.hass, mode, params, prompt_hash
        )

        # 2. Determine if System Prompt is required (New session or Dynamic Update)
        force_system = not prev_id or (prompt_hash and stored_hash != prompt_hash)
        if force_system:
            reason = "new session" if not prev_id else "prompt updated"
            LOGGER.debug(
                "[gateway] prompt: INJECTING SYSTEM PROMPT (reason: %s)", reason
            )
            system_prompt = params.system_prompt or self.prompt_manager.get_prompt(
                mode, params.config, orchestrator=orchestrator
            )

            if prev_id and prompt_hash != stored_hash:
                LOGGER.info(
                    "[gateway] system prompt updated mid-conversation (hash: %s -> %s)",
                    stored_hash[:8] if stored_hash else "none",
                    prompt_hash[:8],
                )
        else:
            LOGGER.debug("[gateway] prompt: NO PROMPT SENT (using server context)")

        # 3. ZDR encrypted blob retrieval (Session restoration)
        encrypted_blob = None
        if params.use_encrypted_content and conv_key:
            memory = self.hass.data[DOMAIN]["conversation_memory"]
            encrypted_blob = await memory.async_get_encrypted_blob(conv_key)

        # 4. Build SDK payload via centralized helper
        sdk_messages = prepare_sdk_payload(
            messages=params.messages or [],
            params=params,
            system_prompt=system_prompt,
            session_context=build_session_context_info(self.hass, params.config),
            encrypted_blob=encrypted_blob,
        )

        # 5. Finalize and log request
        chat_args = assemble_chat_args(
            params,
            sdk_messages,
            store_messages=params.store_messages,
            previous_response_id=prev_id,
        )
        log_api_request(
            sdk_messages, model, service_type, params=params, is_stateless=False
        )

        params.prompt_hash = prompt_hash
        return self.create_client().chat.create(**chat_args), conv_key, model, params

    @require_xai_sdk
    async def async_generate_image(
        self,
        prompt: str,
        model: str,
        options: ChatOptions | None = None,
        entity: Any | None = None,
    ) -> Any:
        """Centralized image generation via xAI SDK.

        This method ensures image generation follows the same telemetry
        and logging patterns as chat completions.
        """
        client = self.create_client()
        opts = options or ChatOptions()

        # Use provided timer or create a new one
        if opts.timer:
            timer_cm = contextlib.nullcontext(opts.timer)
        else:
            timer_cm = LogTimeServices(
                LOGGER,
                "image_generation",
                {"model": model, "prompt_length": len(prompt)},
            )

        async with timer_cm as timer:
            async with timer.record_api_call():
                response = await client.image.sample(
                    model=model,
                    prompt=prompt,
                    image_format="base64",
                )

            # Log completion for billing/usage
            await async_log_completion(
                response=response,
                service_type="ai_task",
                options=replace(opts, timer=timer),
                model_name=model,
                entity=entity,
            )
            return response

    async def execute_stateless_chat(
        self,
        messages: list[dict],
        service_type: str = "ai_task",
        options: ChatOptions | None = None,
        entity: Any | None = None,
        hass: Any | None = None,
    ) -> str | None:
        """Execute a one-shot chat and return content.

        Stateless: uses parameters as passed, with configuration lookups for defaults.
        """
        params = resolve_chat_parameters(service_type, self.entry, options=options)
        model = params.model
        hass = hass or self.hass

        # System prompt: use resolved value or get from shared manager if still missing
        system_prompt = params.system_prompt
        if not system_prompt:
            system_prompt = self.prompt_manager.get_prompt(
                params.mode or "ai_task", params.config
            )

        # Build SDK payload
        sdk_messages = prepare_sdk_payload(
            messages=messages,
            params=params,
            system_prompt=system_prompt,
            session_context=build_session_context_info(hass, params.config),
        )

        # Log and Execute
        chat_args = assemble_chat_args(params, sdk_messages, store_messages=False)
        log_api_request(
            sdk_messages, model, service_type, params=params, is_stateless=True
        )

        chat = self.create_client().chat.create(**chat_args)

        # Execute and log result
        if params.timer:
            timer_cm = contextlib.nullcontext(params.timer)
        else:
            timer_cm = LogTimeServices(LOGGER, f"{service_type}_exec", {"mode": "exec"})

        async with timer_cm as timer:
            try:
                async with timer.record_api_call():
                    response = await chat.sample()
                content = getattr(response, "content", "")

                # Save metadata and track usage
                res_holder = {"model": model}
                extract_response_metadata(response, res_holder, entity)

                # Extract reasoning_tokens for timer logging (reasoning models)
                if usage := res_holder.get("usage"):
                    reasoning = getattr(usage, "reasoning_tokens", 0) or 0
                    if reasoning == 0:
                        details = getattr(usage, "completion_tokens_details", None)
                        if details:
                            reasoning = getattr(details, "reasoning_tokens", 0) or 0
                    if reasoning > 0:
                        timer.reasoning_tokens = reasoning

                # Append citations if enabled
                citations = res_holder.get("citations")
                if citations and params.show_citations:
                    content += format_citations(citations)

                # Update token stats
                await async_log_completion(
                    response, service_type, options=params, entity=entity, hass=hass
                )
                return content
            except Exception as err:
                handle_api_error(err, timer.start_time, f"{service_type} execution")
                raise

    async def async_delete_remote_completions(self, response_ids: list[str]) -> int:
        """Delete stored completion IDs from xAI server asynchronously."""
        if not response_ids:
            return 0

        deleted_count = 0
        try:
            client = self.create_client()
            for pid in response_ids:
                try:
                    await client.chat.delete_stored_completion(pid)
                    deleted_count += 1
                except Exception:
                    continue
        except Exception:
            pass

        return deleted_count
