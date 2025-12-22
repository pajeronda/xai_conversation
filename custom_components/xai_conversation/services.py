"""Service handlers for xAI Conversation integration.

This module contains all service implementations, keeping __init__.py clean.
Each service is implemented as a dedicated class for better organization and testability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant as HA_HomeAssistant
    from homeassistant.config_entries import ConfigEntry as HA_ConfigEntry

# Local application imports
from .const import (
    DOMAIN,
    LOGGER,
    RECOMMENDED_HISTORY_LIMIT_TURNS,
)
from .exceptions import (
    raise_generic_error,
    raise_validation_error,
)
from .helpers import (
    parse_grok_code_response,
    parse_id_list,
    LogTimeServices,
    get_xai_entity,
    extract_user_id,
)
from .xai_gateway import XAIGateway

import base64
from datetime import datetime
import json
from homeassistant.core import (
    ServiceCall as HA_ServiceCall,
    ServiceResponse as HA_ServiceResponse,
    SupportsResponse,
)
from homeassistant.helpers import entity_registry as ha_entity_registry
from homeassistant.util.json import json_loads as ha_json_loads


# ==============================================================================
# SERVICE: ask (Generate Response)
# ==============================================================================


class AskService:
    """Service handler for 'ask' - Generate stateless response from prompt/data."""

    def __init__(self, hass: HA_HomeAssistant, entry: HA_ConfigEntry):
        """Initialize the service.

        Args:
            hass: Home Assistant instance
            entry: Config entry for xAI Conversation
        """
        self.hass = hass
        self.entry = entry
        self.gateway = XAIGateway(hass, entry)

    async def async_handle(self, call: HA_ServiceCall) -> HA_ServiceResponse:
        """Handle the ask service call."""
        instructions = call.data.get("instructions", "").strip()
        input_data = call.data.get("input_data", "").strip()
        model_override = call.data.get("model")
        max_tokens_override = call.data.get("max_tokens")
        temp_override = call.data.get("temperature")

        if not instructions or not input_data:
            raise_validation_error("Instructions and Input Data are required")

        response_text = await self.gateway.execute_stateless_chat(
            input_data=input_data,
            system_prompt=instructions,
            service_type="ask",
            model_override=model_override,
            max_tokens_override=max_tokens_override,
            temp_override=temp_override,
        )

        return {
            "response_text": response_text,
            "timestamp": datetime.now().isoformat(),
        }


# ==============================================================================
# SERVICE: photo_analysis
# ==============================================================================


class PhotoAnalysisService:
    """Service handler for photo_analysis - Analyze images using Grok Vision."""

    def __init__(self, hass: HA_HomeAssistant, entry: HA_ConfigEntry):
        """Initialize the service.

        Args:
            hass: Home Assistant instance
            entry: Config entry for xAI Conversation
        """
        self.hass = hass
        self.entry = entry

    async def async_handle(self, call: HA_ServiceCall) -> HA_ServiceResponse:
        """Handle the photo_analysis service call.

        Args:
            call: Service call with prompt, images, and optional user_id

        Returns:
            Service response with analysis text and metadata
        """
        context_id = call.context.id

        # Parse request data
        prompt = call.data.get("prompt", "").strip()
        if not prompt:
            raise_validation_error("Prompt is required for photo analysis")

        # Get images from multiple sources
        images = call.data.get("images", [])
        if isinstance(images, str):
            images = [images]  # Single image string to list

        # Get attachments if provided (from HA automation context)
        attachments = call.data.get("attachments")

        # Validate at least one image source
        if not images and not attachments:
            raise_validation_error(
                "At least one image (path/URL or attachment) is required"
            )

        LOGGER.debug(
            "[Context: %s] Request data: prompt_length=%d, images=%d, has_attachments=%s",
            context_id,
            len(prompt),
            len(images),
            bool(attachments),
        )

        # Get AI Task entity (uses its config for vision_model, temperature, etc.)
        ai_task_entity = self._get_ai_task_entity()

        # Call entity method for photo analysis
        analysis_text = await ai_task_entity.analyze_photo(
            prompt=prompt,
            images=images if images else None,
            attachments=attachments,
        )

        return {
            "analysis": analysis_text,
            "timestamp": datetime.now().isoformat(),
        }

    def _get_ai_task_entity(self):
        """Get AI Task entity to access gateway and config."""
        entity = get_xai_entity(self.hass, domain_type="ai_task")
        if not entity:
            raise_generic_error(
                "AI Task entity not found. Please add AI Task in xAI integration settings."
            )
        return entity


# ==============================================================================
# SERVICE: grok_code_fast
# ==============================================================================


class GrokCodeFastService:
    """Service handler for grok_code_fast - Direct API proxy to xAI for code generation."""

    def __init__(self, hass: HA_HomeAssistant, entry: HA_ConfigEntry):
        """Initialize the service.

        Args:
            hass: Home Assistant instance
            entry: Config entry for xAI Conversation
        """
        self.hass = hass
        self.entry = entry
        self.code_memory = hass.data[DOMAIN]["conversation_memory"]
        self.gateway = XAIGateway(hass, entry)

    async def async_handle(self, call: HA_ServiceCall) -> HA_ServiceResponse:
        """Handle the Grok Code Fast service call.

        Args:
            call: Service call object with data and context

        Returns:
            Service response with response_text, response_code, and metadata
        """
        context_id = call.context.id

        async with LogTimeServices(LOGGER, "grok_code_fast") as timer:
            try:
                # Parse and validate request data
                request_data = self._parse_request(call, context_id)
                user_prompt = request_data["prompt"]
                user_id = request_data["user_id"]
                current_code = request_data.get("code")
                attachments = request_data.get("attachments", [])

                # Get configuration and store_messages setting
                config = self.gateway.get_service_config("code_fast")
                store_messages = config["store_messages"]

                # Use unified create_chat from Gateway, which now handles prompt and conv_key generation
                chat, conv_key = await self.gateway.create_chat(
                    service_type="code_fast",
                    scope="user" if user_id else None,
                    identifier=user_id,
                )

                LOGGER.debug(
                    "[Context: %s] Request data: prompt_length=%d, conv_key=%s, has_code=%s, attachments=%d",
                    context_id,
                    len(user_prompt),
                    conv_key or "None",
                    bool(current_code),
                    len(attachments),
                )

                if not self.entry.data.get("api_key"):
                    raise_generic_error("xAI API key not configured")

                # Load history for client-side mode
                if not store_messages and user_id:
                    await self._load_client_side_history(chat, user_id, context_id)

                # Build user message
                user_message = user_prompt
                if current_code:
                    user_message += f"\n\nCurrent code:\n```\n{current_code}\n```"
                for att in attachments:
                    try:
                        att_name = att.get("filename") or att.get("name", "unknown")
                        att_content_base64 = att.get("content", "")
                        content = base64.b64decode(att_content_base64).decode("utf-8")
                        user_message += f"\n\nFile: {att_name}\n```\n{content}\n```"
                    except Exception as e:
                        LOGGER.warning(
                            "Failed to process attachment %s: %s",
                            att.get("filename"),
                            e,
                        )

                chat.append(XAIGateway.user_msg(user_message))

                # Save user message to chat history
                if user_id:
                    chat_history = self.hass.data.get(DOMAIN, {}).get("chat_history")
                    if chat_history:
                        user_message_for_history = json.dumps(
                            {
                                "response_text": user_prompt,
                                "response_code": current_code or "",
                            }
                        )
                        chat_history.save_message_async(
                            user_id, "code", "user", user_message_for_history
                        )

                # Call xAI API
                async with timer.record_api_call():
                    response = await chat.sample()

                if response is None:
                    raise_generic_error(
                        "Failed to get a response from xAI API after retries."
                    )

                content = getattr(response, "content", "")
                response_id = getattr(response, "id", None)

                # Use centralized logging from Gateway
                await self.gateway.async_log_completion(
                    response=response,
                    service_type="code_fast",
                    conv_key=conv_key,
                )

                response_text, response_code = self._parse_response(content, context_id)
                await self._save_to_chat_history(user_id, response_text, response_code)
                self._fire_response_event(user_prompt, response_text, response_code)

                return {
                    "response_text": response_text,
                    "response_code": response_code,
                    "response_id": response_id or "",
                    "timestamp": datetime.now().isoformat(),
                }
            except Exception as err:
                LOGGER.error(
                    "[Context: %s] Error in Grok Code Fast service: %s",
                    context_id,
                    err,
                    exc_info=True,
                )
                raise err

    def _parse_request(self, call: HA_ServiceCall, context_id: str) -> dict:
        """Parse and validate service call request data."""
        instructions_str = call.data.get("instructions", "{}")
        try:
            request_data = ha_json_loads(instructions_str)
        except Exception:
            request_data = {"prompt": instructions_str}

        user_prompt = request_data.get("prompt", "")
        if not user_prompt:
            raise_generic_error("Prompt is required")

        # Get user_id using the helper
        user_id = extract_user_id(call)
        if user_id:
            LOGGER.debug(
                "[Context: %s] Using user_id from context: %s", context_id, user_id
            )

        request_data["user_id"] = user_id
        request_data["prompt"] = user_prompt
        return request_data

    async def _load_client_side_history(
        self,
        chat,
        user_id: str,
        context_id: str,
    ) -> None:
        """Load conversation history for client-side mode."""
        chat_history = self.hass.data.get(DOMAIN, {}).get("chat_history")
        if not chat_history:
            LOGGER.warning(
                "[Context: %s] Client-side mode but chat_history service not available",
                context_id,
            )
            return

        limit = RECOMMENDED_HISTORY_LIMIT_TURNS * 2
        messages = await chat_history.load_history(user_id, "code", limit)

        LOGGER.debug(
            "[Context: %s] Client-side mode: loading %d messages from history",
            context_id,
            len(messages),
        )

        for msg in messages:
            full_content = ""
            try:
                # Unified parsing for both user and assistant messages
                parsed = json.loads(msg["content"])
                text = parsed.get("response_text", "")
                code = parsed.get("response_code", "")

                # Reconstruct the full message content as the API expects it
                full_content = text
                if code:
                    # User messages have a specific preamble for code
                    if msg["role"] == "user":
                        full_content += f"\n\nCurrent code:\n```\n{code}\n```"
                    # Assistant messages just append the code block
                    else:
                        full_content += f"\n\n```\n{code}\n```"

            except (json.JSONDecodeError, TypeError):
                # Fallback for old plain-text messages
                LOGGER.warning(
                    "[Context: %s] Failed to parse message as JSON, using raw content. Role: %s",
                    context_id,
                    msg["role"],
                )
                full_content = msg["content"]

            # Append to the chat object for the API call
            if msg["role"] == "user":
                chat.append(XAIGateway.user_msg(full_content))
            elif msg["role"] == "assistant":
                chat.append(XAIGateway.assistant_msg(full_content))

    def _parse_response(self, content: str, context_id: str) -> tuple[str, str]:
        """Parse Grok Code Fast response content using modular strategies."""
        LOGGER.debug(
            "[Context: %s] Raw xAI response content (full): %s", context_id, content
        )
        response_text, response_code = parse_grok_code_response(content)
        LOGGER.debug(
            "[Context: %s] Parsed - text_length=%d, code_length=%d",
            context_id,
            len(response_text),
            len(response_code),
        )
        if response_code:
            LOGGER.debug(
                "[Context: %s] Response code (first 200 chars): %s",
                context_id,
                response_code[:200],
            )
        return response_text, response_code

    async def _save_to_chat_history(
        self,
        user_id: str | None,
        response_text: str,
        response_code: str,
    ) -> None:
        """Save assistant response to chat history."""
        if not user_id:
            return

        chat_history = self.hass.data.get(DOMAIN, {}).get("chat_history")
        if chat_history:
            assistant_json = json.dumps(
                {"response_text": response_text, "response_code": response_code}
            )
            chat_history.save_message_async(
                user_id, "code", "assistant", assistant_json
            )

    def _fire_response_event(
        self,
        prompt: str,
        response_text: str,
        response_code: str,
    ) -> None:
        """Fire event for frontend listeners."""
        self.hass.bus.async_fire(
            "xai_conversation.grok_code_fast_response",
            {
                "prompt": prompt,
                "response_text": response_text,
                "response_code": response_code,
                "timestamp": datetime.now().isoformat(),
            },
        )


# ==============================================================================
# SERVICE: clear_memory
# ==============================================================================


class ClearMemoryService:
    """Service handler for clear_memory - Clear conversation memory for users/satellites."""

    def __init__(self, hass: HA_HomeAssistant):
        """Initialize the service."""
        self.hass = hass

    async def async_handle(self, call: HA_ServiceCall) -> HA_ServiceResponse:
        """Handle the clear_memory service call."""
        target_entity = get_xai_entity(self.hass)
        conversation_memory = self.hass.data[DOMAIN]["conversation_memory"]

        # Check for physical delete
        if call.data.get("delete_storage_file", False):
            await self._clear_scope(
                target_entity,
                conversation_memory,
                physical_delete=True,
                remote_delete=call.data.get("remote_delete", True),
            )
            return {"status": "ok", "message": "Memory storage file physically deleted"}

        # Get user and device IDs
        user_ids = self._get_user_ids_from_persons(call.data.get("user_id", []))
        device_ids = self._get_device_ids_from_satellites(
            call.data.get("satellite_id", [])
        )

        # Validate selection
        if not user_ids and not device_ids:
            raise_validation_error(
                "Must select users, satellites, or enable 'Delete Storage File'"
            )

        remote_delete = call.data.get("remote_delete", True)
        cleared_items = []

        # Clear memory for users
        for uid in user_ids:
            await self._clear_scope(
                target_entity,
                conversation_memory,
                scope="user",
                target_id=uid,
                remote_delete=remote_delete,
            )
            cleared_items.append(f"user:{uid[:8]}")

        # Clear memory for devices
        for device_id in device_ids:
            await self._clear_scope(
                target_entity,
                conversation_memory,
                scope="device",
                target_id=device_id,
                remote_delete=remote_delete,
            )
            cleared_items.append(f"device:{device_id[:8]}")

        return {
            "status": "ok",
            "message": f"Memory cleared (remote: {remote_delete}): {', '.join(cleared_items)}",
        }

    def _get_user_ids_from_persons(self, person_entity_ids) -> list[str]:
        """Extract user IDs from person entity IDs."""
        person_entity_ids = parse_id_list(person_entity_ids)
        return [
            state.attributes["user_id"]
            for entity_id in person_entity_ids
            if (state := self.hass.states.get(entity_id))
            and state.attributes.get("user_id")
        ]

    def _get_device_ids_from_satellites(self, satellite_entity_ids) -> list[str]:
        """Extract device IDs from satellite entity IDs."""
        satellite_entity_ids = parse_id_list(satellite_entity_ids)
        ent_reg = ha_entity_registry.async_get(self.hass)
        return [
            entity.device_id
            for entity_id in satellite_entity_ids
            if (entity := ent_reg.async_get(entity_id)) and entity.device_id
        ]

    async def _clear_scope(
        self,
        entity,
        conversation_memory,
        scope: str | None = None,
        target_id: str | None = None,
        physical_delete: bool = False,
        remote_delete: bool = True,
    ) -> None:
        """Clear memory locally and optionally on remote xAI servers.

        Args:
            entity: XAI conversation entity (for gateway access)
            conversation_memory: ConversationMemory instance
            scope: Memory scope ("user" or "device")
            target_id: User ID or device ID
            physical_delete: If True, physically delete storage file
            remote_delete: If True, attempt remote deletion on xAI servers

        Raises:
            XAIValidationError: If parameters are invalid
        """
        try:
            # Determine operation and get response IDs
            if physical_delete:
                response_ids = await conversation_memory.async_physical_delete()
                desc = "physical_delete"
            else:
                if not scope or not target_id:
                    raise_validation_error(
                        "scope and target_id required for non-physical delete"
                    )
                if scope not in ("user", "device"):
                    raise_validation_error("scope must be 'user' or 'device'")

                response_ids = await conversation_memory.async_clear_context(
                    scope, target_id
                )
                desc = f"{scope}:{target_id}"

            # Perform remote deletion if enabled
            if remote_delete and response_ids:
                await entity.gateway.delete_remote_completions(
                    response_ids, context="clear_memory"
                )
                LOGGER.info(
                    "Memory cleared: %s (remote: %d entries deleted)",
                    desc,
                    len(response_ids),
                )
            elif response_ids:
                LOGGER.info("Memory cleared: %s (remote deletion disabled)", desc)
            else:
                LOGGER.debug("Memory cleared: %s (no entries found)", desc)

        except (OSError, PermissionError) as err:
            LOGGER.warning("clear_memory: storage access denied: %s", err)
        except Exception as err:
            LOGGER.error(
                "clear_memory failed for %s: %s",
                desc if "desc" in locals() else "unknown",
                err,
            )
            raise


# ==============================================================================
# SERVICE: clear_code_memory
# ==============================================================================


class ClearCodeMemoryService:
    """Service handler for clear_code_memory - Clear Grok Code Fast conversation memory."""

    def __init__(self, hass: HA_HomeAssistant):
        """Initialize the service."""
        self.hass = hass
        self.code_memory = hass.data[DOMAIN]["conversation_memory"]

    async def async_handle(self, call: HA_ServiceCall) -> HA_ServiceResponse:
        """Handle the clear_code_memory service call."""
        user_id = extract_user_id(call)
        if not user_id:
            raise_validation_error("user_id is required")

        # Get conversation entity for gateway access
        target_entity = get_xai_entity(self.hass)

        # Clear memory and get response_ids for remote deletion
        # Filter strictly by "code" mode
        response_ids = await self.code_memory.async_clear_context(
            "user", user_id, mode="code"
        )

        # Attempt remote deletion via gateway
        if response_ids:
            await target_entity.gateway.delete_remote_completions(
                response_ids, context="clear_code_memory"
            )

        # Clear chat history
        chat_history = self.hass.data.get(DOMAIN, {}).get("chat_history")
        if chat_history:
            await chat_history.clear_history(user_id, "code")

        LOGGER.info(
            "Cleared code conversation memory and chat history for user %s",
            user_id[:8],
        )
        return {
            "status": "ok",
            "message": "Code conversation memory and chat history cleared for user {user_id[:8]}",
        }


# ==============================================================================
# SERVICE: sync_chat_history
# ==============================================================================


class SyncChatHistoryService:
    """Service handler for sync_chat_history - Sync chat history from server.

    Primarily used by Grok Code Fast frontend to restore conversation across devices.
    Can sync history for any mode (code, pipeline, tools) but currently only 'code' is used.
    """

    def __init__(self, hass: HA_HomeAssistant):
        """Initialize the service."""
        self.hass = hass

    async def async_handle(self, call: HA_ServiceCall) -> HA_ServiceResponse:
        """Handle the sync_chat_history service call."""
        user_id = extract_user_id(call)
        if not user_id:
            raise_validation_error("user_id is required")

        # Get parameters
        mode = call.data.get("mode", "code")
        limit = call.data.get("limit", RECOMMENDED_HISTORY_LIMIT_TURNS * 2)

        # Load history
        chat_history = self.hass.data.get(DOMAIN, {}).get("chat_history")
        if not chat_history:
            raise_generic_error("Chat history service not initialized")

        messages = await chat_history.load_history(user_id, mode, limit)

        LOGGER.info(
            "Synced chat history for user %s, mode=%s, messages=%d",
            user_id[:8],
            mode,
            len(messages),
        )
        return {"status": "ok", "messages": messages, "count": len(messages)}


# ==============================================================================
# SERVICE: reset_token_stats
# ==============================================================================


class ResetTokenStatsService:
    """Service handler for reset_token_stats - Reset token statistics sensors."""

    def __init__(self, hass: HA_HomeAssistant):
        """Initialize the service."""
        self.hass = hass

    async def async_handle(self, call: HA_ServiceCall) -> HA_ServiceResponse:
        """Handle the reset_token_stats service call."""
        # V2: Use new TokenStats class
        storage = self.hass.data.get(DOMAIN, {}).get("token_stats")

        try:
            # Reset all data in storage.
            # The storage will notify all registered sensors (listeners) to update.
            await storage.reset_stats()

            LOGGER.info(
                "reset_token_stats: Successfully reset all token statistics via storage."
            )

            return {
                "status": "ok",
                "message": "Successfully reset all token statistics",
            }

        except Exception as err:
            LOGGER.error(
                "reset_token_stats: failed to clear storage: %s", err, exc_info=True
            )
            return {
                "status": "error",
                "message": f"Failed to reset token stats: {err}",
            }


# ==============================================================================
# SERVICE: reload_pricing
# ==============================================================================


class ReloadPricingService:
    """Service handler for reload_pricing - Force refresh of model pricing data."""

    def __init__(self, hass: HA_HomeAssistant, entry: HA_ConfigEntry):
        """Initialize the service.

        Args:
            hass: Home Assistant instance
            entry: Config entry for xAI Conversation
        """
        self.hass = hass
        self.entry = entry

    async def async_handle(self, call: HA_ServiceCall) -> HA_ServiceResponse:
        """Handle the reload_pricing service call.

        Args:
            call: Service call object

        Returns:
            Service response with status and message
        """
        LOGGER.info(
            "reload_pricing service called - forcing model and pricing update from API"
        )

        try:
            # Get ModelManager
            model_manager = self.hass.data.get(DOMAIN, {}).get("model_manager")

            # Create a gateway instance to pass to the model manager
            gateway = XAIGateway(self.hass, self.entry)

            # Force update via manager
            await model_manager.async_update_models(gateway)

            # Check updated data count
            xai_data = self.hass.data.get(DOMAIN, {}).get("xai_models_data", {})
            model_count = len(xai_data)

            return {
                "status": "ok",
                "message": f"Successfully refreshed data for {model_count} models from xAI API",
            }

        except Exception as err:
            LOGGER.error("Failed to reload pricing data: %s", err, exc_info=True)
            return {"status": "error", "message": f"Failed to reload pricing: {err}"}


# ==============================================================================
# SERVICES METADATA - AT THE BOTTOM AFTER ALL THE CLASSES
# ==============================================================================

SERVICES_METADATA = [
    (
        "grok_code_fast",
        GrokCodeFastService,
        SupportsResponse.ONLY,
        True,
    ),
    (
        "clear_memory",
        ClearMemoryService,
        SupportsResponse.OPTIONAL,
        False,
    ),
    (
        "clear_code_memory",
        ClearCodeMemoryService,
        SupportsResponse.OPTIONAL,
        False,
    ),
    (
        "sync_chat_history",
        SyncChatHistoryService,
        SupportsResponse.ONLY,
        False,
    ),
    (
        "reset_token_stats",
        ResetTokenStatsService,
        SupportsResponse.OPTIONAL,
        False,
    ),
    (
        "reload_pricing",
        ReloadPricingService,
        SupportsResponse.OPTIONAL,
        True,
    ),
    (
        "photo_analysis",
        PhotoAnalysisService,
        SupportsResponse.ONLY,
        True,
    ),
    (
        "ask",
        AskService,
        SupportsResponse.ONLY,
        True,
    ),
]

# ==============================================================================
# HELPER: Service Registration & Unregistration
# ==============================================================================


def register_services(hass: HA_HomeAssistant, entry: HA_ConfigEntry) -> None:
    """Register all xAI services.

    Args:
        hass: Home Assistant instance
        entry: Config entry for xAI Conversation
    """
    for name, cls, response_type, requires_entry in SERVICES_METADATA:
        if requires_entry:
            instance = cls(hass, entry)
        else:
            instance = cls(hass)

        hass.services.async_register(
            DOMAIN,
            name,
            instance.async_handle,
            supports_response=response_type,
        )

    LOGGER.debug(
        "All xAI services registered successfully (%d services)", len(SERVICES_METADATA)
    )


def unregister_services(hass: HA_HomeAssistant) -> None:
    """Unregister all xAI services.

    Args:
        hass: Home Assistant instance
    """
    for name, _, _, _ in SERVICES_METADATA:
        if hass.services.has_service(DOMAIN, name):
            hass.services.async_remove(DOMAIN, name)

    LOGGER.debug("All xAI services unregistered successfully")
