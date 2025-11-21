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
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_STORE_MESSAGES,
    CONF_TEMPERATURE,
    DOMAIN,
    GROK_CODE_FAST_PROMPT,
    LOGGER,
    RECOMMENDED_HISTORY_LIMIT_TURNS,
    SUBENTRY_TYPE_CODE_TASK,
)
from .exceptions import raise_generic_error, raise_validation_error
from .helpers import parse_grok_code_response, parse_id_list
from .helpers.sensors import update_token_sensors
from .sensor import (
    XAITokenSensorBase,
    XAINewModelsDetectorSensor,
    async_update_pricing_sensors_periodically,
)


# gRPC imports for error handling
try:
    from grpc import StatusCode
    from grpc._channel import _InactiveRpcError
    GRPC_AVAILABLE = True
except ImportError:
    StatusCode = None
    _InactiveRpcError = None
    GRPC_AVAILABLE = False

# Import shared objects from __init__.py (loaded once, shared across integration)
from . import (
    base64,
    datetime,
    json,
    time,
    HA_ServiceCall,
    HA_ServiceResponse,
    SupportsResponse,
    XAI_SDK_AVAILABLE,
    ha_entity_registry,
    ha_json_loads,
    xai_assistant,
    xai_system,
    xai_user,
)

# xAI SDK Client (conditional import)
if XAI_SDK_AVAILABLE:
    from . import XAI_CLIENT_CLASS as XAIClient
else:
    XAIClient = None


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

    async def async_handle(self, call: HA_ServiceCall) -> HA_ServiceResponse:
        """Handle the Grok Code Fast service call.

        Args:
            call: Service call object with data and context

        Returns:
            Service response with response_text, response_code, and metadata
        """
        context_id = call.context.id
        total_start_time = time.time()
        LOGGER.debug(f"[Context: {context_id}] Service 'grok_code_fast' called.")

        try:
            if not XAI_SDK_AVAILABLE:
                raise_generic_error("xAI SDK not available")

            # Parse and validate request data
            request_data = self._parse_request(call, context_id)
            user_prompt = request_data["prompt"]
            previous_response_id = request_data.get("previous_response_id")
            current_code = request_data.get("code")
            attachments = request_data.get("attachments", [])
            user_id = request_data["user_id"]

            # Get configuration
            config = self._get_service_config()
            store_messages = config["store_messages"]

            # Validate and recover previous_response_id
            previous_response_id = await self._validate_and_recover_response_id(
                user_id, previous_response_id, store_messages, context_id
            )

            LOGGER.debug(
                f"[Context: {context_id}] Request data: "
                f"prompt_length={len(user_prompt)}, "
                f"previous_response_id={previous_response_id[:8] if previous_response_id else 'None'}, "
                f"has_code={bool(current_code)}, "
                f"attachments={len(attachments)}"
            )

            # Create xAI client and chat
            api_start_time = time.time()
            api_key = self.entry.data.get("api_key")
            if not api_key:
                raise_generic_error("xAI API key not configured")

            client = None
            try:
                client = XAIClient(api_key=api_key)

                response = None
                for attempt in range(2):  # Allow one retry
                    try:
                        chat = client.chat.create(
                            model=config["model"],
                            max_tokens=config["max_tokens"],
                            temperature=config["temperature"],
                            store_messages=config["store_messages"],
                            previous_response_id=previous_response_id if store_messages else None
                        )

                        # Build and send messages
                        await self._build_and_send_messages(
                            chat,
                            user_prompt,
                            current_code,
                            attachments,
                            user_id,
                            previous_response_id if store_messages else None,
                            store_messages,
                            context_id
                        )

                        # Call xAI API
                        response = await self._call_xai_api(chat, config["model"], context_id)
                        break  # Success, exit loop

                    except _InactiveRpcError as err:
                        if GRPC_AVAILABLE and err.code() == StatusCode.NOT_FOUND and attempt == 0:
                            LOGGER.warning(
                                f"[Context: {context_id}] Conversation context not found on server. "
                                "Clearing local memory and retrying with a new conversation."
                            )
                            if user_id:
                                await self.code_memory.clear_memory(user_id, "code")
                            previous_response_id = None  # Ensure next attempt is a fresh start
                            continue  # Retry
                        # It's a different gRPC error or the second attempt failed, so raise
                        raise

                # This part is reached after a successful attempt (or if the loop finishes)
                if response is None:
                    raise_generic_error("Failed to get a response from xAI API after retries.")

                api_elapsed = time.time() - api_start_time
                LOGGER.debug("xai_api_end: service=code_fast duration=%.2fs", api_elapsed)

                content = getattr(response, "content", "")
                response_id = getattr(response, "id", None)
                usage = getattr(response, "usage", None)

                # Update token sensors
                update_token_sensors(
                    self.hass,
                    self.entry.entry_id,
                    usage,
                    model=config["model"],
                    service_type="code_fast"
                )

                # Save response_id
                await self._save_response_id(user_id, response_id, store_messages, context_id)

                # Parse response
                response_text, response_code = self._parse_response(content, context_id)

                # Save to chat history
                await self._save_to_chat_history(user_id, response_text, response_code)

                # Fire event for frontend
                self._fire_response_event(user_prompt, response_text, response_code)

                total_elapsed = time.time() - total_start_time
                LOGGER.info("grok_code_fast_end: service=code_fast total_duration=%.2fs api_duration=%.2fs", total_elapsed, api_elapsed)

                return {
                    "response_text": response_text,
                    "response_code": response_code,
                    "response_id": response_id or "",
                    "timestamp": datetime.now().isoformat(),
                }
            finally:
                # Close client using context manager protocol (SDK v1.4.0+)
                if client is not None:
                    try:
                        if hasattr(client, '__exit__'):
                            client.__exit__(None, None, None)
                            LOGGER.debug(f"[Context: {context_id}] xAI client closed successfully")
                    except Exception as close_err:
                        LOGGER.warning(f"[Context: {context_id}] Error closing xAI client: %s", close_err)

        except Exception as err:
            LOGGER.error(f"[Context: {context_id}] Error in Grok Code Fast service: %s", err, exc_info=True)
            return {"error": str(err)}

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

        # Get user_id
        user_id = request_data.get("user_id") or call.context.user_id
        if user_id and not request_data.get("user_id"):
            LOGGER.debug(f"[Context: {context_id}] Using user_id from context: {user_id}")

        request_data["user_id"] = user_id
        request_data["prompt"] = user_prompt
        return request_data

    def _get_service_config(self) -> dict:
        """Get code_task subentry configuration."""
        for subentry in self.entry.subentries.values():
            if subentry.subentry_type == SUBENTRY_TYPE_CODE_TASK:
                return {
                    "prompt": subentry.data.get(CONF_PROMPT, GROK_CODE_FAST_PROMPT),
                    "model": subentry.data.get(CONF_CHAT_MODEL, "grok-code-fast-1"),
                    "temperature": float(subentry.data.get(CONF_TEMPERATURE, 0.1)),
                    "max_tokens": int(subentry.data.get(CONF_MAX_TOKENS, 4000)),
                    "store_messages": bool(subentry.data.get(CONF_STORE_MESSAGES, True)),
                }
        raise_generic_error("No code_task configuration found. Please configure Grok Code Fast in xAI integration settings.")

    async def _validate_and_recover_response_id(
        self,
        user_id: str | None,
        previous_response_id: str | None,
        store_messages: bool,
        context_id: str
    ) -> str | None:
        """Validate response_id compatibility and recover from memory if needed."""
        if not user_id:
            return previous_response_id

        # Try to recover from memory if not provided (only if store_messages is enabled)
        if not previous_response_id and store_messages:
            previous_response_id = await self.code_memory.get_response_id(user_id, "code")
            if previous_response_id:
                LOGGER.debug(f"[Context: {context_id}] Recovered previous_response_id from memory: {previous_response_id[:8]}")

        # Validate compatibility with current mode
        if previous_response_id:
            is_valid = await self.code_memory.validate_response_id(
                user_id, "code", previous_response_id, store_messages
            )
            if not is_valid:
                LOGGER.warning(
                    f"[Context: {context_id}] Invalidating previous_response_id={previous_response_id[:8]} "
                    f"(incompatible with current store_messages={store_messages}). "
                    "Starting fresh conversation."
                )
                return None

        return previous_response_id

    async def _build_and_send_messages(
        self,
        chat,
        user_prompt: str,
        current_code: str | None,
        attachments: list,
        user_id: str | None,
        previous_response_id: str | None,
        store_messages: bool,
        context_id: str
    ) -> None:
        """Build system/user messages and append to chat."""
        config = self._get_service_config()

        # Add system prompt (only on first message in server-side mode)
        if not previous_response_id:
            chat.append(xai_system(config["prompt"]))
            LOGGER.debug(f"[Context: {context_id}] Server-side mode - FIRST MESSAGE: Sending system prompt")
        else:
            LOGGER.debug(f"[Context: {context_id}] Server-side mode - SUBSEQUENT MESSAGE: Skipping system prompt (using previous_response_id)")

        # Load history for client-side mode
        if not store_messages and user_id:
            await self._load_client_side_history(chat, user_id, context_id)

        # Build user message
        user_message = user_prompt
        if current_code:
            user_message += f"\n\nCurrent code:\n```\n{current_code}\n```"

        # Process attachments
        for att in attachments:
            try:
                att_name = att.get("filename") or att.get("name", "unknown")
                att_content_base64 = att.get("content", "")

                content = base64.b64decode(att_content_base64).decode("utf-8")

                # Try UTF-8 first, fallback to latin-1
                try:
                    if isinstance(content, bytes):
                        try:
                            content = content.decode("utf-8")
                        except UnicodeDecodeError:
                            content = content.decode("latin-1")
                except Exception:
                    content = att_content_base64

                LOGGER.debug(f"[Context: {context_id}] Processing attachment: {att_name}, content_length={len(content)}")
                user_message += f"\n\nFile: {att_name}\n```\n{content}\n```"
            except Exception as e:
                LOGGER.warning(f"[Context: {context_id}] Failed to process attachment %s: %s", att.get("filename", att.get("name")), e)

        chat.append(xai_user(user_message))

        # Save user message to chat history in a structured format
        if user_id:
            chat_history = self.hass.data.get(DOMAIN, {}).get("chat_history")
            if chat_history:
                user_message_for_history = json.dumps({
                    "response_text": user_prompt,
                    "response_code": current_code if current_code else ""
                })
                chat_history.save_message_async(user_id, "code", "user", user_message_for_history)

    async def _load_client_side_history(self, chat, user_id: str, context_id: str) -> None:
        """Load conversation history for client-side mode."""
        chat_history = self.hass.data.get(DOMAIN, {}).get("chat_history")
        if not chat_history:
            LOGGER.warning(f"[Context: {context_id}] Client-side mode but chat_history service not available")
            return

        limit = RECOMMENDED_HISTORY_LIMIT_TURNS * 2
        messages = await chat_history.load_history(user_id, "code", limit)

        LOGGER.debug(f"[Context: {context_id}] Client-side mode: loading {len(messages)} messages from history")

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
                LOGGER.warning(f"[Context: {context_id}] Failed to parse message as JSON, using raw content. Role: {msg['role']}")
                full_content = msg["content"]

            # Append to the chat object for the API call
            if msg["role"] == "user":
                chat.append(xai_user(full_content))
            elif msg["role"] == "assistant":
                chat.append(xai_assistant(full_content))

    async def _call_xai_api(self, chat, model: str, context_id: str):
        """Call xAI API and return response."""
        LOGGER.debug(f"[Context: {context_id}] Calling xAI API with model '{model}'.")
        response = await chat.sample()

        # Convert dict to JSON string if needed
        content = getattr(response, "content", "")
        if isinstance(content, dict):
            content = json.dumps(content)

        return response


    async def _save_response_id(
        self,
        user_id: str | None,
        response_id: str | None,
        store_messages: bool,
        context_id: str
    ) -> None:
        """Save response_id to memory."""
        if not response_id or not user_id:
            if not user_id:
                LOGGER.warning(f"[Context: {context_id}] No user_id provided, cannot save conversation")
            return

        # Only save if store_messages is enabled (server-side memory)
        if not store_messages:
            LOGGER.debug(f"[Context: {context_id}] Skipping save (store_messages=False, client-side mode)")
            return

        await self.code_memory.save_response_id(user_id, "code", response_id, store_messages=store_messages)
        LOGGER.debug(f"[Context: {context_id}] Saved response_id={response_id[:8]} for user={user_id[:8]} mode=code (server-side)")

    def _parse_response(self, content: str, context_id: str) -> tuple[str, str]:
        """Parse Grok Code Fast response content using modular strategies."""
        LOGGER.debug(f"[Context: {context_id}] Raw xAI response content (full): %s", content)
        response_text, response_code = parse_grok_code_response(content)
        LOGGER.debug(f"[Context: {context_id}] Parsed - text_length={len(response_text)}, code_length={len(response_code)}")
        if response_code:
            LOGGER.debug(f"[Context: {context_id}] Response code (first 200 chars): %s", response_code[:200])
        return response_text, response_code

    async def _save_to_chat_history(
        self,
        user_id: str | None,
        response_text: str,
        response_code: str
    ) -> None:
        """Save assistant response to chat history."""
        if not user_id:
            return

        chat_history = self.hass.data.get(DOMAIN, {}).get("chat_history")
        if chat_history:
            assistant_json = json.dumps({
                "response_text": response_text,
                "response_code": response_code
            })
            chat_history.save_message_async(user_id, "code", "assistant", assistant_json)

    def _fire_response_event(
        self,
        prompt: str,
        response_text: str,
        response_code: str
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
        # Find xAI conversation entity (for gateway access)
        target_entity = self._get_conversation_entity()

        # Get global ConversationMemory instance
        conversation_memory = self.hass.data[DOMAIN]["conversation_memory"]

        # Check for physical delete
        if call.data.get("delete_storage_file", False):
            await self._clear_memory_with_remote_deletion(
                target_entity,
                conversation_memory,
                physical_delete=True
            )
            return {"status": "ok", "message": "Memory storage file physically deleted"}

        # Get user IDs from person entities
        user_ids, user_names = self._get_user_ids_from_persons(call.data.get("user_id", []))

        # Get device IDs from satellite entities
        satellite_device_ids, satellite_names = self._get_device_ids_from_satellites(call.data.get("satellite_id", []))

        # Validate selection
        if not user_ids and not satellite_device_ids:
            raise_validation_error("Must select users, satellites, or enable 'Delete Storage File'")

        # Clear memory for users
        cleared_items = []
        for uid, name in zip(user_ids, user_names):
            await self._clear_memory_with_remote_deletion(
                target_entity,
                conversation_memory,
                scope="user",
                target_id=uid
            )
            cleared_items.append(f"user:{name}")

        # Clear memory for satellites
        for did, name in zip(satellite_device_ids, satellite_names):
            await self._clear_memory_with_remote_deletion(
                target_entity,
                conversation_memory,
                scope="device",
                target_id=did
            )
            cleared_items.append(f"satellite:{name}")

        return {"status": "ok", "message": f"Memory cleared for: {', '.join(cleared_items)}"}

    def _get_conversation_entity(self):
        """Get xAI conversation entity."""
        ent_reg = ha_entity_registry.async_get(self.hass)
        xai_entities = [
            e for e in ent_reg.entities.values()
            if e.platform == DOMAIN and e.domain == "conversation"
        ]

        if not xai_entities:
            raise_validation_error("No xAI conversation entity found")

        target_entity = self.hass.data.get("entity_components", {}).get("conversation").get_entity(xai_entities[0].entity_id)
        if not target_entity:
            raise_generic_error("Target entity not found")

        return target_entity

    def _get_user_ids_from_persons(self, person_entity_ids) -> tuple[list, list]:
        """Convert person entity IDs to user IDs."""
        if isinstance(person_entity_ids, str):
            person_entity_ids = [person_entity_ids] if person_entity_ids else []

        user_ids = []
        user_names = []
        for person_entity_id in person_entity_ids:
            person_state = self.hass.states.get(person_entity_id)
            if person_state and person_state.attributes.get("user_id"):
                user_ids.append(person_state.attributes["user_id"])
                user_names.append(person_state.attributes.get("friendly_name") or person_state.name)

        return user_ids, user_names

    def _get_device_ids_from_satellites(self, satellite_entity_ids) -> tuple[list, list]:
        """Extract device IDs from satellite entity IDs."""
        satellite_entity_ids = parse_id_list(satellite_entity_ids)
        satellite_device_ids = []
        satellite_names = []

        ent_reg = ha_entity_registry.async_get(self.hass)
        for satellite_entity_id in satellite_entity_ids:
            satellite_state = self.hass.states.get(satellite_entity_id)
            if satellite_state:
                entity_entry = ent_reg.async_get(satellite_entity_id)
                if entity_entry and entity_entry.device_id:
                    satellite_device_ids.append(entity_entry.device_id)
                    satellite_names.append(satellite_state.attributes.get("friendly_name") or satellite_state.name)

        return satellite_device_ids, satellite_names

    async def _clear_memory_with_remote_deletion(
        self,
        entity,
        conversation_memory,
        clear_all: bool = False,
        scope: str | None = None,
        target_id: str | None = None,
        physical_delete: bool = False,
    ) -> None:
        """Clear memory locally and remotely.

        Args:
            entity: XAI conversation entity (for gateway access)
            conversation_memory: ConversationMemory instance
            clear_all: If True, clear all memory entries
            scope: Memory scope ("user" or "device")
            target_id: User ID or device ID
            physical_delete: If True, physically delete storage file

        Raises:
            XAIValidationError: If parameters are invalid
        """
        try:
            # Get response IDs from ConversationMemory based on operation type
            if physical_delete:
                response_ids = await conversation_memory.physical_delete_storage()
                context = "physical_delete"
            elif clear_all:
                response_ids = await conversation_memory.clear_all_memory()
                context = "clear_all"
            else:
                # Validate scope-specific parameters
                if not scope:
                    raise_validation_error("scope is required when not clearing all")
                if scope not in ("user", "device"):
                    raise_validation_error("invalid scope, must be 'user' or 'device'")
                if not target_id:
                    raise_validation_error("target_id is required when scope is user or device")

                response_ids = await conversation_memory.clear_memory_by_scope(scope, target_id)
                context = "clear_memory"

            # Attempt remote deletion via gateway
            if response_ids:
                await entity.gateway.delete_remote_completions(response_ids, context=context)

            # Log success
            scope_desc = (
                "physical_delete" if physical_delete
                else "all" if clear_all
                else f"{scope}:{target_id}"
            )
            LOGGER.info("Memory cleared successfully for scope: %s", scope_desc)

        except (OSError, PermissionError) as err:
            LOGGER.warning("clear_memory: storage access denied: %s", err)
        except Exception as err:
            LOGGER.error("clear_memory failed: %s", err)
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
        # Get user_id
        user_id = call.data.get("user_id") or call.context.user_id
        if user_id and not call.data.get("user_id"):
            LOGGER.debug(f"clear_code_memory: Using user_id from context: {user_id}")

        if not user_id:
            raise_validation_error("user_id is required (not found in data or context)")

        # Get conversation entity for gateway access
        target_entity = self._get_conversation_entity()

        # Clear memory and get response_ids for remote deletion
        response_ids = await self.code_memory.clear_memory(user_id, "code")

        # Attempt remote deletion via gateway
        if response_ids:
            await target_entity.gateway.delete_remote_completions(
                response_ids, context="clear_code_memory"
            )

        # Clear chat history
        chat_history = self.hass.data.get(DOMAIN, {}).get("chat_history")
        if chat_history:
            await chat_history.clear_history(user_id, "code")

        LOGGER.info(f"Cleared code conversation memory and chat history for user {user_id[:8]}")
        return {"status": "ok", "message": "Code conversation memory and chat history cleared"}

    def _get_conversation_entity(self):
        """Get xAI conversation entity for gateway access."""
        ent_reg = ha_entity_registry.async_get(self.hass)
        xai_entities = [
            e for e in ent_reg.entities.values()
            if e.platform == DOMAIN and e.domain == "conversation"
        ]

        if not xai_entities:
            raise_validation_error("No xAI conversation entity found")

        target_entity = self.hass.data.get("entity_components", {}).get("conversation").get_entity(xai_entities[0].entity_id)
        if not target_entity:
            raise_generic_error("Target entity not found")

        return target_entity


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
        # Get user_id
        user_id = call.data.get("user_id") or call.context.user_id
        if user_id and not call.data.get("user_id"):
            LOGGER.debug(f"sync_chat_history: Using user_id from context: {user_id}")

        if not user_id:
            raise_validation_error("user_id is required (not found in data or context)")

        # Get parameters
        mode = call.data.get("mode", "code")
        limit = call.data.get("limit", 50)

        # Load history
        chat_history = self.hass.data.get(DOMAIN, {}).get("chat_history")
        if not chat_history:
            raise_generic_error("Chat history service not initialized")

        messages = await chat_history.load_history(user_id, mode, limit)

        LOGGER.info(f"Synced chat history for user {user_id[:8]}, mode={mode}, messages={len(messages)}")
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
        config_entries = self.hass.config_entries.async_entries(DOMAIN)

        if not config_entries:
            raise_validation_error("No xAI Conversation integration entries found")

        sensors_reset = 0

        for entry in config_entries:
            sensors = self.hass.data.get(DOMAIN, {}).get(f"{entry.entry_id}_sensors")
            if not sensors:
                LOGGER.debug("reset_token_stats: no sensors found for entry %s", entry.entry_id)
                continue

            for sensor in sensors:
                # Reset token counters
                if isinstance(sensor, XAITokenSensorBase):
                    try:
                        sensor.reset_statistics()
                        LOGGER.debug("reset_token_stats: reset sensor %s", sensor.entity_id)
                        sensors_reset += 1
                    except Exception as err:
                        LOGGER.error(
                            "reset_token_stats: failed to reset sensor %s: %s",
                            getattr(sensor, "entity_id", "unknown"), err
                        )
                # Also dismiss new models detector
                elif isinstance(sensor, XAINewModelsDetectorSensor):
                    try:
                        sensor.dismiss_new_models()
                        LOGGER.debug("reset_token_stats: dismissed new models detector")
                    except Exception as err:
                        LOGGER.error("reset_token_stats: failed to dismiss new models: %s", err)

        if sensors_reset == 0:
            LOGGER.warning("reset_token_stats: no token sensors found to reset")
            return {"status": "ok", "sensors_reset": 0, "message": "No token sensors found"}

        return {
            "status": "ok",
            "sensors_reset": sensors_reset,
            "message": f"Successfully reset {sensors_reset} token sensors"
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
        LOGGER.info("reload_pricing service called - forcing pricing update")

        try:
            # Call the existing periodic update function which does all the work
            await async_update_pricing_sensors_periodically(self.hass, self.entry)

            return {
                "status": "ok",
                "message": "Pricing data refreshed successfully from xAI API"
            }

        except Exception as err:
            LOGGER.error("Failed to reload pricing data: %s", err, exc_info=True)
            return {
                "status": "error",
                "message": f"Failed to reload pricing: {err}"
            }


class DismissNewModelsService:
    """Service handler for dismiss_new_models - Manually dismiss new models notification."""

    def __init__(self, hass: HA_HomeAssistant, entry: HA_ConfigEntry):
        self.hass = hass
        self.entry = entry

    async def async_handle(self, call: HA_ServiceCall) -> HA_ServiceResponse:
        """Handle the dismiss_new_models service call."""
        LOGGER.info("dismiss_new_models service called")

        # Find the new models detector sensor
        from .sensor import XAINewModelsDetectorSensor
        sensors = self.hass.data.get(DOMAIN, {}).get(f"{self.entry.entry_id}_sensors", [])

        detector_sensor = None
        for sensor in sensors:
            if isinstance(sensor, XAINewModelsDetectorSensor):
                detector_sensor = sensor
                break

        if not detector_sensor:
            LOGGER.warning("New models detector sensor not found")
            return {
                "status": "error",
                "message": "New models detector sensor not found"
            }

        detector_sensor.dismiss_new_models()
        return {
            "status": "ok",
            "message": "New models notification dismissed successfully"
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
        start_time = time.time()
        context_id = call.context.id
        LOGGER.debug(f"[Context: {context_id}] Service 'photo_analysis' called")

        try:
            if not XAI_SDK_AVAILABLE:
                raise_generic_error("xAI SDK not available")

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
                raise_validation_error("At least one image (path/URL or attachment) is required")

            LOGGER.debug(
                f"[Context: {context_id}] Request data: "
                f"prompt_length={len(prompt)}, images={len(images)}, "
                f"has_attachments={bool(attachments)}"
            )

            # Get AI Task entity (uses its config for vision_model, temperature, etc.)
            ai_task_entity = self._get_ai_task_entity()

            # Call gateway for photo analysis
            analysis_text = await ai_task_entity.gateway.analyze_photo(
                prompt=prompt,
                images=images if images else None,
                attachments=attachments,
            )

            elapsed = time.time() - start_time
            LOGGER.info(
                "photo_analysis_end: total_duration=%.2fs images=%d",
                elapsed,
                len(images) + (len(attachments) if attachments else 0)
            )

            return {
                "analysis": analysis_text,
                "timestamp": datetime.now().isoformat(),
                "status": "ok",
            }

        except Exception as err:
            LOGGER.error(
                f"[Context: {context_id}] Error in photo analysis service: %s",
                err,
                exc_info=True
            )
            return {
                "status": "error",
                "error": str(err)
            }

    def _get_ai_task_entity(self):
        """Get AI Task entity to access gateway and config."""
        from .const import SUBENTRY_TYPE_AI_TASK

        # Find ai_task subentry
        ai_task_subentry = None
        for subentry in self.entry.subentries.values():
            if subentry.subentry_type == SUBENTRY_TYPE_AI_TASK:
                ai_task_subentry = subentry
                break

        if not ai_task_subentry:
            raise_generic_error(
                "AI Task not configured. Please add AI Task in xAI integration settings."
            )

        # Get entity from entity component
        entity_component = self.hass.data.get("entity_components", {}).get("ai_task")
        if not entity_component:
            raise_generic_error("AI Task component not loaded")

        # Find entity by matching entry and subentry
        for entity in entity_component.entities:
            if (hasattr(entity, 'entry') and entity.entry.entry_id == self.entry.entry_id and
                hasattr(entity, 'subentry') and entity.subentry.subentry_id == ai_task_subentry.subentry_id):
                return entity

        raise_generic_error(
            "AI Task entity not found. Please reload the xAI integration."
        )


# ==============================================================================
# HELPER: Service Registration
# ==============================================================================

def register_services(hass: HA_HomeAssistant, entry: HA_ConfigEntry) -> None:
    """Register all xAI services.

    Args:
        hass: Home Assistant instance
        entry: Config entry for xAI Conversation
    """
    # Initialize service handlers
    grok_code_fast = GrokCodeFastService(hass, entry)
    clear_memory = ClearMemoryService(hass)
    clear_code_memory = ClearCodeMemoryService(hass)
    sync_chat_history = SyncChatHistoryService(hass)
    reset_token_stats = ResetTokenStatsService(hass)
    reload_pricing = ReloadPricingService(hass, entry)
    dismiss_new_models = DismissNewModelsService(hass, entry)
    photo_analysis = PhotoAnalysisService(hass, entry)

    # Register services
    hass.services.async_register(
        DOMAIN,
        "grok_code_fast",
        grok_code_fast.async_handle,
        supports_response=SupportsResponse.ONLY,
    )

    hass.services.async_register(
        DOMAIN,
        "clear_memory",
        clear_memory.async_handle,
        supports_response=SupportsResponse.OPTIONAL,
    )

    hass.services.async_register(
        DOMAIN,
        "clear_code_memory",
        clear_code_memory.async_handle,
        supports_response=SupportsResponse.OPTIONAL,
    )

    hass.services.async_register(
        DOMAIN,
        "sync_chat_history",
        sync_chat_history.async_handle,
        supports_response=SupportsResponse.ONLY,
    )

    hass.services.async_register(
        DOMAIN,
        "reset_token_stats",
        reset_token_stats.async_handle,
        supports_response=SupportsResponse.OPTIONAL,
    )

    hass.services.async_register(
        DOMAIN,
        "reload_pricing",
        reload_pricing.async_handle,
        supports_response=SupportsResponse.OPTIONAL,
    )

    hass.services.async_register(
        DOMAIN,
        "dismiss_new_models",
        dismiss_new_models.async_handle,
        supports_response=SupportsResponse.OPTIONAL,
    )

    hass.services.async_register(
        DOMAIN,
        "photo_analysis",
        photo_analysis.async_handle,
        supports_response=SupportsResponse.ONLY,
    )

    LOGGER.debug("All xAI services registered successfully")
