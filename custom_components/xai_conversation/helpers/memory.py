"""Conversation memory management for xAI conversation."""

from __future__ import annotations

import hashlib
import time

from ..const import (
    CONF_MEMORY_DEVICE_MAX_TURNS,
    CONF_MEMORY_DEVICE_TTL_HOURS,
    CONF_MEMORY_USER_MAX_TURNS,
    CONF_MEMORY_USER_TTL_HOURS,
    LOGGER,
    RECOMMENDED_MEMORY_DEVICE_MAX_TURNS,
    RECOMMENDED_MEMORY_DEVICE_TTL_HOURS,
    RECOMMENDED_MEMORY_USER_MAX_TURNS,
    RECOMMENDED_MEMORY_USER_TTL_HOURS,
)


class ConversationMemory:
    """Standalone conversation memory manager with TTL and max_turns support.

    This class provides persistent storage for conversation response IDs,
    with automatic cleanup of expired responses based on TTL and max_turns limits.
    """

    @staticmethod
    def _prompt_hash(prompt: str) -> str:
        """Generate a short SHA256 hash of the prompt.

        Args:
            prompt: String to hash

        Returns:
            First 8 characters of SHA256 hash
        """
        return hashlib.sha256(prompt.encode()).hexdigest()[:8]

    def calculate_conv_key_simple(
        self, user_id: str, mode: str, base_prompt: str, memory_scope: str = "user"
    ) -> str:
        """Calculate conversation key for code_fast service.

        Builds a key like: user:{id}:mode:code:ph:{hash}
        This enables automatic prompt-based conversation invalidation.

        Args:
            user_id: User ID (UUID from Home Assistant)
            mode: Conversation mode (e.g., "code")
            base_prompt: Base system prompt (before user instructions)
            memory_scope: "user" or "device" (default: "user")

        Returns:
            Full conversation key for memory lookup
        """
        prompt_hash = self._prompt_hash(base_prompt)
        return f"{memory_scope}:{user_id}:mode:{mode}:ph:{prompt_hash}"

    def __init__(self, hass, storage_path: str, entry):
        """Initialize conversation memory.

        Args:
            hass: Home Assistant instance (needed for Store)
            storage_path: Path for storage file (e.g., "xai_conversation/xai_conversation.memory")
            entry: Config entry (for accessing TTL and max_turns configuration)
        """
        from homeassistant.helpers.storage import Store

        self.hass = hass
        self.entry = entry
        self._store = Store(hass, 1, storage_path)
        self._memory: dict = {}
        self._loaded = False
        self._dirty = False

    async def _ensure_loaded(self):
        """Ensure memory is loaded from storage."""
        if self._loaded:
            return

        try:
            data = await self._store.async_load()
            if isinstance(data, dict):
                self._memory = data
            else:
                self._memory = {}
            self._loaded = True
        except Exception as err:
            LOGGER.warning("Failed to load conversation memory: %s", err)
            self._memory = {}
            self._loaded = True

    async def async_flush(self):
        """Flush pending changes to disk."""
        if not self._dirty:
            return

        try:
            await self._store.async_save(self._memory)
            self._dirty = False
            LOGGER.debug("Conversation memory flushed to disk")
        except Exception as err:
            LOGGER.error("Failed to save conversation memory: %s", err)

    async def save_response_id(
        self,
        user_id: str,
        mode: str,
        response_id: str,
        suffix: str = "",
        store_messages: bool = None,
    ):
        """Save response ID for user and mode with metadata.

        Args:
            user_id: User ID (UUID from Home Assistant)
            mode: Conversation mode (e.g., "code", "pipeline", "tools")
            response_id: Response ID from xAI API
            suffix: Optional suffix for the key (e.g., ":an:hash" for conversation)
            store_messages: Whether this response was created with server-side memory (True) or client-side (False)
        """
        await self._ensure_loaded()

        # Build key: user:{user_id}:mode:{mode}{suffix}
        conv_key = f"user:{user_id}:mode:{mode}{suffix}"

        # Get or create entry
        if conv_key not in self._memory:
            self._memory[conv_key] = {"responses": []}

        # Add new response with timestamp and store_messages metadata
        response_entry = {"id": response_id, "timestamp": time.time()}

        # Add store_messages only if provided (for backward compatibility)
        if store_messages is not None:
            response_entry["store_messages"] = store_messages

        self._memory[conv_key]["responses"].append(response_entry)
        self._dirty = True

    def _get_memory_params(self, conv_key: str) -> tuple[float, int]:
        """Get TTL and max_turns based on conv_key (device vs user).

        Args:
            conv_key: Conversation key (starts with "device:" or "user:")

        Returns:
            Tuple of (ttl_hours, max_turns)
        """
        is_device = conv_key.startswith("device:")
        if is_device:
            ttl_hours = self.entry.data.get(
                CONF_MEMORY_DEVICE_TTL_HOURS, RECOMMENDED_MEMORY_DEVICE_TTL_HOURS
            )
            max_turns = self.entry.data.get(
                CONF_MEMORY_DEVICE_MAX_TURNS, RECOMMENDED_MEMORY_DEVICE_MAX_TURNS
            )
        else:
            ttl_hours = self.entry.data.get(
                CONF_MEMORY_USER_TTL_HOURS, RECOMMENDED_MEMORY_USER_TTL_HOURS
            )
            max_turns = self.entry.data.get(
                CONF_MEMORY_USER_MAX_TURNS, RECOMMENDED_MEMORY_USER_MAX_TURNS
            )
        return ttl_hours, max_turns

    def _filter_valid_responses(
        self, responses: list, ttl_hours: float, max_turns: int, now: float
    ) -> tuple[list, list]:
        """Filter responses by TTL and max_turns.

        Args:
            responses: List of response dicts with 'id' and 'timestamp' keys
            ttl_hours: Time-to-live in hours
            max_turns: Maximum number of turns to keep
            now: Current timestamp for TTL calculation

        Returns:
            Tuple of (valid_responses, expired_ids)
        """
        if not responses:
            return [], []

        ttl_seconds = ttl_hours * 3600
        expired_ids = []
        valid_responses = []

        for r in responses:
            if (now - r.get("timestamp", now)) <= ttl_seconds:
                valid_responses.append(r)
            else:
                if r.get("id"):
                    expired_ids.append(r["id"])

        if len(valid_responses) > max_turns:
            responses_to_remove = valid_responses[:-max_turns]
            valid_responses = valid_responses[-max_turns:]
            for response in responses_to_remove:
                if response.get("id"):
                    expired_ids.append(response["id"])

        return valid_responses, expired_ids

    async def get_response_id(
        self, user_id: str, mode: str, suffix: str = ""
    ) -> str | None:
        """Get last response ID for user and mode.

        Args:
            user_id: User ID (UUID from Home Assistant)
            mode: Conversation mode (e.g., "code", "pipeline", "tools")
            suffix: Optional suffix for the key (e.g., ":an:hash" for conversation)

        Returns:
            Last response ID or None if not found
        """
        await self._ensure_loaded()

        conv_key = f"user:{user_id}:mode:{mode}{suffix}"
        memory_entry = self._memory.get(conv_key)
        if not memory_entry:
            return None

        responses = memory_entry.get("responses", [])
        if not responses:
            return None

        return responses[-1]["id"] if responses else None

    async def get_response_id_by_key(self, conv_key: str) -> str | None:
        """Get last response_id using full conversation key.

        This method accepts a full key like "user:{user_id}:mode:{mode}:an:{hash}"
        and retrieves the last response_id.

        Args:
            conv_key: Full conversation key (e.g., "user:{id}:mode:pipeline:an:abc123")

        Returns:
            Last response_id or None if not found
        """
        await self._ensure_loaded()

        memory_entry = self._memory.get(conv_key)
        if not memory_entry:
            return None

        responses = memory_entry.get("responses", [])
        if not responses:
            return None

        return responses[-1]["id"] if responses else None

    async def validate_response_id(
        self,
        user_id: str,
        mode: str,
        response_id: str,
        current_store_messages: bool,
        suffix: str = "",
    ) -> bool:
        """Validate if response_id is compatible with current store_messages mode.

        This prevents using a response_id created in one mode (server/client)
        with a different mode, which would cause xAI API errors.

        Args:
            user_id: User ID (UUID from Home Assistant)
            mode: Conversation mode (e.g., "code", "pipeline", "tools")
            response_id: Response ID to validate
            current_store_messages: Current store_messages configuration (True=server-side, False=client-side)
            suffix: Optional suffix for the key

        Returns:
            True if response_id is compatible with current mode, False otherwise
        """
        await self._ensure_loaded()

        conv_key = f"user:{user_id}:mode:{mode}{suffix}"
        memory_entry = self._memory.get(conv_key)
        if not memory_entry:
            return False  # No memory entry, response_id is invalid

        responses = memory_entry.get("responses", [])
        if not responses:
            return False  # No responses, response_id is invalid

        # Find the response in history (search from end for efficiency)
        for r in reversed(responses):
            if r.get("id") == response_id:
                # Found the response, check store_messages compatibility
                stored_mode = r.get("store_messages")

                if stored_mode is None:
                    # Old format (before this fix) - assume server-side (True)
                    # This provides backward compatibility with existing response_ids
                    return current_store_messages

                # Check if modes match
                return stored_mode == current_store_messages

        # response_id not found in memory
        return False

    async def save_response_id_by_key(self, conv_key: str, response_id: str):
        """Save response ID using full conversation key.

        This method accepts a full key like "user:{user_id}:mode:{mode}:an:{hash}"
        and saves the response_id with timestamp.

        Args:
            conv_key: Full conversation key (e.g., "user:{id}:mode:pipeline:an:abc123")
            response_id: Response ID from xAI API
        """
        await self._ensure_loaded()

        if conv_key not in self._memory:
            self._memory[conv_key] = {"responses": []}

        self._memory[conv_key]["responses"].append(
            {"id": response_id, "timestamp": time.time()}
        )
        self._dirty = True

    async def clear_memory(self, user_id: str, mode: str) -> list[str]:
        """Clear memory for specific user and mode (all keys matching prefix).

        This will delete all keys starting with user:{user_id}:mode:{mode}
        including any suffixes like :an:hash.

        Args:
            user_id: User ID (UUID from Home Assistant)
            mode: Conversation mode (e.g., "code", "pipeline", "tools")

        Returns:
            List of response_ids that were deleted (for remote cleanup)
        """
        await self._ensure_loaded()

        # Build prefix: user:{user_id}:mode:{mode}
        key_prefix = f"user:{user_id}:mode:{mode}"

        # Find all keys matching the prefix
        keys_to_delete = [k for k in self._memory.keys() if k.startswith(key_prefix)]

        # Collect response IDs for remote deletion
        response_ids = []
        for key in keys_to_delete:
            memory_entry = self._memory.get(key, {})
            for response in memory_entry.get("responses", []):
                if response.get("id"):
                    response_ids.append(response["id"])

        # Delete local entries
        if keys_to_delete:
            for key in keys_to_delete:
                del self._memory[key]

            self._dirty = True
            await self.async_flush()

        return response_ids

    async def clear_memory_by_scope(self, scope: str, target_id: str) -> list[str]:
        """Clear memory for specific scope and target_id (all modes).

        This will delete all keys matching {scope}:{target_id}:mode:*
        including any suffixes like :an:hash.

        Args:
            scope: "user" or "device"
            target_id: User ID or device ID

        Returns:
            List of response_ids that were deleted (for remote cleanup)
        """
        await self._ensure_loaded()

        # Build prefix: {scope}:{target_id}
        base_key = f"{scope}:{target_id}"

        # Find all keys matching the prefix (exact match or with :mode: suffix)
        keys_to_delete = [
            k
            for k in self._memory.keys()
            if k == base_key or k.startswith(base_key + ":mode:")
        ]

        # Collect response IDs for remote deletion
        response_ids = []
        for key in keys_to_delete:
            memory_entry = self._memory.get(key, {})
            for response in memory_entry.get("responses", []):
                if response.get("id"):
                    response_ids.append(response["id"])

        # Delete local entries
        if keys_to_delete:
            for key in keys_to_delete:
                del self._memory[key]

            self._dirty = True
            await self.async_flush()

        return response_ids

    async def clear_all_memory(self) -> list[str]:
        """Clear all memory entries.

        Returns:
            List of response_ids that were deleted (for remote cleanup)
        """
        await self._ensure_loaded()

        # Collect all response IDs for remote deletion
        response_ids = []
        for key, entry in self._memory.items():
            if isinstance(entry, dict):
                for response in entry.get("responses", []):
                    if response.get("id"):
                        response_ids.append(response["id"])

        # Clear all entries
        self._memory.clear()
        self._dirty = True
        await self.async_flush()

        return response_ids

    def get_memory_key(
        self,
        user_input,
        mode: str,
        subentry_data: dict,
        memory_scope: str = "user",
        base_prompt: str | None = None,
    ) -> str:
        """Calculate consistent memory key for conversation chaining.

        The key is built using:
        - Base: "user:{user_id}" or "device:{device_id}"
        - Mode: ":mode:{mode}"
        - Chat-only flag: ":chatonly" if home control is disabled
        - Base prompt hash: ":ph:{hash}" - hash of the complete base system prompt

        CRITICAL: Including chat-only flag ensures that switching between
        "home control enabled" and "home control disabled" starts a NEW
        conversation chain, preventing contradictory instructions in the
        same xAI server-side memory.

        CRITICAL: Including base prompt hash ensures that ANY change to the
        system prompt (assistant name, custom instructions, mode settings)
        starts a NEW conversation, since the system prompt cannot be modified
        in existing conversations with previous_response_id.

        Args:
            user_input: ConversationInput object
            mode: "pipeline" or "tools"
            subentry_data: Configuration dictionary from subentry.data
            memory_scope: "user" or "device" (default: "user")
            base_prompt: Optional pre-built base prompt to hash. If None, will be built.

        Returns:
            Memory key string in format: user:id:mode:{mode}[:chatonly]:ph:{hash}
        """
        from .conversation import extract_device_id, extract_user_id
        from ..const import CONF_ALLOW_SMART_HOME_CONTROL
        from .prompt_manager import PromptManager

        # Build base key
        if memory_scope == "user":
            identifier = extract_user_id(user_input) or "unknown"
            base = f"user:{identifier}"
        else:  # device
            identifier = extract_device_id(user_input) or "unknown"
            base = f"device:{identifier}"

        # Add mode
        key = f"{base}:mode:{mode}"

        # CRITICAL: Add chat-only flag to force new conversation when toggling home control
        allow_control = subentry_data.get(CONF_ALLOW_SMART_HOME_CONTROL, True)
        if not allow_control:
            key = f"{key}:chatonly"

        # Build or use provided base prompt for hashing
        if base_prompt is None:
            prompt_mgr = PromptManager(subentry_data, mode)
            # For tools mode, we build without static_context/tool_definitions
            # since those are dynamic and shouldn't affect memory key
            base_prompt = prompt_mgr.build_base_prompt_with_user_instructions()

        # Always add hash of complete base prompt
        ph = self._prompt_hash(base_prompt)
        key = f"{key}:ph:{ph}"

        return key

    async def get_conv_key_and_prev_id(
        self, user_input, mode: str, subentry_data: dict
    ) -> tuple[str, str | None]:
        """Get conversation key and previous response ID.

        Args:
            user_input: ConversationInput object
            mode: "pipeline" or "tools"
            subentry_data: Configuration dictionary from subentry.data

        Returns:
            A tuple containing (conversation_key, previous_response_id or None)
        """
        from .conversation import extract_user_id
        from ..const import CONF_STORE_MESSAGES

        # Check if server-side memory is enabled
        store_messages = subentry_data.get(CONF_STORE_MESSAGES, True)

        if not store_messages:
            # Client-side mode: no need to calculate key or retrieve previous_response_id
            LOGGER.debug(
                "ConversationMemory.get_conv_key_and_prev_id: store_messages=False, skipping key calculation"
            )
            return "", None

        # Memory scope logic (applies to both pipeline and tools modes):
        # - If user_id is available: use "user" scope to share conversation across all user's devices
        # - If no user_id (e.g., voice satellites without user context): use "device" scope for device-specific memory
        user_id = extract_user_id(user_input)
        if user_id:
            memory_scope = "user"
        else:
            # Fallback to device scope for voice satellites or other devices without user context
            memory_scope = "device"

        conv_key = self.get_memory_key(user_input, mode, subentry_data, memory_scope)
        prev_id = await self.get_response_id_by_key(conv_key)

        LOGGER.debug(
            "ConversationMemory.get_conv_key_and_prev_id: mode=%s scope=%s conv_key=%s prev_id=%s",
            mode,
            memory_scope,
            conv_key,
            prev_id[:8] if prev_id else None,
        )
        return conv_key, prev_id

    async def physical_delete_storage(self) -> list[str]:
        """Physically delete the storage file.

        Returns:
            List of response_ids that were in storage (for remote cleanup)
        """
        await self._ensure_loaded()

        # Collect all response IDs for remote deletion
        response_ids = []
        for key, entry in self._memory.items():
            if isinstance(entry, dict):
                for response in entry.get("responses", []):
                    if response.get("id"):
                        response_ids.append(response["id"])

        # Delete the storage file
        try:
            await self._store.async_remove()
            self._memory.clear()
            self._loaded = False
            self._dirty = False
            LOGGER.info("Memory storage file physically deleted")
        except Exception as err:
            LOGGER.error(f"Failed to physically delete memory storage file: {err}")
            raise

        return response_ids

    def setup_periodic_cleanup(self, cleanup_interval_hours: int):
        """Setup periodic cleanup task for conversation memory.

        This method creates and returns an async cleanup callback and its unsubscribe function.
        Should be called from async_setup_entry in __init__.py.

        Args:
            cleanup_interval_hours: Interval in hours between cleanup runs

        Returns:
            Tuple of (cleanup_callback, unsubscribe_function)
        """
        from datetime import timedelta
        from homeassistant.helpers.event import async_track_time_interval

        async def periodic_cleanup(_now):
            """Periodic cleanup task for conversation memory."""
            LOGGER.debug(
                "Running periodic memory cleanup (interval: %s hours)",
                cleanup_interval_hours,
            )
            stats = await self.async_cleanup_expired()
            if stats["keys_removed"] > 0 or stats["keys_cleaned"] > 0:
                LOGGER.debug(
                    "Memory cleanup: cleaned %d keys, removed %d keys, deleted %d responses",
                    stats["keys_cleaned"],
                    stats["keys_removed"],
                    stats["responses_removed"],
                )

        cleanup_interval = timedelta(hours=cleanup_interval_hours)
        cleanup_unsub = async_track_time_interval(
            self.hass, periodic_cleanup, cleanup_interval
        )

        LOGGER.debug(
            "Started periodic memory cleanup task (interval: %s hours)",
            cleanup_interval_hours,
        )

        return cleanup_unsub

    async def async_cleanup_expired(self) -> dict:
        """Periodic cleanup task: remove expired response IDs based on TTL and max_turns.

        This method should be called periodically (e.g., every 24 hours) to clean up
        old response IDs without doing it on every get_response_id() call.

        Returns:
            Dict with cleanup statistics: {
                "keys_cleaned": int,
                "keys_removed": int,
                "responses_removed": int
            }
        """
        await self._ensure_loaded()

        if not self._memory:
            return {"keys_cleaned": 0, "keys_removed": 0, "responses_removed": 0}

        now = time.time()
        keys_cleaned = 0
        keys_removed = 0
        responses_removed = 0
        keys_to_delete = []

        for conv_key, memory_entry in list(self._memory.items()):
            if not isinstance(memory_entry, dict):
                continue

            responses = memory_entry.get("responses", [])
            if not responses:
                keys_to_delete.append(conv_key)
                continue

            # Get TTL and max_turns for this key
            ttl_hours, max_turns = self._get_memory_params(conv_key)
            valid_responses, expired_ids = self._filter_valid_responses(
                responses, ttl_hours, max_turns, now
            )

            if not valid_responses:
                # All responses expired, mark key for deletion
                keys_to_delete.append(conv_key)
                responses_removed += len(responses)
                keys_removed += 1
            elif len(valid_responses) != len(responses):
                # Some responses expired, update entry
                memory_entry["responses"] = valid_responses
                self._memory[conv_key] = memory_entry
                responses_removed += len(responses)
                keys_cleaned += 1

        # Delete empty keys
        for key in keys_to_delete:
            self._memory.pop(key, None)

        # Save if anything changed
        if keys_cleaned > 0 or keys_removed > 0:
            self._dirty = True
            await self.async_flush()
            LOGGER.debug(
                "Memory cleanup completed: %d keys cleaned, %d keys removed, %d responses removed",
                keys_cleaned,
                keys_removed,
                responses_removed,
            )

        return {
            "keys_cleaned": keys_cleaned,
            "keys_removed": keys_removed,
            "responses_removed": responses_removed,
        }