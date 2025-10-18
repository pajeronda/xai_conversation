"""Conversation memory management for xAI conversation."""
from __future__ import annotations

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
            LOGGER.warning(f"Failed to load conversation memory: {err}")
            self._memory = {}
            self._loaded = True

    async def save_response_id(self, user_id: str, mode: str, response_id: str, suffix: str = ""):
        """Save response ID for user and mode.

        Args:
            user_id: User ID (UUID from Home Assistant)
            mode: Conversation mode (e.g., "code", "pipeline", "tools")
            response_id: Response ID from xAI API
            suffix: Optional suffix for the key (e.g., ":an:hash" for conversation)
        """
        await self._ensure_loaded()

        # Build key: user:{user_id}:mode:{mode}{suffix}
        conv_key = f"user:{user_id}:mode:{mode}{suffix}"

        # Get or create entry
        if conv_key not in self._memory:
            self._memory[conv_key] = {"responses": []}

        # Add new response with timestamp
        self._memory[conv_key]["responses"].append({
            "id": response_id,
            "timestamp": time.time()
        })

        # Save to disk
        try:
            await self._store.async_save(self._memory)
        except Exception as err:
            LOGGER.error(f"Failed to save conversation memory: {err}")

    def _get_memory_params(self, conv_key: str) -> tuple[float, int]:
        """Get TTL and max_turns based on conv_key (device vs user).

        Args:
            conv_key: Conversation key (starts with "device:" or "user:")

        Returns:
            Tuple of (ttl_hours, max_turns)
        """
        is_device = conv_key.startswith("device:")
        if is_device:
            ttl_hours = self.entry.data.get(CONF_MEMORY_DEVICE_TTL_HOURS, RECOMMENDED_MEMORY_DEVICE_TTL_HOURS)
            max_turns = self.entry.data.get(CONF_MEMORY_DEVICE_MAX_TURNS, RECOMMENDED_MEMORY_DEVICE_MAX_TURNS)
        else:
            ttl_hours = self.entry.data.get(CONF_MEMORY_USER_TTL_HOURS, RECOMMENDED_MEMORY_USER_TTL_HOURS)
            max_turns = self.entry.data.get(CONF_MEMORY_USER_MAX_TURNS, RECOMMENDED_MEMORY_USER_MAX_TURNS)
        return ttl_hours, max_turns

    def _filter_valid_responses(self, responses: list, ttl_hours: float, max_turns: int, now: float) -> tuple[list, list]:
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

    async def get_response_id(self, user_id: str, mode: str, suffix: str = "") -> str | None:
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

        self._memory[conv_key]["responses"].append({
            "id": response_id,
            "timestamp": time.time()
        })

        await self._store.async_save(self._memory)

    async def clear_memory(self, user_id: str, mode: str):
        """Clear memory for specific user and mode (all keys matching prefix).

        This will delete all keys starting with user:{user_id}:mode:{mode}
        including any suffixes like :an:hash.

        Args:
            user_id: User ID (UUID from Home Assistant)
            mode: Conversation mode (e.g., "code", "pipeline", "tools")
        """
        await self._ensure_loaded()

        # Build prefix: user:{user_id}:mode:{mode}
        key_prefix = f"user:{user_id}:mode:{mode}"

        # Find all keys matching the prefix
        keys_to_delete = [k for k in self._memory.keys() if k.startswith(key_prefix)]

        if keys_to_delete:
            for key in keys_to_delete:
                del self._memory[key]

            # Save to disk
            try:
                await self._store.async_save(self._memory)
            except Exception as err:
                LOGGER.error(f"Failed to save conversation memory after clear: {err}")

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
        keys_to_delete = [k for k in self._memory.keys() if k == base_key or k.startswith(base_key + ":mode:")]

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

            # Save to disk
            try:
                await self._store.async_save(self._memory)
            except Exception as err:
                LOGGER.error(f"Failed to save conversation memory after clear: {err}")

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

        # Save to disk
        try:
            await self._store.async_save(self._memory)
        except Exception as err:
            LOGGER.error(f"Failed to save conversation memory after clear: {err}")

        return response_ids

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
            LOGGER.info("Memory storage file physically deleted")
        except Exception as err:
            LOGGER.error(f"Failed to physically delete memory storage file: {err}")
            raise

        return response_ids

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
            valid_responses, expired_ids = self._filter_valid_responses(responses, ttl_hours, max_turns, now)

            if not valid_responses:
                # All responses expired, mark key for deletion
                keys_to_delete.append(conv_key)
                responses_removed += len(responses)
                keys_removed += 1
            elif len(valid_responses) != len(responses):
                # Some responses expired, update entry
                memory_entry["responses"] = valid_responses
                self._memory[conv_key] = memory_entry
                responses_removed += len(responses) - len(valid_responses)
                keys_cleaned += 1

        # Delete empty keys
        for key in keys_to_delete:
            self._memory.pop(key, None)

        # Save if anything changed
        if keys_cleaned > 0 or keys_removed > 0:
            try:
                await self._store.async_save(self._memory)
                LOGGER.info(
                    "Memory cleanup completed: %d keys cleaned, %d keys removed, %d responses removed",
                    keys_cleaned, keys_removed, responses_removed
                )
            except Exception as err:
                LOGGER.error(f"Failed to save memory after cleanup: {err}")

        return {
            "keys_cleaned": keys_cleaned,
            "keys_removed": keys_removed,
            "responses_removed": responses_removed
        }
