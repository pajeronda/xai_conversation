"""Conversation memory management for xAI conversation."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, TypedDict, NotRequired

from homeassistant.helpers.storage import Store
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

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant
    from homeassistant.config_entries import ConfigEntry


class ResponseData(TypedDict):
    """Structure for a stored response."""

    id: str
    timestamp: float
    store_messages: NotRequired[bool]


class MemoryEntry(TypedDict):
    """Structure for a conversation memory entry."""

    responses: list[ResponseData]


class MemoryManager:
    """Standalone conversation memory manager with TTL and max_turns support.

    Provides persistent storage for conversation response IDs,
    with automatic cleanup of expired responses based on TTL and limits.
    """

    def __init__(self, hass: HomeAssistant, storage_path: str, entry: ConfigEntry):
        """Initialize memory manager.

        Args:
            hass: Home Assistant instance.
            storage_path: Path for storage file.
            entry: Config entry (for accessing TTL settings).
        """
        self.hass = hass
        self.entry = entry
        self._store = Store(hass, 1, storage_path)
        self._memory: dict[str, MemoryEntry] = {}
        self._loaded = False
        self._dirty = False

    @staticmethod
    def generate_key(
        scope: str,
        identifier: str,
        mode: str,
        prompt_hash: str,
        allow_control: bool = True,
    ) -> str:
        """Generate a consistent memory key.

        Format: {scope}:{identifier}:mode:{mode}[:chatonly]:ph:{prompt_hash}

        Args:
            scope: "user" or "device".
            identifier: User ID or Device ID.
            mode: "pipeline", "tools", "code".
            prompt_hash: Short hash of the stable system prompt.
            allow_control: Whether smart home control is enabled (affects key).
        """
        base = f"{scope}:{identifier}:mode:{mode}"

        # Add chat-only flag to separate conversation chains with/without control
        if not allow_control:
            base = f"{base}:chatonly"

        return f"{base}:ph:{prompt_hash}"

    async def async_get_last_response_id(self, key: str) -> str | None:
        """Retrieve the last response ID for a given key."""
        await self._ensure_loaded()

        entry = self._memory.get(key)
        if not entry or not entry["responses"]:
            return None

        return entry["responses"][-1]["id"]

    async def async_save_response(
        self, key: str, response_id: str, store_messages: bool | None = None
    ) -> None:
        """Save a response ID to memory."""
        await self._ensure_loaded()

        if key not in self._memory:
            self._memory[key] = {"responses": []}

        data: ResponseData = {
            "id": response_id,
            "timestamp": time.time(),
        }
        if store_messages is not None:
            data["store_messages"] = store_messages

        self._memory[key]["responses"].append(data)
        self._dirty = True

    async def async_validate_response_compatibility(
        self, key: str, response_id: str, current_store_messages: bool
    ) -> bool:
        """Check if a response ID is compatible with current store_messages setting."""
        await self._ensure_loaded()

        entry = self._memory.get(key)
        if not entry:
            return False

        # Search specifically for this ID
        for r in reversed(entry["responses"]):
            if r["id"] == response_id:
                stored_mode = r.get("store_messages")
                # Legacy compatibility: assume True (Server-Side) if missing
                return stored_mode is None or stored_mode == current_store_messages

        return False

    async def async_flush(self):
        """Flush pending changes to disk."""
        if not self._dirty:
            return
        try:
            await self._store.async_save(self._memory)
            self._dirty = False
            LOGGER.debug("Memory flushed to disk")
        except Exception as err:
            LOGGER.error("Failed to save memory: %s", err)

    async def async_clear_context(
        self, scope: str, identifier: str, mode: str | None = None
    ) -> list[str]:
        """Clear memory for a specific scope/ID, optionally filtered by mode.

        Args:
            scope: "user" or "device".
            identifier: User ID or Device ID.
            mode: Optional mode to filter (e.g. "code", "pipeline").

        Returns:
            List of deleted response_ids.
        """
        await self._ensure_loaded()
        prefix = f"{scope}:{identifier}"
        if mode:
            prefix = f"{prefix}:mode:{mode}"
        return await self._delete_keys_by_prefix(prefix)

    async def async_clear_key(self, key: str) -> list[str]:
        """Clear memory for a specific conversation key.

        Returns:
            List of deleted response_ids.
        """
        await self._ensure_loaded()
        deleted_ids = []
        entry = self._memory.get(key)
        if entry:
            for r in entry["responses"]:
                if r.get("id"):
                    deleted_ids.append(r["id"])
            del self._memory[key]
            self._dirty = True
            await self.async_flush()
        return deleted_ids

    async def async_clear_all(self) -> list[str]:
        """Clear entire memory storage."""
        await self._ensure_loaded()
        deleted_ids = self._collect_all_ids()
        self._memory.clear()
        self._dirty = True
        await self.async_flush()
        return deleted_ids

    async def async_physical_delete(self) -> list[str]:
        """Physically remove the storage file."""
        await self._ensure_loaded()
        deleted_ids = self._collect_all_ids()
        try:
            await self._store.async_remove()
            self._memory.clear()
            self._loaded = False
            self._dirty = False
            LOGGER.info("Memory storage file deleted")
        except Exception as err:
            LOGGER.error("Failed to delete memory file: %s", err)
            raise
        return deleted_ids

    # --- Internal / Maintenance ---

    async def _ensure_loaded(self):
        """Ensure memory is loaded from storage."""
        if self._loaded:
            return
        try:
            data = await self._store.async_load()
            self._memory = data if isinstance(data, dict) else {}
        except Exception as err:
            LOGGER.warning("Failed to load memory: %s", err)
            self._memory = {}
        finally:
            self._loaded = True

    def _collect_all_ids(self) -> list[str]:
        """Collect all response IDs currently in memory."""
        ids = []
        for entry in self._memory.values():
            for r in entry.get("responses", []):
                if r.get("id"):
                    ids.append(r["id"])
        return ids

    async def _delete_keys_by_prefix(self, prefix: str) -> list[str]:
        """Delete all keys starting with prefix."""
        keys_to_delete = [k for k in self._memory.keys() if k.startswith(prefix)]
        deleted_ids = []

        for key in keys_to_delete:
            entry = self._memory.get(key)
            if entry:
                for r in entry["responses"]:
                    if r.get("id"):
                        deleted_ids.append(r["id"])
            del self._memory[key]

        if keys_to_delete:
            self._dirty = True
            await self.async_flush()

        return deleted_ids

    # --- Cleanup Logic ---

    def setup_periodic_cleanup(self, cleanup_interval_hours: int):
        """Setup periodic cleanup task."""
        from datetime import timedelta
        from homeassistant.helpers.event import async_track_time_interval

        async def _cleanup(_now):
            LOGGER.debug(
                "Running memory cleanup (interval: %dh)", cleanup_interval_hours
            )
            stats = await self.async_cleanup_expired()
            if stats["keys_removed"] > 0 or stats["keys_cleaned"] > 0:
                LOGGER.debug("Cleanup stats: %s", stats)

        interval = timedelta(hours=cleanup_interval_hours)
        unsub = async_track_time_interval(self.hass, _cleanup, interval)
        LOGGER.debug("Memory cleanup scheduled every %dh", cleanup_interval_hours)
        return unsub

    async def async_cleanup_expired(self) -> dict:
        """Remove expired entries based on TTL and Max Turns."""
        await self._ensure_loaded()
        if not self._memory:
            return {"keys_cleaned": 0, "keys_removed": 0, "responses_removed": 0}

        now = time.time()
        stats = {"keys_cleaned": 0, "keys_removed": 0, "responses_removed": 0}
        keys_to_delete = []

        for key, entry in self._memory.items():
            responses = entry.get("responses", [])
            if not responses:
                keys_to_delete.append(key)
                continue

            # Determine limits based on key type
            is_device = key.startswith("device:")
            if is_device:
                ttl = self.entry.data.get(
                    CONF_MEMORY_DEVICE_TTL_HOURS, RECOMMENDED_MEMORY_DEVICE_TTL_HOURS
                )
                max_turns = self.entry.data.get(
                    CONF_MEMORY_DEVICE_MAX_TURNS, RECOMMENDED_MEMORY_DEVICE_MAX_TURNS
                )
            else:
                ttl = self.entry.data.get(
                    CONF_MEMORY_USER_TTL_HOURS, RECOMMENDED_MEMORY_USER_TTL_HOURS
                )
                max_turns = self.entry.data.get(
                    CONF_MEMORY_USER_MAX_TURNS, RECOMMENDED_MEMORY_USER_MAX_TURNS
                )

            # Filter
            valid, removed_count = self._filter_responses(
                responses, ttl, max_turns, now
            )

            if not valid:
                keys_to_delete.append(key)
                stats["responses_removed"] += removed_count
                stats["keys_removed"] += 1
            elif removed_count > 0:
                entry["responses"] = valid
                stats["responses_removed"] += removed_count
                stats["keys_cleaned"] += 1

        for key in keys_to_delete:
            self._memory.pop(key, None)

        if stats["keys_cleaned"] > 0 or stats["keys_removed"] > 0:
            self._dirty = True
            await self.async_flush()

        return stats

    def _filter_responses(
        self,
        responses: list[ResponseData],
        ttl_hours: float,
        max_turns: int,
        now: float,
    ) -> tuple[list[ResponseData], int]:
        """Filter responses, returning (valid_list, removed_count)."""
        ttl_seconds = ttl_hours * 3600
        valid = []

        # TTL Check
        for r in responses:
            if (now - r["timestamp"]) <= ttl_seconds:
                valid.append(r)

        removed_ttl = len(responses) - len(valid)

        # Max Turns Check
        if len(valid) > max_turns:
            removed_turns = len(valid) - max_turns
            valid = valid[-max_turns:]
        else:
            removed_turns = 0

        return valid, (removed_ttl + removed_turns)
