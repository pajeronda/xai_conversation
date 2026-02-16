"""Conversation memory management for xAI conversation."""

from __future__ import annotations

import time
import asyncio
from contextvars import ContextVar
from datetime import timedelta
from typing import TYPE_CHECKING, TypedDict, NotRequired, Iterable, Callable, Any

from homeassistant.helpers.event import async_track_time_interval
from homeassistant.helpers.storage import Store

from ..const import (
    CONF_MEMORY_DEVICE_TTL_HOURS,
    CONF_MEMORY_USER_TTL_HOURS,
    LOGGER,
    RECOMMENDED_MEMORY_DEVICE_TTL_HOURS,
    RECOMMENDED_MEMORY_USER_TTL_HOURS,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant
    from homeassistant.config_entries import ConfigEntry


class ResponseData(TypedDict):
    """Structure for a stored response."""

    id: str
    timestamp: float
    root_id: NotRequired[str]  # Root ID of the xAI conversation chain (Session)
    store_messages: NotRequired[bool]


# Context variable to track the active parent response ID during an execution flow.
# This allows automatic session tagging without modifying external function signatures.
_ACTIVE_PARENTID: ContextVar[str | None] = ContextVar(
    "xai_active_parentid", default=None
)


class MemoryEntry(TypedDict):
    """Structure for a conversation memory entry."""

    responses: list[ResponseData]
    prompt_hash: NotRequired[str]  # Hash of the prompt used to start/update the session
    encrypted_blob: NotRequired[str]  # ZDR Blob (latest)
    last_updated: NotRequired[float]  # Last activity timestamp


class MemoryManager:
    """Standalone conversation memory manager.

    Provides persistent storage for conversation response IDs with automatic
    TTL-based cleanup and size limiting.
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
        self._store = Store(hass, 2, storage_path, minor_version=0)
        self._memory: dict[str, MemoryEntry] = {}
        self._loaded = False
        self._dirty = False
        self._deleted_keys: set[str] = set()
        self._listeners: list[Callable[[], Any]] = []

    # --- Listener Management (Observer Pattern) ---

    def register_listener(self, callback: Callable[[], Any]) -> Callable[[], None]:
        """Register a listener to be notified when memory changes."""
        self._listeners.append(callback)

        def unsubscribe():
            if callback in self._listeners:
                self._listeners.remove(callback)

        return unsubscribe

    def _notify_listeners(self) -> None:
        """Notify all registered listeners that memory has changed."""
        for callback in list(self._listeners):
            try:
                res = callback()
                if asyncio.iscoroutine(res):
                    self.hass.async_create_task(res)
            except Exception as err:
                LOGGER.error("[memory] listener notification failed: %s", err)

    @staticmethod
    def generate_key(
        scope: str,
        identifier: str,
        mode: str,
        subentry_id: str,
    ) -> str:
        """Generate a consistent memory key.

        Format: {scope}:{identifier}:sub:{subentry_id}:mode:{mode}

        Args:
            scope: "user" or "device".
            identifier: User ID or Device ID.
            mode: "ai_task", "pipeline", "tools", "vision", "chatonly".
            subentry_id: Subentry ID for conversation isolation.
        """
        return f"{scope}:{identifier}:sub:{subentry_id}:mode:{mode}"

    @staticmethod
    def parse_key(key: str) -> tuple[str, str, str, str] | None:
        """Parse a memory key into (scope, identifier, subentry_id, mode).

        Expected format: {scope}:{identifier}:sub:{subentry_id}:mode:{mode}
        Returns None if the key does not match the expected format.
        """
        if ":sub:" not in key or ":mode:" not in key:
            return None

        scope_and_id, _, rest = key.partition(":sub:")
        if not scope_and_id or not rest:
            return None

        if ":" not in scope_and_id:
            return None

        scope, identifier = scope_and_id.split(":", 1)
        subentry_id, _, mode = rest.partition(":mode:")
        if not scope or not identifier or not subentry_id or not mode:
            return None

        return scope, identifier, subentry_id, mode

    async def async_get_last_response_id(self, key: str) -> str | None:
        """Retrieve the last response ID for a given key."""
        await self._ensure_loaded()

        # New conversation turn: clear deletion guard so new saves are accepted
        self._deleted_keys.discard(key)

        entry = self._memory.get(key)
        if not entry or not entry["responses"]:
            _ACTIVE_PARENTID.set(None)
            return None

        last_id = entry["responses"][-1]["id"]
        # Set as active parent for subsequent save_response calls in same task
        _ACTIVE_PARENTID.set(last_id)
        return last_id

    async def async_get_turn_count(self, key: str) -> int:
        """Return number of stored turns for a given memory key."""
        await self._ensure_loaded()
        entry = self._memory.get(key)
        if not entry:
            return 0
        return len(entry.get("responses", []))

    async def async_get_turn_counts(
        self,
    ) -> list[tuple[str, str, str, str, int]]:
        """Return turn counts for all memory keys.

        Returns a list of tuples:
        (scope, identifier, subentry_id, mode, turns)
        """
        await self._ensure_loaded()
        results: list[tuple[str, str, str, str, int]] = []
        for key, entry in self._memory.items():
            responses = entry.get("responses", [])
            if not responses:
                continue

            parsed = self.parse_key(key)
            if not parsed:
                continue

            scope, identifier, subentry_id, mode = parsed
            results.append((scope, identifier, subentry_id, mode, len(responses)))

        return results

    async def async_get_stored_hash(self, key: str) -> str | None:
        """Retrieve the stored prompt hash for a given key."""
        await self._ensure_loaded()
        entry = self._memory.get(key)
        return entry.get("prompt_hash") if entry else None

    async def async_save_response(
        self,
        key: str,
        response_id: str,
        store_messages: bool | None = None,
        prompt_hash: str | None = None,
    ) -> None:
        """Save a response ID to memory with automatic root_id (session) inheritance."""
        await self._ensure_loaded()

        if key in self._deleted_keys:
            return

        if key not in self._memory:
            self._memory[key] = {"responses": []}

        if prompt_hash:
            self._memory[key]["prompt_hash"] = prompt_hash

        parent_id = _ACTIVE_PARENTID.get()
        root_id = response_id  # Default for new sessions

        # If a parent exists in current context, inherit its root_id
        if parent_id:
            for r in reversed(self._memory[key]["responses"]):
                if r["id"] == parent_id:
                    root_id = r.get("root_id") or r["id"]
                    break

        data: ResponseData = {
            "id": response_id,
            "timestamp": time.time(),
            "root_id": root_id,
        }
        if store_messages is not None:
            data["store_messages"] = store_messages

        self._memory[key]["responses"].append(data)
        self._dirty = True

        # Update active parent to the new response ID for subsequent turns (e.g., tool loops)
        _ACTIVE_PARENTID.set(response_id)
        self._notify_listeners()

    async def async_save_encrypted_blob(self, key: str, blob: str) -> None:
        """Save an encrypted blob for ZDR mode (overwrites previous)."""
        await self._ensure_loaded()

        if key in self._deleted_keys:
            return

        if key not in self._memory:
            self._memory[key] = {"responses": []}

        self._memory[key]["encrypted_blob"] = blob
        self._memory[key]["last_updated"] = time.time()
        self._dirty = True

    async def async_get_encrypted_blob(self, key: str) -> str | None:
        """Get the encrypted blob for a key."""
        await self._ensure_loaded()
        entry = self._memory.get(key)
        if not entry:
            return None
        return entry.get("encrypted_blob")

    async def async_flush(self):
        """Flush pending changes to disk."""
        if not self._dirty:
            return
        try:
            await self._store.async_save(self._memory)
            self._dirty = False
            LOGGER.debug("[memory] flushed to disk")
        except Exception as err:
            LOGGER.error("[memory] save failed: %s", err)

    async def async_delete(
        self,
        scope: str | None = None,
        identifier: str | None = None,
        key: str | None = None,
        all: bool = False,
        physical: bool = False,
    ) -> dict[str, list[str]]:
        """Unified deletion entry point. Removes memory locally.

        Args:
            scope: "user" or "device".
            identifier: User ID or Device ID.
            key: Specific conversation key.
            all: Clear entire memory storage.
            physical: Physically remove the storage file.

        Returns:
            Dict with:
                - "all": All deleted response IDs
                - "server_stored": Only IDs that were stored on xAI servers
                  (store_messages=True), suitable for remote deletion
        """
        await self._ensure_loaded()

        if physical:
            entries = list(self._memory.values())
            all_ids = self._get_ids(entries)
            server_ids = self._get_ids(entries, server_stored_only=True)
            try:
                self._deleted_keys.update(self._memory.keys())
                await self._store.async_remove()
                self._memory.clear()
                self._loaded = False
                self._dirty = False
                LOGGER.debug("[memory] storage file deleted")
            except Exception as err:
                LOGGER.error("[memory] delete failed: %s", err)
                raise
            return {"all": all_ids, "server_stored": server_ids}

        if all:
            keys_to_delete = list(self._memory.keys())
        elif key:
            keys_to_delete = [key] if key in self._memory else []
        elif scope and identifier:
            prefix = f"{scope}:{identifier}"
            keys_to_delete = [k for k in self._memory.keys() if k.startswith(prefix)]
        else:
            return {"all": [], "server_stored": []}

        return await self._delete_keys(keys_to_delete)

    # --- Internal / Maintenance ---

    async def _ensure_loaded(self):
        """Ensure memory is loaded from storage."""
        if self._loaded:
            return
        try:
            data = await self._store.async_load()
            self._memory = data if isinstance(data, dict) else {}
        except Exception as err:
            LOGGER.warning("[memory] load failed: %s", err)
            self._memory = {}
        finally:
            self._loaded = True

    def _get_ids(
        self, entries: Iterable[MemoryEntry], server_stored_only: bool = False
    ) -> list[str]:
        """Collect response IDs from provided memory entries.

        Args:
            entries: Memory entries to collect IDs from.
            server_stored_only: If True, only return IDs that were stored on xAI
                servers (store_messages=True). Used to filter what can be deleted remotely.

        Returns:
            List of response IDs.
        """
        ids = []
        for entry in entries:
            for r in entry.get("responses", []):
                if not r.get("id"):
                    continue
                if server_stored_only and not r.get("store_messages", False):
                    continue
                ids.append(r["id"])
        return ids

    async def _delete_keys(
        self, keys: list[str], flush: bool = True
    ) -> dict[str, list[str]]:
        """Internal helper to delete keys and return their response IDs."""
        if not keys:
            return {"all": [], "server_stored": []}

        entries = [self._memory[k] for k in keys if k in self._memory]
        all_ids = self._get_ids(entries)
        server_ids = self._get_ids(entries, server_stored_only=True)

        for key in keys:
            self._memory.pop(key, None)
            self._deleted_keys.add(key)

        self._dirty = True
        if flush:
            await self.async_flush()
        self._notify_listeners()

        return {"all": all_ids, "server_stored": server_ids}

    # --- Cleanup Logic ---

    def setup_periodic_cleanup(
        self,
        cleanup_interval_hours: int,
        on_cleanup: Callable[[list[str]], Any] | None = None,
    ):
        """Setup periodic cleanup task."""

        async def _cleanup(_now):
            deleted_ids = await self.async_cleanup_expired()
            if deleted_ids and deleted_ids.get("all"):
                LOGGER.debug(
                    "[memory] cleanup: %d sessions expired", len(deleted_ids["all"])
                )
                if on_cleanup:
                    await on_cleanup(deleted_ids)

        interval = timedelta(hours=cleanup_interval_hours)
        unsub = async_track_time_interval(self.hass, _cleanup, interval)
        return unsub

    async def async_cleanup_expired(
        self, scope: str | None = None, identifier: str | None = None
    ) -> dict[str, list[str]]:
        """Remove expired sessions based on inactivity.

        Args:
            scope: Optional scope (user/device) to filter cleanup.
            identifier: Optional identifier to filter cleanup.

        Returns:
            Dict with:
                - "all": All deleted response IDs
                - "server_stored": Only IDs stored on xAI servers
        """
        await self._ensure_loaded()
        if not self._memory:
            return {"all": [], "server_stored": []}

        # 1. Determine TTLs
        device_ttl = self.entry.data.get(
            CONF_MEMORY_DEVICE_TTL_HOURS, RECOMMENDED_MEMORY_DEVICE_TTL_HOURS
        ) * 3600
        user_ttl = self.entry.data.get(
            CONF_MEMORY_USER_TTL_HOURS, RECOMMENDED_MEMORY_USER_TTL_HOURS
        ) * 3600

        now = time.time()
        all_deleted_ids: list[str] = []
        server_stored_ids: list[str] = []
        keys_to_delete: list[str] = []

        # 2. Iterate Memory Entries
        for key, entry in self._memory.items():
            # Parse key to get scope
            parsed = self.parse_key(key)
            if not parsed:
                continue
            k_scope, k_id, _ = parsed

            # Filter if arguments provided
            if scope and k_scope != scope:
                continue
            if identifier and k_id != identifier:
                continue

            ttl = device_ttl if k_scope == "device" else user_ttl

            responses = entry.get("responses", [])
            # Check empty entries (no responses and no ZDR blob)
            if not responses and not entry.get("encrypted_blob"):
                keys_to_delete.append(key)
                continue

            # Group by root_id (Session)
            sessions: dict[str, list[ResponseData]] = {}
            for r in responses:
                rid = r.get("root_id") or r["id"]
                if rid not in sessions:
                    sessions[rid] = []
                sessions[rid].append(r)

            valid_responses: list[ResponseData] = []
            entry_modified = False

            for session_messages in sessions.values():
                # Check expiration (Inactive logic: whole session expires)
                last_used = max(m["timestamp"] for m in session_messages)
                if (now - last_used) > ttl:
                    # Expired
                    for m in session_messages:
                        if m.get("id"):
                            all_deleted_ids.append(m["id"])
                            if m.get("store_messages", False):
                                server_stored_ids.append(m["id"])
                    entry_modified = True
                else:
                    valid_responses.extend(session_messages)

            # Update entry if changed
            if entry_modified:
                if valid_responses:
                    entry["responses"] = sorted(valid_responses, key=lambda x: x["timestamp"])
                else:
                    entry["responses"] = []
                self._save_needed = True

            # If entry is now empty, mark for deletion
            if not entry.get("responses") and not entry.get("encrypted_blob"):
                keys_to_delete.append(key)

        # 3. Cleanup Keys
        for key in keys_to_delete:
            if key in self._memory:
                self._memory.pop(key)
                self._save_needed = True

        # 4. Rebuild Index and Finalize
        if all_deleted_ids:
            # Rebuild flat index to match memory
            new_responses = {}
            for entry in self._memory.values():
                for r in entry.get("responses", []):
                    new_responses[r["id"]] = r
            self._responses = new_responses

            LOGGER.info(
                "Cleanup: removed %d messages from expired sessions",
                len(all_deleted_ids),
            )
            # Cleanup ZDR blobs referenced by deleted sessions
            self._cleanup_encrypted_blobs(list(self._responses.values()))

        if self._save_needed:
            await self.async_flush()
            self._notify_listeners()

        if server_stored_ids:
            LOGGER.debug("Cleanup: %d server-side IDs flagged", len(server_stored_ids))

        return {"all": all_deleted_ids, "server_stored": server_stored_ids}
