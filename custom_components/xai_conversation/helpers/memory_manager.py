"""Conversation memory management for xAI conversation."""

from __future__ import annotations

import time
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
        """Remove expired sessions based on inactivity (Session-Aware).

        A session (all messages with same root_id) is expired only if its
        MOST RECENT message is older than the TTL.

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

        # Determine target key if scope/identifier provided
        target_key = f"{scope}:{identifier}" if scope and identifier else None

        now = time.time()
        all_deleted_ids: list[str] = []
        server_stored_ids: list[str] = []
        keys_to_delete = []

        # Iterate only over the target key or the whole memory
        items_to_process = (
            [(target_key, self._memory[target_key])]
            if target_key and target_key in self._memory
            else list(self._memory.items())
        )

        for key, entry in items_to_process:
            responses = entry.get("responses", [])
            last_updated = entry.get("last_updated", 0)
            blob = entry.get("encrypted_blob")

            if not responses and not blob:
                keys_to_delete.append(key)
                continue

            # Determine TTL based on key type
            key_scope = "device" if key.startswith("device:") else "user"
            if key_scope == "device":
                ttl_hours = self.entry.data.get(
                    CONF_MEMORY_DEVICE_TTL_HOURS, RECOMMENDED_MEMORY_DEVICE_TTL_HOURS
                )
            else:
                ttl_hours = self.entry.data.get(
                    CONF_MEMORY_USER_TTL_HOURS, RECOMMENDED_MEMORY_USER_TTL_HOURS
                )

            ttl_seconds = ttl_hours * 3600
            # Group by root_id
            sessions: dict[str, list[ResponseData]] = {}
            for r in responses:
                rid = r.get("root_id") or r["id"]
                if rid not in sessions:
                    sessions[rid] = []
                sessions[rid].append(r)

            valid_responses: list[ResponseData] = []
            session_deleted_count = 0

            for root_id, session_messages in sessions.items():
                # Check inactivity: only the most recent message matters
                last_used = max(m["timestamp"] for m in session_messages)
                if last_updated > last_used:
                    last_used = last_updated

                if (now - last_used) > ttl_seconds:
                    # Session expired - collect IDs
                    for m in session_messages:
                        if m.get("id"):
                            all_deleted_ids.append(m["id"])
                            if m.get("store_messages", False):
                                server_stored_ids.append(m["id"])
                    session_deleted_count += 1
                else:
                    # Session active
                    valid_responses.extend(session_messages)

            if not valid_responses and not blob:
                keys_to_delete.append(key)
            elif session_deleted_count > 0:
                entry["responses"] = sorted(
                    valid_responses, key=lambda x: x["timestamp"]
                )
                self._dirty = True

        for key in keys_to_delete:
            self._memory.pop(key, None)
            self._dirty = True

        if self._dirty:
            await self.async_flush()

        return {"all": all_deleted_ids, "server_stored": server_stored_ids}
