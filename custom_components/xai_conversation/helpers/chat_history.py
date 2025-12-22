"""Specialized Chat History Service for Grok Code Fast.

This service manages the persistent storage of complete chat messages (text and code)
exclusively for the Grok Code Fast frontend experience. It enables:
- Cross-device synchronization of the Grok Code Fast card UI.
- Persistent history for the Grok Code Fast service.
- Cross-session continuity for Grok Code Fast developers.

ARCHITECTURE NOTE:
This is SEPARATE from the standard conversation memory.
- ChatHistoryService: Stores FULL message content (text/code) for Grok Code Fast only.
- ConversationMemory: Stores ONLY response_ids for xAI server-side memory chaining
  (used by pipeline and tools modes).

Used by:
- grok_code_fast service: Saves/loads chat messages for frontend card synchronization.
- clear_code_memory service: Clears stored chat history for a specific user.
- sync_chat_history service: Provides the data for cross-device synchronization.
"""

from __future__ import annotations

import time
from typing import Any

from ..const import LOGGER, RECOMMENDED_CHAT_HISTORY_MAX_MESSAGES


class ChatHistoryService:
    """Autonomous chat history service for Grok Code Fast.

    Manages full message storage (user prompts + assistant responses) for:
    - Frontend card UI synchronization across devices
    - Client-side memory mode (when store_messages=False)
    - Chat history export/import

    Features:
    - Asynchronous message saving (fire-and-forget, zero latency impact)
    - Synchronous history loading (for frontend sync button)
    - Automatic cleanup on conversation reset
    - Per-user, per-mode message storage

    Note: Separate from ConversationMemory which only stores response_ids for
    server-side memory chaining. This service stores complete message text.
    """

    def __init__(self, hass, storage_path: str, entry):
        """Initialize chat history service.

        Args:
            hass: Home Assistant instance
            storage_path: Storage file path (e.g., "xai_conversation/chat_history")
            entry: Config entry (for TTL/max_messages configuration)
        """
        from homeassistant.helpers.storage import Store

        self.hass = hass
        self.entry = entry
        self._store = Store(hass, 1, storage_path)
        self._history: dict[str, dict] = {}
        self._loaded = False

    async def _ensure_loaded(self):
        """Ensure history is loaded from storage."""
        if self._loaded:
            return

        try:
            data = await self._store.async_load()
            if isinstance(data, dict):
                self._history = data
            else:
                self._history = {}
            self._loaded = True
            LOGGER.debug("Chat history loaded: %d conversations", len(self._history))
        except Exception as err:
            LOGGER.warning("Failed to load chat history: %s", err)
            self._history = {}
            self._loaded = True

    def _get_history_key(self, user_id: str, mode: str) -> str:
        """Build history key for user and mode.

        Args:
            user_id: User ID
            mode: Chat mode (code, tools, pipeline)

        Returns:
            History key string
        """
        return f"user:{user_id}:mode:{mode}"

    async def _do_save(self, user_id: str, mode: str, role: str, content: str):
        """Internal save implementation (runs async).

        Args:
            user_id: User ID
            mode: Chat mode
            role: Message role (user/assistant)
            content: Message content
        """
        await self._ensure_loaded()

        history_key = self._get_history_key(user_id, mode)

        # Get or create history entry
        if history_key not in self._history:
            self._history[history_key] = {
                "messages": [],
                "created_at": time.time(),
                "last_updated": time.time(),
            }

        # Add message with timestamp
        self._history[history_key]["messages"].append(
            {"role": role, "content": content, "timestamp": time.time()}
        )
        self._history[history_key]["last_updated"] = time.time()

        # Apply limits (keep last N messages max)
        if (
            len(self._history[history_key]["messages"])
            > RECOMMENDED_CHAT_HISTORY_MAX_MESSAGES
        ):
            self._history[history_key]["messages"] = self._history[history_key][
                "messages"
            ][-RECOMMENDED_CHAT_HISTORY_MAX_MESSAGES:]

        # Save to disk (async, non-blocking)
        try:
            await self._store.async_save(self._history)
            LOGGER.debug(
                "Chat history saved: %s role=%s len=%d", history_key, role, len(content)
            )
        except Exception as err:
            LOGGER.error("Failed to save chat history: %s", err)

    def save_message_async(self, user_id: str, mode: str, role: str, content: str):
        """Save message asynchronously (fire-and-forget).

        This method returns immediately without waiting for save completion.
        Zero impact on chat response latency.

        Args:
            user_id: User ID
            mode: Chat mode (code, tools, pipeline)
            role: Message role (user/assistant)
            content: Message content
        """
        self.hass.async_create_task(self._do_save(user_id, mode, role, content))

    async def async_flush(self):
        """Flush pending changes to disk.

        Current implementation saves on every message, so this is primarily
        to satisfy the coordinator's unload protocol.
        """
        # No-op for now as we save immediately in _do_save.
        # Future optimization: Implement buffering/dirty flags like MemoryManager.
        pass

    async def load_history(
        self, user_id: str, mode: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Load chat history synchronously (for frontend sync).

        Args:
            user_id: User ID
            mode: Chat mode
            limit: Maximum number of messages to return (default: 50)

        Returns:
            List of messages with role, content, timestamp
        """
        await self._ensure_loaded()

        history_key = self._get_history_key(user_id, mode)
        history_entry = self._history.get(history_key)

        if not history_entry:
            LOGGER.debug("No chat history found for %s", history_key)
            return []

        messages = history_entry.get("messages", [])

        # Return last N messages
        result = messages[-limit:] if limit else messages

        LOGGER.debug(
            "Chat history loaded: %s messages=%d (limited to %d)",
            history_key,
            len(messages),
            len(result),
        )

        return result

    async def clear_history(self, user_id: str, mode: str):
        """Clear chat history for user and mode.

        Called when conversation memory is reset.

        Args:
            user_id: User ID
            mode: Chat mode
        """
        await self._ensure_loaded()

        history_key = self._get_history_key(user_id, mode)

        if history_key in self._history:
            del self._history[history_key]

            try:
                await self._store.async_save(self._history)
                LOGGER.info("Chat history cleared: %s", history_key)
            except Exception as err:
                LOGGER.error("Failed to save after clearing history: %s", err)
        else:
            LOGGER.debug("No chat history to clear for %s", history_key)
