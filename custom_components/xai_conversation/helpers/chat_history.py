"""Chat history service for xAI conversation.

This service provides autonomous, asynchronous chat history management.
It runs in parallel to the main chat flow without adding latency.
"""
from __future__ import annotations

import time
from typing import Any

from ..const import LOGGER


class ChatHistoryService:
    """Autonomous chat history service.

    Features:
    - Asynchronous message saving (fire-and-forget, zero latency impact)
    - Synchronous history loading (for frontend sync button)
    - Automatic cleanup on conversation reset
    - TTL and max_messages support
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
                "last_updated": time.time()
            }

        # Add message with timestamp
        self._history[history_key]["messages"].append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
        self._history[history_key]["last_updated"] = time.time()

        # Apply limits (keep last 100 messages max)
        max_messages = 100
        if len(self._history[history_key]["messages"]) > max_messages:
            self._history[history_key]["messages"] = \
                self._history[history_key]["messages"][-max_messages:]

        # Save to disk (async, non-blocking)
        try:
            await self._store.async_save(self._history)
            LOGGER.debug(
                "Chat history saved: %s role=%s len=%d",
                history_key, role, len(content)
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
        self.hass.async_create_task(
            self._do_save(user_id, mode, role, content)
        )

    async def load_history(
        self,
        user_id: str,
        mode: str,
        limit: int = 50
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
            history_key, len(messages), len(result)
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

    async def clear_all_history(self) -> int:
        """Clear all chat history.

        Returns:
            Number of conversations cleared
        """
        await self._ensure_loaded()

        count = len(self._history)
        self._history.clear()

        try:
            await self._store.async_save(self._history)
            LOGGER.info("All chat history cleared: %d conversations", count)
        except Exception as err:
            LOGGER.error("Failed to save after clearing all history: %s", err)

        return count

    async def get_stats(self) -> dict[str, Any]:
        """Get chat history statistics.

        Returns:
            Dictionary with stats (total conversations, total messages, etc.)
        """
        await self._ensure_loaded()

        total_conversations = len(self._history)
        total_messages = sum(
            len(entry.get("messages", []))
            for entry in self._history.values()
        )

        return {
            "total_conversations": total_conversations,
            "total_messages": total_messages,
            "conversations": list(self._history.keys())
        }
