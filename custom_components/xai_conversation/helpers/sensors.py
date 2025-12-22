"""Token statistics manager for xAI Conversation integration - V2 Simplified.

This module provides a clean, focused implementation of token tracking:
- Single responsibility: manage token statistics and pricing storage
- No sensor logic mixed in
- Clear separation between data and presentation
- Simplified concurrency model
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Callable

from homeassistant.util import dt as dt_util

from ..const import (
    CONF_TOKENS_PER_MILLION,
    CONF_COST_PER_TOOL_CALL,
    LOGGER,
    RECOMMENDED_TOKENS_PER_MILLION,
    RECOMMENDED_COST_PER_TOOL_CALL,
    TOOL_PRICING,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant
    from .. import XAIConfigEntry


class TokenStats:
    """Central storage and calculator for token statistics and costs.

    This class is responsible for:
    - Accumulating usage data from services (fire-and-forget).
    - Storing pricing data (pushed by XAIModelManager).
    - Managing a persistent list of known models for new model detection.
    - Calculating aggregated statistics and costs.
    - Persisting data to disk.
    - Notifying listeners (sensors) of changes.

    NOTE: This class is NOT responsible for fetching model data or detecting new models.
    That responsibility lies with XAIModelManager.
    """

    def __init__(
        self, hass: HomeAssistant, storage_path: str, entry: XAIConfigEntry
    ) -> None:
        """Initialize token statistics manager.

        Args:
            hass: Home Assistant instance
            storage_path: Path for JSON storage file
            entry: Config entry
        """
        from homeassistant.helpers.storage import Store

        self.hass = hass
        self.entry = entry
        self._store = Store(hass, 1, storage_path)
        self._data: dict = {}
        self._loaded = False
        self._dirty = False
        self._lock = asyncio.Lock()
        self._listeners: list[Callable[[], None]] = []

        # Track background tasks to prevent garbage collection
        self._background_tasks: set = set()

    def _get_tokens_per_million(self) -> int:
        """Get tokens per million from sensors subentry config."""
        for subentry in self.entry.subentries.values():
            if subentry.subentry_type == "sensors":
                return int(
                    subentry.data.get(
                        CONF_TOKENS_PER_MILLION, RECOMMENDED_TOKENS_PER_MILLION
                    )
                )
        return RECOMMENDED_TOKENS_PER_MILLION

    def _get_cost_per_tool_call(self) -> float:
        """Get cost per tool call from sensors subentry config."""
        for subentry in self.entry.subentries.values():
            if subentry.subentry_type == "sensors":
                return float(
                    subentry.data.get(
                        CONF_COST_PER_TOOL_CALL, RECOMMENDED_COST_PER_TOOL_CALL
                    )
                )
        return RECOMMENDED_COST_PER_TOOL_CALL

    # =========================================================================
    # LISTENER MANAGEMENT (Observer Pattern)
    # =========================================================================

    def register_listener(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Register a listener to be notified when stats change.

        Args:
            callback: Function to call when stats are updated

        Returns:
            Unsubscribe function
        """
        self._listeners.append(callback)

        def unsubscribe():
            if callback in self._listeners:
                self._listeners.remove(callback)

        return unsubscribe

    def _notify_listeners(self) -> None:
        """Notify all registered listeners that data has changed."""
        for callback in self._listeners:
            try:
                res = callback()
                if asyncio.iscoroutine(res):
                    self.hass.async_create_task(res)
            except Exception as err:
                LOGGER.error("Error notifying token stats listener: %s", err)

    # =========================================================================
    # INTERNAL: STORAGE MANAGEMENT
    # =========================================================================

    async def _ensure_loaded(self) -> None:
        """Ensure data is loaded from storage (must hold lock)."""
        if self._loaded:
            return

        try:
            data = await self._store.async_load()
            if isinstance(data, dict):
                self._data = data
            else:
                self._data = {}

            # MIGRATION V1 -> V2 (One-time)
            # If we have 'new_models' (V1) but 'known_models' (V2) is empty or missing:
            if "new_models" in self._data:
                old_data = self._data["new_models"]
                if isinstance(old_data, dict):
                    old_acknowledged = old_data.get("acknowledged_models", [])
                    if old_acknowledged:
                        current_known = self._data.get("known_models", [])
                        if not current_known:
                            LOGGER.info(
                                "Migrating legacy known models data to V2 format..."
                            )
                            self._data["known_models"] = list(
                                set(old_acknowledged)
                            )  # deduplicate
                            # Remove legacy key
                            del self._data["new_models"]
                            # Force save
                            self._dirty = True
                            await self.async_flush()

            # Ensure known_models exists
            if "known_models" not in self._data:
                self._data["known_models"] = []

            # Ensure reset_timestamp exists
            if "reset_timestamp" not in self._data:
                self._data["reset_timestamp"] = dt_util.now().timestamp()

            # Ensure processed_ids exists for deduplication
            if "processed_ids" not in self._data:
                self._data["processed_ids"] = []

            self._loaded = True
            LOGGER.debug("Loaded token stats from storage")
        except Exception as err:
            LOGGER.warning("Failed to load token stats storage: %s", err)
            self._data = {}
            # Ensure critical fields exist even if load failed
            self._data["known_models"] = []
            self._data["reset_timestamp"] = dt_util.now().timestamp()
            self._loaded = True

    async def async_flush(self) -> None:
        """Flush pending changes to disk."""
        if not self._dirty:
            return

        try:
            await self._store.async_save(self._data)
            self._dirty = False
            LOGGER.debug("Token stats flushed to disk")
        except Exception as err:
            LOGGER.error("Failed to save token stats: %s", err)

    # =========================================================================
    # PUBLIC API: USAGE TRACKING
    # =========================================================================

    async def async_update_usage(
        self,
        service_type: str,
        model: str,
        usage: Any,
        mode: str = "pipeline",
        is_fallback: bool = False,
        store_messages: bool = True,
        server_side_tool_usage: dict | None = None,
        num_sources_used: int = 0,
        response_id: str | None = None,
    ) -> None:
        """Update token usage statistics (fire-and-forget).

        This method returns immediately without blocking the caller.
        The actual processing happens asynchronously in the background.

        CRITICAL: This ensures conversation/ai_task/code_fast services
        are NOT delayed by token tracking I/O operations.

        Args:
            service_type: "conversation", "ai_task", or "code_fast"
            model: Model name
            usage: xAI response usage object or dict
            mode: "pipeline" or "tools" (conversation only)
            is_fallback: True if fallback occurred (conversation only)
            store_messages: Memory persistence mode (conversation only)
            server_side_tool_usage: Usage stats for server-side tools (e.g. {"web_search": 1})
            num_sources_used: Number of search sources used (for cost calculation)
            response_id: Unique xAI response ID for deduplication
        """
        if not model or model == "null":
            return

        # Create background task - returns immediately
        task = self.hass.async_create_task(
            self._process_usage_update(
                service_type,
                model,
                usage,
                mode,
                is_fallback,
                store_messages,
                server_side_tool_usage,
                num_sources_used,
                response_id,
            )
        )

        # Track task to prevent garbage collection
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _process_usage_update(
        self,
        service_type: str,
        model: str,
        usage: Any,
        mode: str,
        is_fallback: bool,
        store_messages: bool,
        server_side_tool_usage: dict | None = None,
        num_sources_used: int = 0,
        response_id: str | None = None,
    ) -> None:
        """Internal: Process usage update with lock and I/O (runs in background)."""
        try:
            async with self._lock:
                await self._ensure_loaded()

                # Deduplication check
                if response_id:
                    if "processed_ids" not in self._data:
                        self._data["processed_ids"] = []

                    if response_id in self._data["processed_ids"]:
                        LOGGER.debug(
                            "TokenStats: Skipping duplicate response_id: %s",
                            response_id,
                        )
                        return

                    # Add to processed list and keep it manageable (max 100)
                    self._data["processed_ids"].append(response_id)
                    if len(self._data["processed_ids"]) > 100:
                        self._data["processed_ids"].pop(0)

                # Initialize structure
                if "token_stats" not in self._data:
                    self._data["token_stats"] = {}

                # 1. Update service-specific stats
                if service_type not in self._data["token_stats"]:
                    self._data["token_stats"][service_type] = {}

                self._update_stats_dict(
                    self._data["token_stats"][service_type],
                    usage,
                    model,
                    is_service_stats=True,
                    mode=mode,
                    is_fallback=is_fallback,
                    store_messages=store_messages,
                )

                # 2. Update aggregated stats
                if "aggregated" not in self._data["token_stats"]:
                    self._data["token_stats"]["aggregated"] = {}

                self._update_stats_dict(
                    self._data["token_stats"]["aggregated"],
                    usage,
                    model,
                    is_service_stats=False,
                )

                # 3. Update Server-Side Tool Stats (Accumulation)
                if server_side_tool_usage or num_sources_used > 0:
                    if "server_tools" not in self._data:
                        self._data["server_tools"] = {
                            "total_invocations": 0,
                            "total_sources": 0,
                            "by_tool": {},
                            "by_service": {},
                        }

                    # Ensure keys exist (helpful for migrations or partial inits)
                    st_data = self._data["server_tools"]
                    if "total_invocations" not in st_data:
                        st_data["total_invocations"] = 0
                    if "total_sources" not in st_data:
                        st_data["total_sources"] = 0
                    if "by_tool" not in st_data:
                        st_data["by_tool"] = {}
                    if "by_service" not in st_data:
                        st_data["by_service"] = {}

                    # Accumulate sources
                    self._data["server_tools"]["total_sources"] += num_sources_used

                    if server_side_tool_usage:
                        # Accumulate totals
                        for tool_type, count in server_side_tool_usage.items():
                            self._data["server_tools"]["total_invocations"] += count

                            # Normalize tool name (SDK uses SERVER_SIDE_TOOL_* prefix)
                            normalized_tool = tool_type.lower().replace(
                                "server_side_tool_", ""
                            )

                            # By tool type (using normalized name)
                            if (
                                normalized_tool
                                not in self._data["server_tools"]["by_tool"]
                            ):
                                self._data["server_tools"]["by_tool"][
                                    normalized_tool
                                ] = 0
                            self._data["server_tools"]["by_tool"][normalized_tool] += (
                                count
                            )

                            # By service type
                            if (
                                service_type
                                not in self._data["server_tools"]["by_service"]
                            ):
                                self._data["server_tools"]["by_service"][
                                    service_type
                                ] = 0
                            self._data["server_tools"]["by_service"][service_type] += (
                                count
                            )

                    LOGGER.debug(
                        "Tracked server tool usage: %s, sources=%d (service=%s)",
                        server_side_tool_usage,
                        num_sources_used,
                        service_type,
                    )

                # 4. Mark as dirty (write-behind)
                self._dirty = True

            # 5. Notify listeners (outside lock to avoid deadlock)
            self._notify_listeners()

        except Exception as err:
            LOGGER.error(
                "Failed to process token usage update for %s/%s: %s",
                service_type,
                model,
                err,
                exc_info=True,
            )

    def _update_stats_dict(
        self,
        stats: dict,
        usage: Any,
        model: str,
        is_service_stats: bool,
        mode: str = "pipeline",
        is_fallback: bool = False,
        store_messages: bool = True,
    ) -> None:
        """Calculate and accumulate stats into a dictionary.

        Args:
            stats: Dictionary to update (modified in-place)
            usage: xAI response usage object
            model: Model name
            is_service_stats: True for service-specific, False for aggregated
            mode: Conversation mode
            is_fallback: Fallback flag
            store_messages: Memory mode
        """
        # Extract token counts
        completion = self._get_token_count(usage, "completion_tokens")
        prompt = self._get_token_count(usage, "prompt_tokens")
        cached = self._get_token_count(usage, "cached_prompt_text_tokens")
        reasoning = self._get_token_count(usage, "reasoning_tokens")

        # Initialize counters if missing
        defaults = {
            "cumulative_completion_tokens": 0,
            "cumulative_prompt_tokens": 0,
            "cumulative_cached_tokens": 0,
            "cumulative_reasoning_tokens": 0,
            "message_count": 0,
            "tokens_by_model": {},
        }
        for k, v in defaults.items():
            if k not in stats:
                stats[k] = v

        # Accumulate totals
        stats["cumulative_completion_tokens"] += completion
        stats["cumulative_prompt_tokens"] += prompt
        stats["cumulative_cached_tokens"] += cached
        stats["cumulative_reasoning_tokens"] += reasoning
        stats["message_count"] += 1

        # Update per-model breakdown
        if model not in stats["tokens_by_model"]:
            stats["tokens_by_model"][model] = {
                "completion": 0,
                "prompt": 0,
                "cached": 0,
                "reasoning": 0,
                "count": 0,
            }

        m_stats = stats["tokens_by_model"][model]
        m_stats["completion"] += completion
        m_stats["prompt"] += prompt
        m_stats["cached"] += cached
        m_stats["reasoning"] += reasoning
        m_stats["count"] += 1

        # Update "last message" snapshot
        stats.update(
            {
                "last_completion_tokens": completion,
                "last_prompt_tokens": prompt,
                "last_cached_tokens": cached,
                "last_reasoning_tokens": reasoning,
                "last_model": model,
                "last_timestamp": dt_util.now().isoformat(),
            }
        )

        # Service-specific metadata
        if is_service_stats:
            stats["last_mode"] = "tools" if is_fallback else mode
            stats["last_store_messages"] = store_messages

            # Update cache breakdown buckets (conversation only)
            self._update_cache_breakdown(
                stats, prompt, cached, mode, is_fallback, store_messages
            )

    def _update_cache_breakdown(
        self,
        stats: dict,
        prompt: int,
        cached: int,
        mode: str,
        is_fallback: bool,
        store_messages: bool,
    ) -> None:
        """Update cache breakdown buckets for conversation analysis."""
        # Initialize buckets if missing
        buckets = [
            "tokens_pipeline_server",
            "tokens_pipeline_client",
            "tokens_tools_server",
            "tokens_tools_client",
        ]
        for b in buckets:
            if b not in stats:
                stats[b] = {"prompt": 0, "cached": 0, "count": 0}

        # Determine target bucket
        if mode == "pipeline" and not is_fallback:
            target = (
                "tokens_pipeline_server" if store_messages else "tokens_pipeline_client"
            )
        else:
            target = "tokens_tools_server" if store_messages else "tokens_tools_client"

        # Update bucket
        stats[target]["prompt"] += prompt
        stats[target]["cached"] += cached
        stats[target]["count"] += 1

    @staticmethod
    def _get_token_count(usage: Any, attr: str) -> int:
        """Safely extract token count from object or dict."""
        if isinstance(usage, dict):
            val = usage.get(attr, 0)
        else:
            val = getattr(usage, attr, 0)
        return int(val) if val is not None else 0

    # =========================================================================
    # PUBLIC API: GETTERS (Read-Only)
    # =========================================================================

    async def get_service_stats(self, service_type: str) -> dict:
        """Get stats for a specific service.

        Args:
            service_type: "conversation", "ai_task", or "code_fast"

        Returns:
            Dictionary with service-specific statistics
        """
        async with self._lock:
            await self._ensure_loaded()
            return self._data.get("token_stats", {}).get(service_type, {}).copy()

    async def get_aggregated_stats(self) -> dict:
        """Get aggregated stats across all services.

        Returns:
            Dictionary with aggregated statistics
        """
        async with self._lock:
            await self._ensure_loaded()
            stats = self._data.get("token_stats", {}).get("aggregated", {}).copy()
            # Inject reset_timestamp from root data
            stats["reset_timestamp"] = self._data.get("reset_timestamp")
            return stats

    async def get_server_tool_stats(self) -> dict:
        """Get server-side tool usage statistics.

        Returns:
            Dictionary with server tool usage data
        """
        async with self._lock:
            await self._ensure_loaded()
            return self._data.get("server_tools", {}).copy()

    async def get_pricing(self, model: str, price_type: str) -> float | None:
        """Get pricing data for a model.

        Args:
            model: Model name
            price_type: "input_price", "output_price", or "cached_input_price"

        Returns:
            Price per million tokens, or None if not set
        """
        async with self._lock:
            await self._ensure_loaded()
            return self._data.get("pricing_data", {}).get(model, {}).get(price_type)

    async def get_all_pricing_data(self) -> dict:
        """Get all stored pricing data.

        Returns:
            Dictionary mapping model_name -> {price_type: price}
        """
        async with self._lock:
            await self._ensure_loaded()
            return self._data.get("pricing_data", {}).copy()

    async def get_costs(self) -> dict:
        """Calculate and return cost data ready for presentation.

        Calculates costs on-demand based on current tokens and pricing.
        This ensures costs are always up-to-date even if pricing changes.

        Returns:
            Dictionary with:
                - total_cost: float (USD)
                - cost_by_model: dict with per-model breakdown
                - tokens_by_model: dict with token counts
        """
        async with self._lock:
            await self._ensure_loaded()

            # Get aggregated stats
            stats = self._data.get("token_stats", {}).get("aggregated", {})
            tokens_by_model = stats.get("tokens_by_model", {})

            # Get pricing data
            pricing_data = self._data.get("pricing_data", {})

            # Calculate costs
            total_cost = 0.0
            cost_breakdown = {}

            # Get tokens_per_million from config
            tokens_per_million = self._get_tokens_per_million()

            for model, counts in tokens_by_model.items():
                model_pricing = pricing_data.get(model, {})
                input_price = model_pricing.get("input_price", 0.0)
                output_price = model_pricing.get("output_price", 0.0)
                cached_price = model_pricing.get("cached_input_price", input_price)

                # Calculate costs per type (per million tokens)
                c_prompt = (counts.get("prompt", 0) / tokens_per_million) * input_price
                c_cached = (counts.get("cached", 0) / tokens_per_million) * cached_price
                c_completion = (
                    counts.get("completion", 0) / tokens_per_million
                ) * output_price

                model_total = c_prompt + c_cached + c_completion
                total_cost += model_total

                cost_breakdown[model] = {
                    "prompt_cost": round(c_prompt, 4),
                    "cached_cost": round(c_cached, 4),
                    "completion_cost": round(c_completion, 4),
                    "total_cost": round(model_total, 4),
                    "tokens": counts.copy(),
                }

            # Calculate Tool Costs (Invocation based with per-tool pricing)
            server_tools = self._data.get("server_tools", {})
            by_tool = server_tools.get("by_tool", {})

            # Calculate cost for each tool type using official pricing
            tool_cost = 0.0
            tool_cost_breakdown = {}

            for tool_name, invocations in by_tool.items():
                # Get price for this specific tool (fallback to default if not in map)
                price = TOOL_PRICING.get(tool_name, self._get_cost_per_tool_call())
                cost = invocations * price
                tool_cost += cost
                tool_cost_breakdown[tool_name] = {
                    "invocations": invocations,
                    "cost_per_call": price,
                    "total_cost": round(cost, 4),
                }

            total_cost += tool_cost

            return {
                "total_cost": round(total_cost, 4),
                "tool_cost": round(tool_cost, 4),
                "tool_cost_breakdown": tool_cost_breakdown,
                "cost_by_model": cost_breakdown,
                "tokens_by_model": tokens_by_model.copy(),
            }

    async def get_known_models(self) -> list[str]:
        """Get the list of models that have been previously detected and acknowledged."""
        async with self._lock:
            await self._ensure_loaded()
            return self._data.get("known_models", []).copy()

    async def add_known_models(self, models: list[str]) -> None:
        """Add new models to the list of known models and persist to disk.

        Args:
            models: A list of model names to add to the known models.
        """
        if not models:
            return
        async with self._lock:
            await self._ensure_loaded()
            updated = False
            for model in models:
                if model not in self._data["known_models"]:
                    self._data["known_models"].append(model)
                    updated = True
            if updated:
                self._dirty = True
                await self.async_flush()
        self._notify_listeners()

    # =========================================================================
    # PUBLIC API: SETTERS (Admin Operations)
    # =========================================================================

    async def save_pricing(self, model: str, price_type: str, price: float) -> None:
        """Save pricing data for a model.

        Args:
            model: Model name
            price_type: "input_price", "output_price", or "cached_input_price"
            price: Price per million tokens
        """
        async with self._lock:
            await self._ensure_loaded()
            if "pricing_data" not in self._data:
                self._data["pricing_data"] = {}
            if model not in self._data["pricing_data"]:
                self._data["pricing_data"][model] = {}

            self._data["pricing_data"][model][price_type] = price
            self._data["pricing_data"][model]["last_updated"] = (
                dt_util.now().timestamp()
            )

            self._dirty = True
            await self.async_flush()

        self._notify_listeners()

    async def save_pricing_batch(
        self, pricing_updates: dict[str, dict[str, float]]
    ) -> None:
        """Save pricing data for multiple models at once.

        Args:
            pricing_updates: Dict mapping model_name -> {price_type: price}
        """
        async with self._lock:
            await self._ensure_loaded()
            if "pricing_data" not in self._data:
                self._data["pricing_data"] = {}

            updated = False
            now_ts = dt_util.now().timestamp()

            for model, prices in pricing_updates.items():
                if model not in self._data["pricing_data"]:
                    self._data["pricing_data"][model] = {}

                for price_type, price in prices.items():
                    self._data["pricing_data"][model][price_type] = price
                    self._data["pricing_data"][model]["last_updated"] = now_ts
                    updated = True

            if updated:
                self._dirty = True
                await self.async_flush()

        self._notify_listeners()

    # =========================================================================
    # PUBLIC API: SERVICES INTEGRATION
    # =========================================================================

    async def reset_stats(self) -> None:
        """Reset token statistics (preserve pricing data and known models).

        Called by: ResetTokenStatsService
        """
        async with self._lock:
            await self._ensure_loaded()

            # Preserve pricing data, known models, and server tools?
            # Assuming server tools usage should ALSO be reset as it's a usage stat.
            # Preserving: pricing_data, known_models.
            pricing_data = self._data.get("pricing_data", {})
            known_models = self._data.get("known_models", [])

            # Clear stats but keep persistent data
            self._data = {
                "token_stats": {},
                "pricing_data": pricing_data,
                "known_models": known_models,
                # Reset server tools too
                "server_tools": {},
                # Update reset timestamp
                "reset_timestamp": dt_util.now().timestamp(),
            }

            self._dirty = True
            await self.async_flush()
            LOGGER.info("Token statistics reset (pricing and known models preserved)")

        self._notify_listeners()
