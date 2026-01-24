"""Token statistics manager for xAI Conversation integration.

Manages token statistics, pricing storage, and concurrent usage tracking.
Separates data management from sensor presentation logic.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Callable

from homeassistant.util import dt as dt_util

from ..const import (
    CONF_TOKENS_PER_MILLION,
    CONF_XAI_PRICING_CONVERSION_FACTOR,
    DEFAULT_TOOL_PRICE_RAW,
    LOGGER,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant
    from homeassistant.config_entries import ConfigEntry


class TokenStats:
    """Central storage and calculator for token statistics and costs.

    Accumulates usage data, stores pricing, manages known models list,
    and handles persistence and change notifications.
    """

    def __init__(
        self, hass: HomeAssistant, storage_path: str, entry: ConfigEntry
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
        self._store = Store(hass, 2, storage_path, minor_version=0)
        self._data: dict = {}
        self._loaded = False
        self._dirty = False
        self._lock = asyncio.Lock()
        self._listeners: list[Callable[[], None]] = []

        # Track background tasks to prevent garbage collection
        self._background_tasks: set = set()

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
                LOGGER.error("[stats] listener notification failed: %s", err)

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
                            LOGGER.debug("[stats] migrating legacy known models to V2")
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
        except Exception as err:
            LOGGER.warning("[stats] load failed: %s", err)
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
            LOGGER.debug("[stats] flushed to disk")
        except Exception as err:
            LOGGER.error("[stats] save failed: %s", err)

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
        reasoning_tokens: int = 0,
    ) -> None:
        """Update token usage statistics (fire-and-forget).

        This method returns immediately without blocking the caller.
        The actual processing happens asynchronously in the background.

        CRITICAL: This ensures conversation/ai_task services
        are NOT delayed by token tracking I/O operations.

        Args:
            service_type: "conversation", "ai_task"
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
                reasoning_tokens,
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
        reasoning_tokens: int = 0,
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
                    reasoning_tokens=reasoning_tokens,
                )

                # 2. Update aggregated stats
                if "aggregated" not in self._data["token_stats"]:
                    self._data["token_stats"]["aggregated"] = {}

                self._update_stats_dict(
                    self._data["token_stats"]["aggregated"],
                    usage,
                    model,
                    is_service_stats=False,
                    reasoning_tokens=reasoning_tokens,
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

                # 4. Mark as dirty (write-behind)
                self._dirty = True

            # 5. Notify listeners (outside lock to avoid deadlock)
            self._notify_listeners()

        except Exception as err:
            LOGGER.error(
                "[stats] update failed %s/%s: %s",
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
        reasoning_tokens: int = 0,
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
        reasoning = reasoning_tokens
        if reasoning == 0:
            reasoning = self._get_token_count(usage, "reasoning_tokens")
            if reasoning == 0:
                # Try nested structure (xAI / OpenAI standard)
                details = getattr(usage, "completion_tokens_details", None)
                if details:
                    reasoning = getattr(details, "reasoning_tokens", 0) or 0

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
    # PUBLIC API: STATISTICS, PRICING & COST ANALYSIS
    # =========================================================================

    async def get_service_stats(self, service_type: str) -> dict:
        """Get stats for a specific service.

        Args:
            service_type: "conversation" or "ai_task"

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

    async def get_pricing(self, model_name: str, price_type: str) -> float | None:
        """Get pricing data for a model.

        Args:
            model_name: Model name
            price_type: "input_price", "output_price", or "cached_input_price"

        Returns:
            Price per million tokens, or None if not set
        """
        async with self._lock:
            await self._ensure_loaded()
            return (
                self._data.get("pricing_data", {}).get(model_name, {}).get(price_type)
            )

    async def get_costs(self) -> dict:
        """Calculate and return raw cost data ready for presentation.

        Accumulates raw totals (units * raw_price) without conversion.
        Conversion to USD happens in the sensor layer.

        Returns:
            Dictionary with:
                - total_raw: float
                - tool_raw: float
                - cost_by_model: dict with per-model raw breakdown
                - tokens_by_model: dict with token counts
        """
        async with self._lock:
            await self._ensure_loaded()

            # Get aggregated stats
            stats = self._data.get("token_stats", {}).get("aggregated", {})
            tokens_by_model = stats.get("tokens_by_model", {})

            # Get pricing data
            pricing_data = self._data.get("pricing_data", {})

            # Calculate raw costs (Units * Raw API Price)
            total_raw = 0.0
            cost_breakdown = {}

            for model, counts in tokens_by_model.items():
                model_pricing = pricing_data.get(model, {})

                # Use RAW API VALUES directly
                input_price = model_pricing.get("input_price", 0.0)
                output_price = model_pricing.get("output_price", 0.0)
                cached_price = model_pricing.get("cached_input_price", 0.0)
                if cached_price == 0:
                    cached_price = input_price

                # Calculate raw values per type
                r_prompt = counts.get("prompt", 0) * input_price
                r_cached = counts.get("cached", 0) * cached_price
                r_completion = counts.get("completion", 0) * output_price

                model_total_raw = r_prompt + r_cached + r_completion
                total_raw += model_total_raw

                cost_breakdown[model] = {
                    "prompt_raw": r_prompt,
                    "cached_raw": r_cached,
                    "completion_raw": r_completion,
                    "total_raw": model_total_raw,
                    "tokens": counts.copy(),
                    "pricing_raw": {
                        "input": input_price,
                        "output": output_price,
                        "cached": cached_price,
                    },
                }

            # Calculate Tool Costs (Raw based on per-tool pricing)
            server_tools = self._data.get("server_tools", {})
            by_tool = server_tools.get("by_tool", {})
            tool_raw = 0.0
            tool_cost_breakdown = {}

            # Search price from API is also "per 1M calls" in raw units
            dynamic_search_price_raw = pricing_data.get("grok-2-1212", {}).get(
                "search_price", 0.0
            )

            for tool_name, invocations in by_tool.items():
                price_raw = 0.0
                if tool_name in ["web_search", "x_search"]:
                    price_raw = dynamic_search_price_raw or DEFAULT_TOOL_PRICE_RAW

                c_raw = invocations * price_raw
                tool_raw += c_raw
                tool_cost_breakdown[tool_name] = {
                    "invocations": invocations,
                    "price_raw": price_raw,
                    "total_raw": c_raw,
                }

            total_raw += tool_raw

            return {
                "total_raw": total_raw,
                "tool_raw": tool_raw,
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

    async def set_known_models(self, models: list[str]) -> None:
        """Overwrite the list of known models and persist to disk.

        Args:
            models: The exact list of model names to store.
        """
        async with self._lock:
            await self._ensure_loaded()
            self._data["known_models"] = sorted(list(set(models)))
            self._dirty = True
            await self.async_flush()
        self._notify_listeners()

    # =========================================================================
    # PUBLIC API: SETTERS (Admin Operations)
    # =========================================================================

    async def save_pricing_batch(
        self, pricing_updates: dict[str, dict[str, Any]]
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

    async def prune_pricing(self, supported_models: list[str]) -> None:
        """Remove pricing for models no longer supported/available.

        Args:
            supported_models: List of model names/aliases that should be kept.
        """
        async with self._lock:
            await self._ensure_loaded()
            if "pricing_data" not in self._data:
                return

            keep = set(supported_models)
            current_models = list(self._data["pricing_data"].keys())
            removed = []

            for model in current_models:
                if model not in keep:
                    del self._data["pricing_data"][model]
                    removed.append(model)

            if removed:
                self._dirty = True
                await self.async_flush()

        if removed:
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
            LOGGER.debug("[stats] reset (pricing and known models preserved)")

        self._notify_listeners()


# =============================================================================
# CONFIG HELPERS (for pricing display conversion)
# =============================================================================


def _get_sensors_config(entry: ConfigEntry) -> dict:
    """Get sensors subentry config."""
    for subentry in entry.subentries.values():
        if subentry.subentry_type == "sensors":
            config = dict(subentry.data)
            if hasattr(subentry, "options") and subentry.options:
                config.update(subentry.options)
            return config
    return {}


def get_pricing_conversion_factor(entry: ConfigEntry) -> float:
    """Resolve pricing conversion factor from sensors subentry."""
    config = _get_sensors_config(entry)
    return config.get(CONF_XAI_PRICING_CONVERSION_FACTOR)


def get_tokens_per_million(entry: ConfigEntry) -> int:
    """Resolve tokens per million from sensors subentry."""
    config = _get_sensors_config(entry)
    return config.get(CONF_TOKENS_PER_MILLION)
