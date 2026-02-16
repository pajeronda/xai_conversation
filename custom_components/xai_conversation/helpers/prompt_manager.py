"""Prompt management for xAI conversation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..const import (
    CHAT_MODE_CHATONLY,
    CHAT_MODE_PIPELINE,
    CHAT_MODE_TOOLS,
    PROMPT_SEARCH_USAGE,
    CONF_AI_TASK_PROMPT,
    CONF_ALLOW_SMART_HOME_CONTROL,
    CONF_ASSISTANT_NAME,
    CONF_STORE_MESSAGES,
    CONF_ZDR,
    CONF_LIVE_SEARCH,
    CONF_PROMPT_PIPELINE,
    CONF_PROMPT_TOOLS,
    GROK_AI_TASK_PROMPT,
    LIVE_SEARCH_OFF,
    PROMPT_IDENTITY,
    PROMPT_MEMORY_SERVERSIDE,
    PROMPT_MEMORY_ZDR,
    PROMPT_MEMORY_CLIENTSIDE,
    PROMPT_ROLE_BASE,
    PROMPT_NO_CONTROL,
    PROMPT_SMART_HOME_RECOGNITION,
    PROMPT_CUSTOM_RULES,
    PROMPT_PIPELINE_DECISION_LOGIC,
    PROMPT_PIPELINE_EXAMPLES,
    PROMPT_PIPELINE_INTENT_TOPICS,
    PROMPT_MODE_TOOLS,
    PROMPT_INTENT_EXECUTION,
    PROMPT_OUTPUT_FORMAT,
    CONF_VISION_PROMPT,
    VISION_ANALYSIS_PROMPT,
)
from .utils import hash_text

if TYPE_CHECKING:
    pass


class PromptManager:
    """Autonomous prompt management - builds and caches prompts."""

    def __init__(self):
        """Initialize shared prompt manager.

        Configuration and context are now passed per request to ensure pure,
        stateless operation and integration-level sharing.
        """
        self._cache: dict[str, str] = {}
        self._hash_cache: dict[str, str] = {}

    # ==========================================================================
    # PUBLIC API - Gateway chiede solo MODE
    # ==========================================================================

    def get_prompt(self, mode: str, config: dict, orchestrator=None) -> str:
        """Get prompt for mode. Cached.

        Args:
            mode: "pipeline", "tools", "chatonly", "ai_task", "vision"
            config: Configuration dict to use.
            orchestrator: Optional ToolOrchestrator for static context.
        """
        cache_key = self._get_cache_key(mode, config, orchestrator)
        if cache_key in self._cache:
            return self._cache[cache_key]

        prompt = self._build_prompt(mode, config, orchestrator)
        self._cache[cache_key] = prompt
        return prompt

    def get_prompt_hash(self, mode: str, config: dict, orchestrator=None) -> str:
        """Get hash of prompt for memory keys. Cached."""
        cache_key = self._get_cache_key(mode, config, orchestrator)
        if cache_key in self._hash_cache:
            return self._hash_cache[cache_key]

        prompt = self.get_prompt(mode, config, orchestrator)
        prompt_hash = hash_text(prompt)
        self._hash_cache[cache_key] = prompt_hash
        return prompt_hash

    # ==========================================================================
    # INTERNAL - Build prompts autonomously
    # ==========================================================================

    def _get_cache_key(self, mode: str, config: dict, orchestrator=None) -> str:
        """Generate cache key based on mode and variable components."""
        components = [
            mode,
            config.get(CONF_ASSISTANT_NAME, ""),
            str(config.get(CONF_STORE_MESSAGES, False)),
            str(config.get(CONF_ALLOW_SMART_HOME_CONTROL, True)),
            str(config.get(CONF_LIVE_SEARCH, LIVE_SEARCH_OFF) != LIVE_SEARCH_OFF),
            config.get(CONF_AI_TASK_PROMPT, ""),
            self._get_user_instructions(mode, config),
        ]

        # For tools/pipeline mode, include intents context (and CSV for tools)
        if mode in (CHAT_MODE_TOOLS, CHAT_MODE_PIPELINE) and orchestrator:
            intents_context = orchestrator.get_custom_intents_context()
            components.append(intents_context)

        if mode == CHAT_MODE_TOOLS and orchestrator:
            csv_content = orchestrator.get_static_context_csv()
            components.append(csv_content)

        return f"{mode}:{hash_text('|'.join(components))[:12]}"

    def _build_prompt(self, mode: str, config: dict, orchestrator=None) -> str:
        """Build prompt for mode."""
        if mode == "ai_task":
            return config.get(CONF_AI_TASK_PROMPT) or GROK_AI_TASK_PROMPT

        if mode == "vision":
            return config.get(CONF_VISION_PROMPT) or VISION_ANALYSIS_PROMPT

        if mode in (CHAT_MODE_TOOLS, CHAT_MODE_PIPELINE, CHAT_MODE_CHATONLY):
            return self._build_conversation_prompt(mode, config, orchestrator)

        return ""

    def _build_conversation_prompt(
        self, mode: str, config: dict, orchestrator=None
    ) -> str:
        """Build prompt for tools/pipeline/chatonly modes."""
        blocks = []
        instructions = self._get_user_instructions(mode, config)

        # --- A. Identity ---
        assistant_name = config.get(CONF_ASSISTANT_NAME)
        if assistant_name:
            blocks.append(PROMPT_IDENTITY.format(assistant_name=assistant_name))

        # --- B. Memory Management ---
        if config.get(CONF_STORE_MESSAGES):
            blocks.append(PROMPT_MEMORY_SERVERSIDE)
        elif config.get(CONF_ZDR):
            blocks.append(PROMPT_MEMORY_ZDR)
        else:
            blocks.append(PROMPT_MEMORY_CLIENTSIDE)

        # --- C. Base Role ---
        blocks.append(PROMPT_ROLE_BASE)

        # --- D. Mode-specific content ---
        allow_control = config.get(CONF_ALLOW_SMART_HOME_CONTROL, True)

        if not allow_control:
            # Universal block when control is disabled across all modes
            blocks.append(PROMPT_NO_CONTROL)
        elif mode == CHAT_MODE_PIPELINE:
            blocks.append(PROMPT_SMART_HOME_RECOGNITION)
            custom_intents = (
                orchestrator.get_custom_intents_context() if orchestrator else ""
            )
            if custom_intents:
                blocks.append(
                    PROMPT_PIPELINE_INTENT_TOPICS.format(
                        custom_intents=custom_intents
                    ).strip()
                )
            if instructions:
                blocks.append(PROMPT_CUSTOM_RULES)
            blocks.extend(
                [
                    PROMPT_PIPELINE_DECISION_LOGIC.strip(),
                    PROMPT_PIPELINE_EXAMPLES.strip(),
                ]
            )
        elif mode == CHAT_MODE_TOOLS:
            static_context = self._get_static_context(orchestrator)
            blocks.append(
                PROMPT_MODE_TOOLS.format(static_context=static_context).strip()
            )
            custom_intents = (
                orchestrator.get_custom_intents_context() if orchestrator else ""
            )
            if custom_intents:
                blocks.append(
                    PROMPT_INTENT_EXECUTION.format(
                        custom_intents=custom_intents
                    ).strip()
                )

        # --- E. User Instructions ---
        if instructions:
            blocks.append(f"# --- USER INSTRUCTIONS ---\n{instructions}")

        # --- F. Search Usage Rule ---
        if config.get(CONF_LIVE_SEARCH, LIVE_SEARCH_OFF) != LIVE_SEARCH_OFF:
            blocks.append(PROMPT_SEARCH_USAGE)

        # --- G. Output Formatting ---
        blocks.append(PROMPT_OUTPUT_FORMAT)

        return "\n\n".join(blocks).strip()

    def _get_user_instructions(self, mode: str, config: dict) -> str:
        """Get user instructions from config."""
        if mode == CHAT_MODE_PIPELINE:
            return config.get(CONF_PROMPT_PIPELINE, "").strip()
        if mode == CHAT_MODE_TOOLS:
            return config.get(CONF_PROMPT_TOOLS, "").strip()
        return ""

    def _get_static_context(self, orchestrator=None) -> str:
        """Get static context CSV from ToolOrchestrator."""
        if orchestrator:
            return orchestrator.get_static_context_csv()
        return ""
