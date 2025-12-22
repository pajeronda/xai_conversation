"Prompt management for xAI conversation."

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..const import (
    CONF_ALLOW_SMART_HOME_CONTROL,
    CONF_ASSISTANT_NAME,
    CONF_PROMPT,
    CONF_PROMPT_TOOLS,
    CONF_PROMPT_CODE,
    CONF_PROMPT_PIPELINE,
    CONF_STORE_MESSAGES,
    CONF_USE_EXTENDED_TOOLS,
    DEFAULT_GROK_CODE_FAST_NAME,
    GROK_AI_TASK_PROMPT,
    VISION_ANALYSIS_PROMPT,
    PROMPT_CODE_OUTPUT_FORMAT,
    PROMPT_CODE_ROLE,
    PROMPT_CUSTOM_RULES,
    PROMPT_IDENTITY,
    PROMPT_MEMORY_CLIENTSIDE,
    PROMPT_MEMORY_SERVERSIDE,
    PROMPT_MODE_TOOLS,
    PROMPT_NO_CONTROL,
    PROMPT_OUTPUT_FORMAT,
    PROMPT_PIPELINE_DECISION_LOGIC,
    PROMPT_PIPELINE_EXAMPLES,
    PROMPT_ROLE_BASE,
    PROMPT_SMART_HOME_RECOGNITION,
    RECOMMENDED_ASSISTANT_NAME,
)
from .utils import hash_text, build_session_context_info

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant


def build_system_prompt(
    entity: Any | None = None,
    mode: str = "tools",
    hass: HomeAssistant | None = None,
    config: dict | None = None,
) -> str:
    """Helper function to build a system prompt with a single call.
    Can be called as build_system_prompt(entity, mode=mode).
    """
    # 1. Resolve Hass
    active_hass = hass or (entity.hass if entity else None)
    if not active_hass:
        raise ValueError("HomeAssistant instance required to build prompt")

    # 2. Resolve Config
    active_config = config
    if not active_config and entity and hasattr(entity, "gateway") and entity.gateway:
        active_config = entity.gateway.get_service_config(
            "conversation", entity.subentry.subentry_id
        )

    # 3. Build Prompt
    manager = PromptManager(config=active_config or {}, mode=mode)
    return manager.build_system_prompt(active_hass, entity)


class PromptManager:
    """Centralized, stateless prompt management."""

    def __init__(self, config: dict, mode: str):
        """Initialize PromptManager.

        Args:
            config: Configuration dictionary (merged data and options).
            mode: "pipeline", "tools", or "code".
        """
        self.config = config
        self.mode = mode

        if self.mode not in ("pipeline", "tools", "code", "ai_task", "vision"):
            raise ValueError(
                f"Invalid mode '{self.mode}', must be 'pipeline', 'tools', 'code', 'ai_task', or 'vision'"
            )

    def build_system_prompt(
        self,
        hass: HomeAssistant,
        entity: Any | None = None,
    ) -> str:
        """Build the complete system prompt including dynamic context.

        Args:
            hass: Home Assistant instance (for time/location context).
            entity: The entity instance, used to derive the orchestrator in tools mode.
        """
        # 1. Get dynamic context if needed
        static_context = ""

        if self.mode == "tools" and entity and hasattr(entity, "_tools_processor"):
            orchestrator = entity._tools_processor._orchestrator
            static_context = orchestrator.get_static_context_csv()

        # 2. Build base prompt content
        base_prompt = self._build_base_prompt_content(static_context)

        # 3. Append session context (time, location) - always dynamic
        session_context = build_session_context_info(hass)

        return f"{base_prompt}\n\n{session_context}".strip()

    def build_base_prompt_with_user_instructions(self) -> str:
        """Build the base system prompt for hash calculation.

        Excludes dynamic context (time, location) and tool definitions
        to ensure a stable hash for memory keys.
        """
        return self._build_base_prompt_content(static_context="")

    @staticmethod
    def get_stable_hash_from_prompt(prompt: str) -> str:
        """Get the hash of a given prompt string."""
        return hash_text(prompt)

    def get_stable_hash(self) -> str:
        """Get the hash of the stable base prompt."""
        return self.get_stable_hash_from_prompt(
            self.build_base_prompt_with_user_instructions()
        )

    def _build_base_prompt_content(
        self,
        static_context: str,
    ) -> str:
        """Build the core logical blocks of the prompt."""
        # AI Task mode: return dedicated prompt only (no dynamic context)
        if self.mode == "ai_task":
            return GROK_AI_TASK_PROMPT

        # Vision mode: return dedicated vision prompt
        if self.mode == "vision":
            return VISION_ANALYSIS_PROMPT

        blocks = []

        # --- A. Identity ---
        name_key = CONF_ASSISTANT_NAME
        default_name = (
            DEFAULT_GROK_CODE_FAST_NAME
            if self.mode == "code"
            else RECOMMENDED_ASSISTANT_NAME
        )
        assistant_name = self.config.get(name_key, default_name)
        blocks.append(PROMPT_IDENTITY.format(assistant_name=assistant_name))

        # --- B. Base Role (Conversation modes only) ---
        if self.mode in ("pipeline", "tools"):
            blocks.append(PROMPT_ROLE_BASE)

        # --- C. Memory Management ---
        store_messages = self.config.get(CONF_STORE_MESSAGES, True)
        if store_messages:
            blocks.append(PROMPT_MEMORY_SERVERSIDE)
        elif self.mode in ("tools", "code"):
            blocks.append(PROMPT_MEMORY_CLIENTSIDE)

        # --- D. Mode-Specific Logic ---
        allow_control = self.config.get(CONF_ALLOW_SMART_HOME_CONTROL, True)
        user_instructions = self._get_user_instructions()

        if self.mode == "code":
            blocks.append(PROMPT_CODE_ROLE)
            if user_instructions:
                blocks.append(self._format_user_instructions(user_instructions))
            blocks.append(PROMPT_CODE_OUTPUT_FORMAT)

        elif not allow_control:
            blocks.append(PROMPT_NO_CONTROL)
            if user_instructions:
                blocks.append(self._format_user_instructions(user_instructions))

        elif self.mode == "pipeline":
            blocks.append(PROMPT_SMART_HOME_RECOGNITION)
            if user_instructions:
                blocks.append(PROMPT_CUSTOM_RULES)
            blocks.append(PROMPT_PIPELINE_DECISION_LOGIC)
            blocks.append(PROMPT_PIPELINE_EXAMPLES)
            # Legacy behavior: append user instructions at the end for pipeline too
            if user_instructions:
                blocks.append(self._format_user_instructions(user_instructions))
            blocks.append(PROMPT_OUTPUT_FORMAT)

        elif self.mode == "tools":
            blocks.append(
                PROMPT_MODE_TOOLS.format(
                    static_context=static_context,
                )
            )
            if user_instructions:
                blocks.append(self._format_user_instructions(user_instructions))
            blocks.append(PROMPT_OUTPUT_FORMAT)

        return "\n\n".join(blocks).strip()

    def _get_user_instructions(self) -> str:
        """Retrieve mode-specific user instructions."""
        if self.mode == "pipeline":
            return self.config.get(CONF_PROMPT_PIPELINE, "")
        elif self.mode == "code":
            return self.config.get(CONF_PROMPT_CODE, "")
        # Tools mode (fallback to old CONF_PROMPT for backward compatibility)
        return self.config.get(CONF_PROMPT_TOOLS, self.config.get(CONF_PROMPT, ""))

    def _format_user_instructions(self, instructions: str) -> str:
        """Format user instructions with a standard header."""
        return f"# --- USER INSTRUCTIONS (appended to default) ---\n{instructions}"
