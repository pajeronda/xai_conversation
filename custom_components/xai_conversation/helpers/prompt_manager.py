"""Prompt management for xAI conversation."""
from __future__ import annotations

from ..const import (
    CONF_ALLOW_SMART_HOME_CONTROL,
    CONF_ASSISTANT_NAME,
    CONF_PROMPT,
    CONF_PROMPT_PIPELINE,
    CONF_STORE_MESSAGES,
    PROMPT_CUSTOM_RULES,
    PROMPT_IDENTITY,
    PROMPT_MEMORY_CLIENTSIDE,
    PROMPT_MEMORY_SERVERSIDE,
    PROMPT_NO_CONTROL,
    PROMPT_OUTPUT_FORMAT,
    PROMPT_PIPELINE_DECISION_LOGIC,
    PROMPT_PIPELINE_EXAMPLES,
    PROMPT_ROLE_BASE,
    PROMPT_SMART_HOME_RECOGNITION,
    PROMPT_TOOLS_USAGE,
    RECOMMENDED_ASSISTANT_NAME,
)


class PromptManager:
    """Centralized prompt management for both pipeline and tools modes.

    This class handles:
    - Building system prompts based on mode (pipeline/tools)
    - Managing user custom instructions
    - Handling mode-specific configurations
    """

    def __init__(self, subentry_data: dict, mode: str):
        """Initialize PromptManager.

        Args:
            subentry_data: Configuration dictionary from subentry.data
            mode: Either "pipeline" or "tools"
        """
        if mode not in ("pipeline", "tools"):
            raise ValueError(f"Invalid mode '{mode}', must be 'pipeline' or 'tools'")

        self.mode = mode
        self.data = subentry_data

    def is_chat_only_mode(self) -> bool:
        """Check if home control is disabled for the current mode.

        Returns:
            True if chat-only mode (no home control), False otherwise.
        """
        allow_control = self.data.get(CONF_ALLOW_SMART_HOME_CONTROL, True)
        return not allow_control

    def get_user_custom_prompt(self) -> str:
        """Get user custom instructions based on mode.

        Returns:
            User's custom prompt
        """
        if self.mode == "pipeline":
            return self.data.get(CONF_PROMPT_PIPELINE, "")
        else:  # tools mode
            return self.data.get(CONF_PROMPT, "")

    def get_base_system_prompt(
        self,
        static_context: str = "",
        tool_definitions: str = ""
    ) -> str:
        """Get the base system prompt for this mode using modular building blocks.

        This method now uses the modular prompt composition system.
        Kept for backward compatibility with any code that might call it directly.

        Args:
            static_context: Static context for tools mode (optional)
            tool_definitions: Tool definitions for tools mode (optional)

        Returns:
            Complete system prompt assembled from building blocks
        """
        return self.build_system_prompt(
            static_context=static_context,
            tool_definitions=tool_definitions
        )

    def _build_pipeline_control_prompt_blocks(self, custom_rules: str) -> list[str]:
        """Build prompt blocks for pipeline mode with home control."""
        blocks = [PROMPT_SMART_HOME_RECOGNITION]
        if custom_rules:
            blocks.append(PROMPT_CUSTOM_RULES)
        blocks.append(PROMPT_PIPELINE_DECISION_LOGIC)
        blocks.append(PROMPT_PIPELINE_EXAMPLES)
        return blocks

    def _build_tools_control_prompt_blocks(
        self, static_context: str, tool_definitions: str
    ) -> list[str]:
        """Build prompt blocks for tools mode with home control."""
        tools_block = PROMPT_TOOLS_USAGE.format(
            static_context=static_context,
            tool_definitions=tool_definitions,
        )
        return [tools_block]

    def _build_chat_only_prompt_blocks(self) -> list[str]:
        """Build prompt blocks for chat-only mode."""
        return [PROMPT_NO_CONTROL]

    def build_system_prompt(
        self,
        static_context: str = "",
        tool_definitions: str = "",
    ) -> str:
        """Build system prompt dynamically using modular blocks.

        This method assembles the final system prompt from reusable building blocks
        defined in const.py, based on the current configuration.
        """
        assistant_name = self.data.get(CONF_ASSISTANT_NAME, RECOMMENDED_ASSISTANT_NAME)
        store_messages = self.data.get(CONF_STORE_MESSAGES, True)
        allow_control = not self.is_chat_only_mode()
        custom_rules = self.get_user_custom_prompt()

        blocks = []

        # 1. Identity and Role
        blocks.append(PROMPT_IDENTITY.format(assistant_name=assistant_name))
        blocks.append(PROMPT_ROLE_BASE)

        # 2. Memory Instructions
        if store_messages:
            blocks.append(PROMPT_MEMORY_SERVERSIDE)
        elif self.mode == "tools":
            # Only add client-side memory block in tools mode when server memory is off
            blocks.append(PROMPT_MEMORY_CLIENTSIDE)

        # 3. Mode-Specific Blocks (delegated to helper methods)
        if allow_control:
            if self.mode == "pipeline":
                blocks.extend(self._build_pipeline_control_prompt_blocks(custom_rules))
            else:  # tools mode
                blocks.extend(
                    self._build_tools_control_prompt_blocks(
                        static_context, tool_definitions
                    )
                )
        else:
            blocks.extend(self._build_chat_only_prompt_blocks())

        # 4. User Instructions (only appended if control is enabled)
        if custom_rules and allow_control:
            blocks.append(f"# --- USER INSTRUCTIONS (appended to default) ---\n{custom_rules}")

        # 5. Output Format
        blocks.append(PROMPT_OUTPUT_FORMAT)

        return "\n\n".join(blocks)

    def build_base_prompt_with_user_instructions(
        self,
        static_context: str = "",
        tool_definitions: str = ""
    ) -> str:
        """Build base system prompt + user custom instructions using modular system.

        Uses the new modular prompt composition system (build_system_prompt)
        which assembles prompt from building blocks based on configuration.

        For backward compatibility, falls back to get_base_system_prompt() if needed.

        Args:
            static_context: Static context for tools mode (optional)
            tool_definitions: Tool definitions for tools mode (optional)

        Returns:
            Complete system prompt with user instructions integrated
        """
        # Use new modular system
        return self.build_system_prompt(
            static_context=static_context,
            tool_definitions=tool_definitions
        )
