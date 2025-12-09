"""Prompt management for xAI conversation."""

from __future__ import annotations

from ..const import (
    CONF_ALLOW_SMART_HOME_CONTROL,
    CONF_ASSISTANT_NAME,
    CONF_PROMPT,
    CONF_PROMPT_CODE,
    CONF_PROMPT_PIPELINE,
    CONF_STORE_MESSAGES,
    DEFAULT_CODE_FAST_ASSISTANT_NAME,
    PROMPT_CODE_OUTPUT_FORMAT,
    PROMPT_CODE_ROLE,
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
            mode: Either "pipeline", "tools", or "code"
        """
        if mode not in ("pipeline", "tools", "code"):
            raise ValueError(
                f"Invalid mode '{mode}', must be 'pipeline', 'tools', or 'code'"
            )

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
        elif self.mode == "code":
            return self.data.get(CONF_PROMPT_CODE, "")
        else:  # tools mode
            return self.data.get(CONF_PROMPT, "")

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

    def _build_code_prompt_blocks(self, custom_instructions: str) -> list[str]:
        """Build prompt blocks for code mode (Grok Code Fast service).

        Args:
            custom_instructions: User's custom instructions for code generation

        Returns:
            List of prompt blocks specific to code mode
        """
        blocks = [PROMPT_CODE_ROLE]
        if custom_instructions:
            blocks.append(
                f"# --- USER INSTRUCTIONS (appended to default) ---\n{custom_instructions}"
            )
        blocks.append(PROMPT_CODE_OUTPUT_FORMAT)
        return blocks

    def build_system_prompt(
        self,
        static_context: str = "",
        tool_definitions: str = "",
    ) -> str:
        """Build system prompt dynamically using modular blocks.

        This method assembles the final system prompt from reusable building blocks
        defined in const.py, based on the current configuration.

        Composition order:
        - Code mode: Identity → Memory → Code Role → [User Instructions] → Code Output Format
        - Pipeline/Tools: Identity → Role Base → Memory → Mode Blocks → [User Instructions] → Output Format
        """
        # Get assistant name with mode-specific default
        if self.mode == "code":
            default_name = self.data.get(
                CONF_ASSISTANT_NAME, DEFAULT_CODE_FAST_ASSISTANT_NAME
            )
        else:
            default_name = self.data.get(
                CONF_ASSISTANT_NAME, RECOMMENDED_ASSISTANT_NAME
            )
        assistant_name = default_name

        store_messages = self.data.get(CONF_STORE_MESSAGES, True)
        allow_control = not self.is_chat_only_mode()
        custom_rules = self.get_user_custom_prompt()

        blocks = []

        # 1. Identity (always present)
        blocks.append(PROMPT_IDENTITY.format(assistant_name=assistant_name))

        # 2. Role Base (only for conversation modes: pipeline/tools)
        if self.mode in ("pipeline", "tools"):
            blocks.append(PROMPT_ROLE_BASE)

        # 3. Memory Instructions (all modes support memory)
        if store_messages:
            blocks.append(PROMPT_MEMORY_SERVERSIDE)
        elif self.mode in ("tools", "code"):
            # Only add client-side memory block in tools/code modes when server memory is off
            blocks.append(PROMPT_MEMORY_CLIENTSIDE)

        # 4. Mode-Specific Blocks (delegated to helper methods)
        if self.mode == "code":
            # Code mode: special handling with code-specific blocks
            blocks.extend(self._build_code_prompt_blocks(custom_rules))
        elif allow_control:
            if self.mode == "pipeline":
                blocks.extend(self._build_pipeline_control_prompt_blocks(custom_rules))
            else:  # tools mode
                blocks.extend(
                    self._build_tools_control_prompt_blocks(
                        static_context, tool_definitions
                    )
                )
        else:
            # Chat-only mode (conversation with control disabled)
            blocks.extend(self._build_chat_only_prompt_blocks())

        # 5. User Instructions (only for conversation modes with control enabled)
        if custom_rules and allow_control and self.mode in ("pipeline", "tools"):
            blocks.append(
                f"# --- USER INSTRUCTIONS (appended to default) ---\n{custom_rules}"
            )

        # 6. Output Format (only for conversation modes, code mode has its own)
        if self.mode in ("pipeline", "tools"):
            blocks.append(PROMPT_OUTPUT_FORMAT)

        return "\n\n".join(blocks).strip()

    def build_base_prompt_with_user_instructions(
        self, static_context: str = "", tool_definitions: str = ""
    ) -> str:
        """Build base system prompt + user custom instructions using modular system.

        Uses the modular prompt composition system (build_system_prompt)
        which assembles prompt from building blocks based on configuration.

        Args:
            static_context: Static context for tools mode (optional)
            tool_definitions: Tool definitions for tools mode (optional)

        Returns:
            Complete system prompt with user instructions integrated
        """
        return self.build_system_prompt(
            static_context=static_context, tool_definitions=tool_definitions
        )
