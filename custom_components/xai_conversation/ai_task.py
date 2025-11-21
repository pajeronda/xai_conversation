"""AI Task integration for xAI."""

from __future__ import annotations

# Standard library imports
from json import JSONDecodeError
from typing import TYPE_CHECKING

# Local application imports
from .__init__ import (
    ConfigEntry,
    HA_AddConfigEntryEntitiesCallback,
    HA_HomeAssistant,
    ha_ai_task,
    ha_conversation,
)
from .const import (
    LOGGER,
    SUBENTRY_TYPE_AI_TASK,
)
from .helpers import (
    parse_with_strategies,
    parse_strict_json,
    parse_json_fenced,
    parse_balanced_braces,
)
from .entity import XAIBaseLLMEntity
from .exceptions import raise_generic_error

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigSubentry

    from . import XAIConfigEntry


async def async_setup_entry(
    hass: HA_HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: HA_AddConfigEntryEntitiesCallback,
) -> None:
    """Set up AI Task entities."""
    for subentry in config_entry.subentries.values():
        # Handle standard AI Task entities (stateless one-shot)
        if subentry.subentry_type == SUBENTRY_TYPE_AI_TASK:
            async_add_entities(
                [XAITaskEntity(config_entry, subentry)],
                config_subentry_id=subentry.subentry_id,
            )
        # Note: code_task is now handled via grok_code_fast service (no entity needed)


class XAITaskEntity(
    ha_ai_task.AITaskEntity,
    XAIBaseLLMEntity,
):
    """xAI AI Task entity."""

    def __init__(self, entry: XAIConfigEntry, subentry: ConfigSubentry) -> None:
        """Initialize the entity."""
        super().__init__(entry, subentry)

        # Configure supported features
        self._attr_supported_features = (
            ha_ai_task.AITaskEntityFeature.GENERATE_DATA
            | ha_ai_task.AITaskEntityFeature.GENERATE_IMAGE
            | ha_ai_task.AITaskEntityFeature.SUPPORT_ATTACHMENTS
        )

    async def _async_generate_data(
        self,
        task: ha_ai_task.GenDataTask,
        chat_log: ha_conversation.ChatLog,
    ) -> ha_ai_task.GenDataTaskResult:
        """Handle a generate data task.

        AI Task is stateless - performs one-shot data generation without memory.
        For structured outputs, provide a structure parameter in the task.
        Supports attachments (images/media) via task.attachments.
        """
        # Process chat log in stateless mode with optional attachments
        # task.attachments contains list[Attachment] with path, mime_type, media_content_id
        # Note: token sensors updated inside gateway.execute_stateless_chat with service_type="ai_task"
        await self.gateway.execute_stateless_chat(chat_log, attachments=task.attachments, service_type="ai_task")

        if not isinstance(chat_log.content[-1], ha_conversation.AssistantContent):
            raise_generic_error("Last content in chat log is not an AssistantContent")

        text = chat_log.content[-1].content or ""

        if not task.structure:
            return ha_ai_task.GenDataTaskResult(
                conversation_id=chat_log.conversation_id,
                data=text,
            )
        # Use centralized parsing strategies
        data = parse_with_strategies(
            text,
            strategies=[
                parse_strict_json,
                parse_json_fenced,
                parse_balanced_braces,
            ]
        )

        try:
            if data is None:
                raise JSONDecodeError("Could not extract JSON", text, 0)
        except JSONDecodeError as err:
            LOGGER.error(
                "Failed to parse JSON response: %s. Response: %s",
                err,
                text,
            )
            raise_generic_error("Error with xAI structured response")

        return ha_ai_task.GenDataTaskResult(
            conversation_id=chat_log.conversation_id,
            data=data,
        )

    async def _async_generate_image(
        self,
        task: ha_ai_task.GenImageTask,
        chat_log: ha_conversation.ChatLog,
    ) -> ha_ai_task.GenImageTaskResult:
        """Handle an image generation task using grok-2-image model.

        Args:
            task: Image generation task containing prompt and optional attachments
            chat_log: Conversation context for tracking

        Returns:
            GenImageTaskResult with image data and metadata

        Raises:
            Various exceptions via handle_api_error for auth, network, API errors
        """
        # Use gateway to generate image (uses RECOMMENDED_IMAGE_MODEL from const.py)
        result = await self.gateway.generate_image(
            prompt=task.instructions,
            chat_log=chat_log
        )

        # Convert gateway result to GenImageTaskResult
        return ha_ai_task.GenImageTaskResult(
            image_data=result["image_data"],
            conversation_id=chat_log.conversation_id,
            mime_type=result["mime_type"],
            model=result["model"],
            revised_prompt=result.get("revised_prompt"),
        )
