"""AI Task integration for xAI."""

from __future__ import annotations

# Standard library imports
import json
import time
from json import JSONDecodeError
from typing import TYPE_CHECKING

# Home Assistant imports
from homeassistant.config_entries import ConfigEntry

# Local application imports
from .__init__ import (
    HA_AddConfigEntryEntitiesCallback,
    HA_HomeAssistant,
    XAI_SDK_AVAILABLE,
    ha_ai_task,
    ha_conversation,
    ha_json_loads,
    xai_user,
    xai_system,
)
from .const import (
    CONF_PROMPT,
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
        # Note: token sensors updated inside _async_process_chat_log_stateless with service_type="ai_task"
        await self._async_process_chat_log_stateless(chat_log, attachments=task.attachments, service_type="ai_task")

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
        import time
        from .exceptions import raise_validation_error, handle_api_error

        if not XAI_SDK_AVAILABLE:
            raise_generic_error("xAI SDK not available")

        # Validate prompt is not empty
        if not task.instructions or not task.instructions.strip():
            raise_validation_error("Image generation prompt cannot be empty")

        start_time = time.time()
        client = self._create_client()

        try:
            # Use base64 format to get raw bytes directly from xAI
            # Note: grok-2-image does not support image-to-image (attachments ignored)
            response = await self.hass.async_add_executor_job(
                lambda: client.image.sample(
                    model="grok-2-image",
                    prompt=task.instructions,
                    image_format="base64",
                )
            )

            # Validate response structure
            if not hasattr(response, "image") or not response.image:
                raise_generic_error("xAI API returned invalid image response")

            if not isinstance(response.image, bytes):
                raise_generic_error(f"Expected bytes, got {type(response.image).__name__}")

            # Extract image data and optional metadata
            image_bytes = response.image
            revised_prompt = getattr(response, "prompt", None)

            elapsed = time.time() - start_time
            LOGGER.info(
                "Image generated successfully: size=%d bytes, prompt_length=%d, elapsed=%.2fs",
                len(image_bytes),
                len(task.instructions),
                elapsed
            )

            return ha_ai_task.GenImageTaskResult(
                image_data=image_bytes,
                conversation_id=chat_log.conversation_id,
                mime_type="image/jpeg",  # xAI generates jpg format
                model="grok-2-image",
                revised_prompt=revised_prompt,
            )

        except Exception as err:
            handle_api_error(err, start_time, "image generation")
