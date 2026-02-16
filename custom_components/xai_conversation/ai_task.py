"""AI Task integration for xAI."""

from __future__ import annotations

import asyncio
from json import JSONDecodeError
from typing import TYPE_CHECKING

# Home Assistant imports
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant as HA_HomeAssistant
from homeassistant.helpers.entity_platform import (
    AddEntitiesCallback as HA_AddConfigEntryEntitiesCallback,
)
from homeassistant.components import ai_task as ha_ai_task
from homeassistant.components import conversation as ha_conversation

# Local application imports
from .const import (
    CONF_IMAGE_MODEL,
    DEFAULT_IMAGE_MIME_TYPE,
    LOGGER,
    RECOMMENDED_IMAGE_MODEL,
    XAIConfigEntry,
)
from .helpers import (
    parse_with_strategies,
    parse_strict_json,
    parse_json_fenced,
    parse_balanced_braces,
    LogTimeServices,
    ChatOptions,
    convert_ha_schema_to_xai,
    async_prepare_attachments,
)
from .entity import XAIBaseLLMEntity
from .exceptions import raise_generic_error, raise_validation_error

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigSubentry

# No local _read_file_sync or base64 needed here anymore (moved to entity.py)


async def async_setup_entry(
    hass: HA_HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: HA_AddConfigEntryEntitiesCallback,
) -> None:
    """Set up AI Task entities."""
    for subentry in config_entry.subentries.values():
        # Handle standard AI Task entities (stateless one-shot)
        if subentry.subentry_type == "ai_task":
            async_add_entities(
                [XAITaskEntity(config_entry, subentry)],
                config_subentry_id=subentry.subentry_id,
            )


class XAITaskEntity(
    ha_ai_task.AITaskEntity,
    XAIBaseLLMEntity,
):
    """Entity for executing one-shot AI tasks (Data Generation, Image Generation)."""

    def __init__(self, entry: XAIConfigEntry, subentry: ConfigSubentry) -> None:
        """Initialize the entity."""
        super().__init__(entry, subentry)

        # Configure supported features
        self._attr_supported_features = (
            ha_ai_task.AITaskEntityFeature.GENERATE_DATA
            | ha_ai_task.AITaskEntityFeature.GENERATE_IMAGE
            | ha_ai_task.AITaskEntityFeature.SUPPORT_ATTACHMENTS
        )

    async def async_added_to_hass(self) -> None:
        """Ensure XAIBaseLLMEntity initialization logic runs."""
        # Call XAIBaseLLMEntity's async_added_to_hass directly to ensure gateway is initialized
        # even if ha_ai_task.AITaskEntity doesn't call super() correctly.
        # Note: XAIBaseLLMEntity calls super().async_added_to_hass() internally.
        await XAIBaseLLMEntity.async_added_to_hass(self)

    # _prepare_attachments removed, moved to XAIBaseLLMEntity

    async def _async_generate_data(
        self,
        task: ha_ai_task.GenDataTask,
        chat_log: ha_conversation.ChatLog,
    ) -> ha_ai_task.GenDataTaskResult:
        """Execute a data generation task.

        Performs a stateless, one-shot inference to generate text or structured data.
        Supports optional attachments for multimodal input.
        """
        # Convert HA selector structure to JSON Schema for the LLM
        json_schema = (
            convert_ha_schema_to_xai(task.structure, task.name)
            if task.structure
            else None
        )

        # Use the unified stateless logic from XAIBaseLLMEntity
        await self._async_handle_stateless_task(
            chat_log,
            task_name=task.name,
            task_structure=json_schema,
            extra_attachments=task.attachments,
            options=ChatOptions(response_format=json_schema) if json_schema else None,
        )

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
            ],
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
        """Execute an image generation task."""
        prompt = task.instructions
        if not prompt or not prompt.strip():
            raise_validation_error("Image generation prompt cannot be empty")

        # Read model directly from entity config
        config = self.get_config_dict()
        model = config.get(CONF_IMAGE_MODEL, RECOMMENDED_IMAGE_MODEL)

        # Process attachments for image editing (img2img)
        image_url = None
        attachments = getattr(task, "attachments", None)
        final_instructions = task.instructions
        if attachments:
            prepared = await async_prepare_attachments(self.hass, attachments)
            if prepared.uris:
                image_url = prepared.uris[0]
            if prepared.has_skipped:
                skipped_list = ", ".join(prepared.skipped)
                final_instructions += f"\n\n[System Note: The following files were skipped due to unsupported formats: {skipped_list}]"

        context = {"mode": "image", "model": model, "prompt_length": len(prompt)}
        async with LogTimeServices(LOGGER, "ai_task", context) as timer:
            response = await self.gateway.async_generate_image(
                prompt=final_instructions,
                model=model,
                image_url=image_url,
                options=ChatOptions(timer=timer),
                entity=self,
            )

            try:
                image_bytes = response.image
                # If image_bytes is a coroutine, await it (SDK variation)
                if asyncio.iscoroutine(image_bytes):
                    image_bytes = await image_bytes
            except AttributeError:
                raise_generic_error("xAI API returned invalid image response")

            if not image_bytes:
                raise_generic_error("xAI API returned empty image data")

            revised_prompt = getattr(response, "prompt", None)
            timer.context_info["image_size_bytes"] = len(image_bytes)

            return ha_ai_task.GenImageTaskResult(
                image_data=image_bytes,
                conversation_id=chat_log.conversation_id,
                mime_type=DEFAULT_IMAGE_MIME_TYPE,
                model=model,
                revised_prompt=revised_prompt,
            )
