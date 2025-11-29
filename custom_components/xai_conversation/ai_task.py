"""AI Task integration for xAI."""

from __future__ import annotations

import base64

# Standard library imports
from json import JSONDecodeError
from typing import TYPE_CHECKING, Any

# Local application imports
from .__init__ import ConfigEntry
from homeassistant.core import HomeAssistant as HA_HomeAssistant
from homeassistant.helpers.entity_platform import (
    AddEntitiesCallback as HA_AddConfigEntryEntitiesCallback,
)
from homeassistant.components import ai_task as ha_ai_task
from homeassistant.components import conversation as ha_conversation

from .const import (
    LOGGER,
    SUBENTRY_TYPE_AI_TASK,
    CONF_IMAGE_MODEL,
    CONF_VISION_MODEL,
    RECOMMENDED_VISION_MODEL,
    RECOMMENDED_IMAGE_MODEL,
    CONF_VISION_PROMPT,
    VISION_ANALYSIS_PROMPT,
    CONF_MAX_TOKENS,
    CONF_TEMPERATURE,
)
from .helpers import (
    parse_with_strategies,
    parse_strict_json,
    parse_json_fenced,
    parse_balanced_braces,
)
from .helpers.xai_gateway import XAIGateway
from .helpers.log_time_services import LogTimeServices
from .helpers.response import save_response_metadata
from .entity import XAIBaseLLMEntity
from .exceptions import raise_generic_error, raise_validation_error

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

    async def _prepare_attachments(
        self, attachments: list | None, images: list[str] | None = None
    ) -> list[Any]:
        """Prepare image messages from attachments and image paths.

        Handles file I/O and base64 conversion.

        Args:
            attachments: List of HA Attachment objects (from AI Task)
            images: List of image paths or URLs (from service call)

        Returns:
            List of xAI image message objects ready for the gateway.
        """
        messages = []

        # Process local file paths / URLs
        if images:
            for img in images:
                if isinstance(img, str):
                    if img.startswith(("http://", "https://")):
                        messages.append(XAIGateway.img_msg(img))
                    else:
                        try:

                            def _read_file_sync(p):
                                with open(p, "rb") as f:
                                    return f.read()

                            image_bytes = await self.hass.async_add_executor_job(
                                _read_file_sync, img
                            )
                            base64_image = base64.b64encode(image_bytes).decode("utf-8")
                            # Simple mime type detection
                            mime_type = "image/jpeg"
                            if img.lower().endswith(".png"):
                                mime_type = "image/png"
                            elif img.lower().endswith(".webp"):
                                mime_type = "image/webp"

                            data_uri = f"data:{mime_type};base64,{base64_image}"
                            messages.append(XAIGateway.img_msg(data_uri))
                        except Exception as err:
                            LOGGER.warning("Failed to read image file %s: %s", img, err)

        # Process HA attachments
        if attachments:
            for attachment in attachments:
                try:

                    def _read_att_sync(p):
                        with open(p, "rb") as f:
                            return f.read()

                    image_bytes = await self.hass.async_add_executor_job(
                        _read_att_sync, attachment.path
                    )
                    base64_image = base64.b64encode(image_bytes).decode("utf-8")
                    data_uri = f"data:{attachment.mime_type};base64,{base64_image}"
                    messages.append(XAIGateway.img_msg(data_uri))
                except Exception as err:
                    LOGGER.warning(
                        "Failed to process attachment %s: %s", attachment.path, err
                    )

        return messages

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
        # Pre-process attachments into image messages
        extra_messages = await self._prepare_attachments(task.attachments)

        # Execute stateless chat via gateway
        await self.gateway.execute_stateless_chat(
            chat_log, extra_messages=extra_messages, service_type="ai_task"
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
        """Handle an image generation task using grok-2-image model.

        Args:
            task: Image generation task containing prompt and optional attachments
            chat_log: Conversation context for tracking

        Returns:
            GenImageTaskResult with image data and metadata

        Raises:
            Various exceptions via handle_api_error for auth, network, API errors
        """
        # Generate image using private method
        result = await self._generate_image(prompt=task.instructions, chat_log=chat_log)

        # Convert result to GenImageTaskResult
        return ha_ai_task.GenImageTaskResult(
            image_data=result["image_data"],
            conversation_id=chat_log.conversation_id,
            mime_type=result["mime_type"],
            model=result["model"],
            revised_prompt=result.get("revised_prompt"),
        )

    async def _generate_image(
        self,
        prompt: str,
        chat_log: ha_conversation.ChatLog,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Generate image using xAI image generation model.

        Args:
            prompt: Image generation prompt
            chat_log: Chat log for context
            model: Optional model override, defaults to configured image_model

        Returns:
            Dict with image_data, mime_type, model, revised_prompt
        """
        if model is None:
            model = self._get_option(CONF_IMAGE_MODEL, RECOMMENDED_IMAGE_MODEL)

        if not prompt or not prompt.strip():
            raise_validation_error("Image generation prompt cannot be empty")

        context = {"model": model, "prompt_length": len(prompt)}
        async with LogTimeServices(LOGGER, "image_generation", context) as timer:
            client = self.gateway.create_client()

            async with timer.record_api_call():
                response = await client.image.sample(
                    model=model,
                    prompt=prompt,
                    image_format="base64",
                )

            # Get image data (response.image is a coroutine property in SDK)
            try:
                image_bytes = await response.image
            except AttributeError:
                raise_generic_error("xAI API returned invalid image response")

            if not image_bytes:
                raise_generic_error("xAI API returned empty image data")

            if not isinstance(image_bytes, bytes):
                raise_generic_error(
                    f"Expected bytes, got {type(image_bytes).__name__}"
                )

            revised_prompt = getattr(response, "prompt", None)
            timer.context_info["image_size_bytes"] = len(image_bytes)

            # Record usage for cost calculation (1 image = 1 output token)
            class ImageUsage:
                completion_tokens = 1
                prompt_tokens = 0
                reasoning_tokens = 0
                cached_prompt_text_tokens = 0

            await save_response_metadata(
                hass=self.hass,
                entry_id=self.entry.entry_id,
                usage=ImageUsage(),
                model=model,
                service_type="image_generation",
                store_messages=False,
            )

            return {
                "image_data": image_bytes,
                "mime_type": "image/jpeg",
                "model": model,
                "revised_prompt": revised_prompt,
            }

    async def analyze_photo(
        self,
        prompt: str,
        images: list[str] | None = None,
        attachments: list | None = None,
        model: str | None = None,
    ) -> str:
        """Analyze photos using vision model.

        Args:
            prompt: Analysis prompt
            images: List of image paths or URLs
            attachments: List of attachment objects
            model: Optional model override

        Returns:
            Analysis text result
        """
        if not prompt or not prompt.strip():
            raise_validation_error("Photo analysis prompt cannot be empty")
        if not images and not attachments:
            raise_validation_error(
                "At least one image (path/URL or attachment) is required"
            )

        if model is None:
            model = self._get_option(CONF_VISION_MODEL, RECOMMENDED_VISION_MODEL)

        # Pre-process images using the shared helper
        image_messages = await self._prepare_attachments(attachments, images)

        context = {"model": model, "images_count": len(image_messages)}
        async with LogTimeServices(LOGGER, "photo_analysis", context) as timer:
            # Use gateway helper to create chat with vision model
            client = self.gateway.create_client()
            chat = self.gateway.create_chat(
                client=client, tools=None, previous_response_id=None, model=model
            )

            vision_prompt = self._get_option(CONF_VISION_PROMPT, VISION_ANALYSIS_PROMPT)
            if vision_prompt:
                chat.append(XAIGateway.system_msg(vision_prompt))

            # Construct user message: [text, img1, img2...]
            message_content = [prompt]
            message_content.extend(image_messages)

            # Add single user message with mixed content
            chat.append(XAIGateway.user_msg(message_content))

            async with timer.record_api_call():
                response = await chat.sample()

            content_text = getattr(response, "content", "")
            usage = getattr(response, "usage", None)

            # Use correct metadata saver
            await save_response_metadata(
                hass=self.hass,
                entry_id=self.entry.entry_id,
                usage=usage,
                model=model,
                service_type="photo_analysis",
                store_messages=False,
            )

            return content_text
