"""Photo Analysis Service handler."""

from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING

from ..const import (
    CONF_VISION_MODEL,
    LOGGER,
    MODEL_TARGET_VISION,
    RECOMMENDED_VISION_MODEL,
    STATUS_OK,
)
from ..exceptions import raise_validation_error
from ..helpers import (
    LogTimeServices,
    ChatOptions,
    async_prepare_attachments,
    async_parse_image_input,
)
from .base import GatewayMixin

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant, ServiceCall, ServiceResponse
    from homeassistant.config_entries import ConfigEntry


class PhotoAnalysisService(GatewayMixin):
    """Service handler for photo_analysis - Analyze images using Grok Vision."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry):
        """Initialize the service."""
        self.hass = hass
        self.entry = entry

    async def async_handle(self, call: ServiceCall) -> ServiceResponse:
        """Handle the photo_analysis service call."""
        context_id = call.context.id
        temperature = call.data.get("temperature")
        top_p = call.data.get("top_p")
        # Model priority: service call parameter > config flow > recommended default
        configured_model = RECOMMENDED_VISION_MODEL
        for subentry in self.entry.subentries.values():
            if subentry.subentry_type == "ai_task":
                configured_model = subentry.data.get(
                    CONF_VISION_MODEL, RECOMMENDED_VISION_MODEL
                )
                break
        model = call.data.get("model") or configured_model

        # Parse request data
        prompt = call.data.get("prompt", "").strip()
        if not prompt:
            raise_validation_error("Prompt is required for photo analysis")

        # Get images from multiple sources
        images = async_parse_image_input(call.data.get("images"))

        # Get attachments if provided (from HA automation context)
        attachments = call.data.get("attachments")

        # Validate at least one image source
        if not images and not attachments:
            raise_validation_error(
                "At least one image (path/URL or attachment) is required"
            )

        context = {
            "mode": "vision",
            "prompt_length": len(prompt),
            "images": len(images) if images else 0,
            "model": model,
        }
        async with LogTimeServices(LOGGER, "photo_analysis", context) as timer:
            LOGGER.debug(
                "[Context: %s] Request data: prompt_length=%d, images=%d, has_attachments=%s, model=%s",
                context_id,
                len(prompt),
                len(images),
                bool(attachments),
                model,
            )

            # 1. Prepare attachments (functional helper, no entity needed)
            prepared = await async_prepare_attachments(
                self.hass, attachments, images
            )
            extra_content = prepared.uris

            # 2. Setup mixed content messages
            final_prompt = prompt
            if prepared.has_skipped:
                skipped_list = ", ".join(prepared.skipped)
                final_prompt += (
                    f"\n\n[System Note: The following files were skipped due to unsupported formats "
                    f"(xAI supports JPEG, PNG, WebP): {skipped_list}]"
                )

            user_content = [final_prompt] + (extra_content or [])
            messages = [
                {
                    "role": "user",
                    "content": user_content
                    if len(user_content) > 1
                    else user_content[0],
                }
            ]

            # 3. Call Gateway directly - truly stateless with explicit model
            analysis_text = await self.gateway.execute_stateless_chat(
                messages=messages,
                service_type="photo_analysis",
                options=ChatOptions(
                    model=model,
                    mode_override="vision",
                    model_target=MODEL_TARGET_VISION,
                    temperature=temperature,
                    top_p=top_p,
                    timer=timer,
                ),
                hass=self.hass,
            )

            return {
                "status": STATUS_OK,
                "analysis": analysis_text,
                "timestamp": datetime.now().isoformat(),
            }
