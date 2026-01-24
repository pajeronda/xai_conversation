"""Ask Service handler."""

from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING

from ..const import LOGGER, STATUS_OK, RECOMMENDED_CHAT_MODEL
from ..exceptions import raise_validation_error
from ..helpers import LogTimeServices, ChatOptions
from .base import GatewayMixin

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant, ServiceCall, ServiceResponse
    from homeassistant.config_entries import ConfigEntry


class AskService(GatewayMixin):
    """Service handler for 'ask' - Generate stateless response from prompt/data."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry):
        """Initialize the service."""
        self.hass = hass
        self.entry = entry

    async def async_handle(self, call: ServiceCall) -> ServiceResponse:
        """Handle the ask service call."""
        instructions = call.data.get("instructions", "").strip()
        input_data = call.data.get("input_data", "").strip()
        model = call.data.get("model") or RECOMMENDED_CHAT_MODEL
        max_tokens = call.data.get("max_tokens")
        temperature = call.data.get("temperature")
        top_p = call.data.get("top_p")
        reasoning_effort = call.data.get("reasoning_effort")
        live_search = call.data.get("live_search", "off")
        show_citations = call.data.get("show_citations", False)

        if not instructions or not input_data:
            raise_validation_error("Instructions and Input Data are required")

        context = {
            "mode": "stateless",
            "model": model,
            "prompt_length": len(input_data),
        }
        async with LogTimeServices(LOGGER, "ask", context) as timer:
            messages = [{"role": "user", "content": input_data}]

            response_text = await self.gateway.execute_stateless_chat(
                messages=messages,
                service_type="ask",
                options=ChatOptions(
                    system_prompt=instructions,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    reasoning_effort=reasoning_effort,
                    live_search=live_search,
                    show_citations=show_citations,
                    timer=timer,
                ),
                hass=self.hass,
            )

            return {
                "status": STATUS_OK,
                "response_text": response_text,
                "timestamp": datetime.now().isoformat(),
            }
