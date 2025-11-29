"""xAI Grok Conversation agent for Home Assistant."""

from __future__ import annotations

# Standard library imports
from typing import Literal

# Home Assistant imports
from homeassistant.const import MATCH_ALL
from homeassistant.helpers import device_registry as ha_device_registry
from homeassistant.core import HomeAssistant as HA_HomeAssistant
from homeassistant.config_entries import ConfigSubentry as HA_ConfigSubentry
from homeassistant.helpers.entity_platform import (
    AddConfigEntryEntitiesCallback as HA_AddConfigEntryEntitiesCallback,
)

# Local application imports
from .__init__ import (
    ha_conversation,
)
from .const import (
    CONF_ALLOW_SMART_HOME_CONTROL,
    CONF_CHAT_MODEL,
    DEFAULT_MANUFACTURER,
    DOMAIN,
    LOGGER,
    RECOMMENDED_CHAT_MODEL,
    SUBENTRY_TYPE_CONVERSATION,
)
from .entity import XAIBaseLLMEntity


async def async_setup_entry(
    hass: HA_HomeAssistant,
    config_entry,
    async_add_entities: HA_AddConfigEntryEntitiesCallback,
) -> None:
    """Set up xAI conversation entities."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != SUBENTRY_TYPE_CONVERSATION:
            continue

        async_add_entities(
            [XAIConversationEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class XAIConversationEntity(
    XAIBaseLLMEntity,
    ha_conversation.ConversationEntity,
    ha_conversation.AbstractConversationAgent,
):
    """xAI Grok conversation entity with tool-calling support."""

    _attr_has_entity_name = True
    _attr_name = None
    _attr_supports_streaming = True

    def __init__(self, config_entry, subentry: HA_ConfigSubentry) -> None:
        """Initialize the xAI conversation entity."""
        super().__init__(config_entry, subentry)
        self.config_entry = config_entry
        self.subentry = subentry

        # Get model from subentry data
        self._model = subentry.data.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)

        # Entity configuration
        self._attr_unique_id = subentry.subentry_id

        # Set supported features based on unified allow_control setting
        allow_control = self.subentry.data.get(CONF_ALLOW_SMART_HOME_CONTROL, True)

        if allow_control:
            self._attr_supported_features = (
                ha_conversation.ConversationEntityFeature.CONTROL
            )

        # Device info following OpenAI pattern
        self._attr_device_info = ha_device_registry.DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=subentry.title,
            manufacturer=DEFAULT_MANUFACTURER,
            model=self._model,
            entry_type=ha_device_registry.DeviceEntryType.SERVICE,
        )

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return supported languages."""
        return MATCH_ALL

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        ha_conversation.async_set_agent(self.hass, self.config_entry, self)
        LOGGER.debug("XAI Conversation agent registered: %s", self.entity_id)

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant.

        NOTE: This is called both during reload (config change) and actual removal.
        We only unregister the agent here. Memory cleanup is handled in __init__.async_remove_entry()
        which is called only when the integration is actually removed by the user.
        """
        ha_conversation.async_unset_agent(self.hass, self.config_entry)
        await super().async_will_remove_from_hass()

    async def _async_handle_message(
        self,
        user_input: ha_conversation.ConversationInput,
        chat_log: ha_conversation.ChatLog,
    ) -> ha_conversation.ConversationResult:
        """Call the API.

        This method determines which mode to use (pipeline or tools) and delegates
        the actual processing to the appropriate handler. Prompt construction and
        memory management are handled entirely by the processor (entity_pipeline.py or entity_tools.py).
        """
        # Delegate to the appropriate processor
        # Each processor handles its own prompt construction and memory management independently
        await self._async_handle_chat_log(
            user_input,
            chat_log,
        )

        return ha_conversation.async_get_result_from_chat_log(user_input, chat_log)
