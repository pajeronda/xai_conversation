"""xAI text-to-speech provider for Home Assistant Assist."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from homeassistant.components.tts import (
    ATTR_VOICE,
    TextToSpeechEntity,
    TTSAudioRequest,
    TTSAudioResponse,
    TtsAudioType,
    Voice,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import (
    CONF_API_HOST,
    CONF_TTS_DEFAULT_SPEED,
    CONF_TTS_OPTIMIZE_STREAMING_LATENCY,
    CONF_TTS_REPLACE,
    CONF_TTS_TEXT_NORMALIZATION,
    DEFAULT_API_HOST,
    DOMAIN,
    LOGGER,
)
from .voice_api import XAIVoiceError, async_list_voices, async_tts, async_tts_stream
from .voice_const import TTS_LANGUAGES, normalize_tts_language

SPEED_OPTION = "speed"
TEXT_NORMALIZATION_OPTION = "text_normalization"
REPLACE_OPTION = "replace"


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the xAI TTS provider."""
    session = async_get_clientsession(hass)
    voices = await async_list_voices(
        session,
        config_entry.data.get("api_key", ""),
        config_entry.data.get(CONF_API_HOST, DEFAULT_API_HOST),
    )
    voice_subentry = next(
        (
            subentry
            for subentry in config_entry.subentries.values()
            if subentry.subentry_type == "voice"
        ),
        None,
    )
    if voice_subentry is None:
        LOGGER.error("Could not find the xAI voice services subentry for TTS")
        async_add_entities([XaiTtsEntity(config_entry, voices, {})])
        return
    async_add_entities(
        [
            XaiTtsEntity(
                config_entry,
                voices,
                voice_subentry.data if voice_subentry else {},
            )
        ],
        config_subentry_id=voice_subentry.subentry_id,
    )


class XaiTtsEntity(TextToSpeechEntity):
    """TTS engine backed by xAI Voice."""

    _attr_has_entity_name = False
    _attr_name = "xAI TTS"
    _attr_supported_options = [
        ATTR_VOICE,
        SPEED_OPTION,
        TEXT_NORMALIZATION_OPTION,
        REPLACE_OPTION,
    ]
    _attr_supported_languages = TTS_LANGUAGES
    _attr_default_language = "auto"

    def __init__(
        self,
        entry: ConfigEntry,
        voices: dict[str, str],
        voice_options: dict[str, Any],
    ) -> None:
        """Initialize the TTS provider."""
        self._entry = entry
        self._voices = voices
        self._voice_options = voice_options
        self._attr_unique_id = f"{entry.entry_id}_tts"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, f"{entry.entry_id}_tts")},
            name="xAI TTS",
            manufacturer="xAI",
            model="Text-to-Speech",
            entry_type=DeviceEntryType.SERVICE,
        )

    @property
    def default_options(self) -> dict[str, Any]:
        """Return the default xAI synthesis settings."""
        return {
            ATTR_VOICE: "eve",
            SPEED_OPTION: self._voice_options.get(CONF_TTS_DEFAULT_SPEED, 1.0),
            TEXT_NORMALIZATION_OPTION: self._voice_options.get(
                CONF_TTS_TEXT_NORMALIZATION, False
            ),
            REPLACE_OPTION: self._voice_options.get(CONF_TTS_REPLACE) or None,
        }

    @callback
    def async_get_supported_voices(self, language: str) -> list[Voice]:
        """Return built-in and account-specific xAI voices to Assist."""
        return [
            Voice(voice_id, name)
            for voice_id, name in sorted(
                self._voices.items(), key=lambda item: item[1].casefold()
            )
        ]

    async def async_stream_tts_audio(
        self, request: TTSAudioRequest
    ) -> TTSAudioResponse:
        """Stream audio chunks from xAI's TTS WebSocket API."""
        LOGGER.debug(
            "xAI TTS streaming synthesis requested (language=%s)", request.language
        )
        (
            voice,
            speed,
            text_normalization,
            replacements,
        ) = self._options(request.options)
        optimize_streaming_latency = self._voice_options.get(
            CONF_TTS_OPTIMIZE_STREAMING_LATENCY, 0
        )
        optimize_streaming_latency = int(optimize_streaming_latency)
        api_host = self._entry.data.get(CONF_API_HOST, DEFAULT_API_HOST)
        session = async_get_clientsession(self.hass)

        async def data_gen() -> AsyncGenerator[bytes]:
            try:
                async for audio_chunk in async_tts_stream(
                    session,
                    self._entry.data.get("api_key", ""),
                    api_host,
                    text_stream=request.message_gen,
                    voice=voice,
                    language=normalize_tts_language(request.language),
                    speed=speed,
                    text_normalization=text_normalization,
                    replace=replacements,
                    optimize_streaming_latency=optimize_streaming_latency,
                ):
                    yield audio_chunk
            except XAIVoiceError as err:
                LOGGER.warning("xAI TTS stream failed: %s", err)
                raise HomeAssistantError(str(err)) from err

        return TTSAudioResponse(
            "mp3",
            data_gen(),
        )

    async def async_get_tts_audio(
        self, message: str, language: str, options: dict[str, Any]
    ) -> TtsAudioType:
        """Generate an MP3 for TTS previews and one-shot calls."""
        LOGGER.debug("xAI TTS synthesis requested (language=%s)", language)
        voice, speed, text_normalization, replacements = self._options(options)
        session = async_get_clientsession(self.hass)
        try:
            audio = await async_tts(
                session,
                self._entry.data.get("api_key", ""),
                self._entry.data.get(CONF_API_HOST, DEFAULT_API_HOST),
                text=message,
                voice=voice,
                language=normalize_tts_language(language),
                speed=speed,
                text_normalization=text_normalization,
                replace=replacements,
            )
        except XAIVoiceError as err:
            LOGGER.warning("xAI TTS request failed: %s", err)
            raise HomeAssistantError(str(err)) from err
        return "mp3", audio

    def _options(
        self, options: dict[str, Any]
    ) -> tuple[str, float, bool, dict[str, str] | None]:
        voice = options.get(ATTR_VOICE, "eve")
        if not isinstance(voice, str) or not voice.strip():
            raise HomeAssistantError("Select a valid xAI TTS voice")

        try:
            speed = float(
                options.get(
                    SPEED_OPTION,
                    self._voice_options.get(CONF_TTS_DEFAULT_SPEED, 1.0),
                )
            )
        except (TypeError, ValueError) as err:
            raise HomeAssistantError(
                "xAI TTS speed must be a number from 0.7 to 1.5"
            ) from err
        if not 0.7 <= speed <= 1.5:
            raise HomeAssistantError("xAI TTS speed must be a number from 0.7 to 1.5")

        text_normalization = options.get(
            TEXT_NORMALIZATION_OPTION,
            self._voice_options.get(CONF_TTS_TEXT_NORMALIZATION, False),
        )
        if not isinstance(text_normalization, bool):
            raise HomeAssistantError("xAI TTS text_normalization must be a boolean")

        replacements = options.get(
            REPLACE_OPTION, self._voice_options.get(CONF_TTS_REPLACE)
        )
        if replacements is not None:
            if (
                not isinstance(replacements, dict)
                or len(replacements) > 200
                or any(
                    not isinstance(key, str)
                    or not isinstance(value, str)
                    or not key
                    or len(key) > 100
                    or len(value) > 128
                    for key, value in replacements.items()
                )
            ):
                raise HomeAssistantError(
                    "xAI TTS replace must be a valid phrase mapping"
                )

        return voice.strip(), speed, text_normalization, replacements
