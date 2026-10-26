"""xAI speech-to-text provider for Home Assistant Assist."""

from __future__ import annotations

from collections.abc import AsyncIterable
from typing import Any

from homeassistant.components.stt import (
    AudioBitRates,
    AudioChannels,
    AudioCodecs,
    AudioFormats,
    AudioSampleRates,
    SpeechMetadata,
    SpeechResult,
    SpeechResultState,
    SpeechToTextEntity,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util import dt as dt_util

from .const import (
    CONF_API_HOST,
    CONF_STT_FILLER_WORDS,
    CONF_STT_KEYTERMS,
    CONF_STT_VAD_THRESHOLD,
    DEFAULT_API_HOST,
    DOMAIN,
    LOGGER,
)
from .voice_api import XAIVoiceError, async_stt_upload
from .voice_const import STT_LANGUAGES


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the xAI STT provider."""
    voice_subentry = next(
        (
            subentry
            for subentry in config_entry.subentries.values()
            if subentry.subentry_type == "voice"
        ),
        None,
    )
    if voice_subentry is None:
        LOGGER.error("Could not find the xAI voice services subentry for STT")
        async_add_entities([XaiSttEntity(config_entry, {})])
        return
    async_add_entities(
        [XaiSttEntity(config_entry, voice_subentry.data)],
        config_subentry_id=voice_subentry.subentry_id,
    )


class XaiSttEntity(SpeechToTextEntity):
    """STT engine backed by xAI Voice."""

    _attr_has_entity_name = False
    _attr_name = "xAI STT"
    _attr_should_poll = False

    def __init__(self, entry: ConfigEntry, voice_options: dict[str, Any]) -> None:
        """Initialize the STT provider."""
        self._entry = entry
        self._voice_options = voice_options
        self._attr_unique_id = f"{entry.entry_id}_stt"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, f"{entry.entry_id}_stt")},
            name="xAI STT",
            manufacturer="xAI",
            model="Speech-to-Text",
            entry_type=DeviceEntryType.SERVICE,
        )

    @property
    def supported_languages(self) -> list[str]:
        """Return the languages available for selection in Assist."""
        return STT_LANGUAGES

    @property
    def supported_formats(self) -> list[AudioFormats]:
        """Return the audio containers accepted from Assist."""
        return [AudioFormats.WAV, AudioFormats.OGG]

    @property
    def supported_codecs(self) -> list[AudioCodecs]:
        """Return the codecs accepted from Assist."""
        return [AudioCodecs.PCM, AudioCodecs.OPUS]

    @property
    def supported_bit_rates(self) -> list[AudioBitRates]:
        """Return the bit depths accepted from Assist."""
        return [AudioBitRates.BITRATE_16]

    @property
    def supported_sample_rates(self) -> list[AudioSampleRates]:
        """Return Home Assistant sample rates supported by xAI."""
        return [
            AudioSampleRates.SAMPLERATE_8000,
            AudioSampleRates.SAMPLERATE_16000,
            AudioSampleRates.SAMPLERATE_44100,
            AudioSampleRates.SAMPLERATE_48000,
        ]

    @property
    def supported_channels(self) -> list[AudioChannels]:
        """Return the channel layouts supported for Assist audio."""
        return [AudioChannels.CHANNEL_MONO]

    def check_metadata(self, metadata: SpeechMetadata) -> bool:
        """Check whether Home Assistant's requested audio metadata is supported."""
        supported = super().check_metadata(metadata)
        if not supported:
            LOGGER.debug(
                "xAI STT rejected Home Assistant audio metadata: language=%s, "
                "format=%s, codec=%s, bit_rate=%s, sample_rate=%s, channel=%s",
                metadata.language,
                metadata.format,
                metadata.codec,
                metadata.bit_rate,
                metadata.sample_rate,
                metadata.channel,
            )
        return supported

    async def async_process_audio_stream(
        self, metadata: SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> SpeechResult:
        """Transcribe Home Assistant's WAV or OGG audio through xAI REST."""
        # Assist calls async_process_audio_stream directly, bypassing the STT
        # wrapper that normally records this timestamp for API requests.
        self._SpeechToTextEntity__last_processed = dt_util.utcnow().isoformat()
        self.async_write_ha_state()
        LOGGER.debug(
            "xAI STT received audio from Home Assistant: language=%s, format=%s, "
            "codec=%s, bit_rate=%s, sample_rate=%s, channel=%s, "
            "entity_state=%s",
            metadata.language,
            metadata.format,
            metadata.codec,
            metadata.bit_rate,
            metadata.sample_rate,
            metadata.channel,
            self.state,
        )
        session = async_get_clientsession(self.hass)
        api_key = self._entry.data.get("api_key", "")
        api_host = self._entry.data.get(CONF_API_HOST, DEFAULT_API_HOST)

        try:
            LOGGER.debug(
                "xAI STT selected REST upload for %s container audio",
                metadata.format,
            )
            transcript = await async_stt_upload(
                session,
                api_key,
                api_host,
                stream=stream,
                language=metadata.language,
                audio_format=getattr(metadata.format, "value", metadata.format),
                is_raw_pcm=metadata.codec == AudioCodecs.PCM,
                sample_rate=int(metadata.sample_rate),
                channels=int(metadata.channel),
                keyterms=self._voice_options.get(CONF_STT_KEYTERMS, []),
                filler_words=self._voice_options.get(CONF_STT_FILLER_WORDS, False),
                vad_threshold=self._voice_options.get(CONF_STT_VAD_THRESHOLD, 0.5),
            )
        except XAIVoiceError as err:
            LOGGER.warning("xAI STT request failed: %s", err)
            return SpeechResult(None, SpeechResultState.ERROR)

        LOGGER.debug(
            "xAI STT completed successfully; transcript length=%d characters, "
            "entity_state=%s",
            len(transcript),
            self.state,
        )

        if not transcript:
            LOGGER.warning(
                "xAI STT returned an empty transcript; treating it as an error "
                "(language=%s, format=%s, "
                "codec=%s, sample_rate=%s, channels=%s)",
                metadata.language,
                metadata.format,
                metadata.codec,
                metadata.sample_rate,
                metadata.channel,
            )
            return SpeechResult(None, SpeechResultState.ERROR)

        return SpeechResult(transcript, SpeechResultState.SUCCESS)
