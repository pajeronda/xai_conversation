"""Async HTTP and WebSocket clients for xAI Voice."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import struct
from collections.abc import AsyncGenerator, AsyncIterable
from urllib.parse import urlsplit

from aiohttp import (
    ClientError,
    ClientResponseError,
    ClientSession,
    ClientTimeout,
    FormData,
    WSMsgType,
)

from .voice_const import (
    STT_FORMAT_LANGUAGES,
    XAI_VOICES,
    normalize_stt_language,
)

VOICE_TIMEOUT = ClientTimeout(total=120)
VOICE_CONNECT_TIMEOUT = ClientTimeout(total=15)
MAX_BUFFERED_AUDIO = 25 * 1024 * 1024
PCM_SAMPLE_RATES = {8000, 16000, 22050, 24000, 44100, 48000}
LOGGER = logging.getLogger(__package__)


class XAIVoiceError(Exception):
    """Raised when an xAI Voice request fails."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def _endpoint(api_host: str, path: str, websocket: bool = False) -> str:
    """Build a Voice URL from the configured xAI API hostname."""
    host = api_host.strip().lower().rstrip(".")
    parsed = urlsplit(f"https://{host}")
    if (
        not parsed.hostname
        or parsed.netloc != parsed.hostname
        or parsed.path
        or parsed.query
        or parsed.fragment
        or parsed.username
        or parsed.password
        or not (parsed.hostname == "api.x.ai" or parsed.hostname.endswith(".api.x.ai"))
    ):
        raise XAIVoiceError("The configured API host is not a supported xAI endpoint")

    scheme = "wss" if websocket else "https"
    return f"{scheme}://{parsed.hostname}/v1/{path.lstrip('/')}"


def _raise_voice_status(status: int) -> XAIVoiceError:
    if status in (401, 403):
        message = "xAI Voice rejected the API key or the key lacks Voice access"
    elif status == 404:
        message = "The configured xAI API endpoint does not provide this Voice API"
    elif status == 413:
        message = "The audio or text exceeds the xAI Voice request limit"
    elif status == 429:
        message = "xAI Voice rate limit exceeded"
    elif status >= 500:
        message = "xAI Voice is temporarily unavailable"
    else:
        message = f"xAI Voice request failed with HTTP {status}"
    return XAIVoiceError(message, status)


async def async_list_voices(
    session: ClientSession, api_key: str, api_host: str
) -> dict[str, str]:
    """Fetch the account's xAI voices, falling back to built-in voices."""
    try:
        async with session.get(
            _endpoint(api_host, "tts/voices"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=VOICE_CONNECT_TIMEOUT,
        ) as response:
            if response.status != 200:
                return dict(XAI_VOICES)
            data = await response.json(content_type=None)
    except (ClientError, TimeoutError, ValueError, XAIVoiceError):
        return dict(XAI_VOICES)

    voices: dict[str, str] = {}
    items = data.get("voices", []) if isinstance(data, dict) else []
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            voice_id = item.get("voice_id")
            name = item.get("name") or voice_id
            if isinstance(voice_id, str) and voice_id:
                voices[voice_id] = str(name)
    return voices or dict(XAI_VOICES)


def _decode_tts_response(body: bytes, content_type: str) -> bytes:
    if "application/json" not in content_type.lower():
        return body
    try:
        payload = json.loads(body)
        audio = payload.get("audio")
        if not isinstance(audio, str):
            raise ValueError("missing audio field")
        return base64.b64decode(audio, validate=True)
    except (ValueError, binascii.Error, json.JSONDecodeError) as err:
        raise XAIVoiceError("xAI TTS returned an invalid audio response") from err


async def async_tts(
    session: ClientSession,
    api_key: str,
    api_host: str,
    *,
    text: str,
    voice: str,
    language: str,
    speed: float,
    text_normalization: bool,
    replace: dict[str, str] | None,
) -> bytes:
    """Synthesize MP3 audio using xAI's REST TTS endpoint."""
    if len(text) > 60000:
        raise XAIVoiceError("The TTS message exceeds xAI's 60,000 character REST limit")

    payload = {
        "text": text,
        "voice_id": voice,
        "language": language,
        "output_format": {
            "codec": "mp3",
            "sample_rate": 24000,
            "bit_rate": 128000,
        },
        "speed": speed,
        "text_normalization": text_normalization,
    }
    if replace:
        payload["replace"] = replace

    try:
        async with session.post(
            _endpoint(api_host, "tts"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=VOICE_TIMEOUT,
        ) as response:
            body = await response.read()
            if response.status != 200:
                raise _raise_voice_status(response.status)
            return _decode_tts_response(body, response.headers.get("Content-Type", ""))
    except XAIVoiceError:
        raise
    except (ClientError, TimeoutError) as err:
        raise XAIVoiceError("Could not reach the xAI TTS endpoint") from err


async def async_tts_stream(
    session: ClientSession,
    api_key: str,
    api_host: str,
    *,
    text_stream: AsyncIterable[str],
    voice: str,
    language: str,
    speed: float,
    text_normalization: bool,
    replace: dict[str, str] | None,
    optimize_streaming_latency: int = 0,
) -> AsyncGenerator[bytes]:
    """Stream MP3 chunks from xAI's WebSocket TTS endpoint."""
    params = {
        "voice": voice,
        "language": language,
        "codec": "mp3",
        "sample_rate": "24000",
        "bit_rate": "128000",
        "speed": str(speed),
        "text_normalization": str(text_normalization).lower(),
        "optimize_streaming_latency": str(optimize_streaming_latency),
    }
    try:
        async with session.ws_connect(
            _endpoint(api_host, "tts", websocket=True),
            headers={"Authorization": f"Bearer {api_key}"},
            params=params,
            heartbeat=30,
        ) as websocket:
            audio_queue: asyncio.Queue[bytes | Exception | None] = asyncio.Queue(
                maxsize=8
            )

            async def send_text() -> None:
                try:
                    if replace:
                        await websocket.send_json(
                            {"type": "session.update", "replace": replace}
                        )
                    async for chunk in text_stream:
                        for offset in range(0, len(chunk), 60000):
                            await websocket.send_json(
                                {
                                    "type": "text.delta",
                                    "delta": chunk[offset : offset + 60000],
                                }
                            )
                    await websocket.send_json({"type": "text.done"})
                except asyncio.CancelledError:
                    raise
                except XAIVoiceError as err:
                    await audio_queue.put(err)
                except (ClientError, TimeoutError):
                    await audio_queue.put(
                        XAIVoiceError("Could not send text to the xAI TTS stream")
                    )
                except Exception as err:
                    await audio_queue.put(err)

            async def receive_audio() -> None:
                try:
                    while True:
                        message = await websocket.receive(timeout=VOICE_TIMEOUT.total)
                        if message.type == WSMsgType.TEXT:
                            try:
                                event = json.loads(message.data)
                            except json.JSONDecodeError as err:
                                raise XAIVoiceError(
                                    "xAI TTS returned an invalid event"
                                ) from err
                            event_type = event.get("type")
                            if event_type == "audio.delta":
                                try:
                                    audio = base64.b64decode(
                                        event["delta"], validate=True
                                    )
                                except (KeyError, ValueError, binascii.Error) as err:
                                    raise XAIVoiceError(
                                        "xAI TTS returned invalid audio data"
                                    ) from err
                                await audio_queue.put(audio)
                            elif event_type == "audio.done":
                                await audio_queue.put(None)
                                return
                            elif event_type == "error":
                                raise XAIVoiceError(
                                    "xAI TTS could not synthesize the request"
                                )
                        elif message.type == WSMsgType.ERROR:
                            raise XAIVoiceError(
                                "The xAI TTS WebSocket closed with an error"
                            )
                        elif message.type in (WSMsgType.CLOSED, WSMsgType.CLOSE):
                            raise XAIVoiceError(
                                "The xAI TTS WebSocket closed before audio completed"
                            )
                except asyncio.CancelledError:
                    raise
                except XAIVoiceError as err:
                    await audio_queue.put(err)
                except (ClientError, TimeoutError):
                    await audio_queue.put(
                        XAIVoiceError("Could not receive audio from the xAI TTS stream")
                    )
                except Exception as err:
                    await audio_queue.put(err)

            sender_task = asyncio.create_task(send_text())
            receiver_task = asyncio.create_task(receive_audio())
            try:
                while (item := await audio_queue.get()) is not None:
                    if isinstance(item, Exception):
                        raise item
                    yield item
                await sender_task
                await receiver_task
            finally:
                for task in (sender_task, receiver_task):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(sender_task, receiver_task, return_exceptions=True)
    except XAIVoiceError:
        raise
    except ClientResponseError as err:
        raise _raise_voice_status(err.status) from err
    except ClientError as err:
        raise XAIVoiceError("Could not complete the xAI TTS stream") from err
    except TimeoutError as err:
        raise XAIVoiceError("Could not complete the xAI TTS stream") from err


async def async_stt_stream(
    session: ClientSession,
    api_key: str,
    api_host: str,
    *,
    stream: AsyncIterable[bytes],
    language: str | None,
    sample_rate: int,
) -> str:
    """Send raw PCM chunks to xAI's streaming STT endpoint."""
    if sample_rate not in PCM_SAMPLE_RATES:
        raise XAIVoiceError(f"Unsupported xAI PCM sample rate: {sample_rate}")

    params = {
        "model": "grok-voice-transcribe-2.0",
        "encoding": "pcm",
        "sample_rate": str(sample_rate),
        "interim_results": "false",
    }
    normalized_language = normalize_stt_language(language)
    if normalized_language:
        params["language"] = normalized_language
        if normalized_language in STT_FORMAT_LANGUAGES:
            params["format"] = "true"

    LOGGER.debug(
        "Starting xAI STT WebSocket connection (sample_rate=%s, language=%s, "
        "formatting=%s)",
        sample_rate,
        normalized_language or "auto",
        params.get("format", "false"),
    )

    try:
        async with session.ws_connect(
            _endpoint(api_host, "stt", websocket=True),
            headers={"Authorization": f"Bearer {api_key}"},
            params=params,
            heartbeat=30,
        ) as websocket:
            LOGGER.debug("xAI STT WebSocket connected; waiting for ready event")
            ready = await websocket.receive(timeout=15)
            if ready.type != WSMsgType.TEXT:
                LOGGER.debug("xAI STT received non-text ready frame: %s", ready.type)
                raise XAIVoiceError(
                    "xAI STT did not initialize the transcription stream"
                )
            try:
                ready_event = json.loads(ready.data)
            except json.JSONDecodeError as err:
                raise XAIVoiceError(
                    "xAI STT returned an invalid initialization event"
                ) from err
            if ready_event.get("type") == "error":
                LOGGER.debug(
                    "xAI STT rejected WebSocket initialization: %s", ready_event
                )
                raise XAIVoiceError(
                    "xAI STT could not initialize the transcription stream"
                )
            if ready_event.get("type") != "transcript.created":
                LOGGER.debug(
                    "xAI STT received unexpected initialization event: %s",
                    ready_event.get("type"),
                )
                raise XAIVoiceError(
                    "xAI STT returned an unexpected initialization event"
                )

            async def send_audio() -> None:
                sent_audio = False
                sent_bytes = 0
                sent_frames = 0
                frame_size = sample_rate * 2 // 10
                audio_buffer = bytearray()
                async for chunk in stream:
                    if chunk:
                        audio_buffer.extend(chunk)
                        sent_audio = True
                        while len(audio_buffer) >= frame_size:
                            await websocket.send_bytes(audio_buffer[:frame_size])
                            del audio_buffer[:frame_size]
                            sent_bytes += frame_size
                            sent_frames += 1
                if not sent_audio:
                    raise XAIVoiceError("The audio stream was empty")
                if audio_buffer:
                    if len(audio_buffer) % 2:
                        raise XAIVoiceError(
                            "The PCM audio stream ended with an incomplete sample"
                        )
                    await websocket.send_bytes(audio_buffer)
                    sent_bytes += len(audio_buffer)
                    sent_frames += 1
                await websocket.send_json({"type": "audio.done"})
                LOGGER.debug(
                    "Sent PCM audio to xAI STT and signaled audio.done "
                    "(bytes=%d, frames=%d)",
                    sent_bytes,
                    sent_frames,
                )

            async def receive_transcript() -> str:
                while True:
                    message = await websocket.receive(timeout=VOICE_TIMEOUT.total)
                    if message.type == WSMsgType.TEXT:
                        try:
                            event = json.loads(message.data)
                        except json.JSONDecodeError as err:
                            raise XAIVoiceError(
                                "xAI STT returned an invalid event"
                            ) from err
                        event_type = event.get("type")
                        LOGGER.debug("Received xAI STT WebSocket event: %s", event_type)
                        if event_type == "transcript.done":
                            transcript = event.get("text")
                            if isinstance(transcript, str):
                                LOGGER.debug(
                                    "xAI STT returned final transcript "
                                    "(length=%d characters)",
                                    len(transcript.strip()),
                                )
                                return transcript.strip()
                            raise XAIVoiceError("xAI STT returned no transcript")
                        if event_type == "error":
                            error_message = event.get("message")
                            if not isinstance(error_message, str) or not error_message:
                                error_message = "unknown error"
                            LOGGER.debug(
                                "xAI STT returned an error event: %s", error_message
                            )
                            raise XAIVoiceError(
                                f"xAI STT could not transcribe the audio: {error_message}"
                            )
                    elif message.type == WSMsgType.ERROR:
                        raise XAIVoiceError(
                            "The xAI STT WebSocket closed with an error"
                        )
                    elif message.type in (WSMsgType.CLOSED, WSMsgType.CLOSE):
                        raise XAIVoiceError(
                            "The xAI STT WebSocket closed before transcription completed"
                        )

            sender_task = asyncio.create_task(send_audio())
            transcript_task = asyncio.create_task(receive_transcript())
            try:
                await asyncio.gather(sender_task, transcript_task)
            finally:
                for task in (sender_task, transcript_task):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(
                    sender_task, transcript_task, return_exceptions=True
                )
            return transcript_task.result()
    except XAIVoiceError:
        raise
    except ClientResponseError as err:
        LOGGER.debug("xAI STT WebSocket handshake failed with HTTP %s", err.status)
        raise _raise_voice_status(err.status) from err
    except ClientError as err:
        LOGGER.debug("xAI STT WebSocket transport error: %s", err)
        raise XAIVoiceError("Could not complete the xAI STT stream") from err
    except TimeoutError as err:
        LOGGER.debug("xAI STT WebSocket timed out: %s", err)
        raise XAIVoiceError("Could not complete the xAI STT stream") from err


async def async_stt_upload(
    session: ClientSession,
    api_key: str,
    api_host: str,
    *,
    stream: AsyncIterable[bytes],
    language: str | None,
    audio_format: str,
    is_raw_pcm: bool = False,
    sample_rate: int = 16000,
    channels: int = 1,
    keyterms: list[str] | None = None,
    filler_words: bool = False,
    vad_threshold: float = 0.5,
) -> str:
    """Upload container audio or wrap raw PCM before sending it to xAI STT."""
    file_details = {
        "wav": ("audio.wav", "audio/wav"),
        "ogg": ("audio.ogg", "audio/ogg"),
    }
    if audio_format not in file_details:
        raise XAIVoiceError(f"Unsupported xAI STT audio format: {audio_format}")

    LOGGER.debug("Buffering encoded audio for xAI STT REST upload")
    audio = bytearray()
    async for chunk in stream:
        audio.extend(chunk)
        if len(audio) > MAX_BUFFERED_AUDIO:
            raise XAIVoiceError(
                "The buffered audio exceeds the 25 MiB integration limit"
            )
    if not audio:
        raise XAIVoiceError("The audio stream was empty")

    if is_raw_pcm:
        if len(audio) % 2:
            raise XAIVoiceError("The PCM audio stream ended with an incomplete sample")
        audio = bytearray(_pcm16_to_wav(bytes(audio), sample_rate, channels))
        audio_format = "wav"
        LOGGER.debug(
            "Wrapped raw PCM as WAV for xAI STT (sample_rate=%d, channels=%d)",
            sample_rate,
            channels,
        )

    LOGGER.debug("Buffered audio for xAI STT REST upload (bytes=%d)", len(audio))

    form = FormData()
    form.add_field("model", "grok-voice-transcribe-2.0")
    normalized_language = normalize_stt_language(language)
    if normalized_language:
        form.add_field("language", normalized_language)
        if normalized_language in STT_FORMAT_LANGUAGES:
            form.add_field("format", "true")
    if filler_words:
        form.add_field("filler_words", "true")
    if vad_threshold != 0.5:
        form.add_field("vad_threshold", str(vad_threshold))
    for keyterm in keyterms or []:
        form.add_field("keyterm", keyterm)
    filename, content_type = file_details[audio_format]
    form.add_field(
        "file",
        bytes(audio),
        filename=filename,
        content_type=content_type,
    )

    try:
        LOGGER.debug("Sending audio to xAI STT REST endpoint")
        async with session.post(
            _endpoint(api_host, "stt"),
            headers={"Authorization": f"Bearer {api_key}"},
            data=form,
            timeout=VOICE_TIMEOUT,
        ) as response:
            LOGGER.debug("xAI STT REST endpoint returned HTTP %s", response.status)
            if response.status != 200:
                detail = (await response.text())[:500]
                LOGGER.debug(
                    "xAI STT REST error detail (HTTP %s): %s",
                    response.status,
                    detail or "<empty response>",
                )
                status_error = _raise_voice_status(response.status)
                if detail:
                    raise XAIVoiceError(f"{status_error}: {detail}", response.status)
                raise status_error
            payload = await response.json(content_type=None)
    except XAIVoiceError:
        raise
    except (ClientError, TimeoutError, ValueError) as err:
        LOGGER.debug("xAI STT REST transport or response error: %s", err)
        raise XAIVoiceError("Could not complete the xAI STT request") from err

    transcript = payload.get("text") if isinstance(payload, dict) else None
    if not isinstance(transcript, str):
        LOGGER.debug("xAI STT REST response did not contain a text field")
        raise XAIVoiceError("xAI STT returned an invalid transcription response")
    LOGGER.debug(
        "xAI STT REST upload completed (transcript length=%d characters)",
        len(transcript.strip()),
    )
    return transcript.strip()


def _pcm16_to_wav(pcm: bytes, sample_rate: int, channels: int) -> bytes:
    """Wrap raw signed 16-bit little-endian PCM samples in a WAV header."""
    bits_per_sample = 16
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_size = len(pcm)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header + pcm
