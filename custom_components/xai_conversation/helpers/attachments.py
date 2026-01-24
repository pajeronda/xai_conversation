"""Attachment and image processing utilities for xAI."""

from __future__ import annotations

import base64
import mimetypes
from typing import Any, TYPE_CHECKING

from homeassistant.components import media_source, camera

from ..const import LOGGER

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant


async def async_read_image_to_uri(
    hass: HomeAssistant, path: str, mime_type: str = "image/jpeg"
) -> str:
    """Read a local file and return it as a base64-encoded Data URI.

    Uses the executor to avoid blocking the event loop.
    """

    def _read_file_sync(file_path: str) -> bytes:
        if not file_path.startswith("/"):
            raise ValueError(f"Not a local absolute path: {file_path}")
        with open(file_path, "rb") as f:
            return f.read()

    try:
        image_bytes = await hass.async_add_executor_job(_read_file_sync, path)
        base_64_image = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:{mime_type};base64,{base_64_image}"
    except Exception as err:
        LOGGER.debug("Failed to read local image %s: %s", path, err)
        return ""


def async_parse_image_input(images_input: Any | None) -> list[str]:
    """Parse image input (string, list, or single object) into a list of strings.

    Handles:
    - Splitting multiline strings.
    - Stripping YAML list markers (e.g., "- ").
    - Splitting multiple URLs/paths on the same line by whitespace.
    """
    if not images_input:
        return []

    if isinstance(images_input, list):
        # Even for lists, ensure elements are strings and cleaned
        return [str(img).strip() for img in images_input if img]

    if not isinstance(images_input, str):
        return [str(images_input)]

    images: list[str] = []
    # Split by lines and clean up whitespace/YAML list markers
    for line in images_input.splitlines():
        line = line.strip()
        if not line:
            continue

        # Remove common YAML list markers if present at start of line
        if line.startswith("- "):
            line = line[len("- ") :].strip()
        elif line.startswith("-") and not line.startswith(("/", "http", "data:")):
            # Only strip '-' if it's not part of a path or URL starting char
            line = line[1:].strip()

        # Further split by whitespace in case multiple URLs are on one line
        for part in line.split():
            if part:
                images.append(part)

    return images


async def _async_resolve_media_source(
    hass: HomeAssistant, path: str
) -> tuple[str, str | None]:
    """Resolve media-source:// URLs or fetch camera images directly."""
    if path.startswith("media-source://camera/"):
        try:
            entity_id = path.replace("media-source://camera/", "")
            image = await camera.async_get_image(hass, entity_id)
            return (
                f"data:image/jpeg;base64,{base64.b64encode(image.content).decode('utf-8')}",
                "image/jpeg",
            )
        except Exception as err:
            LOGGER.warning("Failed to fetch camera image for %s: %s", path, err)
            return "", None

    if not path.startswith("media-source://"):
        return path, None

    try:
        play_info = await media_source.async_resolve_media(hass, path, None)
        return play_info.url, play_info.mime_type
    except Exception as err:
        LOGGER.warning("Failed to resolve media-source %s: %s", path, err)
        return "", None


async def _async_process_single_source(
    hass: HomeAssistant, path_or_url: str | None, mime: str | None = None
) -> str | None:
    """Unified processor for a single image source."""
    if not path_or_url or not isinstance(path_or_url, str):
        return None

    # 1. Handle direct URLs or already encoded data URIs
    if path_or_url.startswith(("http://", "https://", "data:")):
        return path_or_url

    # 2. Normalize camera entity shorthand
    processed_path = path_or_url
    if processed_path.startswith("camera."):
        processed_path = f"media-source://camera/{processed_path}"

    # 3. Resolve (Camera proxy, Media Source, etc.)
    resolved_path, resolved_mime = await _async_resolve_media_source(
        hass, processed_path
    )
    if not resolved_path:
        return None

    # 4. If resolve produced a data URI, we are done
    if resolved_path.startswith("data:"):
        return resolved_path

    # 5. Process local files or internal URLs into data URIs
    try:
        # Robust Mime Detection (Inlined for efficiency)
        final_mime = resolved_mime or mime
        if not final_mime:
            found_mime, _ = mimetypes.guess_type(resolved_path)
            final_mime = (
                found_mime
                if found_mime and found_mime.startswith("image/")
                else "image/jpeg"
            )

        return await async_read_image_to_uri(hass, resolved_path, final_mime)
    except Exception as err:
        LOGGER.warning("Failed to process image source %s: %s", path_or_url, err)
        return None


async def async_prepare_attachments(
    hass: HomeAssistant,
    attachments: Any | None,
    images: Any | None = None,
) -> list[str]:
    """Prepare image URIs from attachments and image paths.

    Returns:
        List of data URIs or URLs (strings).
    """
    uris: list[str] = []

    # 1. Collect all potential sources (normalized as path, optional_mime)
    sources: list[tuple[str | None, str | None]] = []

    # Direct image list (URL strings, local paths, etc.)
    for img in async_parse_image_input(images):
        sources.append((img, None))

    # HA attachments (from automation context)
    attachment_list = (
        attachments
        if isinstance(attachments, list)
        else ([attachments] if attachments else [])
    )
    for attach in attachment_list:
        if not attach:
            continue
        # Efficient extraction from HA objects or legacy dicts
        path = str(getattr(attach, "path", "")) or (
            isinstance(attach, dict)
            and (
                attach.get("file")
                or attach.get("media_content_id")
                or attach.get("camera_entity")
            )
        )
        mime = getattr(attach, "mime_type", None) or (
            isinstance(attach, dict)
            and (attach.get("mime_type") or attach.get("media_content_type"))
        )
        sources.append((path or None, mime or "image/jpeg"))

    # 2. Process all sources through a single unified loop
    for path, mime in sources:
        if uri := await _async_process_single_source(hass, path, mime):
            uris.append(uri)

    if uris:
        LOGGER.debug("Prepared %d image URIs for AI Task", len(uris))
    else:
        LOGGER.debug("No image URIs prepared for AI Task")

    return uris
