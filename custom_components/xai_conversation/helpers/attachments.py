"""Attachment and image processing utilities for xAI."""

from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from homeassistant.components import media_source, camera
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from ..const import LOGGER

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant


@dataclass
class PreparedAttachments:
    """Dataclass to hold prepared image URIs and skipped items."""

    uris: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def has_skipped(self) -> bool:
        """Check if any items were skipped."""
        return len(self.skipped) > 0


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
        # Prefer .path (filesystem path) for local files, fall back to .url
        resolved = str(play_info.path) if play_info.path else play_info.url
        return resolved, play_info.mime_type
    except Exception as err:
        LOGGER.warning("Failed to resolve media-source %s: %s", path, err)
        return "", None


async def _async_process_single_source(
    hass: HomeAssistant, path_or_url: str | None, mime: str | None = None
) -> tuple[str | None, str | None]:
    """Unified processor for a single image source.

    Returns:
        Tuple of (uri, skipped_reason)
    """
    if not path_or_url or not isinstance(path_or_url, str):
        return None, None

    # 1. Early string-based validation (Skip obvious unsupported formats)
    unsupported_extensions = (".svg", ".pdf", ".gif", ".bmp", ".tiff")
    if any(path_or_url.lower().endswith(ext) for ext in unsupported_extensions):
        LOGGER.warning(
            "Image format in '%s' is likely unsupported by xAI (needs JPEG/PNG/WebP). Skipping.",
            path_or_url,
        )
        return None, f"{path_or_url} (formato non supportato)"

    # 2. Handle direct data URIs
    if path_or_url.startswith("data:"):
        return path_or_url, None

    # 3. Normalize common shorthands
    processed_path = path_or_url.strip()
    if processed_path.startswith("/local/"):
        processed_path = processed_path.replace("/local/", "/config/www/", 1)

    # Normalize camera entity shorthand
    if processed_path.startswith("camera."):
        processed_path = f"media-source://camera/{processed_path}"

    # 4. Handle External/Internal URLs (Download them locally)
    if processed_path.startswith(("http://", "https://")):
        try:
            session = async_get_clientsession(hass)
            # Use HEAD request for early validation
            async with session.head(
                processed_path, timeout=5, allow_redirects=True
            ) as head_res:
                content_type = head_res.headers.get("Content-Type", "").lower()
                supported = ["image/jpeg", "image/jpg", "image/png", "image/webp"]
                if content_type and not any(t in content_type for t in supported):
                    LOGGER.debug(
                        "Unsupported Content-Type '%s' for URL: %s. Skipping (xAI supports JPEG/PNG/WebP).",
                        content_type,
                        processed_path,
                    )
                    return None, f"{processed_path} (tipo {content_type} non supportato)"

            # Proceed to download if potentially valid
            async with session.get(processed_path, timeout=15) as response:
                if response.status != 200:
                    LOGGER.debug(
                        "Failed to download image from %s: status %s",
                        processed_path,
                        response.status,
                    )
                    return None, f"{processed_path} (errore download: {response.status})"

                final_content_type = response.headers.get("Content-Type", content_type)
                image_bytes = await response.read()
                base_64_image = base64.b64encode(image_bytes).decode("utf-8")
                return f"data:{final_content_type};base64,{base_64_image}", None
        except Exception as err:
            LOGGER.warning("Error processing image URL %s: %s", processed_path, err)
            return None, f"{processed_path} (errore elaborazione)"

    # 5. Resolve (Camera proxy, Media Source, etc. for non-HTTP)
    resolved_path, resolved_mime = await _async_resolve_media_source(
        hass, processed_path
    )
    if not resolved_path:
        return None, f"{processed_path} (non risolvibile)"

    # 6. If resolve produced a data URI, we are done
    if resolved_path.startswith("data:"):
        return resolved_path, None

    # 7. Process local files or internal URLs into data URIs
    try:
        final_mime = resolved_mime or mime
        if not final_mime:
            found_mime, _ = mimetypes.guess_type(resolved_path)
            final_mime = (
                found_mime
                if found_mime and found_mime.startswith("image/")
                else "image/jpeg"
            )

        # Validate final mime
        supported = ["image/jpeg", "image/jpg", "image/png", "image/webp"]
        if not any(t in final_mime.lower() for t in supported):
            LOGGER.debug(
                "Final resolved format '%s' for %s is not supported by xAI. Skipping.",
                final_mime,
                resolved_path,
            )
            return None, f"{path_or_url} (tipo {final_mime} non supportato)"

        uri = await async_read_image_to_uri(hass, resolved_path, final_mime)
        return uri, None if uri else f"{path_or_url} (errore lettura file)"
    except Exception as err:
        LOGGER.warning("Failed to process image source %s: %s", path_or_url, err)
        return None, f"{path_or_url} ({str(err)})"


async def async_prepare_attachments(
    hass: HomeAssistant,
    attachments: Any | None,
    images: Any | None = None,
) -> PreparedAttachments:
    """Prepare image URIs from attachments and image paths.

    Returns:
        PreparedAttachments object containing lists of URIs and skipped reasons.
    """
    result = PreparedAttachments()

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
        sources.append((path or None, mime))

    # 2. Process all sources and track results
    for path, mime in sources:
        uri, skipped = await _async_process_single_source(hass, path, mime)
        if uri:
            result.uris.append(uri)
        elif skipped:
            result.skipped.append(skipped)

    if result.uris:
        LOGGER.debug("Prepared %d image URIs for AI Task", len(result.uris))
    if result.skipped:
        LOGGER.debug("Skipped %d images for AI Task", len(result.skipped))

    return result
