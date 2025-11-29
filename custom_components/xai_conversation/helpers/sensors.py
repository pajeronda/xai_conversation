"""Helper functions for sensor management."""

from __future__ import annotations

from homeassistant.core import HomeAssistant, callback

from ..const import DOMAIN, LOGGER


@callback
def update_token_sensors(
    hass: HomeAssistant,
    entry_id: str,
    usage,
    model: str,
    service_type: str,
    mode: str = "pipeline",
    is_fallback: bool = False,
    store_messages: bool = True,
) -> None:
    """Find all sensors for an entry and update their token usage.

    Args:
        hass: Home Assistant instance
        entry_id: Config entry ID
        usage: xAI response.usage object with token counts
        model: Model name from xAI response
        service_type: Service type ("conversation", "ai_task", "code_fast")
        mode: Conversation mode ("pipeline" or "tools")
        is_fallback: True if fallback from pipeline to tools mode
        store_messages: True for server-side memory, False for client-side
    """
    sensors = hass.data.get(DOMAIN, {}).get(f"{entry_id}_sensors")
    if not sensors:
        LOGGER.debug("update_token_sensors: no sensors found for entry %s", entry_id)
        return

    # Count sensors that will actually be updated
    updated_count = 0

    for sensor in sensors:
        # Skip sensors that don't have update_token_usage (e.g., pricing sensors)
        if not hasattr(sensor, "update_token_usage"):
            continue

        sensor_service_type = getattr(sensor, "_service_type", None)

        if sensor_service_type is None or sensor_service_type == service_type:
            try:
                sensor.update_token_usage(
                    usage=usage,
                    model=model,
                    mode=mode,
                    is_fallback=is_fallback,
                    store_messages=store_messages,
                    skip_save=True,  # Skip individual saves during loop
                )
                updated_count += 1
            except (ValueError, TypeError, AttributeError) as e:
                entity_id = getattr(sensor, "entity_id", "unknown")
                LOGGER.error(
                    "Failed to update sensor %s: %s", entity_id, e, exc_info=True
                )

    LOGGER.debug(
        "Updated %d/%d sensors for entry %s, service=%s, model=%s",
        updated_count,
        len(sensors),
        entry_id,
        service_type,
        model,
    )

    # NOTE: Batch save is handled by save_response_metadata() after this function returns
