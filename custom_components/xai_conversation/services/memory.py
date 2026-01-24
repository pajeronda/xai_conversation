"""Memory Management Services."""

from __future__ import annotations
from typing import TYPE_CHECKING

from ..const import (
    DOMAIN,
    LOGGER,
    STATUS_OK,
    MEMORY_SCOPE_DEVICE,
    MEMORY_SCOPE_USER,
)
from ..exceptions import raise_validation_error
from ..helpers import (
    parse_id_list,
)
from .base import GatewayMixin
from homeassistant.helpers import entity_registry as ha_entity_registry

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant, ServiceCall, ServiceResponse
    from homeassistant.config_entries import ConfigEntry


class ClearMemoryService(GatewayMixin):
    """Service handler for clear_memory - Clear conversation memory for users/satellites.

    The service automatically handles both local and server-side data:
    - Local data (ZDR blobs, conversation keys) is always cleared
    - Server data is only deleted if include_server=True AND the conversation
      was actually stored on xAI servers (store_messages=True)
    """

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry):
        """Initialize the service."""
        self.hass = hass
        self.entry = entry

    async def async_handle(self, call: ServiceCall) -> ServiceResponse:
        """Handle the clear_memory service call."""
        conversation_memory = self.hass.data[DOMAIN]["conversation_memory"]

        include_server = call.data.get("include_server", True)
        only_inactive = call.data.get("only_inactive", False)

        # Delete All: clear entire memory storage
        if call.data.get("delete_all", False):
            result = await self._clear_scope(
                conversation_memory,
                physical_delete=True,
                include_server=include_server,
            )
            return {
                "status": STATUS_OK,
                "message": f"All memory cleared (local: {len(result['all'])}, server: {len(result['server_deleted'])})",
            }

        # Get user and device IDs from entity selections
        user_ids = self._get_user_ids_from_persons(call.data.get("user_id", []))
        device_ids = self._get_device_ids_from_satellites(
            call.data.get("satellite_id", [])
        )

        # Validate: must select something
        if not user_ids and not device_ids:
            raise_validation_error(
                "Select at least one user or satellite, or enable 'Delete All'"
            )

        total_local = 0
        total_server = 0
        cleared_items = []

        # Clear memory for users
        for uid in user_ids:
            result = await self._clear_scope(
                conversation_memory,
                scope=MEMORY_SCOPE_USER,
                target_id=uid,
                include_server=include_server,
                only_inactive=only_inactive,
            )
            total_local += len(result["all"])
            total_server += len(result["server_deleted"])
            cleared_items.append(f"user:{uid[:8]}")

        # Clear memory for devices
        for device_id in device_ids:
            result = await self._clear_scope(
                conversation_memory,
                scope=MEMORY_SCOPE_DEVICE,
                target_id=device_id,
                include_server=include_server,
                only_inactive=only_inactive,
            )
            total_local += len(result["all"])
            total_server += len(result["server_deleted"])
            cleared_items.append(f"device:{device_id[:8]}")

        return {
            "status": STATUS_OK,
            "message": f"Memory cleared for {', '.join(cleared_items)} (local: {total_local}, server: {total_server})",
        }

    def _get_user_ids_from_persons(self, person_entity_ids) -> list[str]:
        """Extract user IDs from person entity IDs."""
        person_entity_ids = parse_id_list(person_entity_ids)
        return [
            state.attributes["user_id"]
            for entity_id in person_entity_ids
            if (state := self.hass.states.get(entity_id))
            and state.attributes.get("user_id")
        ]

    def _get_device_ids_from_satellites(self, satellite_entity_ids) -> list[str]:
        """Extract device IDs from satellite entity IDs."""
        satellite_entity_ids = parse_id_list(satellite_entity_ids)
        ent_reg = ha_entity_registry.async_get(self.hass)
        return [
            entity.device_id
            for entity_id in satellite_entity_ids
            if (entity := ent_reg.async_get(entity_id)) and entity.device_id
        ]

    async def _clear_scope(
        self,
        conversation_memory,
        scope: str | None = None,
        target_id: str | None = None,
        physical_delete: bool = False,
        include_server: bool = True,
        only_inactive: bool = False,
    ) -> dict[str, list[str]]:
        """Clear memory locally and optionally on remote xAI servers.

        Returns:
            Dict with:
                - "all": All locally deleted IDs
                - "server_deleted": IDs that were deleted from xAI servers
        """
        desc = "unknown"
        output = {"all": [], "server_deleted": []}

        try:
            # Determine operation and get response IDs
            if physical_delete:
                result = await conversation_memory.async_delete(physical=True)
                desc = "delete_all"
            else:
                if not scope or not target_id:
                    raise_validation_error(
                        "scope and target_id required for non-physical delete"
                    )
                if scope not in (MEMORY_SCOPE_USER, MEMORY_SCOPE_DEVICE):
                    raise_validation_error(
                        f"scope must be '{MEMORY_SCOPE_USER}' or '{MEMORY_SCOPE_DEVICE}'"
                    )

                if only_inactive:
                    result = await conversation_memory.async_cleanup_expired(
                        scope=scope, identifier=target_id
                    )
                    desc = f"{scope}:{target_id} (inactive only)"
                else:
                    result = await conversation_memory.async_delete(
                        scope=scope, identifier=target_id
                    )
                    desc = f"{scope}:{target_id}"

            # Copy local deletion count
            output["all"] = result.get("all", [])

            # Perform remote deletion only for server-stored conversations
            server_ids = result.get("server_stored", [])
            if include_server and server_ids:
                deleted = await self.gateway.async_delete_remote_completions(server_ids)
                output["server_deleted"] = server_ids
                LOGGER.debug(
                    "Memory cleared: %s (local: %d, server: %d/%d deleted)",
                    desc,
                    len(output["all"]),
                    deleted,
                    len(server_ids),
                )
            elif output["all"]:
                LOGGER.debug(
                    "Memory cleared: %s (local: %d, server: skipped)",
                    desc,
                    len(output["all"]),
                )
            else:
                LOGGER.debug("Memory cleared: %s (no entries found)", desc)

        except (OSError, PermissionError) as err:
            LOGGER.warning("clear_memory: storage access denied: %s", err)
        except Exception as err:
            LOGGER.error("clear_memory failed for %s: %s", desc, err)
            raise

        return output
