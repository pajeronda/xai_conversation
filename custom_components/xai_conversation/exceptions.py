"""Exceptions for xAI conversation integration."""

from __future__ import annotations

import grpc
import time
from homeassistant.exceptions import (
    HomeAssistantError as HA_HomeAssistantError,
    ConfigEntryNotReady as HA_ConfigEntryNotReady,
    ServiceValidationError,
)
from .const import LOGGER


# ==============================================================================
# EXCEPTIONS: xAI Core
# ==============================================================================


class XAIConnectionError(HA_HomeAssistantError):
    """Error connecting to xAI service."""

    def __init__(self, message: str) -> None:
        """Initialize error."""
        super().__init__(f"xAI connection error: {message}")
        self.message = message

    def __str__(self) -> str:
        """Return string representation."""
        return f"Unable to connect to xAI service: {self.message}"

    @classmethod
    def from_grpc_error(cls, error: grpc.RpcError) -> XAIConnectionError:
        """Create XAIConnectionError from gRPC error with user-friendly message."""
        status_code = error.code()
        error_details = error.details()

        # Map gRPC status codes to user-friendly messages
        if status_code == grpc.StatusCode.UNAUTHENTICATED:
            LOGGER.error("Authentication failed: Invalid API key")
            message = (
                "Authentication failed. Please check your xAI API key configuration."
            )
        elif status_code == grpc.StatusCode.PERMISSION_DENIED:
            LOGGER.error("Permission denied: API key lacks required permissions")
            message = "Permission denied. Your API key may not have the required permissions for this operation."
        elif status_code == grpc.StatusCode.RESOURCE_EXHAUSTED:
            LOGGER.warning("Rate limit or quota exceeded")
            message = "Rate limit exceeded or quota depleted. Please try again later or check your xAI account usage."
        elif status_code == grpc.StatusCode.UNAVAILABLE:
            LOGGER.warning("xAI service temporarily unavailable")
            message = "The xAI service is temporarily unavailable. Please try again in a few moments."
        elif status_code == grpc.StatusCode.DEADLINE_EXCEEDED:
            LOGGER.warning("Request timeout")
            message = "Request timed out. The operation took longer than expected. Please try again."
        elif status_code == grpc.StatusCode.INTERNAL:
            LOGGER.error("Internal server error from xAI API")
            message = "Internal server error occurred. Please try again later."
        elif status_code == grpc.StatusCode.INVALID_ARGUMENT:
            LOGGER.error("Invalid request parameters: %s", error_details)
            message = "Invalid request parameters. Please check your configuration."
        elif status_code == grpc.StatusCode.NOT_FOUND:
            LOGGER.error("Model or endpoint not found: %s", error_details)
            message = "The requested model or endpoint was not found. Please check your model configuration."
        elif status_code == grpc.StatusCode.CANCELLED:
            LOGGER.info("Request was cancelled")
            message = "The request was cancelled. Please try again."
        elif status_code == grpc.StatusCode.DATA_LOSS:
            LOGGER.error(
                "Data loss error (likely failed to fetch image from URL): %s",
                error_details,
            )
            message = "Failed to fetch image from provided URL. Please check the image URL is accessible."
        else:
            LOGGER.error("Unknown gRPC error: %s - %s", status_code, error_details)
            message = (
                f"An error occurred while communicating with xAI: {status_code.name}"
            )

        return cls(message)


class XAIToolConversionError(HA_HomeAssistantError):
    """Error converting tools between HA and xAI formats."""

    def __init__(self, tool_name: str, message: str) -> None:
        """Initialize error."""
        super().__init__(f"Tool conversion error for '{tool_name}': {message}")
        self.tool_name = tool_name
        self.message = message

    def __str__(self) -> str:
        """Return string representation."""
        return f"Failed to convert tool '{self.tool_name}': {self.message}"


class XAIConfigurationError(HA_HomeAssistantError):
    """Error in xAI integration configuration."""

    def __init__(self, message: str) -> None:
        """Initialize error."""
        super().__init__(f"xAI configuration error: {message}")
        self.message = message

    def __str__(self) -> str:
        """Return string representation."""
        return f"Configuration error: {self.message}"


# ==============================================================================
# HELPER FUNCTIONS: xAI Error Handling
# ==============================================================================


def raise_auth_error(
    message: str = "Authentication failed. Please check your xAI API key.",
) -> None:
    """Raise authentication error with consistent messaging."""
    raise XAIConnectionError(message)


def raise_communication_error(
    message: str = "Communication error with xAI service. Please try again.",
) -> None:
    """Raise communication error with consistent messaging."""
    raise XAIConnectionError(message)


def raise_config_error(field: str, issue: str) -> None:
    """Raise configuration error with consistent messaging."""
    raise XAIConfigurationError(f"Invalid {field}: {issue}")


def raise_tool_error(tool_name: str, issue: str) -> None:
    """Raise tool conversion error with consistent messaging."""
    raise XAIToolConversionError(tool_name, issue)


def raise_validation_error(message: str) -> None:
    """Raise validation error using HA native exception."""
    raise ServiceValidationError(message)


def raise_generic_error(message: str) -> None:
    """Raise generic HA error with consistent messaging."""
    raise HA_HomeAssistantError(message)


def raise_config_not_ready(message: str) -> None:
    """Raise ConfigEntryNotReady with consistent messaging."""
    raise HA_ConfigEntryNotReady(message)


def handle_api_error(
    err: Exception, start_time: float, context: str = "API call"
) -> None:
    """Centralized error handling for xAI API calls with timing and classification.

    Args:
        err: The exception that was raised
        start_time: The time.time() when the API call started
        context: Description of the operation for logging (e.g., "tools API call", "pipeline call")

    Raises:
        XAIConnectionError: For gRPC and network errors
        Various exceptions: Based on error type classification
    """
    elapsed = time.time() - start_time

    # Handle gRPC errors specifically
    if isinstance(err, grpc.RpcError):
        LOGGER.error(
            "gRPC error in %s after %.2f seconds: %s (status: %s)",
            context,
            elapsed,
            err.details(),
            err.code(),
        )
        raise XAIConnectionError.from_grpc_error(err) from err

    # Handle other exceptions with error message classification
    error_msg = str(err)

    if "Tool has no" in error_msg and "field" in error_msg:
        LOGGER.error(
            "Tool validation error in %s after %.2f seconds: %s",
            context,
            elapsed,
            err,
            exc_info=True,
        )
        raise_tool_error(
            "tool_validation",
            "Tool configuration error. Please check the integration setup.",
        )
    elif "grpc" in error_msg.lower() or "channel" in error_msg.lower():
        LOGGER.error(
            "gRPC communication error in %s after %.2f seconds: %s",
            context,
            elapsed,
            err,
            exc_info=True,
        )
        raise_communication_error()
    elif "authentication" in error_msg.lower() or "api_key" in error_msg.lower():
        LOGGER.error(
            "Authentication error in %s after %.2f seconds: %s",
            context,
            elapsed,
            err,
            exc_info=True,
        )
        raise_auth_error()
    else:
        LOGGER.error(
            "Unexpected error in %s after %.2f seconds: %s",
            context,
            elapsed,
            err,
            exc_info=True,
        )
        raise_generic_error(
            f"An unexpected error occurred during {context}: {error_msg}"
        )


async def handle_response_not_found_error(
    err: Exception,
    attempt: int,
    memory,
    conv_key: str | None,
    context_id: str = "",
) -> bool:
    """Handle NOT_FOUND error for expired response_id on xAI server.

    When xAI returns NOT_FOUND for a previous_response_id, it means the
    conversation context has expired server-side. This function clears
    the local memory and signals to retry with a fresh conversation.

    Args:
        err: The exception that was raised
        attempt: Current attempt number (0-indexed)
        memory: Instance of MemoryManager
        conv_key: Conversation key for memory cleanup
        context_id: Optional context identifier for logging

    Returns:
        True if should retry (cleared memory), False if should re-raise error
    """
    # Check if it's a gRPC NOT_FOUND error (handles both sync and async)
    is_not_found = False
    try:
        from grpc import StatusCode
        from grpc._channel import _InactiveRpcError
        try:
            from grpc.aio import AioRpcError
        except ImportError:
            AioRpcError = _InactiveRpcError  # Fallback

        if isinstance(err, (_InactiveRpcError, AioRpcError)):
            if err.code() == StatusCode.NOT_FOUND:
                is_not_found = True
    except ImportError:
        pass

    if not is_not_found:
        return False

    # Only retry on first attempt
    if attempt != 0:
        return False

    # Log the retry
    context_prefix = f"[Context: {context_id}] " if context_id else ""
    LOGGER.warning(
        "%sConversation context not found on xAI server (expired response_id). "
        "Clearing local memory and retrying with fresh conversation.",
        context_prefix,
    )

    # Clear memory by key if available (assuming MemoryManager is used)
    if conv_key:
        try:
            await memory.async_clear_key(conv_key)
            LOGGER.debug("Cleared expired memory for conv_key=%s", conv_key)
            return True
        except Exception as clear_err:
            LOGGER.warning(
                "Failed to clear memory for conv_key=%s: %s", conv_key, clear_err
            )

    return True  # Signal to retry even if clear specific key failed, hoping a fresh start helps


# ==============================================================================
# EXCEPTIONS: Extended Tools (Legacy from Extended OpenAI Conversation)
# ==============================================================================


class EntityNotFound(HA_HomeAssistantError):
    """When referenced entity not found."""

    def __init__(self, entity_id: str) -> None:
        """Initialize error."""
        super().__init__(self, f"entity {entity_id} not found")
        self.entity_id = entity_id

    def __str__(self) -> str:
        """Return string representation."""
        return f"Unable to find entity {self.entity_id}"


class EntityNotExposed(HA_HomeAssistantError):
    """When referenced entity not exposed."""

    def __init__(self, entity_id: str) -> None:
        """Initialize error."""
        super().__init__(self, f"entity {entity_id} not exposed")
        self.entity_id = entity_id

    def __str__(self) -> str:
        """Return string representation."""
        return f"entity {self.entity_id} is not exposed"


class CallServiceError(HA_HomeAssistantError):
    """Error during service calling."""

    def __init__(self, domain: str, service: str, data: object) -> None:
        """Initialize error."""
        super().__init__(
            self,
            f"unable to call service {domain}.{service} with data {data}. One of 'entity_id', 'area_id', or 'device_id' is required",
        )
        self.domain = domain
        self.service = service
        self.data = data

    def __str__(self) -> str:
        """Return string representation."""
        return f"unable to call service {self.domain}.{self.service} with data {self.data}. One of 'entity_id', 'area_id', or 'device_id' is required"


class FunctionNotFound(HA_HomeAssistantError):
    """When referenced function not found."""

    def __init__(self, function: str) -> None:
        """Initialize error."""
        super().__init__(self, f"function '{function}' does not exist")
        self.function = function

    def __str__(self) -> str:
        """Return string representation."""
        return f"function '{self.function}' does not exist"


class NativeNotFound(HA_HomeAssistantError):
    """When native function not found."""

    def __init__(self, name: str) -> None:
        """Initialize error."""
        super().__init__(self, f"native function '{name}' does not exist")
        self.name = name

    def __str__(self) -> str:
        """Return string representation."""
        return f"native function '{self.name}' does not exist"


class InvalidFunction(HA_HomeAssistantError):
    """When function validation failed."""

    def __init__(self, function_name: str) -> None:
        """Initialize error."""
        super().__init__(
            self,
            f"failed to validate function `{function_name}`",
        )
        self.function_name = function_name

    def __str__(self) -> str:
        """Return string representation."""
        return f"failed to validate function `{self.function_name}` ({self.__cause__})"


# ==============================================================================
# EXPORTS: Public API
# ==============================================================================

__all__ = [
    # xAI custom exceptions
    "XAIConnectionError",
    "XAIToolConversionError",
    "XAIConfigurationError",
    # Extended Tools exceptions (legacy)
    "EntityNotFound",
    "EntityNotExposed",
    "CallServiceError",
    "FunctionNotFound",
    "NativeNotFound",
    "InvalidFunction",
    # xAI helper functions
    "raise_auth_error",
    "raise_communication_error",
    "raise_config_error",
    "raise_tool_error",
    "raise_validation_error",
    "raise_generic_error",
    "raise_config_not_ready",
    "handle_api_error",
    "handle_response_not_found_error",
    # Re-exported HA exceptions
    "HA_HomeAssistantError",
    "HA_ConfigEntryNotReady",
    "ServiceValidationError",
]
