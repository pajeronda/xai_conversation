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


# Centralized error helper functions
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


# Re-export HA exceptions for centralized access
__all__ = [
    # Custom xAI exceptions
    "XAIConnectionError",
    "XAIToolConversionError",
    "XAIConfigurationError",
    # Helper functions
    "raise_auth_error",
    "raise_communication_error",
    "raise_config_error",
    "raise_tool_error",
    "raise_validation_error",
    "raise_generic_error",
    "raise_config_not_ready",
    "handle_api_error",
    # Re-exported HA exceptions
    "HA_HomeAssistantError",
    "HA_ConfigEntryNotReady",
    "ServiceValidationError",
]
