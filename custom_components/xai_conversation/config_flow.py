"""Config flow for xAI Conversation integration."""

from __future__ import annotations

# Standard library imports
from typing import Any

# Third-party imports
import voluptuous as vol

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigEntryState,
    ConfigFlow,
    ConfigFlowResult,
    ConfigSubentryFlow,
    SubentryFlowResult,
    OptionsFlow,
)
from homeassistant.const import CONF_API_KEY
from homeassistant.core import callback as ha_callback
from homeassistant.helpers import selector

from .xai_gateway import XAIGateway
from .helpers import ExtendedToolsRegistry

from .const import (
    # Configuration keys
    CONF_ALLOW_SMART_HOME_CONTROL,
    CONF_API_HOST,
    CONF_ASSISTANT_NAME,
    CONF_CHAT_MODEL,
    CONF_EXTENDED_TOOLS_YAML,
    CONF_IMAGE_MODEL,
    CONF_LIVE_SEARCH,
    CONF_LOCATION_CONTEXT,
    LIVE_SEARCH_OPTIONS,
    CONF_MAX_TOKENS,
    CONF_MEMORY_CLEANUP_INTERVAL_HOURS,
    CONF_MEMORY_DEVICE_TTL_HOURS,
    CONF_MEMORY_USER_TTL_HOURS,
    CONF_MEMORY_REMOTE_DELETE,
    CONF_PRICING_UPDATE_INTERVAL_HOURS,
    CONF_AI_TASK_PROMPT,
    CONF_PROMPT_TOOLS,
    CONF_PROMPT_PIPELINE,
    CONF_REASONING_EFFORT,
    CONF_SEND_USER_NAME,
    CONF_SHOW_CITATIONS,
    CONF_STORE_MESSAGES,
    CONF_TEMPERATURE,
    CONF_TOKENS_PER_MILLION,
    CONF_TOP_P,
    CONF_USE_EXTENDED_TOOLS,
    CONF_USE_INTELLIGENT_PIPELINE,
    CONF_VISION_MODEL,
    CONF_VISION_PROMPT,
    CONF_XAI_PRICING_CONVERSION_FACTOR,
    CONF_ZDR,
    # Default names
    DEFAULT_AI_TASK_NAME,
    DEFAULT_API_HOST,
    DEFAULT_CONVERSATION_NAME,
    DEFAULT_MANUFACTURER,
    DEFAULT_SENSORS_NAME,
    # Other constants
    DOMAIN,
    LOGGER,
    MEMORY_DEFAULTS,
    REASONING_EFFORT_MODELS,
    RECOMMENDED_AI_TASK_OPTIONS,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_PIPELINE_OPTIONS,
    RECOMMENDED_REASONING_EFFORT,
    RECOMMENDED_SENSORS_OPTIONS,
    RECOMMENDED_ZDR_MODEL,
    SUPPORTED_MODELS,
)


# =============================================================================
# Selector Helpers - Reduce boilerplate for common selector patterns
# =============================================================================


def _model_selector() -> selector.SelectSelector:
    """Dropdown selector for supported models."""
    return selector.SelectSelector(
        selector.SelectSelectorConfig(
            options=SUPPORTED_MODELS,
            mode=selector.SelectSelectorMode.DROPDOWN,
        )
    )


def _live_search_selector() -> selector.SelectSelector:
    """Dropdown selector for live search options."""
    return selector.SelectSelector(
        selector.SelectSelectorConfig(
            options=LIVE_SEARCH_OPTIONS,
            mode=selector.SelectSelectorMode.DROPDOWN,
        )
    )


def _reasoning_effort_selector() -> selector.SelectSelector:
    """Dropdown selector for reasoning effort levels."""
    return selector.SelectSelector(
        selector.SelectSelectorConfig(
            options=["low", "medium", "high", "max"],
            mode=selector.SelectSelectorMode.DROPDOWN,
        )
    )


def _number_box(
    min_val: float,
    max_val: float,
    step: float = 1,
    unit: str | None = None,
) -> selector.NumberSelector:
    """Number input box with optional unit."""
    config = selector.NumberSelectorConfig(
        min=min_val,
        max=max_val,
        step=step,
        mode=selector.NumberSelectorMode.BOX,
    )
    if unit:
        config = selector.NumberSelectorConfig(
            min=min_val,
            max=max_val,
            step=step,
            mode=selector.NumberSelectorMode.BOX,
            unit_of_measurement=unit,
        )
    return selector.NumberSelector(config)


# Commonly used selectors (immutable, safe to reuse)
BOOL_SELECTOR = selector.BooleanSelector()
TEMPLATE_SELECTOR = selector.TemplateSelector()


class XAIConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for xAI Conversation."""

    VERSION = 2

    def __init__(self) -> None:
        """Initialize the config flow."""
        self._user_input: dict[str, Any] | None = None
        self._validate_info: dict[str, Any] | None = None

    def _build_user_schema(self, defaults: dict[str, Any] | None = None) -> vol.Schema:
        """Build schema for user step, with optional defaults from previous input."""
        d = defaults or {}
        R = RECOMMENDED_PIPELINE_OPTIONS
        return vol.Schema(
            {
                vol.Required(CONF_API_KEY, default=d.get(CONF_API_KEY, "")): str,
                vol.Optional(
                    CONF_API_HOST, default=d.get(CONF_API_HOST, R[CONF_API_HOST])
                ): str,
                vol.Optional(
                    CONF_ASSISTANT_NAME,
                    default=d.get(CONF_ASSISTANT_NAME, R[CONF_ASSISTANT_NAME]),
                ): str,
                vol.Optional(
                    CONF_LIVE_SEARCH,
                    default=d.get(CONF_LIVE_SEARCH, R[CONF_LIVE_SEARCH]),
                ): _live_search_selector(),
                vol.Optional(
                    CONF_SEND_USER_NAME,
                    default=d.get(CONF_SEND_USER_NAME, R[CONF_SEND_USER_NAME]),
                ): BOOL_SELECTOR,
            }
        )

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        if user_input is None:
            return self.async_show_form(
                step_id="user",
                data_schema=self._build_user_schema(),
            )

        errors: dict[str, str] = {}

        try:
            # Validate API key via XAIGateway (handles real API call)
            info = await XAIGateway.async_validate_api_key(
                api_key=user_input[CONF_API_KEY],
                api_host=user_input.get(CONF_API_HOST),
            )
        except ValueError as err:
            errors["base"] = str(err)
        except Exception:  # pylint: disable=broad-except
            LOGGER.exception("Unexpected exception during API key validation")
            errors["base"] = "unknown"
        else:
            # Store data for the next step
            self._user_input = user_input
            self._validate_info = info
            return await self.async_step_confirm()

        return self.async_show_form(
            step_id="user",
            data_schema=self._build_user_schema(user_input),
            errors=errors,
        )

    async def async_step_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the confirmation step."""
        if user_input is not None:
            return self._async_create_entry_from_input()

        return self.async_show_form(
            step_id="confirm",
            description_placeholders={
                "name": self._validate_info["name"] if self._validate_info else "N/A",
                "team_id": self._validate_info["team_id"]
                if self._validate_info
                else "N/A",
            },
        )

    def _async_create_entry_from_input(self) -> ConfigFlowResult:
        """Create the config entry from validated input."""
        user_input = self._user_input
        if not user_input:
            return self.async_abort(reason="unknown")

        # Ensure API Host is valid (not empty), fallback to default
        if not user_input.get(CONF_API_HOST, "").strip():
            user_input[CONF_API_HOST] = DEFAULT_API_HOST

        # Default to Intelligent Pipeline for conversation subentry
        conv_defaults = RECOMMENDED_PIPELINE_OPTIONS.copy()

        # Override API host only if user provided a custom one
        user_api_host = user_input.get(CONF_API_HOST)
        if (
            user_api_host
            and user_api_host != RECOMMENDED_PIPELINE_OPTIONS[CONF_API_HOST]
        ):
            conv_defaults[CONF_API_HOST] = user_api_host

        # Set live search from user input
        conv_defaults[CONF_LIVE_SEARCH] = user_input.get(
            CONF_LIVE_SEARCH, RECOMMENDED_PIPELINE_OPTIONS[CONF_LIVE_SEARCH]
        )

        # Propagate assistant name if provided
        user_assistant_name = user_input.get(CONF_ASSISTANT_NAME)
        if user_assistant_name:
            conv_defaults[CONF_ASSISTANT_NAME] = user_assistant_name

        # Propagate send user name toggle
        conv_defaults[CONF_SEND_USER_NAME] = user_input.get(
            CONF_SEND_USER_NAME, RECOMMENDED_PIPELINE_OPTIONS[CONF_SEND_USER_NAME]
        )

        # Add memory configuration to entry data (shared by all entities)
        entry_data = user_input.copy()
        entry_data.update(MEMORY_DEFAULTS)

        # Prepare AI Task defaults with custom API host if provided
        ai_task_defaults = RECOMMENDED_AI_TASK_OPTIONS.copy()
        if user_api_host and user_api_host != ai_task_defaults.get(CONF_API_HOST):
            ai_task_defaults[CONF_API_HOST] = user_api_host

        # Prepare Sensors defaults with custom API host (sensors inherit global but keep for consistency)
        sensors_defaults = RECOMMENDED_SENSORS_OPTIONS.copy()
        if user_api_host:
            sensors_defaults[CONF_API_HOST] = user_api_host

        return self.async_create_entry(
            title=DEFAULT_MANUFACTURER,
            data=entry_data,
            subentries=[
                {
                    "subentry_type": "conversation",
                    "data": conv_defaults,
                    "title": DEFAULT_CONVERSATION_NAME,
                    "unique_id": f"{DOMAIN}:conversation",
                },
                {
                    "subentry_type": "ai_task",
                    "data": ai_task_defaults,
                    "title": DEFAULT_AI_TASK_NAME,
                    "unique_id": f"{DOMAIN}:ai_task",
                },
                {
                    "subentry_type": "sensors",
                    "data": {
                        "name": DEFAULT_SENSORS_NAME,
                        **sensors_defaults,
                    },
                    "title": DEFAULT_SENSORS_NAME,
                    "unique_id": f"{DOMAIN}:sensors",
                },
            ],
        )

    @staticmethod
    @ha_callback
    def async_get_options_flow(
        config_entry: ConfigEntry,
    ) -> "XAIIntegrationOptionsFlow":
        """Get the options flow for this integration."""
        return XAIIntegrationOptionsFlow()

    @classmethod
    @ha_callback
    def async_get_supported_subentry_types(
        cls, config_entry: ConfigEntry
    ) -> dict[str, type[ConfigSubentryFlow]]:
        """Return subentries supported by this integration."""
        return {
            "conversation": XAIConversationOptionsFlow,
            "ai_task": XAIAITaskOptionsFlow,
            "sensors": XAISensorsOptionsFlow,
        }


class XAIIntegrationOptionsFlow(OptionsFlow):
    """Handle options flow for xAI integration (main entry, not subentries)."""

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage memory configuration options."""
        errors = {}

        if user_input is not None:
            # Ensure API Host is valid (not empty), fallback to default
            if CONF_API_HOST in user_input and not user_input[CONF_API_HOST].strip():
                user_input[CONF_API_HOST] = DEFAULT_API_HOST

            # Get current and new values for extended tools toggle
            current_use_extended = self.config_entry.data.get(
                CONF_USE_EXTENDED_TOOLS, False
            )
            new_use_extended = user_input.get(CONF_USE_EXTENDED_TOOLS, False)

            # If YAML is provided, validate it
            yaml_config = user_input.get(CONF_EXTENDED_TOOLS_YAML)
            if new_use_extended and yaml_config:
                is_valid, error_msg = ExtendedToolsRegistry.validate_yaml(yaml_config)
                if not is_valid:
                    errors["base"] = "invalid_yaml"
                    # We can pass the specific error message to the logger or find a way to show it
                    # For now, let's log it and show generic error
                    LOGGER.error("Extended tools YAML validation failed: %s", error_msg)
                    # Ideally we could use placeholders in strings.json if we had one for this error

            if not errors:
                # Reload form if extended tools toggle changed
                if current_use_extended != new_use_extended:
                    # Update data temporarily and reload form
                    new_data = {**self.config_entry.data, **user_input}
                    self.hass.config_entries.async_update_entry(
                        self.config_entry, data=new_data
                    )
                    return await self.async_step_init()

                # Update entry data with new settings
                new_data = {**self.config_entry.data, **user_input}
                self.hass.config_entries.async_update_entry(
                    self.config_entry, data=new_data
                )

                # Propagate api_host to subentries only if it actually changed
                new_host = user_input.get(CONF_API_HOST)
                old_host = self.config_entry.data.get(CONF_API_HOST)
                if new_host and new_host != old_host:
                    updated_count = 0
                    for subentry in self.config_entry.subentries.values():
                        if subentry.data.get(CONF_API_HOST) != new_host:
                            new_sub_data = dict(subentry.data)
                            new_sub_data[CONF_API_HOST] = new_host
                            self.hass.config_entries.async_update_subentry(
                                self.config_entry, subentry, data=new_sub_data
                            )
                            updated_count += 1
                    if updated_count:
                        LOGGER.debug(
                            "Propagated API Host change to %d subentries", updated_count
                        )

                return self.async_create_entry(title="", data={})

        # Load current values
        data = self.config_entry.data
        M = MEMORY_DEFAULTS

        def _get(key: str, fallback: dict = M):
            """Get value from user_input, data, or fallback."""
            if user_input:
                return user_input.get(key, data.get(key, fallback.get(key)))
            return data.get(key, fallback.get(key))

        use_extended = _get(CONF_USE_EXTENDED_TOOLS, {CONF_USE_EXTENDED_TOOLS: False})

        # Build schema
        schema_dict = {
            vol.Optional(
                CONF_API_HOST,
                description={"suggested_value": data.get(CONF_API_HOST, "")},
            ): str,
            # Memory settings
            vol.Optional(
                CONF_MEMORY_USER_TTL_HOURS, default=_get(CONF_MEMORY_USER_TTL_HOURS)
            ): _number_box(24, 24 * 30, 24, "hours"),
            vol.Optional(
                CONF_MEMORY_DEVICE_TTL_HOURS, default=_get(CONF_MEMORY_DEVICE_TTL_HOURS)
            ): _number_box(24, 24 * 30, 24, "hours"),
            vol.Optional(
                CONF_MEMORY_REMOTE_DELETE, default=_get(CONF_MEMORY_REMOTE_DELETE)
            ): BOOL_SELECTOR,
            vol.Optional(
                CONF_MEMORY_CLEANUP_INTERVAL_HOURS,
                default=_get(CONF_MEMORY_CLEANUP_INTERVAL_HOURS),
            ): _number_box(0.25, 168, 1, "hours"),
            # Extended Tools
            vol.Optional(CONF_USE_EXTENDED_TOOLS, default=use_extended): BOOL_SELECTOR,
        }

        # Show YAML editor if enabled
        if use_extended:
            schema_dict[
                vol.Optional(
                    CONF_EXTENDED_TOOLS_YAML, default=_get(CONF_EXTENDED_TOOLS_YAML, {})
                )
            ] = TEMPLATE_SELECTOR

        return self.async_show_form(
            step_id="init", data_schema=vol.Schema(schema_dict), errors=errors
        )


class XAIOptionsFlowBase(ConfigSubentryFlow):
    """Base class for xAI subentry options flows."""

    options: dict[str, Any]

    @property
    def _is_new(self) -> bool:
        """Return if this is a new subentry."""
        return self.source == "user"

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Add a subentry."""
        self.options = self._get_default_options()
        return await self.async_step_init()

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Handle reconfiguration of a subentry."""
        self.options = self._get_reconfigure_subentry().data.copy()
        return await self.async_step_init()

    def _get_default_options(self) -> dict[str, Any]:
        """Get default options for this subentry type. Must be implemented by subclasses."""
        raise NotImplementedError

    def _get_default_name(self) -> str:
        """Get default name for this subentry type. Must be implemented by subclasses."""
        raise NotImplementedError

    async def _async_save(self, user_input: dict[str, Any]) -> SubentryFlowResult:
        """Update options and save."""
        self.options.update(user_input)
        options = self.options.copy()

        if self._is_new:
            return self.async_create_entry(
                title=options.pop("name", self._get_default_name()),
                data=options,
            )
        return self.async_update_and_abort(
            self._get_entry(),
            self._get_reconfigure_subentry(),
            data=options,
        )

    def _get_base_schema(self) -> dict:
        """Get common schema fields."""
        schema = {}
        if self._is_new:
            schema[vol.Required("name", default=self._get_default_name())] = str
        return schema

    def _opt(self, key: str, fallback: dict | None = None) -> vol.Optional:
        """Create vol.Optional with default from self.options, falling back to fallback dict."""
        fallback = fallback or {}
        value = self.options.get(key)
        if value is None:
            value = fallback.get(key)
        return vol.Optional(key, default=value)

    def _opt_tpl(self, key: str, fallback: dict | None = None) -> vol.Optional:
        """Create vol.Optional for template fields using suggested_value."""
        fallback = fallback or {}
        value = self.options.get(key)
        if value is None:
            value = fallback.get(key, "")
        return vol.Optional(
            key,
            description={"suggested_value": value},
        )

    def _needs_reload_for_model(self, model: str, new_model: str) -> bool:
        """Check if form needs reload based on model change (reasoning features)."""
        return (model in REASONING_EFFORT_MODELS) != (
            new_model in REASONING_EFFORT_MODELS
        )


class XAICoreLLMOptionsFlow(XAIOptionsFlowBase):
    """Base class for LLM-based subentries (Conversation, AI Task, Code Fast)."""

    def _get_llm_schema_fields(self, model: str, include_model: bool = True) -> dict:
        """Get common LLM schema fields (max_tokens, temperature, top_p, reasoning, optionally model)."""
        fields = {}
        if include_model:
            fields[vol.Optional(CONF_CHAT_MODEL, default=model)] = _model_selector()
        fields.update(
            {
                self._opt(CONF_MAX_TOKENS): _number_box(1, 8192),
                self._opt(CONF_TEMPERATURE): _number_box(0.0, 2.0, 0.1),
                self._opt(CONF_TOP_P): _number_box(0.0, 1.0, 0.05),
            }
        )
        if model in REASONING_EFFORT_MODELS:
            fields[
                self._opt(
                    CONF_REASONING_EFFORT,
                    {CONF_REASONING_EFFORT: RECOMMENDED_REASONING_EFFORT},
                )
            ] = _reasoning_effort_selector()
        return fields

    async def _handle_model_reload(
        self, user_input: dict, current_model: str
    ) -> SubentryFlowResult | None:
        """Handle dynamic reload when model changes (affecting reasoning fields)."""
        new_model = user_input.get(CONF_CHAT_MODEL, current_model)
        if self._needs_reload_for_model(current_model, new_model):
            self.options.update(user_input)
            return await self.async_step_init()
        return None


class XAIConversationOptionsFlow(XAICoreLLMOptionsFlow):
    """Options flow for conversation subentries."""

    def _get_default_options(self) -> dict[str, Any]:
        return RECOMMENDED_PIPELINE_OPTIONS.copy()

    def _get_default_name(self) -> str:
        return DEFAULT_CONVERSATION_NAME

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Manage the options."""
        if self._get_entry().state != ConfigEntryState.LOADED:
            return self.async_abort(reason="entry_not_loaded")

        options = self.options
        use_pipeline = options.get(CONF_USE_INTELLIGENT_PIPELINE, True)
        model = options.get(
            CONF_CHAT_MODEL, RECOMMENDED_PIPELINE_OPTIONS[CONF_CHAT_MODEL]
        )

        if user_input is not None:
            # Detection logic for dynamic reload
            new_use_pipeline = user_input.get(
                CONF_USE_INTELLIGENT_PIPELINE, use_pipeline
            )

            # 1. Check for pipeline toggle reload
            if use_pipeline != new_use_pipeline:
                options.update(user_input)
                return await self.async_step_init()

            # 2. Check for ZDR vs Store Messages Mutual Exclusivity & Auto Model Switch
            new_zdr = user_input.get(CONF_ZDR, False)
            current_zdr = options.get(CONF_ZDR, False)
            new_store = user_input.get(CONF_STORE_MESSAGES, False)
            current_store = options.get(CONF_STORE_MESSAGES, False)

            if new_zdr != current_zdr:
                if new_zdr:
                    user_input[CONF_STORE_MESSAGES] = False
                    if (
                        user_input.get(CONF_CHAT_MODEL, model)
                        not in REASONING_EFFORT_MODELS
                    ):
                        user_input[CONF_CHAT_MODEL] = RECOMMENDED_ZDR_MODEL
                        LOGGER.debug(
                            "[config] ZDR enabled: auto-switching to reasoning"
                        )
                else:
                    user_input[CONF_CHAT_MODEL] = RECOMMENDED_CHAT_MODEL
                    LOGGER.debug("[config] ZDR disabled: reverting to fast model")

                options.update(user_input)
                return await self.async_step_init()

            if new_store != current_store and new_store:
                # If enabling store_messages, disable ZDR and revert model
                user_input[CONF_ZDR] = False
                user_input[CONF_CHAT_MODEL] = RECOMMENDED_CHAT_MODEL
                LOGGER.debug(
                    "[config] Memory enabled: disabling ZDR and reverting model"
                )

                options.update(user_input)
                return await self.async_step_init()

            # 3. Check for model-based reload (reasoning fields) via helper
            if reload_result := await self._handle_model_reload(user_input, model):
                return reload_result

            return await self._async_save(user_input)

        # Build schema
        R = RECOMMENDED_PIPELINE_OPTIONS
        step_schema = self._get_base_schema()
        step_schema.update(
            {
                self._opt(CONF_STORE_MESSAGES, R): BOOL_SELECTOR,
                self._opt(CONF_ZDR, R): BOOL_SELECTOR,
                vol.Optional(
                    CONF_USE_INTELLIGENT_PIPELINE, default=use_pipeline
                ): BOOL_SELECTOR,
                self._opt(CONF_ALLOW_SMART_HOME_CONTROL, R): BOOL_SELECTOR,
                self._opt(CONF_USE_EXTENDED_TOOLS, R): BOOL_SELECTOR,
                self._opt(CONF_SEND_USER_NAME, R): BOOL_SELECTOR,
                self._opt(CONF_SHOW_CITATIONS, R): BOOL_SELECTOR,
                self._opt(CONF_ASSISTANT_NAME, R): str,
                vol.Optional(
                    CONF_LOCATION_CONTEXT,
                    description={"suggested_value": options.get(CONF_LOCATION_CONTEXT)},
                ): str,
                self._opt(CONF_LIVE_SEARCH, R): _live_search_selector(),
            }
        )

        # Mode specific prompt field
        prompt_key = CONF_PROMPT_PIPELINE if use_pipeline else CONF_PROMPT_TOOLS
        step_schema[self._opt_tpl(prompt_key)] = TEMPLATE_SELECTOR

        # Common LLM fields
        step_schema.update(self._get_llm_schema_fields(model))

        return self.async_show_form(step_id="init", data_schema=vol.Schema(step_schema))


class XAIAITaskOptionsFlow(XAICoreLLMOptionsFlow):
    """Options flow for AI Task subentries."""

    def _get_default_options(self) -> dict[str, Any]:
        return RECOMMENDED_AI_TASK_OPTIONS.copy()

    def _get_default_name(self) -> str:
        return DEFAULT_AI_TASK_NAME

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        if self._get_entry().state != ConfigEntryState.LOADED:
            return self.async_abort(reason="entry_not_loaded")

        R = RECOMMENDED_AI_TASK_OPTIONS
        model = self.options.get(CONF_CHAT_MODEL, R[CONF_CHAT_MODEL])

        if user_input is not None:
            if reload_result := await self._handle_model_reload(user_input, model):
                return reload_result
            return await self._async_save(user_input)

        step_schema = self._get_base_schema()
        step_schema.update(
            {
                self._opt_tpl(CONF_AI_TASK_PROMPT, R): TEMPLATE_SELECTOR,
                self._opt_tpl(CONF_VISION_PROMPT, R): TEMPLATE_SELECTOR,
                vol.Optional(CONF_CHAT_MODEL, default=model): _model_selector(),
                self._opt(CONF_IMAGE_MODEL, R): _model_selector(),
                self._opt(CONF_VISION_MODEL, R): _model_selector(),
            }
        )
        # Add LLM fields without model (already added above)
        step_schema.update(self._get_llm_schema_fields(model, include_model=False))

        return self.async_show_form(step_id="init", data_schema=vol.Schema(step_schema))


class XAISensorsOptionsFlow(XAIOptionsFlowBase):
    """Options flow for sensors subentries."""

    def _get_default_options(self) -> dict[str, Any]:
        return {"name": DEFAULT_SENSORS_NAME, **RECOMMENDED_SENSORS_OPTIONS}

    def _get_default_name(self) -> str:
        return DEFAULT_SENSORS_NAME

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        if self._get_entry().state != ConfigEntryState.LOADED:
            return self.async_abort(reason="entry_not_loaded")

        if user_input is not None:
            return await self._async_save(user_input)

        R = RECOMMENDED_SENSORS_OPTIONS
        step_schema = self._get_base_schema()
        step_schema.update(
            {
                self._opt(CONF_TOKENS_PER_MILLION, R): _number_box(1, 10_000_000),
                self._opt(CONF_XAI_PRICING_CONVERSION_FACTOR, R): _number_box(
                    1.0, 100_000.0, 0.1
                ),
                self._opt(CONF_PRICING_UPDATE_INTERVAL_HOURS, R): _number_box(1, 168),
            }
        )

        return self.async_show_form(step_id="init", data_schema=vol.Schema(step_schema))
