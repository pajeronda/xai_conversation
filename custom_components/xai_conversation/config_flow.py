"""Config flow for xAI Conversation integration."""
from __future__ import annotations

# Standard library imports
from typing import Any

# Third-party imports
import voluptuous as vol

from homeassistant.config_entries import OptionsFlow

from .__init__ import (
    HA_ConfigSubentry, ha_callback, ha_llm, ha_device_registry,
    ha_entity_registry, ha_area_registry, HA_Entity,
    HA_AddConfigEntryEntitiesCallback, ha_conversation, ha_ai_task,
    ha_json_loads, CONF_API_KEY, CONF_LLM_HASS_API, HA_HomeAssistant, HA_ServiceCall,
    HA_ServiceResponse, HA_HomeAssistantError, HA_ConfigEntryNotReady,
    ConfigEntry, ConfigEntryState, ConfigFlow, ConfigFlowResult,
    ConfigSubentryFlow, SubentryFlowResult, selector, get_async_client,
    async_validate_api_key,
)

from .const import (
    CONF_API_HOST,
    CONF_ASSISTANT_NAME,
    CONF_CHAT_MODEL,
    CONF_LIVE_SEARCH,
    CONF_MAX_TOKENS,
    CONF_MEMORY_CLEANUP_INTERVAL_HOURS,
    CONF_ALLOW_SMART_HOME_CONTROL,
    CONF_MEMORY_USER_MAX_TURNS,
    CONF_MEMORY_USER_TTL_HOURS,
    CONF_MEMORY_DEVICE_MAX_TURNS,
    CONF_MEMORY_DEVICE_TTL_HOURS,
    CONF_PROMPT,
    CONF_PROMPT_PIPELINE,
    CONF_REASONING_EFFORT,
    CONF_STORE_MESSAGES,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_USE_INTELLIGENT_PIPELINE,
    DEFAULT_AI_TASK_NAME,
    DEFAULT_API_HOST,
    DEFAULT_CONVERSATION_NAME,
    DEFAULT_DEVICE_NAME,
    DEFAULT_GROK_CODE_FAST_NAME,
    DEFAULT_MODEL_PRICING,
    DEFAULT_SENSORS_NAME,
    DOMAIN,
    LOGGER,
    REASONING_EFFORT_MODELS,
    RECOMMENDED_AI_TASK_OPTIONS,
    RECOMMENDED_ASSISTANT_NAME,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_GROK_CODE_FAST_OPTIONS,
    RECOMMENDED_LIVE_SEARCH,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_MEMORY_CLEANUP_INTERVAL_HOURS,
    RECOMMENDED_MEMORY_USER_MAX_TURNS,
    RECOMMENDED_MEMORY_USER_TTL_HOURS,
    RECOMMENDED_MEMORY_DEVICE_MAX_TURNS,
    RECOMMENDED_MEMORY_DEVICE_TTL_HOURS,
    RECOMMENDED_PIPELINE_OPTIONS,
    RECOMMENDED_REASONING_EFFORT,
    RECOMMENDED_STORE_MESSAGES,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOOLS_OPTIONS,
    RECOMMENDED_TOP_P,
    SUPPORTED_MODELS,
)




class XAIConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for xAI Conversation."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        if user_input is None:
            return self.async_show_form(
                step_id="user",
                data_schema=vol.Schema(
                    {
                        vol.Required(CONF_API_KEY): str,
                        vol.Optional(CONF_API_HOST, default=RECOMMENDED_TOOLS_OPTIONS.get(CONF_API_HOST, DEFAULT_API_HOST)): str,
                        vol.Optional(CONF_ASSISTANT_NAME, default=RECOMMENDED_ASSISTANT_NAME): str,
                        vol.Optional(CONF_LIVE_SEARCH, default=RECOMMENDED_LIVE_SEARCH): selector.SelectSelector(
                            selector.SelectSelectorConfig(
                                options=["off", "auto", "on"],
                                mode=selector.SelectSelectorMode.DROPDOWN,
                            )
                        ),
                    }
                ),
            )

        errors: dict[str, str] = {}

        try:
            # Validate API key
            await async_validate_api_key(self.hass, user_input["api_key"])
        except ValueError as err:
            if "xai_sdk not installed" in str(err):
                errors["base"] = "missing_dependency"
            else:
                errors["base"] = "invalid_api_key"
        except Exception:  # pylint: disable=broad-except
            LOGGER.exception("Unexpected exception")
            errors["base"] = "unknown"
        else:
            # Default to Intelligent Pipeline for conversation subentry
            conv_defaults = RECOMMENDED_PIPELINE_OPTIONS.copy()

            # Override API host only if user provided a custom one
            user_api_host = user_input.get(CONF_API_HOST)
            if user_api_host and user_api_host != DEFAULT_API_HOST:
                conv_defaults[CONF_API_HOST] = user_api_host

            # Set live search from user input
            conv_defaults[CONF_LIVE_SEARCH] = user_input.get(CONF_LIVE_SEARCH, RECOMMENDED_LIVE_SEARCH)

            # Add memory configuration to entry data (shared by all entities)
            entry_data = user_input.copy()
            entry_data.update({
                CONF_MEMORY_USER_TTL_HOURS: RECOMMENDED_MEMORY_USER_TTL_HOURS,
                CONF_MEMORY_USER_MAX_TURNS: RECOMMENDED_MEMORY_USER_MAX_TURNS,
                CONF_MEMORY_DEVICE_TTL_HOURS: RECOMMENDED_MEMORY_DEVICE_TTL_HOURS,
                CONF_MEMORY_DEVICE_MAX_TURNS: RECOMMENDED_MEMORY_DEVICE_MAX_TURNS,
                CONF_MEMORY_CLEANUP_INTERVAL_HOURS: RECOMMENDED_MEMORY_CLEANUP_INTERVAL_HOURS,
            })

            return self.async_create_entry(
                title=DEFAULT_DEVICE_NAME,
                data=entry_data,
                subentries=[
                    {
                        "subentry_type": "conversation",
                        "data": conv_defaults,
                        "title": DEFAULT_CONVERSATION_NAME,
                        "unique_id": f"{DOMAIN}:conversation",
                    },
                    {
                        "subentry_type": "ai_task_data",
                        "data": RECOMMENDED_AI_TASK_OPTIONS,
                        "title": DEFAULT_AI_TASK_NAME,
                        "unique_id": f"{DOMAIN}:ai_task",
                    },
                    {
                        "subentry_type": "code_task",
                        "data": RECOMMENDED_GROK_CODE_FAST_OPTIONS,
                        "title": DEFAULT_GROK_CODE_FAST_NAME,
                        "unique_id": f"{DOMAIN}:grok_code_fast",
                    },
                    {
                        "subentry_type": "sensors",
                        "data": {"name": DEFAULT_SENSORS_NAME},
                        "title": DEFAULT_SENSORS_NAME,
                        "unique_id": f"{DOMAIN}:sensors",
                    },
                ],
            )

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_API_KEY): str,
                    vol.Optional(CONF_API_HOST, default=RECOMMENDED_TOOLS_OPTIONS.get(CONF_API_HOST, DEFAULT_API_HOST)): str,
                }
            ),
            errors=errors,
        )

    @staticmethod
    @ha_callback
    def async_get_options_flow(config_entry: ConfigEntry) -> "XAIIntegrationOptionsFlow":
        """Get the options flow for this integration."""
        return XAIIntegrationOptionsFlow()

    @classmethod
    @ha_callback
    def async_get_supported_subentry_types(
        cls, config_entry: ConfigEntry
    ) -> dict[str, type[ConfigSubentryFlow]]:
        """Return subentries supported by this integration."""
        return {
            "conversation": XAIOptionsFlow,
            "ai_task_data": XAIOptionsFlow,
            "code_task": XAIOptionsFlow,
            "sensors": XAIOptionsFlow,
        }


class XAIIntegrationOptionsFlow(OptionsFlow):
    """Handle options flow for xAI integration (main entry, not subentries)."""

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage memory configuration options."""
        if user_input is not None:
            # Update entry data with new memory settings
            new_data = {**self.config_entry.data, **user_input}
            self.hass.config_entries.async_update_entry(
                self.config_entry, data=new_data
            )
            return self.async_create_entry(title="", data={})

        # Build schema with current values
        schema = vol.Schema({
            vol.Optional(
                CONF_MEMORY_USER_TTL_HOURS,
                default=self.config_entry.data.get(
                    CONF_MEMORY_USER_TTL_HOURS, RECOMMENDED_MEMORY_USER_TTL_HOURS
                ),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=1, max=24*365, step=24, mode=selector.NumberSelectorMode.BOX,
                    unit_of_measurement="hours"
                )
            ),
            vol.Optional(
                CONF_MEMORY_USER_MAX_TURNS,
                default=self.config_entry.data.get(
                    CONF_MEMORY_USER_MAX_TURNS, RECOMMENDED_MEMORY_USER_MAX_TURNS
                ),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=1, max=10000, step=1, mode=selector.NumberSelectorMode.BOX
                )
            ),
            vol.Optional(
                CONF_MEMORY_DEVICE_TTL_HOURS,
                default=self.config_entry.data.get(
                    CONF_MEMORY_DEVICE_TTL_HOURS, RECOMMENDED_MEMORY_DEVICE_TTL_HOURS
                ),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=1, max=24*30, step=24, mode=selector.NumberSelectorMode.BOX,
                    unit_of_measurement="hours"
                )
            ),
            vol.Optional(
                CONF_MEMORY_DEVICE_MAX_TURNS,
                default=self.config_entry.data.get(
                    CONF_MEMORY_DEVICE_MAX_TURNS, RECOMMENDED_MEMORY_DEVICE_MAX_TURNS
                ),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=1, max=1000, step=1, mode=selector.NumberSelectorMode.BOX
                )
            ),
            vol.Optional(
                CONF_MEMORY_CLEANUP_INTERVAL_HOURS,
                default=self.config_entry.data.get(
                    CONF_MEMORY_CLEANUP_INTERVAL_HOURS, RECOMMENDED_MEMORY_CLEANUP_INTERVAL_HOURS
                ),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=0.25, max=168, step=1, mode=selector.NumberSelectorMode.BOX,
                    unit_of_measurement="hours"
                )
            ),
        })

        return self.async_show_form(step_id="init", data_schema=schema)


class XAIOptionsFlow(ConfigSubentryFlow):
    """xAI config flow options handler."""

    options: dict[str, Any]

    @property
    def _is_new(self) -> bool:
        """Return if this is a new subentry."""
        return self.source == "user"

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Add a subentry."""
        if self._subentry_type == "ai_task_data":
            self.options = RECOMMENDED_AI_TASK_OPTIONS.copy()
        elif self._subentry_type == "code_task":
            self.options = RECOMMENDED_GROK_CODE_FAST_OPTIONS.copy()
        elif self._subentry_type == "sensors":
            # Sensors subentry has no configurable options
            self.options = {"name": DEFAULT_SENSORS_NAME}
        else:
            # Default to Intelligent Pipeline options when adding a conversation subentry
            self.options = RECOMMENDED_PIPELINE_OPTIONS.copy()
        return await self.async_step_init()

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Handle reconfiguration of a subentry."""
        self.options = self._get_reconfigure_subentry().data.copy()
        return await self.async_step_init()

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Manage the options."""
        # abort if entry is not loaded
        if self._get_entry().state != ConfigEntryState.LOADED:
            return self.async_abort(reason="entry_not_loaded")

        options = self.options

        # Build schema - start with name field for new entries
        step_schema = {}

        if self._is_new:
            if self._subentry_type == "ai_task_data":
                default_name = DEFAULT_AI_TASK_NAME
            elif self._subentry_type == "code_task":
                default_name = DEFAULT_GROK_CODE_FAST_NAME
            elif self._subentry_type == "sensors":
                default_name = DEFAULT_SENSORS_NAME
            else:
                default_name = DEFAULT_CONVERSATION_NAME
            step_schema[vol.Required("name", default=default_name)] = str

        # Sensors subentry now has per-model pricing configuration
        if self._subentry_type == "sensors":
            # Generate pricing fields dynamically for each model in DEFAULT_MODEL_PRICING
            for model_name, pricing in DEFAULT_MODEL_PRICING.items():
                # Create config keys like "grok_4_input_price" (replace - with _)
                model_key = model_name.replace("-", "_")
                input_key = f"{model_key}_input_price"
                cached_input_key = f"{model_key}_cached_input_price"
                output_key = f"{model_key}_output_price"

                step_schema.update({
                    vol.Optional(
                        input_key,
                        default=options.get(input_key, pricing["input"]),
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=0.0, max=100.0, step=0.01, mode=selector.NumberSelectorMode.BOX
                        )
                    ),
                    vol.Optional(
                        cached_input_key,
                        default=options.get(cached_input_key, pricing["cached_input"]),
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=0.0, max=100.0, step=0.01, mode=selector.NumberSelectorMode.BOX
                        )
                    ),
                    vol.Optional(
                        output_key,
                        default=options.get(output_key, pricing["output"]),
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=0.0, max=100.0, step=0.01, mode=selector.NumberSelectorMode.BOX
                        )
                    ),
                })

            if user_input is not None:
                options.update(user_input)
                if self._is_new:
                    return self.async_create_entry(
                        title=options.pop("name", DEFAULT_SENSORS_NAME),
                        data=options,
                    )
                return self.async_update_and_abort(
                    self._get_entry(),
                    self._get_reconfigure_subentry(),
                    data=options,
                )

            return self.async_show_form(
                step_id="init",
                data_schema=vol.Schema(step_schema),
            )

        # Conversation subentry supports two modes via the toggle CONF_USE_INTELLIGENT_PIPELINE
        model = options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)

        if self._subentry_type == "conversation":
            # Unified parameters (appear for both pipeline and tools modes)
            step_schema.update({
                vol.Optional(
                    CONF_USE_INTELLIGENT_PIPELINE,
                    default=options.get(CONF_USE_INTELLIGENT_PIPELINE, True),
                ): selector.BooleanSelector(),
                vol.Optional(
                    CONF_ALLOW_SMART_HOME_CONTROL,
                    default=options.get(CONF_ALLOW_SMART_HOME_CONTROL, True),
                ): selector.BooleanSelector(),
                vol.Optional(
                    CONF_STORE_MESSAGES,
                    default=options.get(CONF_STORE_MESSAGES, RECOMMENDED_STORE_MESSAGES),
                ): selector.BooleanSelector(),
                vol.Optional(
                    CONF_ASSISTANT_NAME,
                    default=options.get(CONF_ASSISTANT_NAME, RECOMMENDED_ASSISTANT_NAME),
                ): str,
                vol.Optional(
                    CONF_LIVE_SEARCH,
                    default=options.get(CONF_LIVE_SEARCH, RECOMMENDED_LIVE_SEARCH),
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=["off", "auto", "on"],
                        mode=selector.SelectSelectorMode.DROPDOWN,
                    )
                ),
            })
            
            use_pipeline = options.get(CONF_USE_INTELLIGENT_PIPELINE, True)

            if use_pipeline:
                # Intelligent Pipeline default config
                step_schema.update({
                    vol.Optional(
                        CONF_PROMPT_PIPELINE,
                        description={"suggested_value": options.get(CONF_PROMPT_PIPELINE, "")},
                    ): selector.TemplateSelector(),
                    vol.Optional(
                        CONF_API_HOST,
                        default=options.get(CONF_API_HOST, DEFAULT_API_HOST),
                    ): str,
                    vol.Optional(
                        CONF_CHAT_MODEL,
                        default=model,
                    ): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=SUPPORTED_MODELS,
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Optional(
                        CONF_MAX_TOKENS,
                        default=options.get(CONF_MAX_TOKENS, RECOMMENDED_PIPELINE_OPTIONS.get(CONF_MAX_TOKENS)),
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=1, max=8192, mode=selector.NumberSelectorMode.BOX
                        )
                    ),
                    vol.Optional(
                        CONF_TEMPERATURE,
                        default=options.get(CONF_TEMPERATURE, RECOMMENDED_PIPELINE_OPTIONS.get(CONF_TEMPERATURE)),
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=0.0, max=2.0, step=0.1, mode=selector.NumberSelectorMode.BOX
                        )
                    ),
                    vol.Optional(
                        CONF_TOP_P,
                        default=options.get(CONF_TOP_P, RECOMMENDED_PIPELINE_OPTIONS.get(CONF_TOP_P)),
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=0.0, max=1.0, step=0.05, mode=selector.NumberSelectorMode.BOX
                        )
                    ),
                })

                if model in REASONING_EFFORT_MODELS:
                    step_schema.update({
                        vol.Optional(
                            CONF_REASONING_EFFORT,
                            default=options.get(CONF_REASONING_EFFORT, RECOMMENDED_REASONING_EFFORT),
                        ): selector.SelectSelector(
                            selector.SelectSelectorConfig(
                                options=["low", "medium", "high", "max"],
                                mode=selector.SelectSelectorMode.DROPDOWN,
                            )
                        ),
                    })
            else:
                # Home Assistant LLM API mode (Tools mode)
                step_schema.update({
                    vol.Optional(
                        CONF_API_HOST, default=options.get(CONF_API_HOST, RECOMMENDED_TOOLS_OPTIONS.get(CONF_API_HOST, "api.x.ai")),
                    ): str,
                    vol.Optional(
                        CONF_PROMPT,
                        description={"suggested_value": options.get(CONF_PROMPT, "")},
                    ): selector.TemplateSelector(),
                    vol.Optional(
                        CONF_CHAT_MODEL,
                        default=model,
                    ): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=SUPPORTED_MODELS,
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Optional(
                        CONF_MAX_TOKENS,
                        default=options.get(CONF_MAX_TOKENS, RECOMMENDED_TOOLS_OPTIONS.get(CONF_MAX_TOKENS)),
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=1, max=8192, mode=selector.NumberSelectorMode.BOX
                        )
                    ),
                    vol.Optional(
                        CONF_TEMPERATURE,
                        default=options.get(CONF_TEMPERATURE, RECOMMENDED_TOOLS_OPTIONS.get(CONF_TEMPERATURE)),
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=0.0, max=2.0, step=0.1, mode=selector.NumberSelectorMode.BOX
                        )
                    ),
                    vol.Optional(
                        CONF_TOP_P,
                        default=options.get(CONF_TOP_P, RECOMMENDED_TOOLS_OPTIONS.get(CONF_TOP_P)),
                    ): selector.NumberSelector(
                        selector.NumberSelectorConfig(
                            min=0.0, max=1.0, step=0.05, mode=selector.NumberSelectorMode.BOX
                        )
                    ),
                })

                if model in REASONING_EFFORT_MODELS:
                    step_schema.update({
                        vol.Optional(
                            CONF_REASONING_EFFORT,
                            default=options.get(CONF_REASONING_EFFORT, RECOMMENDED_REASONING_EFFORT),
                        ): selector.SelectSelector(
                            selector.SelectSelectorConfig(
                                options=["low", "medium", "high", "max"],
                                mode=selector.SelectSelectorMode.DROPDOWN,
                            )
                        ),
                    })

            # Memory settings are now at integration level, not subentry level
            # Only CONF_STORE_MESSAGES remains as a per-conversation toggle
        else:
            # ai_task_data and code_task schemas stay as they are
            model = options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)
            step_schema.update({
                vol.Optional(
                    CONF_PROMPT,
                    description={"suggested_value": options.get(CONF_PROMPT, "")},
                ): selector.TemplateSelector(),
                vol.Optional(
                    CONF_API_HOST,
                    default=options.get(CONF_API_HOST, DEFAULT_API_HOST),
                ): str,
                vol.Optional(
                    CONF_CHAT_MODEL,
                    default=model,
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=SUPPORTED_MODELS,
                        mode=selector.SelectSelectorMode.DROPDOWN,
                    )
                ),
                vol.Optional(
                    CONF_MAX_TOKENS,
                    default=options.get(CONF_MAX_TOKENS),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=1, max=8192, mode=selector.NumberSelectorMode.BOX
                    )
                ),
                vol.Optional(
                    CONF_TEMPERATURE,
                    default=options.get(CONF_TEMPERATURE),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0.0, max=2.0, step=0.1, mode=selector.NumberSelectorMode.BOX
                    )
                ),
                vol.Optional(
                    CONF_TOP_P,
                    default=options.get(CONF_TOP_P),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0.0, max=1.0, step=0.05, mode=selector.NumberSelectorMode.BOX
                    )
                ),
            })

            # Add code_task specific fields
            if self._subentry_type == "code_task":
                step_schema.update({
                    vol.Optional(
                        CONF_STORE_MESSAGES,
                        default=options.get(CONF_STORE_MESSAGES, True),
                    ): selector.BooleanSelector(),
                    vol.Optional(
                        CONF_LIVE_SEARCH,
                        default=options.get(CONF_LIVE_SEARCH, RECOMMENDED_LIVE_SEARCH),
                    ): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=["off", "auto", "on"],
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        )
                    ),
                })

            # Add reasoning_effort for models that support it
            if model in REASONING_EFFORT_MODELS:
                step_schema.update({
                    vol.Optional(
                        CONF_REASONING_EFFORT,
                        default=options.get(CONF_REASONING_EFFORT, RECOMMENDED_REASONING_EFFORT),
                    ): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=["low", "medium", "high", "max"],
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        )
                    ),
                })

        if user_input is not None:
            # Handle conversation subentry specific logic
            if self._subentry_type == "conversation":
                # Get current and new values
                current_use_pipeline = options.get(CONF_USE_INTELLIGENT_PIPELINE, True)
                new_use_pipeline = user_input.get(CONF_USE_INTELLIGENT_PIPELINE, current_use_pipeline)

                current_model = options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)
                new_model = user_input.get(CONF_CHAT_MODEL, current_model)

                current_allow_control = options.get(CONF_ALLOW_SMART_HOME_CONTROL, True)
                new_allow_control = user_input.get(CONF_ALLOW_SMART_HOME_CONTROL, current_allow_control)

                # Map CONF_ALLOW_SMART_HOME_CONTROL to CONF_LLM_HASS_API when in tools mode
                if not new_use_pipeline:
                    user_input[CONF_LLM_HASS_API] = [ha_llm.LLM_API_ASSIST] if new_allow_control else []

                # Reload form if any dynamic field changed
                needs_reload = (
                    current_use_pipeline != new_use_pipeline or
                    current_allow_control != new_allow_control or
                    (current_model in REASONING_EFFORT_MODELS) != (new_model in REASONING_EFFORT_MODELS)
                )

                if needs_reload:
                    options.update(user_input)
                    self.options = options
                    return await self.async_step_init()

            # Update options and save
            options.update(user_input)
            if self._is_new:
                return self.async_create_entry(
                    title=options.pop("name", default_name),
                    data=options,
                )
            return self.async_update_and_abort(
                self._get_entry(),
                self._get_reconfigure_subentry(),
                data=options,
            )

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(step_schema),
        )
