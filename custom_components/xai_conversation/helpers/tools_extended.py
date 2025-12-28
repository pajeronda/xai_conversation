"""Extended Tools system - Compatible with extended_openai_conversation format."""

from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Any
import asyncio
import os
import sqlite3
import time
from urllib import parse

from bs4 import BeautifulSoup
import voluptuous as vol
import yaml

from homeassistant.components import (
    automation,
    energy,
    recorder,
    rest,
    scrape,
)
from homeassistant.components.automation.config import _async_validate_config_item
from homeassistant.components.script.config import SCRIPT_ENTITY_SCHEMA
from homeassistant.config import AUTOMATION_CONFIG_PATH
from homeassistant.const import (
    CONF_ATTRIBUTE,
    CONF_METHOD,
    CONF_NAME,
    CONF_PAYLOAD,
    CONF_RESOURCE,
    CONF_RESOURCE_TEMPLATE,
    CONF_TIMEOUT,
    CONF_VALUE_TEMPLATE,
    CONF_VERIFY_SSL,
    SERVICE_RELOAD,
)
from homeassistant.core import HomeAssistant, Context, State
from homeassistant.exceptions import HomeAssistantError, ServiceNotFound
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.script import Script
from homeassistant.helpers.template import Template
from homeassistant.helpers.event import async_call_later
import homeassistant.util.dt as dt_util

from ..const import (
    DOMAIN,
    LOGGER,
    CONF_PAYLOAD_TEMPLATE,
    EVENT_AUTOMATION_REGISTERED,
)

from ..exceptions import (
    CallServiceError,
    EntityNotExposed,
    EntityNotFound,
    FunctionNotFound,
    InvalidFunction,
    NativeNotFound,
)


class ExtendedToolError(HomeAssistantError):
    """Base error for extended tools."""


def get_function_executor(value: str):
    function_executor = FUNCTION_EXECUTORS.get(value)
    if function_executor is None:
        raise FunctionNotFound(value)
    return function_executor


def convert_to_template(
    settings,
    template_keys=["data", "event_data", "target", "service"],
    hass: HomeAssistant | None = None,
):
    _convert_to_template(settings, template_keys, hass, [])


def _convert_to_template(settings, template_keys, hass, parents: list[str]):
    if isinstance(settings, dict):
        for key, value in settings.items():
            if isinstance(value, str) and (
                key in template_keys or set(parents).intersection(template_keys)
            ):
                settings[key] = Template(value, hass)
            if isinstance(value, dict):
                parents.append(key)
                _convert_to_template(value, template_keys, hass, parents)
                parents.pop()
            if isinstance(value, list):
                parents.append(key)
                for item in value:
                    _convert_to_template(item, template_keys, hass, parents)
                parents.pop()
    if isinstance(settings, list):
        for setting in settings:
            _convert_to_template(setting, template_keys, hass, parents)


def _get_rest_data(hass, rest_config, arguments):
    rest_config.setdefault(CONF_METHOD, rest.const.DEFAULT_METHOD)
    rest_config.setdefault(CONF_VERIFY_SSL, rest.const.DEFAULT_VERIFY_SSL)
    rest_config.setdefault(CONF_TIMEOUT, rest.data.DEFAULT_TIMEOUT)
    rest_config.setdefault(rest.const.CONF_ENCODING, rest.const.DEFAULT_ENCODING)

    resource_template: Template | None = rest_config.get(CONF_RESOURCE_TEMPLATE)
    if resource_template is not None:
        rest_config.pop(CONF_RESOURCE_TEMPLATE)
        rest_config[CONF_RESOURCE] = resource_template.async_render(
            arguments, parse_result=False
        )

    payload_template: Template | None = rest_config.get(CONF_PAYLOAD_TEMPLATE)
    if payload_template is not None:
        rest_config.pop(CONF_PAYLOAD_TEMPLATE)
        rest_config[CONF_PAYLOAD] = payload_template.async_render(
            arguments, parse_result=False
        )

    return rest.create_rest_data_from_config(hass, rest_config)


class FunctionExecutor(ABC):
    def __init__(self, data_schema=vol.Schema({})) -> None:
        """initialize function executor"""
        self.data_schema = data_schema.extend({vol.Required("type"): str})

    def validate_config(self, config: dict) -> dict:
        """Validate function configuration."""
        return self.data_schema(config)

    def to_arguments(self, arguments):
        """to_arguments function"""
        try:
            return self.data_schema(arguments)
        except vol.error.Error as e:
            function_type = next(
                (key for key, value in FUNCTION_EXECUTORS.items() if value == self),
                None,
            )
            raise InvalidFunction(function_type) from e

    def validate_entity_ids(self, hass: HomeAssistant, entity_ids, exposed_entities):
        if any(hass.states.get(entity_id) is None for entity_id in entity_ids):
            raise EntityNotFound(", ".join(entity_ids))
        exposed_entity_ids = map(lambda e: e["entity_id"], exposed_entities)
        if not set(entity_ids).issubset(exposed_entity_ids):
            raise EntityNotExposed(", ".join(entity_ids))

    @abstractmethod
    async def execute(
        self,
        hass: HomeAssistant,
        function,
        arguments,
        context: Context,
        exposed_entities,
    ):
        """execute function"""


class NativeFunctionExecutor(FunctionExecutor):
    def __init__(self) -> None:
        """initialize native function"""
        super().__init__(vol.Schema({vol.Required("name"): str}))

    async def execute(
        self,
        hass: HomeAssistant,
        function,
        arguments,
        context: Context,
        exposed_entities,
    ):
        name = function["name"]
        if name in ("execute_service", "execute_services"):
            return await self.execute_service(
                hass, function, arguments, context, exposed_entities
            )
        if name == "execute_service_single":
            return await self.execute_service_single(
                hass, function, arguments, context, exposed_entities
            )
        if name == "add_automation":
            return await self.add_automation(
                hass, function, arguments, context, exposed_entities
            )
        if name == "get_history":
            return await self.get_history(
                hass, function, arguments, context, exposed_entities
            )
        if name == "get_energy":
            return await self.get_energy(
                hass, function, arguments, context, exposed_entities
            )
        if name == "get_statistics":
            return await self.get_statistics(
                hass, function, arguments, context, exposed_entities
            )
        if name == "get_user_from_user_id":
            return await self.get_user_from_user_id(
                hass, function, arguments, context, exposed_entities
            )

        raise NativeNotFound(name)

    async def execute_service_single(
        self,
        hass: HomeAssistant,
        function,
        service_argument,
        context: Context,
        exposed_entities,
    ):
        domain = service_argument["domain"]
        service = service_argument["service"]
        service_data = service_argument.get(
            "service_data", service_argument.get("data", {})
        )
        entity_id = service_data.get("entity_id", service_argument.get("entity_id"))
        area_id = service_data.get("area_id")
        device_id = service_data.get("device_id")

        if isinstance(entity_id, str):
            entity_id = [e.strip() for e in entity_id.split(",")]
        service_data["entity_id"] = entity_id

        if entity_id is None and area_id is None and device_id is None:
            raise CallServiceError(domain, service, service_data)
        if not hass.services.has_service(domain, service):
            raise ServiceNotFound(domain, service)
        self.validate_entity_ids(hass, entity_id or [], exposed_entities)

        try:
            await hass.services.async_call(
                domain=domain,
                service=service,
                service_data=service_data,
            )
            return {"success": True}
        except HomeAssistantError as e:
            LOGGER.error(e)
            return {"error": str(e)}

    async def execute_service(
        self,
        hass: HomeAssistant,
        function,
        arguments,
        context: Context,
        exposed_entities,
    ):
        result = []
        for service_argument in arguments.get("list", []):
            result.append(
                await self.execute_service_single(
                    hass, function, service_argument, context, exposed_entities
                )
            )
        return result

    async def add_automation(
        self,
        hass: HomeAssistant,
        function,
        arguments,
        context: Context,
        exposed_entities,
    ):
        automation_config = yaml.safe_load(arguments["automation_config"])
        config = {"id": str(round(time.time() * 1000))}
        if isinstance(automation_config, list):
            config.update(automation_config[0])
        if isinstance(automation_config, dict):
            config.update(automation_config)

        await _async_validate_config_item(hass, config, True, False)

        automations = [config]
        with open(
            os.path.join(hass.config.config_dir, AUTOMATION_CONFIG_PATH),
            "r",
            encoding="utf-8",
        ) as f:
            current_automations = yaml.safe_load(f.read())

        with open(
            os.path.join(hass.config.config_dir, AUTOMATION_CONFIG_PATH),
            "a" if current_automations else "w",
            encoding="utf-8",
        ) as f:
            raw_config = yaml.dump(automations, allow_unicode=True, sort_keys=False)
            f.write("\n" + raw_config)

        await hass.services.async_call(automation.config.DOMAIN, SERVICE_RELOAD)
        hass.bus.async_fire(
            EVENT_AUTOMATION_REGISTERED,
            {"automation_config": config, "raw_config": raw_config},
        )
        return "Success"

    async def get_history(
        self,
        hass: HomeAssistant,
        function,
        arguments,
        context: Context,
        exposed_entities,
    ):
        start_time = arguments.get("start_time")
        end_time = arguments.get("end_time")
        entity_ids = arguments.get("entity_ids", [])
        include_start_time_state = arguments.get("include_start_time_state", True)
        significant_changes_only = arguments.get("significant_changes_only", True)
        minimal_response = arguments.get("minimal_response", True)
        no_attributes = arguments.get("no_attributes", True)

        now = dt_util.utcnow()
        one_day = timedelta(days=1)
        start_time = self.as_utc(start_time, now - one_day, "start_time not valid")
        end_time = self.as_utc(end_time, start_time + one_day, "end_time not valid")

        self.validate_entity_ids(hass, entity_ids, exposed_entities)

        with recorder.util.session_scope(hass=hass, read_only=True) as session:
            result = await recorder.get_instance(hass).async_add_executor_job(
                recorder.history.get_significant_states_with_session,
                hass,
                session,
                start_time,
                end_time,
                entity_ids,
                None,
                include_start_time_state,
                significant_changes_only,
                minimal_response,
                no_attributes,
            )

        return [[self.as_dict(item) for item in sublist] for sublist in result.values()]

    async def get_energy(
        self,
        hass: HomeAssistant,
        function,
        arguments,
        context: Context,
        exposed_entities,
    ):
        energy_manager: energy.data.EnergyManager = await energy.async_get_manager(hass)
        return energy_manager.data

    async def get_user_from_user_id(
        self,
        hass: HomeAssistant,
        function,
        arguments,
        context: Context,
        exposed_entities,
    ):
        user = await hass.auth.async_get_user(context.user_id)
        return {"name": user.name if user and hasattr(user, "name") else "Unknown"}

    async def get_statistics(
        self,
        hass: HomeAssistant,
        function,
        arguments,
        context: Context,
        exposed_entities,
    ):
        statistic_ids = arguments.get("statistic_ids", [])
        start_time = dt_util.as_utc(dt_util.parse_datetime(arguments["start_time"]))
        end_time = dt_util.as_utc(dt_util.parse_datetime(arguments["end_time"]))

        return await recorder.get_instance(hass).async_add_executor_job(
            recorder.statistics.statistics_during_period,
            hass,
            start_time,
            end_time,
            statistic_ids,
            arguments.get("period", "day"),
            arguments.get("units"),
            arguments.get("types", {"change"}),
        )

    def as_utc(self, value: str, default_value, parse_error_message: str):
        if value is None:
            return default_value

        parsed_datetime = dt_util.parse_datetime(value)
        if parsed_datetime is None:
            raise HomeAssistantError(parse_error_message)

        return dt_util.as_utc(parsed_datetime)

    def as_dict(self, state: State | dict[str, Any]):
        if isinstance(state, State):
            return state.as_dict()
        return state


class ScriptFunctionExecutor(FunctionExecutor):
    def __init__(self) -> None:
        """initialize script function"""
        super().__init__(SCRIPT_ENTITY_SCHEMA)

    async def execute(
        self,
        hass: HomeAssistant,
        function,
        arguments,
        context: Context,
        exposed_entities,
    ):
        script = Script(
            hass,
            function["sequence"],
            "extended_openai_conversation",
            DOMAIN,
            running_description="[extended_openai_conversation] function",
            logger=LOGGER,
        )

        result = await script.async_run(run_variables=arguments, context=context)
        return result.variables.get("_function_result", "Success")


class TemplateFunctionExecutor(FunctionExecutor):
    def __init__(self) -> None:
        """initialize template function"""
        super().__init__(
            vol.Schema(
                {
                    vol.Required("value_template"): cv.template,
                    vol.Optional("parse_result"): bool,
                }
            )
        )

    async def execute(
        self,
        hass: HomeAssistant,
        function,
        arguments,
        context: Context,
        exposed_entities,
    ):
        return function["value_template"].async_render(
            arguments,
            parse_result=function.get("parse_result", False),
        )


class RestFunctionExecutor(FunctionExecutor):
    def __init__(self) -> None:
        """initialize Rest function"""
        super().__init__(
            vol.Schema(rest.RESOURCE_SCHEMA).extend(
                {
                    vol.Optional("value_template"): cv.template,
                    vol.Optional("payload_template"): cv.template,
                }
            )
        )

    async def execute(
        self,
        hass: HomeAssistant,
        function,
        arguments,
        context: Context,
        exposed_entities,
    ):
        config = function
        rest_data = _get_rest_data(hass, config, arguments)

        await rest_data.async_update()
        value = rest_data.data_without_xml()
        value_template = config.get(CONF_VALUE_TEMPLATE)

        if value is not None and value_template is not None:
            value = value_template.async_render_with_possible_json_value(
                value, None, arguments
            )

        return value


class ScrapeFunctionExecutor(FunctionExecutor):
    def __init__(self) -> None:
        """initialize Scrape function"""
        super().__init__(
            scrape.COMBINED_SCHEMA.extend(
                {
                    vol.Optional("value_template"): cv.template,
                    vol.Optional("payload_template"): cv.template,
                }
            )
        )

    async def execute(
        self,
        hass: HomeAssistant,
        function,
        arguments,
        context: Context,
        exposed_entities,
    ):
        config = function
        rest_data = _get_rest_data(hass, config, arguments)
        coordinator = scrape.coordinator.ScrapeCoordinator(
            hass,
            rest_data,
            scrape.const.DEFAULT_SCAN_INTERVAL,
        )
        await coordinator.async_config_entry_first_refresh()

        new_arguments = dict(arguments)

        for sensor_config in config["sensor"]:
            name: Template = sensor_config.get(CONF_NAME)
            value = self._async_update_from_rest_data(
                coordinator.data, sensor_config, arguments
            )
            new_arguments["value"] = value
            if name:
                new_arguments[name.async_render()] = value

        result = new_arguments["value"]
        value_template = config.get(CONF_VALUE_TEMPLATE)

        if value_template is not None:
            result = value_template.async_render_with_possible_json_value(
                result, None, new_arguments
            )

        return result

    def _async_update_from_rest_data(
        self,
        data: BeautifulSoup,
        sensor_config: dict[str, Any],
        arguments: dict[str, Any],
    ) -> None:
        """Update state from the rest data."""
        value = self._extract_value(data, sensor_config)
        value_template = sensor_config.get(CONF_VALUE_TEMPLATE)

        if value_template is not None:
            value = value_template.async_render_with_possible_json_value(
                value, None, arguments
            )

        return value

    def _extract_value(self, data: BeautifulSoup, sensor_config: dict[str, Any]) -> Any:
        """Parse the html extraction in the executor."""
        value: str | list[str] | None
        select = sensor_config[scrape.const.CONF_SELECT]
        index = sensor_config.get(scrape.const.CONF_INDEX, 0)
        attr = sensor_config.get(CONF_ATTRIBUTE)
        try:
            if attr is not None:
                value = data.select(select)[index][attr]
            else:
                tag = data.select(select)[index]
                if tag.name in ("style", "script", "template"):
                    value = tag.string
                else:
                    value = tag.text
        except IndexError:
            LOGGER.warning("Index '%s' not found", index)
            value = None
        except KeyError:
            LOGGER.warning("Attribute '%s' not found", attr)
            value = None
        LOGGER.debug("Parsed value: %s", value)
        return value


class CompositeFunctionExecutor(FunctionExecutor):
    def __init__(self) -> None:
        """initialize composite function"""
        super().__init__(
            vol.Schema(
                {
                    vol.Required("sequence"): vol.All(
                        cv.ensure_list, [self.function_schema]
                    )
                }
            )
        )

    def function_schema(self, value: Any) -> dict:
        """Validate a composite function schema."""
        if not isinstance(value, dict):
            raise vol.Invalid("expected dictionary")

        composite_schema = {vol.Optional("response_variable"): str}
        function_executor = get_function_executor(value["type"])

        return function_executor.data_schema.extend(composite_schema)(value)

    async def execute(
        self,
        hass: HomeAssistant,
        function,
        arguments,
        context: Context,
        exposed_entities,
    ):
        config = function
        sequence = config["sequence"]

        for executor_config in sequence:
            function_executor = get_function_executor(executor_config["type"])
            result = await function_executor.execute(
                hass, executor_config, arguments, context, exposed_entities
            )

            response_variable = executor_config.get("response_variable")
            if response_variable:
                arguments[response_variable] = result

        return result


class SqliteFunctionExecutor(FunctionExecutor):
    def __init__(self) -> None:
        """initialize sqlite function"""
        super().__init__(
            vol.Schema(
                {
                    vol.Optional("query"): str,
                    vol.Optional("db_url"): str,
                    vol.Optional("single"): bool,
                }
            )
        )

    def is_exposed(self, entity_id, exposed_entities) -> bool:
        return any(
            exposed_entity["entity_id"] == entity_id
            for exposed_entity in exposed_entities
        )

    def is_exposed_entity_in_query(self, query: str, exposed_entities) -> bool:
        exposed_entity_ids = list(
            map(lambda e: f"'{e['entity_id']}'", exposed_entities)
        )
        return any(
            exposed_entity_id in query for exposed_entity_id in exposed_entity_ids
        )

    def raise_error(self, msg="Unexpected error occurred."):
        raise HomeAssistantError(msg)

    def get_default_db_url(self, hass: HomeAssistant) -> str:
        db_file_path = os.path.join(hass.config.config_dir, recorder.DEFAULT_DB_FILE)
        return f"file:{db_file_path}?mode=ro"

    def set_url_read_only(self, url: str) -> str:
        scheme, netloc, path, query_string, fragment = parse.urlsplit(url)
        query_params = parse.parse_qs(query_string)

        query_params["mode"] = ["ro"]
        new_query_string = parse.urlencode(query_params, doseq=True)

        return parse.urlunsplit((scheme, netloc, path, new_query_string, fragment))

    async def execute(
        self,
        hass: HomeAssistant,
        function,
        arguments,
        context: Context,
        exposed_entities,
    ):
        db_url = self.set_url_read_only(
            function.get("db_url", self.get_default_db_url(hass))
        )
        query = function.get("query", "{{query}}")

        template_arguments = {
            "is_exposed": lambda e: self.is_exposed(e, exposed_entities),
            "is_exposed_entity_in_query": lambda q: self.is_exposed_entity_in_query(
                q, exposed_entities
            ),
            "exposed_entities": exposed_entities,
            "raise": self.raise_error,
        }
        template_arguments.update(arguments)

        q = Template(query, hass).async_render(template_arguments)
        LOGGER.info("Rendered query: %s", q)

        with sqlite3.connect(db_url, uri=True) as conn:
            cursor = conn.cursor().execute(q)
            names = [description[0] for description in cursor.description]

            if function.get("single") is True:
                row = cursor.fetchone()
                return {name: val for name, val in zip(names, row)}

            rows = cursor.fetchall()
            result = []
            for row in rows:
                result.append({name: val for name, val in zip(names, row)})
            return result


class ExtendedToolsRegistry:
    """Registry to manage and execute extended tools."""

    def __init__(
        self, hass: HomeAssistant, yaml_config: str, xai_tool_constructor=None
    ):
        self._hass = hass
        self._tools: dict[str, dict] = {}  # name -> full_tool_config
        self._specs_cache: list = []  # xAI tool objects
        self._yaml_config = yaml_config
        self._xai_tool_constructor = xai_tool_constructor
        self._tool_descriptions_cache: str | None = None
        self._parse_config()

    def _parse_config(self) -> None:
        """Parse YAML and populate registry."""
        if not self._yaml_config:
            return

        try:
            tools = yaml.safe_load(self._yaml_config)
        except yaml.YAMLError as err:
            LOGGER.error("Extended tools YAML error: %s", err)
            return

        if not isinstance(tools, list):
            LOGGER.error("Extended tools config must be a list")
            return

        for tool in tools:
            spec = tool.get("spec")
            func = tool.get("function")

            if not spec or not func:
                continue

            name = spec.get("name")
            if not name:
                continue

            # Validate function config using executor's schema
            func_type = func.get("type")
            executor = FUNCTION_EXECUTORS.get(func_type)
            if executor:
                try:
                    # Apply schema validation to convert templates and validate structure
                    validated_func = executor.data_schema(func)
                    func = validated_func
                except vol.error.Error as err:
                    LOGGER.error(
                        "Extended tool %s: function validation failed: %s", name, err
                    )
                    continue
            else:
                LOGGER.error(
                    "Extended tool %s: unknown executor type '%s'", name, func_type
                )
                continue

            # Build xAI tool object if constructor available
            if self._xai_tool_constructor:
                try:
                    xai_tool = self._xai_tool_constructor(
                        name=name,
                        description=spec.get("description", "Extended tool"),
                        parameters=spec.get("parameters", {}),
                    )
                    self._specs_cache.append(xai_tool)
                except Exception as err:
                    LOGGER.error("Failed to create xAI tool for %s: %s", name, err)
                    continue
            else:
                # Fallback: store as dict (for validation without xai_tool_constructor)
                self._specs_cache.append(spec)

            # Store tool with validated function config
            self._tools[name] = {"spec": spec, "function": func}

        # Cache compact descriptions for prompt injection
        descriptions = []
        for name, tool_data in self._tools.items():
            desc = tool_data["spec"].get("description", "No description")
            descriptions.append(f"- **{name}**: {desc}")
        self._tool_descriptions_cache = "\n".join(descriptions)

    @property
    def is_empty(self) -> bool:
        """Check if registry has no tools."""
        return len(self._tools) == 0

    def get_tool_descriptions(self) -> str:
        """Get textual descriptions of all tools for prompt.

        Returns:
            Bulleted list string containing name and description.
        """
        return self._tool_descriptions_cache or ""

    def get_specs_for_llm(self) -> list:
        """Return xAI tool objects for LLM."""
        return self._specs_cache

    def has_function(self, name: str) -> bool:
        """Check if function exists."""
        return name in self._tools

    def get_tool_config(self, name: str) -> dict | None:
        """Get the full tool configuration (spec + function)."""
        return self._tools.get(name)

    def get_delayed_function_config(self, function_config: dict, delay_args: Any) -> dict:
        """Wrap a function config in a composite sequence with a script delay."""
        return {
            "type": "composite",
            "sequence": [
                {
                    "type": "script",
                    "sequence": [{"delay": delay_args}],
                },
                function_config,
            ],
        }

    async def async_execute_raw_config(
        self,
        func_config: dict,
        arguments: dict,
        context: Context,
        exposed_entities: list = None,
    ) -> Any:
        """Execute a raw function configuration."""
        func_type = func_config.get("type")
        executor = FUNCTION_EXECUTORS.get(func_type)
        if not executor:
            raise ExtendedToolError(f"Executor type '{func_type}' not supported")

        try:
            return await executor.execute(
                self._hass, func_config, arguments, context, exposed_entities or []
            )
        except Exception as err:
            LOGGER.exception("Error executing extended tool config")
            raise ExtendedToolError(f"Execution failed: {err}")

    async def execute(
        self,
        name: str,
        arguments: dict,
        context: Context,
        exposed_entities: list = None,
    ) -> Any:
        """Execute a tool by name."""
        tool = self._tools.get(name)
        if not tool:
            raise ExtendedToolError(f"Tool {name} not found")

        func_config = tool["function"]
        func_type = func_config.get("type")

        executor = FUNCTION_EXECUTORS.get(func_type)
        if not executor:
            raise ExtendedToolError(f"Executor type '{func_type}' not supported")

        try:
            return await executor.execute(
                self._hass, func_config, arguments, context, exposed_entities or []
            )
        except Exception as err:
            LOGGER.exception("Error executing extended tool %s", name)
            raise ExtendedToolError(f"Execution failed: {err}")

    @staticmethod
    def validate_yaml(yaml_str: str) -> tuple[bool, str | None]:
        """Static method to validate YAML config during config flow."""
        try:
            tools = yaml.safe_load(yaml_str)
            if not isinstance(tools, list):
                return False, "Configuration must be a list of tools"

            for i, tool in enumerate(tools):
                # Check spec
                if "spec" not in tool:
                    return False, f"Tool {i + 1}: missing 'spec' section"
                spec = tool["spec"]
                if "name" not in spec:
                    return False, f"Tool {i + 1}: missing 'spec.name'"
                if "description" not in spec:
                    return False, f"Tool {i + 1}: missing 'spec.description'"

                # Check function
                if "function" not in tool:
                    return (
                        False,
                        f"Tool {i + 1} ({spec['name']}): missing 'function' section",
                    )
                func = tool["function"]
                if "type" not in func:
                    return (
                        False,
                        f"Tool {i + 1} ({spec['name']}): missing 'function.type'",
                    )

                func_type = func["type"]
                if func_type not in FUNCTION_EXECUTORS:
                    valid_types = ", ".join(FUNCTION_EXECUTORS.keys())
                    return (
                        False,
                        f"Tool {i + 1} ({spec['name']}): unknown type '{func_type}'. Valid: {valid_types}",
                    )

                # Validate executor-specific config
                executor = FUNCTION_EXECUTORS[func_type]
                try:
                    executor.validate_config(func)
                except vol.Error as err:
                    return False, f"Tool {i + 1} ({spec['name']}): {err}"

            return True, None

        except yaml.YAMLError as err:
            return False, f"YAML syntax error: {err}"
        except Exception as err:
            return False, f"Unexpected validation error: {err}"


FUNCTION_EXECUTORS: dict[str, FunctionExecutor] = {
    "native": NativeFunctionExecutor(),
    "script": ScriptFunctionExecutor(),
    "template": TemplateFunctionExecutor(),
    "rest": RestFunctionExecutor(),
    "scrape": ScrapeFunctionExecutor(),
    "composite": CompositeFunctionExecutor(),
    "sqlite": SqliteFunctionExecutor(),
}
