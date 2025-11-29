"""Constants for xAI Grok Conversation integration."""

import logging
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.helpers import llm

LOGGER = logging.getLogger(__package__)

DOMAIN = "xai_conversation"
DEFAULT_DEVICE_NAME = "xAI"
DEFAULT_MANUFACTURER = "xAI"
DEFAULT_CONVERSATION_NAME = "xAI Conversation"
DEFAULT_AI_TASK_NAME = "xAI Task"
DEFAULT_GROK_CODE_FAST_NAME = "Grok Code Fast"
DEFAULT_SENSORS_NAME = "xAI Token Sensors"
DEFAULT_API_HOST = "api.x.ai"

# Default assistant names for different services
DEFAULT_CODE_FAST_ASSISTANT_NAME = "Grok Code Fast"
# xAI specific configuration keys
CONF_CHAT_MODEL = "chat_model"
CONF_IMAGE_MODEL = "image_model"
CONF_VISION_MODEL = "vision_model"
CONF_MAX_TOKENS = "max_tokens"
CONF_PROMPT_PIPELINE = "pipeline_prompt"
# Enable/disable Intelligent Pipeline mode for conversation subentries
CONF_USE_INTELLIGENT_PIPELINE = "use_intelligent_pipeline"
# Enable/disable smart home control (unified for both pipeline and tools mode)
CONF_ALLOW_SMART_HOME_CONTROL = "allow_smart_home_control"
CONF_REASONING_EFFORT = "reasoning_effort"
CONF_TEMPERATURE = "temperature"
CONF_TOP_P = "top_p"
CONF_LIVE_SEARCH = "live_search"
CONF_STORE_MESSAGES = "store_messages"
CONF_SEND_USER_NAME = "send_user_name"
CONF_TIMEOUT = "timeout"
CONF_PROMPT = "prompt"  # User instructions for tools/non-pipeline mode
CONF_PROMPT_CODE = "prompt_code"  # Custom instructions for code_fast mode
CONF_VISION_PROMPT = "vision_prompt"  # System prompt for photo analysis
CONF_API_HOST = "api_host"
CONF_ASSISTANT_NAME = "assistant_name"  # Customizable assistant name
# Memory configuration - separate settings for users and devices
CONF_MEMORY_USER_TTL_HOURS = "memory_user_ttl_hours"
CONF_MEMORY_USER_MAX_TURNS = "memory_user_max_turns"
CONF_MEMORY_DEVICE_TTL_HOURS = "memory_device_ttl_hours"
CONF_MEMORY_DEVICE_MAX_TURNS = "memory_device_max_turns"
CONF_MEMORY_CLEANUP_INTERVAL_HOURS = (
    "memory_cleanup_interval_hours"  # Interval for periodic cleanup
)

# Model pricing is fetched dynamically from xAI API and displayed in XAIPricingSensor entities
# Pricing sensors are created automatically for all available models at startup
# Default model recommendations optimized for grok-4-1-fast-non-reasoning
RECOMMENDED_CHAT_MODEL = "grok-4-1-fast-non-reasoning"
RECOMMENDED_GROK_CODE_FAST_MODEL = "grok-code-fast-1"
RECOMMENDED_AI_TASK_MODEL = "grok-code-fast-1"
RECOMMENDED_IMAGE_MODEL = "grok-2-image-1212"  # Model for image generation (Aurora)
RECOMMENDED_VISION_MODEL = "grok-2-vision-1212"  # Model for photo analysis
RECOMMENDED_MAX_TOKENS = 2000
RECOMMENDED_TEMPERATURE = 1.0
RECOMMENDED_TOP_P = 1.0
RECOMMENDED_REASONING_EFFORT = "low"
RECOMMENDED_LIVE_SEARCH = "off"
RECOMMENDED_STORE_MESSAGES = True
RECOMMENDED_TIMEOUT = 60.0
RECOMMENDED_ASSISTANT_NAME = "Jarvis"
# Memory defaults - separate for users and devices
RECOMMENDED_MEMORY_USER_TTL_HOURS = 24 * 30  # 30 days for users
RECOMMENDED_MEMORY_USER_MAX_TURNS = 1000  # 1000 turns for users
RECOMMENDED_MEMORY_DEVICE_TTL_HOURS = 24 * 7  # 7 days for voice satellites
RECOMMENDED_MEMORY_DEVICE_MAX_TURNS = 100  # 100 turns for voice satellites
RECOMMENDED_MEMORY_CLEANUP_INTERVAL_HOURS = 24  # Run cleanup every 24 hours
RECOMMENDED_CHAT_HISTORY_MAX_MESSAGES = (
    100  # Max messages stored per conversation in ChatHistoryService
)
# Maximum conversation history turns to send when server-side memory is disabled
RECOMMENDED_HISTORY_LIMIT_TURNS = 10  # 10 turns = 20 messages (user+assistant)
# Subentry types (aligned with service_type for consistency)
SUBENTRY_TYPE_CONVERSATION = "conversation"
SUBENTRY_TYPE_AI_TASK = "ai_task"
SUBENTRY_TYPE_CODE_TASK = "code_fast"
SUBENTRY_TYPE_SENSORS = "sensors"


# Dynamic model pricing and details will be fetched from xAI API
# DEFAULT_MODEL_PRICING is removed as per optimization

# Supported models from xAI documentation
# Supported models from xAI documentation - populated dynamically at runtime
SUPPORTED_MODELS: list[str] = []

# Models that support reasoning_effort parameter - populated dynamically at runtime
REASONING_EFFORT_MODELS: list[str] = []

# ==============================================================================
# MODULAR PROMPT SYSTEM - Building Blocks
# ==============================================================================
#
# This section defines reusable prompt components that are dynamically composed
# based on the active configuration (mode, memory type, control permissions, etc.)
#
# COMPOSITION LOGIC:
# The final system prompt is built by helpers.PromptManager.build_system_prompt()
# which assembles blocks in this order:
#
# 1. PROMPT_IDENTITY                    (always)
# 2. PROMPT_ROLE_BASE                   (always for conversation modes)
# 3. PROMPT_MEMORY_* (serverside/client) (based on store_messages)
# 4. MODE-SPECIFIC BLOCKS:
#    - Code mode:                 CODE_ROLE → [USER_INSTRUCTIONS] → CODE_OUTPUT_FORMAT
#    - Pipeline + allow_control:  RECOGNITION → CUSTOM_RULES → DECISION_LOGIC → EXAMPLES
#    - Tools + allow_control:     TOOLS_USAGE
#    - Chat-only or !allow_control: NO_CONTROL
# 5. PROMPT_OUTPUT_FORMAT               (conversation/pipeline/tools modes only)
# ==============================================================================

# -----------------------------------------------------------------------------
# 1. IDENTITY BLOCK (always present)
# -----------------------------------------------------------------------------
PROMPT_IDENTITY = (
    """You are {assistant_name}, a powerfull assistant for Home Assistant."""
)

# -----------------------------------------------------------------------------
# 2. ROLE BASE BLOCK (always present)
# -----------------------------------------------------------------------------
PROMPT_ROLE_BASE = """Function as an advanced ASR/NLU system."""

# -----------------------------------------------------------------------------
# 3. MEMORY BLOCKS (one of these, based on store_messages setting)
# -----------------------------------------------------------------------------
PROMPT_MEMORY_SERVERSIDE = """Full conversation history and timestamps stored server-side. Use stored timestamps from conversation memory to answer time-related questions precisely."""

PROMPT_MEMORY_CLIENTSIDE = """You receive conversation history manually in the messages. Answer based only on the provided context."""

# -----------------------------------------------------------------------------
# 4. SMART HOME CONTROL BLOCKS (for Pipeline mode with allow_control=True)
# -----------------------------------------------------------------------------
PROMPT_SMART_HOME_RECOGNITION = """Recognize Smart Home Commands: actions like "play", "stop", "pause", "turn", "open", "close", "set", or status queries."""

PROMPT_CUSTOM_RULES = """Allow user-added custom rules in any language for Smart Home Commands, translating if needed."""

PROMPT_PIPELINE_DECISION_LOGIC = """Action Decision:
- Single Smart Home Command: '[[HA_LOCAL: {"text": "<the recognized command>"}]]'
- Multiple Smart Home Commands: '[[HA_LOCAL: {"commands": [{"text": "cmd1"}, {"text": "cmd2"}]}]]'
  - Add "sequential": true if commands must execute in order (e.g., "first do X, then Y")
  - Omit "sequential" or set false for independent parallel commands
- General Questions (non-HA related): answer concisely and truthfully.

Priority Rules:
- where you feel it is relevant, present the command in a natural way. Focus on the action in progress, not the completed result.
- Smart Home commands/queries → [[HA_LOCAL]]
- General knowledge questions → Answer directly 
- When in doubt about device state/control → PRIORITIZE SMART HOME CONTROL"""

PROMPT_PIPELINE_EXAMPLES = """Examples:
- "when did I last turn on the light?" → "You turned on the living room light on October 6th at 2:32 PM"
- "turn on the living room light" → your comment, if any and [[HA_LOCAL: {"text": "turn on the living room light"}]]
- "(any text)  what's the temperature in the kitchen (other words)?" → [[HA_LOCAL: {"text": "what's the temperature in the kitchen?"}]]"
- "turn off lights in living room and turn on tv" → your comment, if any and [[HA_LOCAL: {"commands": [{"text": "turn off lights in living room"}, {"text": "turn on tv"}]}]]"
- "first close the blinds, then turn off the lights" → your comment, if any and [[HA_LOCAL: {"commands": [{"text": "close the blinds"}, {"text": "turn off the lights"}], "sequential": true}]]"""

# -----------------------------------------------------------------------------
# 5. TOOLS MODE BLOCK (for Tools mode with allow_control=True)
# -----------------------------------------------------------------------------
PROMPT_TOOLS_USAGE = """An overview of the areas, devices and tools in this smart home:
Static Context:
{static_context}

Available Tools:
{tool_definitions}

Guidelines:
- For device control (turn on/off, set values): use the appropriate tool with targeting parameters (name, area, floor, domain, device_class)
- For state queries (current status, temperature, sensor readings): call GetLiveContext() to retrieve live data, then answer based on the returned information
- In mixed requests (command + non-home automation activity like a poem), always run the home command tools first, then quickly handle the rest."""

# -----------------------------------------------------------------------------
# 6. CHAT-ONLY BLOCK (when allow_control=False)
# -----------------------------------------------------------------------------
PROMPT_NO_CONTROL = """Limitations:
- You are NOT authorized to control smart home devices
- If asked to control a device, reply that the user must enable the functionality in the settings"""

# -----------------------------------------------------------------------------
# 7. CODE MODE BLOCKS (for Code Fast service)
# -----------------------------------------------------------------------------
PROMPT_CODE_ROLE = """You are a sharp Home Assistant dev assistant focused on YAML, Jinja, Python automations, javascript and custom components. Craft tight, scalable code that slots right into HA: 2-space YAML indents, smart Jinja defaults for None/unavailable (|default()), async Python with HA APIs, PEP 8, error logs, and entity validations. Scan uploads for structure, deps, and fixes—improve with specific changes. Use state_attr() safely, set device_class/units, tune scan_intervals."""

PROMPT_CODE_OUTPUT_FORMAT = """Output format: Return a direct JSON object (not stringified) with two fields:
{"response_text": "Concise steps, rationale, setup tips, tests, or troubleshooting.", "response_code": "Pure code without markdown fences."}
Rules: Keep explanations in response_text, raw code in response_code (no ``` fences). No code? Leave response_code empty. File uploads: Return full updated content in response_code. Multi-files: Separate with two blank lines and comment headers like '# file: filename.yaml'. Explain changes in response_text. Escape special characters within JSON strings (\\n for newlines, \\" for quotes). Use the user's language for response_text."""

# -----------------------------------------------------------------------------
# 8. OUTPUT FORMAT BLOCK (conversation/pipeline/tools modes)
# -----------------------------------------------------------------------------
PROMPT_OUTPUT_FORMAT = """Output Format:
- Follow the user's language and communication style.
- Use plain text only: no markdown, no emoji."""

# ==============================================================================
# END OF MODULAR PROMPT SYSTEM
# ==============================================================================

# Default prompts for AI Task and Code Task services
GROK_AI_TASK_PROMPT = """You are a Home Assistant AI Task assistant. Generate responses based on the provided instructions.
IMPORTANT OUTPUT RULES:
1. If a data structure is provided, respond with JSON containing ONLY the specified fields
2. If no data structure is provided, respond with plain text for automations or scripts
3. For structured responses: use JSON format with the exact field names requested
4. For automation responses: provide valid Home Assistant YAML configuration
5. Do not include explanations, comments, or additional text beyond what is requested
6. Ensure all YAML uses 2-space indentation and follows Home Assistant conventions"""


# Vision analysis system prompt - concise and factual
VISION_ANALYSIS_PROMPT = """Be concise and factual in your image analysis. Always respond in the user's language."""


# Default conversation mode options (Intelligent Pipeline)
RECOMMENDED_PIPELINE_OPTIONS = {
    CONF_CHAT_MODEL: RECOMMENDED_CHAT_MODEL,
    CONF_LLM_HASS_API: [],
    CONF_MAX_TOKENS: 1000,
    CONF_TEMPERATURE: 0.1,
    # User can extend the default pipeline prompt; we store only user part here
    CONF_PROMPT_PIPELINE: "",
    CONF_TOP_P: RECOMMENDED_TOP_P,
    CONF_LIVE_SEARCH: RECOMMENDED_LIVE_SEARCH,
    CONF_API_HOST: DEFAULT_API_HOST,
    CONF_ASSISTANT_NAME: RECOMMENDED_ASSISTANT_NAME,
    # Memory enabled by default for conversation
    CONF_STORE_MESSAGES: True,
    # Pipeline is enabled by default in this mode
    CONF_USE_INTELLIGENT_PIPELINE: True,
    # Allow smart home control by default (unified setting)
    CONF_ALLOW_SMART_HOME_CONTROL: True,
}

# Default conversation options for Home Assistant LLM API mode (Tools mode)
# Global endpoint by default; user can override with a regional one
RECOMMENDED_TOOLS_OPTIONS = {
    CONF_CHAT_MODEL: RECOMMENDED_CHAT_MODEL,
    CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
    CONF_PROMPT: "",  # User custom instructions for tools mode or simple chat mode
    CONF_MAX_TOKENS: RECOMMENDED_MAX_TOKENS,
    CONF_TEMPERATURE: RECOMMENDED_TEMPERATURE,
    CONF_TOP_P: RECOMMENDED_TOP_P,
    CONF_LIVE_SEARCH: RECOMMENDED_LIVE_SEARCH,
    CONF_STORE_MESSAGES: True,
    CONF_API_HOST: DEFAULT_API_HOST,
    CONF_ASSISTANT_NAME: RECOMMENDED_ASSISTANT_NAME,
    # Explicitly disable Intelligent Pipeline in this mode
    CONF_USE_INTELLIGENT_PIPELINE: False,
    # Allow smart home control by default (unified setting)
    CONF_ALLOW_SMART_HOME_CONTROL: True,
}

# Default options for the AI Task service
RECOMMENDED_AI_TASK_OPTIONS = {
    CONF_CHAT_MODEL: RECOMMENDED_AI_TASK_MODEL,
    CONF_IMAGE_MODEL: RECOMMENDED_IMAGE_MODEL,
    CONF_VISION_MODEL: RECOMMENDED_VISION_MODEL,
    CONF_LLM_HASS_API: [],
    CONF_MAX_TOKENS: 5000,
    CONF_TEMPERATURE: 0.1,
    CONF_PROMPT: GROK_AI_TASK_PROMPT,
    CONF_VISION_PROMPT: VISION_ANALYSIS_PROMPT,
    CONF_TOP_P: RECOMMENDED_TOP_P,
    CONF_LIVE_SEARCH: RECOMMENDED_LIVE_SEARCH,
    CONF_STORE_MESSAGES: False,
}

# Default options for the Grok Code Fast service
RECOMMENDED_GROK_CODE_FAST_OPTIONS = {
    CONF_CHAT_MODEL: RECOMMENDED_GROK_CODE_FAST_MODEL,
    CONF_MAX_TOKENS: 5000,
    CONF_TEMPERATURE: 0.1,
    CONF_PROMPT_CODE: "",  # Empty default - PromptManager builds prompt from modular blocks
    CONF_TOP_P: RECOMMENDED_TOP_P,
    CONF_LIVE_SEARCH: RECOMMENDED_LIVE_SEARCH,
    CONF_STORE_MESSAGES: True,  # Enable server-side memory via xAI previous_response_id chaining
    CONF_ASSISTANT_NAME: DEFAULT_CODE_FAST_ASSISTANT_NAME,
}

# ==============================================================================
# SENSOR CONFIGURATION CONSTANTS
# ==============================================================================

# Token pricing and cost calculation
TOKENS_PER_MILLION = 1_000_000  # Division factor for token pricing calculations
XAI_PRICING_CONVERSION_FACTOR = (
    10000.0  # API returns prices in units of 0.0001 USD per 1M tokens
)
# Sensor update intervals
PRICING_UPDATE_INTERVAL_HOURS = (
    48  # How often to fetch model pricing from xAI API (prices rarely change)
)
COST_UPDATE_INTERVAL_HOURS = (
    12  # How often to recalculate costs (runs after pricing update)
)

# New models detection
NEW_MODEL_NOTIFICATION_DAYS = (
    7  # How long to show new model notifications before auto-dismissal
)

# ==============================================================================
# END OF SENSOR CONFIGURATION CONSTANTS
# ==============================================================================

# Author's CONF_PROMPT_PIPELINE for improved custom intent recognition
# Treat all messages related to topics such as news, weather updates, almanacs, religious events, feasts or holidays, moon phases, family member locations, garbage collection schedules, or similar as custom Smart Home Command.
