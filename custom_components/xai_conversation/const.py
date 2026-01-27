"""Constants for xAI Grok Conversation integration."""

import logging
import re
from homeassistant.config_entries import ConfigEntry

# ==============================================================================
# SECTION 1: CORE INTEGRATION & DOMAINS
# ==============================================================================
type XAIConfigEntry = ConfigEntry
LOGGER = logging.getLogger(__package__)
DOMAIN = "xai_conversation"
DEFAULT_API_HOST = "api.x.ai"
DEFAULT_MANUFACTURER = "xAI"

# Default Component Names
DEFAULT_AI_TASK_NAME = "xAI Task"
DEFAULT_CONVERSATION_NAME = "xAI Conversation"
DEFAULT_SENSORS_NAME = "xAI Token Sensors"

# ==============================================================================
# SECTION 2: CONFIGURATION KEYS (CONF_*)
# ==============================================================================
# 1. Main Credentials & Global Settings
CONF_API_HOST = "api_host"
CONF_TIMEOUT = "timeout"

# 2. Model Selection
CONF_CHAT_MODEL = "chat_model"
CONF_IMAGE_MODEL = "image_model"
CONF_VISION_MODEL = "vision_model"

# 3. Mode & Capabilities Enablement
CONF_USE_INTELLIGENT_PIPELINE = "use_intelligent_pipeline"
CONF_ALLOW_SMART_HOME_CONTROL = "allow_smart_home_control"
CONF_USE_EXTENDED_TOOLS = "use_extended_tools"

# 4. Parameters & Tuning
CONF_MAX_TOKENS = "max_tokens"
CONF_REASONING_EFFORT = "reasoning_effort"
CONF_TEMPERATURE = "temperature"
CONF_TOP_P = "top_p"
CONF_LIVE_SEARCH = "live_search"
CONF_ZDR = "zero_data_retention"  # Privacy mode: No server-side storage

# 5. UI & Interaction Settings
CONF_STORE_MESSAGES = "store_messages"
CONF_SEND_USER_NAME = "send_user_name"
CONF_SHOW_CITATIONS = "show_citations"
CONF_ASSISTANT_NAME = "assistant_name"
CONF_LOCATION_CONTEXT = "location_context"

# 6. Prompts & Custom Logic
CONF_PROMPT_PIPELINE = "pipeline_prompt"
CONF_AI_TASK_PROMPT = "ai_task_prompt"  # System prompt for AI Task mode
CONF_PROMPT_TOOLS = "prompt_tools"  # Custom instructions for tools mode
CONF_VISION_PROMPT = "vision_prompt"  # System prompt for photo analysis
CONF_EXTENDED_TOOLS_YAML = "extended_tools_yaml"
CONF_PAYLOAD_TEMPLATE = "payload_template"

# Extended Tools Internals
EVENT_AUTOMATION_REGISTERED = (
    f"automation_registered_from_{DEFAULT_CONVERSATION_NAME.lower().replace(' ', '_')}"
)

# 7. Memory Management
CONF_MEMORY_USER_TTL_HOURS = "memory_user_ttl_hours"
CONF_MEMORY_DEVICE_TTL_HOURS = "memory_device_ttl_hours"
CONF_MEMORY_REMOTE_DELETE = "memory_remote_delete"
CONF_MEMORY_CLEANUP_INTERVAL_HOURS = "memory_cleanup_interval_hours"

# 8. Sensor Specifics
CONF_TOKENS_PER_MILLION = "tokens_per_million"
CONF_XAI_PRICING_CONVERSION_FACTOR = "xai_pricing_conversion_factor"
CONF_PRICING_UPDATE_INTERVAL_HOURS = "pricing_update_interval_hours"


# ==============================================================================
# SECTION 3: OPERATIONAL MODES & OPTIONS
# ==============================================================================
# Chat operational modes
CHAT_MODE_AI_TASK = "ai_task"
CHAT_MODE_CHATONLY = "chatonly"
CHAT_MODE_PIPELINE = "pipeline"
CHAT_MODE_TOOLS = "tools"
CHAT_MODE_VISION = "vision"

# Memory scope types
MEMORY_SCOPE_USER = "user"
MEMORY_SCOPE_DEVICE = "device"

# Model target types
MODEL_TARGET_CHAT = "chat"
MODEL_TARGET_VISION = "vision"
MODEL_TARGET_IMAGE = "image"

# Service Response Status Codes
STATUS_OK = "ok"
STATUS_ERROR = "error"

# Image defaults (xAI SDK always returns JPEG)
DEFAULT_IMAGE_MIME_TYPE = "image/jpeg"

# Live search modes
LIVE_SEARCH_OFF = "off"
LIVE_SEARCH_WEB = "web search"
LIVE_SEARCH_X = "x search"
LIVE_SEARCH_FULL = "full"
LIVE_SEARCH_OPTIONS = [
    LIVE_SEARCH_OFF,
    LIVE_SEARCH_WEB,
    LIVE_SEARCH_X,
    LIVE_SEARCH_FULL,
]

# ==============================================================================
# SECTION 4: MODULAR PROMPT SYSTEM
# ==============================================================================
# Final system prompt is built by PromptManager by assembling these blocks:

# 1. Identity Block
PROMPT_IDENTITY = "You are {assistant_name}, a powerful assistant for Home Assistant."

# 2. Role Base Block
PROMPT_ROLE_BASE = "Act as an advanced ASR/NLU system. If you are completely unable to interpret the text, ask with humor, but don't do any command."

# 3. Memory Context Blocks
PROMPT_MEMORY_SERVERSIDE = "Conversation history and timestamps are on the server. Use them for temporal context."
PROMPT_MEMORY_CLIENTSIDE = "History is provided in messages. Use only given context."
PROMPT_MEMORY_ZDR = (
    "Reasoning state maintained between turns. Limited message history available."
)

# 4. Smart Home Control (Pipeline Mode)
PROMPT_SMART_HOME_RECOGNITION = 'Recognize Smart Home Commands: actions like "play", "stop", "pause", "turn", "open", "close", "set", or status queries.'
PROMPT_CUSTOM_RULES = "Allow user-added custom rules, translating if needed. User custom rules take priority over tools."

PROMPT_PIPELINE_DECISION_LOGIC = """Action Decision:
- Single Smart Home Command: '[[HA_LOCAL: {"text": "<the recognized command in the user's language>"}]]'
- Multiple Smart Home Commands: '[[HA_LOCAL: {"commands": [{"text": "cmd1"}, {"text": "cmd2"}]}]]'
  - Add "sequential": true if commands must execute in order (e.g., "first do X, then Y")
  - Omit "sequential" or set false for independent parallel commands

Priority Rules:
- where you feel it is relevant, present the commands in a natural way. Focus on the action in progress.
- Smart Home commands/queries → [[HA_LOCAL]]
- General knowledge questions → Answer directly 
- When in doubt about device state/control → PRIORITIZE SMART HOME CONTROL"""


PROMPT_PIPELINE_EXAMPLES = """Examples:
- "turn on the living room light" → your comment, if any and [[HA_LOCAL: {"text": "turn on the living room light"}]]
- "(any text)  what's the temperature in the kitchen (other words)?" → your comment, if any and [[HA_LOCAL: {"text": "what's the temperature in the kitchen?"}]]
- "turn off lights in living room and turn on tv" → your comment, if any and [[HA_LOCAL: {"commands": [{"text": "turn off lights in living room"}, {"text": "turn on tv"}]}]]
- "first close the blinds, then turn off the lights" → your comment, if any and [[HA_LOCAL: {"commands": [{"text": "close the blinds"}, {"text": "turn off the lights"}], "sequential": true}]]"""

# 5. Smart Home Control (Tools Mode)
PROMPT_MODE_TOOLS = """Smart Home:
Devices available (CSV):
{static_context}
CRITICAL: 
- where you feel it is relevant, first focus on the action in progress.
- Use ONLY names/aliases from the CSV. If not found, inform user.
- Use specific tools before general ones."""

# 6. Restrictions (When control is disabled)
PROMPT_NO_CONTROL = """Limitations:
- You are NOT authorized to control smart home devices
- If user ask to control a device, reply that the user must enable the functionality in the custom component settings"""

# 7. Specialized Capabilities
PROMPT_SEARCH_USAGE = "Use search tools (Web Search or X Search) ONLY when explicitly requested by the user."

# 8. Output Formatting
PROMPT_OUTPUT_FORMAT = """OUTPUT:
- General Questions: answer concisely and truthfully.
- Follow the user's language and communication style.
- No markdown (*, #, `, -, •).
- No emojis.
- Plain text natural sentences (for TTS)."""

# -----------------------------------------------------------------------------
# SPECIFIC SERVICE PROMPTS
# -----------------------------------------------------------------------------
GROK_AI_TASK_PROMPT = (
    "Follow the instructions and respect the requested data structure exactly."
)

VISION_ANALYSIS_PROMPT = """Be concise and factual in your image analysis. Always respond in the user's language."""

# -----------------------------------------------------------------------------
# AUTHOR'S CONFIGURATION EXAMPLE
# -----------------------------------------------------------------------------
# This is an example of a custom prompt that can be used in CONF_PROMPT_PIPELINE
# to improve intent recognition for specific topics.
# Example:
# "Treat all messages related to topics such as news, weather updates, almanacs,
# religious events, feasts or holidays, moon phases, family member locations,
# garbage collection schedules, or similar as custom Smart Home Command."
# -----------------------------------------------------------------------------


# ==============================================================================
# SECTION 5: RECOMMENDED DEFAULTS & LIMITS
# ==============================================================================
# Identity Defaults
RECOMMENDED_ASSISTANT_NAME = "Jarvis"

# Default Model Selection (Optimized for Grok-3 models)
RECOMMENDED_CHAT_MODEL = "grok-4-1-fast-non-reasoning"
RECOMMENDED_ZDR_MODEL = "grok-4-1-fast-reasoning"
RECOMMENDED_AI_TASK_MODEL = "grok-code-fast-1"
RECOMMENDED_IMAGE_MODEL = "grok-2-image-1212"
RECOMMENDED_VISION_MODEL = "grok-2-vision-1212"

# Parameter Recommendations
RECOMMENDED_TEMPERATURE = 0.1
RECOMMENDED_TOP_P = 1
RECOMMENDED_REASONING_EFFORT = "low"
RECOMMENDED_LIVE_SEARCH = "off"
RECOMMENDED_STORE_MESSAGES = True
RECOMMENDED_SEND_USER_NAME = False
RECOMMENDED_SHOW_CITATIONS = False
RECOMMENDED_TIMEOUT = 60.0
RECOMMENDED_ZDR = False

# Memory Defaults
MEMORY_FLUSH_INTERVAL_MINUTES = 30
RECOMMENDED_MEMORY_USER_TTL_HOURS = 672  # 28 days
RECOMMENDED_MEMORY_DEVICE_TTL_HOURS = 672
RECOMMENDED_MEMORY_REMOTE_DELETE = False
RECOMMENDED_MEMORY_CLEANUP_INTERVAL_HOURS = 24

# Processing Limits
RECOMMENDED_HISTORY_LIMIT_TURNS = 10
RECOMMENDED_MAX_ITERATIONS = 5

# Aggregated Defaults
MEMORY_DEFAULTS = {
    CONF_MEMORY_USER_TTL_HOURS: RECOMMENDED_MEMORY_USER_TTL_HOURS,
    CONF_MEMORY_DEVICE_TTL_HOURS: RECOMMENDED_MEMORY_DEVICE_TTL_HOURS,
    CONF_MEMORY_REMOTE_DELETE: RECOMMENDED_MEMORY_REMOTE_DELETE,
    CONF_MEMORY_CLEANUP_INTERVAL_HOURS: RECOMMENDED_MEMORY_CLEANUP_INTERVAL_HOURS,
}

# Subentry Setup Defaults (Intelligent Pipeline)
RECOMMENDED_PIPELINE_OPTIONS = {
    CONF_CHAT_MODEL: RECOMMENDED_CHAT_MODEL,
    CONF_MAX_TOKENS: 1000,
    CONF_TEMPERATURE: RECOMMENDED_TEMPERATURE,
    CONF_PROMPT_PIPELINE: "",
    CONF_TOP_P: RECOMMENDED_TOP_P,
    CONF_LIVE_SEARCH: RECOMMENDED_LIVE_SEARCH,
    CONF_API_HOST: DEFAULT_API_HOST,
    CONF_ASSISTANT_NAME: RECOMMENDED_ASSISTANT_NAME,
    CONF_TIMEOUT: RECOMMENDED_TIMEOUT,
    CONF_SEND_USER_NAME: RECOMMENDED_SEND_USER_NAME,
    CONF_SHOW_CITATIONS: RECOMMENDED_SHOW_CITATIONS,
    CONF_STORE_MESSAGES: RECOMMENDED_STORE_MESSAGES,
    CONF_USE_INTELLIGENT_PIPELINE: True,
    CONF_ALLOW_SMART_HOME_CONTROL: True,
    CONF_USE_EXTENDED_TOOLS: False,
}

# Subentry Setup Defaults (AI Task)
RECOMMENDED_AI_TASK_OPTIONS = {
    CONF_CHAT_MODEL: RECOMMENDED_AI_TASK_MODEL,
    CONF_IMAGE_MODEL: RECOMMENDED_IMAGE_MODEL,
    CONF_VISION_MODEL: RECOMMENDED_VISION_MODEL,
    CONF_MAX_TOKENS: 5000,
    CONF_TEMPERATURE: RECOMMENDED_TEMPERATURE,
    CONF_AI_TASK_PROMPT: GROK_AI_TASK_PROMPT,
    CONF_VISION_PROMPT: VISION_ANALYSIS_PROMPT,
    CONF_TOP_P: RECOMMENDED_TOP_P,
    CONF_LIVE_SEARCH: RECOMMENDED_LIVE_SEARCH,
    CONF_API_HOST: DEFAULT_API_HOST,
    CONF_TIMEOUT: RECOMMENDED_TIMEOUT,
    CONF_STORE_MESSAGES: False,
}

# Subentry Setup Defaults (Sensors)
RECOMMENDED_SENSORS_OPTIONS = {
    CONF_TOKENS_PER_MILLION: 1_000_000,
    CONF_XAI_PRICING_CONVERSION_FACTOR: 10000.0,
    CONF_PRICING_UPDATE_INTERVAL_HOURS: 48,
}


# ==============================================================================
# SECTION 6: MODEL & PRICING CONFIGURATION
# ==============================================================================
# Supported models are populated dynamically at runtime via XAIModelManager
SUPPORTED_MODELS: list[str] = []
REASONING_EFFORT_MODELS: list[str] = []

# Pricing Constants
XAI_PRICING_CONVERSION_FACTOR = 10000.0
RECOMMENDED_TOKENS_PER_MILLION = 1_000_000
RECOMMENDED_PRICING_UPDATE_INTERVAL_HOURS = 48
DEFAULT_TOOL_PRICE_RAW = 5000.0 * XAI_PRICING_CONVERSION_FACTOR

# Sensor Labels
CLEARER_VISION_LABEL = "Vision Input"
CLEARER_SEARCH_LABEL = "Search"
CLEARER_CACHED_LABEL = "Cached Input"


# ==============================================================================
# SECTION 7: PARSING PATTERNS (REGEX)
# ==============================================================================
# HA_LOCAL Tag detection (Intelligent Pipeline Mode)
HA_LOCAL_TAG_PATTERN = re.compile(r"\[\[\s*HA_LOCAL\s*:\s*({.*?})\]\]", re.DOTALL)
HA_LOCAL_TAG_PREFIX = "[["

# Pattern to extract JSON from markdown code fences (```json ... ```)
JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)

# Fallback field parsing (when JSON is malformed)
HA_LOCAL_COMMANDS_PATTERN = re.compile(r'"commands"\s*:\s*(\[.*?\])', re.DOTALL)
HA_LOCAL_SEQUENTIAL_PATTERN = re.compile(r'"sequential"\s*:\s*(true|false)', re.DOTALL)
HA_LOCAL_TEXT_PATTERN = re.compile(r'"text"\s*:\s*"([^"]*)"', re.DOTALL)
