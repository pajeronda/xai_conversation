<div align="center">
  <h1>xAI Conversation</h1>

  <img src="https://github.com/pajeronda/xai_conversation/blob/main/images/logo.png?raw=true" alt="xAI Conversation Logo" width="400">

  <b>Bringing the power of xAI Grok to your Home Assistant.</b>

  [![GitHub Release](https://img.shields.io/github/release/pajeronda/xai_conversation.svg?style=flat-square)](https://github.com/pajeronda/xai_conversation/releases)
  [![Home Assistant](https://img.shields.io/badge/Home%20Assistant-2025.10%2B-green)](https://www.home-assistant.io/blog/categories/core/)
  [![hacs](https://img.shields.io/badge/HACS-Default-blue.svg?style=flat-square)](https://my.home-assistant.io/redirect/config_flow_start/?domain=xai_conversation)
  [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg?style=flat-square)](LICENSE)
</div>

---

# Conversation Agent
A complete conversation agent integration for Home Assistant that offers multiple ways to manage your conversation data and interact with your smart home through Grok.

<p align="center" style="padding:2%">
  <img width="400" alt="Assist Chat" src="https://github.com/pajeronda/xai_conversation/blob/main/images/assist.png" />
</p>

---

## 🚀 Installation

### Via HACS (Recommended)
1. Open **HACS** in Home Assistant.
2. Search for **"xAI Conversation"** and click **Download**.
3. Restart Home Assistant.
4. Go to **Settings** → **Devices & Services** → **Add Integration** → **xAI Conversation**.
5. Enter your **xAI API key** (get one at [x.ai](https://x.ai)).

[![Install via HACS](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=Pajeronda&repository=xai_conversation&category=integration)

[![Configure Integration](https://my.home-assistant.io/badges/config_flow_start.svg)](https://my.home-assistant.io/redirect/config_flow_start/?domain=xai_conversation)

---

## ✨ Key Features

| Feature | Description |
| :--- | :--- |
| **Conversation Continuity** | When using *Server-side Memory*, chats continue seamlessly across any device configured with your user account (Smartphone, Tablet, PC, etc.). |
| **Location Context** | Set your home location (e.g., "Rome, IT") to help Grok with local searches like weather and news. Leave empty to use the Home Assistant time zone instead. |
| **User Recognition** | Enable "Include username in messages" so Grok can identify and address you personally by your `person` entity name (or user display name as fallback). |

## 🔄 Operational Modes
Select the best balance between privacy, cost, and efficiency:

| Mode | Memory Location | Privacy | Cost | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **Server-side Memory (default)** | xAI Cloud | Standard | **Lowest** | Efficiency, speed, and long-running conversations. |
| **Zero Data Retention (ZDR)** | Local blob (Encrypted) | **Maximum** | Medium | Privacy-focused context preservation (requires reasoning model). |
| **Local Chatlog** | Home Assistant | Standard | Highest | Full local history control with stateless turn-by-turn processing. |

> **NOTE**
> - **Automatic Model Selection**: When ZDR is enabled, the integration automatically selects a compatible **reasoning model** to correctly re-initialize the conversation from encrypted state blobs.
> - **Local Chatlog** is active when both *Server-side Memory* and *Zero Data Retention* are disabled.


## 🛠️ Interaction Modes
Grok acts as an advanced **ASR/NLU orchestrator**, providing three distinct ways to interact with your Home Assistant instance:

| Mode | Description |
| :--- | :--- |
| **Intelligent Pipeline (Default)** | Leverages the **Home Assistant Intent Pipeline**. Works flawlessly with official and custom intents. Automatically falls back to Tool Control if needed to guarantee a premium experience. |
| **Tool Control** | Direct interaction via **Tool Calling**. Use **Home Assistant Standard Tools** or **Extended Tools** (YAML). |
| **Chat Only** | A pure conversational experience. Chat with Grok for general knowledge or assistance without interacting with Home Assistant entities. |

> **NOTE**
> * In the configuration agent options:
>   * **Intelligent Pipeline**: Enabled by default, requires **"Allow smart home control"** to be ON.
>   * **Tool Control**: Active when **"Allow smart home control"** is ON and **"Enable Intelligent Pipeline"** is OFF.
>   * **Chat Only**: Active when **"Allow smart home control"** is OFF.
>
> * Switching between these modes starts a new conversation with a separate history.
> * The configuration page may automatically reload when toggling certain options (like *Operational Modes*, *Zero Data Retention*, or *Memory Settings*) to correctly apply internal logic and dynamic fields. This is expected behavior.


## 🔧 Extra Tools

### Extended Tools (YAML)
This component supports **custom tools** in YAML format, fully compatible with [Extended OpenAI Conversation](https://github.com/jekalmin/extended_openai_conversation). Enable and configure them in the integration's global configuration settings, then activate them per conversation agent settings.

<p align="center" style="padding:2%">
  <img src="https://github.com/pajeronda/xai_conversation/blob/main/images/extended_tools_global.png?raw=true" alt="Extended Tools Configuration in global configuration settings" width="400">

</p>

### Home Assistant Extended Tools
This integration extends the standard Home Assistant tools with additional capabilities for controlling **helpers**, **scripts**, and **automations**. These tools are automatically available when **"Allow smart home control"** is enabled and you are using the standard Home Assistant tools.

| Tool | Description |
| :--- | :--- |
| **HassSetInputNumber** | Sets the value of an `input_number` entity (volume, temperature, etc.) |
| **HassSetInputBoolean** | Turns an `input_boolean` entity on or off (toggles, flags) |
| **HassSetInputText** | Sets the text value of an `input_text` entity |
| **HassRunScript** | Executes a Home Assistant script |
| **HassTriggerAutomation** | Triggers a Home Assistant automation |

### xAI Tools
Enable **Live Search** (web) and **𝕏 Search** in the conversation agent options for real-time information with **citations** (if you enable citations in the conversation agent options).

---

# Stateless data generation

## ⚙️ AI Task Integration
Stateless data generation, smart automations, and image generation.

## 💭 ASK (`xai_conversation.ask`)
A highly versatile, stateless service designed for one-shot queries. Similar to AI Task, it doesn't maintain history, but it allows you to override the **model**, **temperature**, and **live search** settings "on the fly" for each individual call. It returns the response directly in a variable.

**Use Case**: Ideal for processing raw sensor data, distilling information from the web, or any task that requires a specific Grok model different from your default assistant.

```yaml
service: xai_conversation.ask
data:
  instructions: "{{ instructions }}"
  input_data: "{{ data_to_send }}"
  model: grok-4-1-fast-non-reasoning  # (optional)
  temperature: 0.7  # (optional)
  max_tokens: 1000  # (optional)
  reasoning_effort: "low"  # (optional) - for reasoning models
  live_search: "web search"  # (optional) - web search, x search, or full
  show_citations: true  # (optional)
response_variable: output_xai
```

## 📸 Photo Analysis (`xai_conversation.photo_analysis`)
A stateless vision service to analyze snapshots or external images. Similarly to the `ask` service, it is fully configurable per call and can be integrated into **Automations**, **Scripts**, or used for manual inspection via **Developer Tools**.

```yaml
service: xai_conversation.photo_analysis
data:
  prompt: "Is there a package at the front door?"
  images:
    - /config/www/doorbell_snapshot.jpg
  temperature: 0.1  # (optional)
  top_p: 1.0        # (optional)
```

---

# 🔧 Maintenance
Services and Sensors for model, price cost, system health, resetting stats, or clearing history.

## 📊 Real-time Monitoring
Detailed sensors for:

- model costs
- token counts
- cache efficiency
- xAI Server tool invocations
- notifications for new model releases.

## 🛠️ Manage Sensors 
Forcefully refresh data or reset counters. Primarily used via **Developer Tools** or triggered by **Automations**.

```yaml
service: xai_conversation.manage_sensors
data:
  reload_pricing: true #(or false) - Force update models and pricing from xAI
  reset_stats: false #(or true) - Reset all token and cost sensors to zero
```

## 🧹 Clear Memory
Clear conversation history for specific users or devices. Use it in **Developer Tools** for maintenance or triggered by **Automations**.

```yaml
service: xai_conversation.clear_memory
data:
  user_id: person.admin  # Clear memory for a specific person
  include_server: true   # Also delete history from xAI servers
```

---

# ⚠️ Deprecation Notice: Grok Code Fast
The **Grok Code Fast** service and the associated card are now **unsupported** (available up to release v2.2.1).

We recommend moving to the new and much more powerful **AI Code Task** integration:

[![GitHub](https://img.shields.io/badge/GitHub-Pajeronda%2FAI_Code_Task-blue)](https://github.com/pajeronda/ai_code_task)

[![Install via your HACS instance.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=Pajeronda&repository=ai_code_task&category=integration)

**AI Code Task** is an agnostic simple IDE provider in a Lovelace card for code generation. It provides a superior workflow for modifying and generating YAML, Jinja2 templates, Python scripts, and more with an integrated pro editor.

# 🤝 Contributing & Support
- 🐛 **Bugs**: [GitHub Issues](https://github.com/pajeronda/xai_conversation/issues)

- 💡 **Feature Request**: [GitHub Issues](https://github.com/pajeronda/xai_conversation/issues)

- 📖 **Documentation**: [Wiki](https://github.com/pajeronda/xai_conversation/wiki) _(coming soon)_

- ⭐ **Love this project? Give it a star to show your support!**

  [![GitHub stars](https://img.shields.io/github/stars/pajeronda/xai_conversation?label=%E2%AD%90%20Stars&style=for-the-badge)](https://github.com/pajeronda/xai_conversation/stargazers)


# Legal Notes

- This is a custom component for Home Assistant developed by [@pajeronda](https://github.com/pajeronda)

- This custom component is released under the [GNU General Public License v3.0](LICENSE)

- This custom component uses the official [xAI Python SDK](https://github.com/xai-org/xai-sdk-python)

- **API Usage**: This custom component requires an active xAI account and a valid API key. Use of the xAI API is subject to [xAI's terms of service](https://x.ai/legal/).

- **Trademarks**: xAI, Grok, and related logos are registered trademarks of xAI Corp. This custom component is an **unofficial** integration for **Home Assistant** and is not affiliated with, sponsored by, or endorsed by xAI Corp.
