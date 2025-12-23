<div align="center">
  <h1>xAI Conversation</h1>

  <img src="https://github.com/pajeronda/xai_conversation/blob/main/images/logo.png?raw=true" alt="xAI Grok Conversation Logo" width="300">
  
  <b>A custom Home Assistant integration that brings xAI Grok AI to your smart home.</b>

  [![GitHub Release](https://img.shields.io/github/release/pajeronda/xai_conversation.svg?style=flat-square)](https://github.com/pajeronda/xai_conversation/releases)
  [![Home Assistant](https://img.shields.io/badge/Home%20Assistant-2025.10%2B-green)](https://www.home-assistant.io/blog/categories/core/)
  [![hacs](https://img.shields.io/badge/HACS-Default-blue.svg?style=flat-square)](https://my.home-assistant.io/redirect/config_flow_start/?domain=xai_conversation)
  [![License](https://img.shields.io/github/license/pajeronda/xai_conversation.svg?style=flat-square)](LICENSE)
</div>

## Key Features
### ⚙️ xAI Python SDK: 
- This component uses [xAI Python SDK](https://github.com/xai-org/xai-sdk-python)

- 🗣️ 🧠 Grok functions as an advanced ASR/NLU system.

### 💬 Intelligent Conversation Agent
- **Two Operating Modes**:
  - **Intelligent Pipeline**: Grok delegates device commands using Home Assistant's **conversation/process** service
  - **Tools Mode**: Direct access to HA tools via tool calling (Home Assistant's standard LLM API)
  - **Conversation Style**: Grok follows your conversation style
  - **Conversation Continuity**: Chat continues seamlessly across any device configured with your user account (Smartphone, Tablet, PC, etc.)
<p align="center">
  <img width="400" height="400" alt="immagine chat Assist 1" src="https://github.com/user-attachments/assets/6aa4677e-80c1-42f9-8461-a7694104fff7" />

  <img width="400" height="400" alt="immagine chat Assist 2" src="https://github.com/user-attachments/assets/83b79a58-be34-4d6f-8967-0eae90d65cc7" />
</p>

### ⚙️ AI Task Service
Integration with Home Assistant's AI Task service for:
- Intelligent automations
- Scripts with dynamic logic
- Context-aware actions based on home state
- Image generation (automatically uses the `grok-2-image` model)

### 💻 Grok Code Fast
Dedicated service for Home Assistant code generation:
- YAML for automations and configurations
- Jinja2 templates
- Custom Python scripts
- **Lovelace Card** with integrated editor and interactive chat


### 📸 Photo Analysis Service
New `photo_analysis` service for analyzing images with AI. Can be used in scripts and automations by passing camera snapshots or external image URLs.

**Analyze a local camera snapshot:**
```yaml
service: xai_conversation.photo_analysis
data:
  prompt: "What do you see in this image?"
  images:
    - /config/www/camera_snapshot.jpg
```

**Analyze an external image:**
```yaml
service: xai_conversation.photo_analysis
data:
  prompt: "Describe this image in detail"
  images:
    - https://design.home-assistant.io/images/brand/logo.png
```

**Analyze multiple images:**
```yaml
service: xai_conversation.photo_analysis
data:
  prompt: "Compare these two images and highlight the differences"
  images:
    - /config/www/before.jpg
    - /config/www/after.jpg
```

### 💭 ASK
**xai_conversation.ask** service allowing stateless LLM queries with raw input data and system instructions, returning the response directly in a variable.
```yaml
 service: xai_conversation.ask
 data:
   max_tokens: 800
   temperature: 1
   instructions:  "{{ instructions }}"
   input_data: "{{ data_to_send }}"
 response_variable: output_ai
```

## 🛠️ Extended Tools

Starting from release 2.2.0, you can use new tools in YAML format, fully compatible with the format used in the [Extended OpenAI Conversation](https://github.com/jekalmin/extended_openai_conversation) integration.

### Configuration

1. **Global Definition**: Go to the integration page and click on the general configuration icon (Configure).

   <img src="https://github.com/pajeronda/xai_conversation/blob/main/images/extended_tools_global.png?raw=true" alt="Global Configuration" width="500">

   When the configuration dialog opens, toggle the **"Enable Extended Tools Configuration"** boolean and click **Submit**.
   Now a new editor field will appear where you can create or paste functions in YAML mode to be used as tools.

   **YAML Example**
   ```yaml
   - spec:
       name: get_attributes
       description: Get attributes of any home assistant entity
       parameters:
         type: object
         properties:
           entity_id:
             type: string
             description: entity_id
         required:
         - entity_id
     function:
       type: template
       value_template: "{{states[entity_id]}}"
   ```

2. **Enable per Agent**: Go to the options of the specific agent (xAI Conversation or any manually configured one) and toggle the **"Use Extended Tools (Global Config)"** boolean. Click **Submit** to save.
   
   *Note: This setting applies only to the selected conversation sub-entry. Other conversation sub-entries will continue to use the default standard HA tools.*

   <img src="https://github.com/pajeronda/xai_conversation/blob/main/images/extended_tools_enable.png?raw=true" alt="Enable per Agent" width="500">


### 💬 Conversation Memory Management
- **Server-side**: Persistent conversations managed by xAI (saves tokens and costs)
- **Client-side**: Conversations managed with local history (Home Assistant standard, more expensive)
- **Separate Configuration** for **users** and **Assist** satellite devices
- **Chat History** synchronizable across user devices in the **grok-code-fast-card** service

<p align="center">
  <img src="https://github.com/pajeronda/xai_conversation/blob/main/images/memory_settings.png?raw=true" alt="Memory Configuration" width="600">
</p>

### 📊 Token and Cost Monitoring
Detailed sensors that track:
- Tokens per service (Conversation, AI Task, Code Fast)
- Cache hit ratio (server-side memory efficiency)
- Real-time cost estimates
- Statistics per model used

<p align="center">
  <img src="https://github.com/pajeronda/xai_conversation/blob/main/images/sensors.png?raw=true" alt="Token Sensors Dashboard" width="600">
</p>

---

<p align="center">
  <img src="https://github.com/pajeronda/xai_conversation/blob/main/images/integration.png?raw=true" alt="Integration Overview" width="700">
</p>

## Installation

### Via HACS (Recommended)

Click this badge to install **xAI Conversation** via **HACS**

[![Install via your HACS instance.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=Pajeronda&repository=xai_conversation&category=integration)

Click this badge after restart Home Assistant to configure **xAI Conversation**

[![Open your Home Assistant instance and start setting up the integration.](https://my.home-assistant.io/badges/config_flow_start.svg)](https://my.home-assistant.io/redirect/config_flow_start/?domain=xai_conversation)


**Manual HACS**
1. Open **HACS** in Home Assistant
2. Go to **Integrations** → **Menu (⋮)** → **Custom repositories**
3. Add: `https://github.com/pajeronda/xai_conversation`
4. Category: **Integration**
5. Search for "**xAI Grok Conversation**" and click **Download**
6. Restart Home Assistant

### Manual Installation
1. Download the latest release of [xAI Grok Conversation](https://github.com/pajeronda/xai_conversation/releases)
   (alternatively, download the zip from the "<> code" badge at the top of this page)
2. Extract and copy the `xai_conversation` folder to `config/custom_components/`
3. Restart Home Assistant

## Grok Code Fast Lovelace Card

The `grok-code-fast-card` custom card offers:
- 💬 Interactive chat with Grok
- 📝 Code editor with syntax highlighting
- 📋 Quick copy of generated code
- 🔄 Automatic synchronization across devices
- 📱 Responsive layout (desktop/mobile)

### Via HACS (Recommended)

Click this badge to install **Grok code fast card** via **HACS** (recommended)

[![Install via your HACS instance.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=Pajeronda&repository=grok-code-fast-card&category=plugin)

**Manual**
- Follow the instructions on the GitHub page [Grok code fast card](https://github.com/pajeronda/grok-code-fast-card/)

## Configuration

### 1. Get xAI API Key
1. Go to [console.x.ai](https://console.x.ai)
2. Create an account or log in
3. Generate a new API key
4. Configure the new API Key

### 2. Add the Integration
1. Go to **Settings** → **Devices & Services** → **Add Integration**
2. Search for "**xAI Grok Conversation**"
3. Enter your xAI API key
4. Configure initial preferences (e.g., name and web search)


### 3. Set as Default Assistant
1. Go to **Settings** → **Voice assistants**
2. Configure a new voice assistant
3. Choose a name and select "**xAI Conversation**" from the conversation agents
4. Complete the configuration with your desired parameters

## Usage Examples

### Conversation Agent
- "Turn on the living room lights"
- "What's the temperature in the kitchen?"
   
### Grok Code Fast
- "Create an automation that turns off the lights at 11 PM"
- "Generate a Jinja2 template to display energy consumption"

## Supported Models

The integration **automatically retrieves all available models** from xAI, including their pricing and capabilities. All model information is exposed through dedicated sensors for real-time monitoring.

**Recommended models by service type:**

- **Conversation Agent**: `grok-4-1-fast-non-reasoning` (fast, economical, optimized for Home Assistant)
- **Grok Code Fast**: `grok-code-fast-1` (optimized for YAML/Python/Jinja2 code generation)
- **AI Task**: `grok-code-fast-1` (optimized for structured responses and automations)
- **Image Generation**: `grok-2-image-1212` (Aurora - automatic when using image generation)
- **Photo Analysis**: `grok-2-vision-1212` (Vision - optimized for image analysis)
- **xai_conversation.ask**: select your prefered model (default: reccomanded Conversation agent model)

> **Note:** When xAI releases **new models**, they are automatically detected and made available in the configuration options.


## Pricing and Token Monitoring

The integration **automatically retrieves real-time pricing** from the xAI API and exposes it through dedicated sensors.

**Available aggregate sensors:**
- **Total tokens**: Overall token consumption across all services
- **Average tokens per message**: Average token usage per interaction
- **Estimated cost**: Real-time cost estimation based on current pricing
- **Cache ratio per service**: Cache hit percentage for Conversation, AI Task, and Code Fast
- **Last tokens per service**: Most recent token usage for each service (Conversation, AI Task, Code Fast)
- **Server tool invocations**: Count of xAI agent tools usage (web_search, code_execution, etc.)
- **Stats reset timestamp**: Last statistics reset time

**Model pricing sensors:**
Real-time pricing (input, cached input, output) for all available models, automatically updated from xAI API.

**New models detection:**
Automatically detects when xAI releases new models and makes them available in configuration options.

All pricing information is fetched dynamically from xAI and kept up-to-date automatically.

## Troubleshooting

### Integration doesn't appear
- Verify the folder is in `custom_components/xai_conversation/`
- Restart Home Assistant
- Check the logs: **Settings** → **System** → **Logs**

### API Key errors on [console.x.ai](https://console.x.ai)
- Verify the key is correct
- Check usage limits in your xAI account
- Ensure the key has sufficient permissions

### Assistant doesn't respond
- Verify the agent is selected in **Settings** → **Voice assistants**
- Check that device control is enabled (if you want device control)
- Check the Home Assistant log

### Low cache ratio
- Normal in initial conversations ("cold" cache)
- Increases with repeated conversations on the same topic
- Server-side memory (`store_messages=True`) improves cache

## Versioning

This integration generally follows [Semantic Versioning (SemVer)](https://semver.org/) to ensure a clear and predictable approach to versioning. Semantic Versioning uses a three-part version number in the format `MAJOR.MINOR.PATCH`, where:

- **MAJOR** version increments indicate backwards-incompatible changes.
- **MINOR** version increments indicate the addition of backward-compatible functionality.
- **PATCH** version increments indicate backward-compatible bug fixes.


## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a branch for your feature (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Support

- 🐛 **Bug Report**: [GitHub Issues](https://github.com/pajeronda/xai_conversation/issues)
- 💡 **Feature Request**: [GitHub Issues](https://github.com/pajeronda/xai_conversation/issues)
- 📖 **Documentation**: [Wiki](https://github.com/pajeronda/xai_conversation/wiki) _(coming soon)_

## Credits

Developed by [@pajeronda](https://github.com/pajeronda)

Integration based on:
- [xAI SDK](https://github.com/xai-org/xai-sdk-python)
- [Home Assistant](https://www.home-assistant.io/)

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## Legal Notes

- **API Usage**: This integration requires an active xAI account and a valid API key. Use of the xAI API is subject to [xAI's terms of service](https://x.ai/legal/).

- **Trademarks**: xAI, Grok, and related logos are registered trademarks of xAI Corp. This project is an **unofficial** integration developed by [@pajeronda](https://github.com/pajeronda) and is not affiliated with, sponsored by, or endorsed by xAI Corp.
