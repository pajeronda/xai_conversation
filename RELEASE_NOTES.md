# Release Notes - v1.2.0

## 🎯 Overview
This release significantly improves reliability and performance of the integration, with special focus on optimization for resource-constrained devices (Raspberry Pi, Proxmox VMs) and critical bug fixes.

---

## ✨ What's New

### 🚀 Performance Improvements
- **Separate API vs Local Timing**: Now you can see how much time the xAI API takes vs local processing
  - Helps identify if slowness comes from the API or your system
  - Visible in debug logs with `api_time` and `local_process_time`

- **Faster Operations**: Save operations (tokens, memory) no longer block conversations
  - Everything is saved in background while you continue talking to the assistant
  - Less waiting time between voice commands

- **Reduced System Load**: Optimizations for Raspberry Pi and low-power devices
  - Reduced CPU usage when not necessary
  - Better log handling (no resource waste when logging is disabled)

### 💰 Cost Tracking Fixed
- **Pricing Issue Resolved**: Cost sensors now show correct values in USD
  - Previous conversion error showing wrong prices fixed
  - Cost display is now accurate and reliable

### 🖼️ Better Image Support
- **Automatic Fallback for Vision Models**: If vision model fails, automatically switches to text model
  - No more blocking errors when sending photos
  - Integration chooses the best available model

### 🐛 Bug Fixes
- **Fixed Crash with Tools Mode and Memory Disabled**:
  - Previous error: `TypeError: assistant() got an unexpected keyword argument 'tool_calls'`
  - Now works perfectly in all configurations

- **Faster Historical Messages**: When server-side memory is disabled, history sending is much faster
  - Previously performed unnecessary checks on old messages
  - Now only checks the last message, saving precious time

---

## 🔧 Technical Improvements

### New Helper Modules
For developers and customizers:
- **`log_time_services.py`**: Automatically handles API call timing
- **`token_storage.py`**: Saves token statistics more efficiently
- **`integration_setup.py`**: Centralizes integration setup logic
- **`model_manager.py`**: Manages model information and pricing

### Code Optimizations
- Improved code organization for easier future maintenance
- Constants instead of "magic numbers" for better clarity
- Better defined data types to reduce errors
- Lazy logging evaluation for performance on low-end devices

---

## 📦 Installation

### Upgrade
This version is **fully backward compatible** with previous versions. No configuration changes required!

1. Update via HACS or manually replace the `xai_conversation` folder
2. Restart Home Assistant
3. All your settings and saved conversations are preserved

### New Files
If installing manually, make sure to include:
- `custom_components/xai_conversation/helpers/log_time_services.py`
- `custom_components/xai_conversation/helpers/token_storage.py`
- `custom_components/xai_conversation/helpers/integration_setup.py`
- `custom_components/xai_conversation/helpers/model_manager.py`

---

## 🎮 Enhanced Multi-Agent Support

Create multiple conversation agents with different configurations:
- Each agent can have independent memory and mode settings
- Perfect for creating specialized assistants (kitchen, bedroom, etc.)
- Costs are tracked globally across the integration

**Practical example:**
- Agent "Fast": tools mode, memory OFF, for quick commands
- Agent "Conversational": pipeline mode, memory ON, for long chats

---

## 📊 What Changes for You

### Before this update:
- ❌ Errors with memory disabled and tools mode
- ❌ Costs displayed incorrectly
- ⚠️ Slowdowns on Raspberry Pi
- ⚠️ Save operations slowing down conversations

### After this update:
- ✅ Everything works in every configuration
- ✅ Accurate and reliable costs
- ✅ Improved performance on all devices
- ✅ Smoother conversations (background saving)
- ✅ Clearer response times in logs

---

## 🧪 Testing Performed

- ✅ Python syntax validation on all files
- ✅ Tools mode tests with memory ON and OFF
- ✅ Pipeline mode tests with memory ON and OFF
- ✅ Cost and pricing accuracy verification
- ✅ Multi-agent testing with different configurations
- ✅ Performance testing on low-resource devices

---

## 📝 Technical Notes

### Change Statistics
- **33 files modified**
- **+4,263 lines added / -2,792 removed**
- **4 new helper modules** for better organization
- **No breaking changes** - fully backward compatible

### Compatibility
- ✅ Home Assistant 2024.1+
- ✅ Python 3.11+
- ✅ xAI SDK latest version
- ✅ Raspberry Pi 3/4/5
- ✅ Proxmox VMs / Docker

---

## 🙏 Acknowledgments

Thanks to all users who reported bugs and requested improvements. This update is the result of an intensive week of development focused on:
- Reliability
- Performance
- User experience

---

## 📅 Release Date
2025-11-29

---

## 🔗 Useful Links
- [GitHub Repository](https://github.com/pajeronda/xai_conversation)
- [Report Bugs](https://github.com/pajeronda/xai_conversation/issues)
- [Documentation](https://www.github.com/pajeronda/xai_conversation)

---

## 🆘 Support

If you have issues after the update:
1. Restart Home Assistant
2. Check logs for any errors
3. Open an issue on GitHub with details

**Enjoy xAI Conversation! 🎉**
