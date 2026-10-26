"""Constants and language helpers for xAI Voice."""

from __future__ import annotations

TTS_LANGUAGES = [
    "auto",
    "en",
    "en-US",
    "en-GB",
    "en-AU",
    "ar-EG",
    "ar-SA",
    "ar-AE",
    "bn",
    "bn-BD",
    "zh",
    "zh-CN",
    "fr",
    "fr-FR",
    "de",
    "de-DE",
    "hi",
    "hi-IN",
    "id",
    "id-ID",
    "it",
    "it-IT",
    "ja",
    "ja-JP",
    "ko",
    "ko-KR",
    "pt-BR",
    "pt-PT",
    "ru",
    "ru-RU",
    "es-MX",
    "es-ES",
    "tr",
    "tr-TR",
    "vi",
    "vi-VN",
]
TTS_LANGUAGES = sorted(
    set(TTS_LANGUAGES) | {language.lower() for language in TTS_LANGUAGES}
)

STT_LANGUAGES = [
    "auto",
    "ar",
    "ar-EG",
    "ar-SA",
    "ar-AE",
    "bs",
    "bs-BA",
    "bg",
    "bg-BG",
    "ca",
    "ca-ES",
    "yue",
    "zh",
    "zh-CN",
    "hr",
    "hr-HR",
    "cs",
    "cs-CZ",
    "da",
    "da-DK",
    "nl",
    "nl-NL",
    "en",
    "en-US",
    "en-GB",
    "en-AU",
    "fi",
    "fi-FI",
    "fil",
    "fil-PH",
    "fr",
    "fr-FR",
    "de",
    "de-DE",
    "el",
    "el-GR",
    "hu",
    "hu-HU",
    "hi",
    "hi-IN",
    "id",
    "id-ID",
    "it",
    "it-IT",
    "ja",
    "ja-JP",
    "ko",
    "ko-KR",
    "mk",
    "mk-MK",
    "ms",
    "ms-MY",
    "nb",
    "nb-NO",
    "fa",
    "fa-IR",
    "pl",
    "pl-PL",
    "pt",
    "pt-BR",
    "pt-PT",
    "ro",
    "ro-RO",
    "ru",
    "ru-RU",
    "sk",
    "sk-SK",
    "es",
    "es-MX",
    "es-ES",
    "sv",
    "sv-SE",
    "th",
    "th-TH",
    "tr",
    "tr-TR",
    "uk",
    "uk-UA",
    "ur",
    "ur-PK",
    "vi",
    "vi-VN",
]
STT_LANGUAGES = sorted(
    set(STT_LANGUAGES) | {language.lower() for language in STT_LANGUAGES}
)

STT_FORMAT_LANGUAGES = {
    "ar",
    "de",
    "en",
    "es",
    "fr",
    "ja",
    "pt",
    "ru",
    "sv",
    "vi",
    "zh",
}

XAI_VOICES = {
    "ara": "Ara",
    "atlas": "Atlas",
    "aurora": "Aurora",
    "carina": "Carina",
    "castor": "Castor",
    "celeste": "Celeste",
    "cosmo": "Cosmo",
    "eve": "Eve",
    "helix": "Helix",
    "helios": "Helios",
    "iris": "Iris",
    "kepler": "Kepler",
    "leo": "Leo",
    "liora": "Liora",
    "lumen": "Lumen",
    "luna": "Luna",
    "lux": "Lux",
    "naksh": "Naksh",
    "orion": "Orion",
    "perseus": "Perseus",
    "rex": "Rex",
    "rigel": "Rigel",
    "sal": "Sal",
    "sirius": "Sirius",
    "ursa": "Ursa",
    "zagan": "Zagan",
    "zenith": "Zenith",
}


def normalize_tts_language(language: str) -> str:
    """Map Home Assistant language tags to xAI's documented TTS codes."""
    normalized = language.replace("_", "-").strip()
    if normalized.lower() == "auto":
        return "auto"

    documented_regional_codes = {
        "ar-AE",
        "ar-EG",
        "ar-SA",
        "es-ES",
        "es-MX",
        "pt-BR",
        "pt-PT",
    }
    for supported in documented_regional_codes:
        if normalized.lower() == supported.lower():
            return supported

    primary = normalized.split("-")[0].lower()
    return primary


def normalize_stt_language(language: str | None) -> str | None:
    """Map Home Assistant language tags to xAI's STT codes."""
    if not language or language.replace("_", "-").strip().lower() == "auto":
        return None

    normalized = language.replace("_", "-").strip()
    regional_codes = {
        "ar-AE",
        "ar-EG",
        "ar-SA",
        "es-ES",
        "es-MX",
        "pt-BR",
        "pt-PT",
    }
    for supported in regional_codes:
        if normalized.lower() == supported.lower():
            return supported

    primary = normalized.split("-")[0].lower()
    return primary
