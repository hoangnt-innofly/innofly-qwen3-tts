from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Speaker:
    id: str
    description: str
    native_language: str


# Canonical names for Qwen3-TTS-12Hz-1.7B-CustomVoice.
# Validation in the official package is case-insensitive.
SPEAKERS: tuple[Speaker, ...] = (
    Speaker("Vivian", "Giọng nữ trẻ, sáng, hơi sắc.", "Chinese"),
    Speaker("Serena", "Giọng nữ trẻ, ấm, dịu.", "Chinese"),
    Speaker("Uncle_Fu", "Giọng nam trưởng thành, trầm, ấm.", "Chinese"),
    Speaker("Dylan", "Giọng nam Bắc Kinh, trẻ, tự nhiên.", "Chinese (Beijing)"),
    Speaker("Eric", "Giọng nam Thành Đô, sống động, hơi khàn.", "Chinese (Sichuan)"),
    Speaker("Ryan", "Giọng nam năng động, nhịp điệu rõ.", "English"),
    Speaker("Aiden", "Giọng nam Mỹ, trong, trung âm.", "English"),
    Speaker("Ono_Anna", "Giọng nữ Nhật, vui, nhẹ.", "Japanese"),
    Speaker("Sohee", "Giọng nữ Hàn, ấm, giàu cảm xúc.", "Korean"),
)

LANGUAGES: tuple[str, ...] = (
    "Auto",
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian",
)

_SPEAKER_BY_KEY = {s.id.lower(): s.id for s in SPEAKERS}
_LANGUAGE_BY_KEY = {lang.lower(): lang for lang in LANGUAGES}

DEFAULT_SPEAKER = "Vivian"
DEFAULT_LANGUAGE = "Auto"


def canonicalize_speaker(value: str | None) -> str:
    if value is None or not str(value).strip():
        return DEFAULT_SPEAKER
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in _SPEAKER_BY_KEY:
        allowed = ", ".join(s.id for s in SPEAKERS)
        raise ValueError(f"Speaker không hợp lệ: {value}. Chọn một trong: {allowed}")
    return _SPEAKER_BY_KEY[key]


def canonicalize_language(value: str | None) -> str:
    if value is None or not str(value).strip():
        return DEFAULT_LANGUAGE
    key = str(value).strip().lower()
    aliases = {
        "zh": "chinese",
        "zh-cn": "chinese",
        "vi": "auto",
        "en": "english",
        "ja": "japanese",
        "jp": "japanese",
        "ko": "korean",
        "kr": "korean",
        "de": "german",
        "fr": "french",
        "ru": "russian",
        "pt": "portuguese",
        "es": "spanish",
        "it": "italian",
        "auto-detect": "auto",
    }
    key = aliases.get(key, key)
    if key not in _LANGUAGE_BY_KEY:
        allowed = ", ".join(LANGUAGES)
        raise ValueError(f"Language không hợp lệ: {value}. Chọn một trong: {allowed}")
    return _LANGUAGE_BY_KEY[key]


def voices_payload() -> dict:
    return {
        "model": "Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "mode": "t2s",
        "languages": list(LANGUAGES),
        "speakers": [
            {
                "id": s.id,
                "description": s.description,
                "native_language": s.native_language,
            }
            for s in SPEAKERS
        ],
    }
