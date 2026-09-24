from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Speaker:
    id: str
    description: str
    native_language: str
    # CustomVoice has only Ono_Anna as a native Japanese preset, and that
    # timbre often fails to EOS on short text (minutes of breath). Route
    # Japanese slots through another premium speaker that can speak Japanese.
    engine_speaker: str | None = None
    force_language: str | None = None


# CustomVoice 1.7B has 9 official timbres. Hana is our Japanese replacement
# (Serena speaking Japanese). Ono_Anna is kept as an alias for old clients.
JAPANESE_CLEAN_INSTRUCT = (
    "Speak Japanese clearly and stop immediately after the last word. "
    "Do not add extra breath, sighs, or trailing sounds."
)

SPEAKERS: tuple[Speaker, ...] = (
    Speaker("Vivian", "Giọng nữ trẻ, sáng, hơi sắc.", "Chinese"),
    Speaker("Serena", "Giọng nữ trẻ, ấm, dịu.", "Chinese"),
    Speaker("Uncle_Fu", "Giọng nam trưởng thành, trầm, ấm.", "Chinese"),
    Speaker("Dylan", "Giọng nam Bắc Kinh, trẻ, tự nhiên.", "Chinese (Beijing)"),
    Speaker("Eric", "Giọng nam Thành Đô, sống động, hơi khàn.", "Chinese (Sichuan)"),
    Speaker("Ryan", "Giọng nam năng động, nhịp điệu rõ.", "English"),
    Speaker("Aiden", "Giọng nam Mỹ, trong, trung âm.", "English"),
    Speaker(
        "Hana",
        "Giọng nữ Nhật thay Ono_Anna (Serena nói tiếng Nhật, ít thở).",
        "Japanese",
        engine_speaker="Serena",
        force_language="Japanese",
    ),
    Speaker(
        "Ono_Anna",
        "Alias → Hana. Preset Nhật gốc hay thở dài với text ngắn.",
        "Japanese",
        engine_speaker="Serena",
        force_language="Japanese",
    ),
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


def get_speaker(speaker_id: str) -> Speaker:
    for speaker in SPEAKERS:
        if speaker.id == speaker_id:
            return speaker
    raise ValueError(f"Speaker không hợp lệ: {speaker_id}")


def resolve_generation(
    speaker: str,
    language: str,
    instruct: str = "",
) -> tuple[str, str, str]:
    """Map public speaker id → CustomVoice speaker / language / instruct."""
    meta = get_speaker(speaker)
    engine_speaker = meta.engine_speaker or meta.id
    if language == "Auto" and meta.force_language:
        language = meta.force_language
    cleaned = (instruct or "").strip()
    if meta.engine_speaker and not cleaned:
        cleaned = JAPANESE_CLEAN_INSTRUCT
    return engine_speaker, language, cleaned


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
