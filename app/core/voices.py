from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Speaker:
    id: str
    description: str
    native_language: str
    # CustomVoice timbre actually used; None means `id` itself.
    engine_speaker: str | None = None
    # Language used for "Auto" when the text is kanji/kana (no English/Korean).
    force_language: str | None = None
    # Always use the tight sampling caps, even in the native language.
    stable: bool = False
    # Per-speaker caps below the global tight ones.
    max_temperature: float | None = None
    subtalker_temperature: float | None = None
    # Default instruction to suppress unwanted vocalizations (e.g. Ono_Anna breaths/laughs)
    default_instruct: str | None = None


@dataclass(frozen=True)
class GenerationPlan:
    speaker: str
    language: str
    instruct: str
    tight: bool
    max_temperature: float | None
    subtalker_temperature: float | None


# CustomVoice 1.7B has only Ono_Anna as a native Japanese preset (and it
# tends to add breath/laughter on short text, hence `stable` + anti-breath instruct).
# The other Japanese names are other premium timbres speaking Japanese.
SPEAKERS: tuple[Speaker, ...] = (
    Speaker("Vivian", "Giọng nữ trẻ, sáng, hơi sắc.", "Chinese"),
    Speaker("Serena", "Giọng nữ trẻ, ấm, dịu.", "Chinese"),
    Speaker("Uncle_Fu", "Giọng nam trưởng thành, trầm, ấm.", "Chinese"),
    Speaker("Dylan", "Giọng nam Bắc Kinh, trẻ, tự nhiên.", "Chinese (Beijing)"),
    Speaker("Eric", "Giọng nam Thành Đô, sống động, hơi khàn.", "Chinese (Sichuan)"),
    Speaker("Ryan", "Giọng nam năng động, nhịp điệu rõ.", "English"),
    Speaker("Aiden", "Giọng nam Mỹ, trong, trung âm.", "English"),
    Speaker("Sohee", "Giọng nữ Hàn, ấm, giàu cảm xúc.", "Korean"),
    Speaker(
        "Hana",
        "Nữ Nhật, ấm, dịu (Serena).",
        "Japanese",
        engine_speaker="Serena",
        force_language="Japanese",
    ),
    Speaker(
        "Yuki",
        "Nữ Nhật, sáng, hơi sắc (Vivian).",
        "Japanese",
        engine_speaker="Vivian",
        force_language="Japanese",
    ),
    Speaker(
        "Aoi",
        "Nữ Nhật, ấm, mềm (Sohee).",
        "Japanese",
        engine_speaker="Sohee",
        force_language="Japanese",
    ),
    Speaker(
        "Ken",
        "Nam Nhật trẻ, rõ ràng (Ryan).",
        "Japanese",
        engine_speaker="Ryan",
        force_language="Japanese",
    ),
    Speaker(
        "Ryo",
        "Nam Nhật, trong, trung âm (Aiden).",
        "Japanese",
        engine_speaker="Aiden",
        force_language="Japanese",
    ),
    Speaker(
        "Hiro",
        "Nam Nhật trưởng thành, trầm, ấm (Uncle_Fu).",
        "Japanese",
        engine_speaker="Uncle_Fu",
        force_language="Japanese",
    ),
    Speaker(
        "Sora",
        "Nam Nhật trẻ, tự nhiên (Dylan).",
        "Japanese",
        engine_speaker="Dylan",
        force_language="Japanese",
    ),
    Speaker(
        "Ono_Anna",
        "Nữ Nhật bản xứ, nhẹ (preset gốc, đã chặn tiếng thở & cười).",
        "Japanese",
        force_language="Japanese",
        stable=True,
        max_temperature=0.35,
        subtalker_temperature=0.35,
        default_instruct="Speak in a clear, calm, and formal tone. Do not laugh, giggle, sigh, or make breathing sounds.",
    ),
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

_KANA_RE = re.compile(r"[\u3041-\u3096\u30a1-\u30fa\u30fc]")
_HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_HANGUL_RE = re.compile(r"[\uac00-\ud7af]")
_ASCII_LETTER_RE = re.compile(r"[A-Za-z]")


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


def _auto_language(text: str, force_language: str) -> str:
    """Language for "Auto" on a speaker that normally reads `force_language`.

    'book' must stay English: read as Japanese it comes out as a stretched,
    echoing 'bookkk'.
    """
    if _KANA_RE.search(text):
        return "Japanese"
    if _HANGUL_RE.search(text):
        return "Korean"
    if _HAN_RE.search(text):
        return force_language
    if _ASCII_LETTER_RE.search(text) and text.isascii():
        return "English"
    return "Auto"


def resolve_generation(
    speaker: str,
    language: str,
    instruct: str = "",
    text: str = "",
) -> GenerationPlan:
    """Map public speaker id → CustomVoice speaker / language / instruct / caps.

    Only the caller's own instruct is sent, falling back to speaker default_instruct
    if available. `tight` is True when the run needs tighter sampling to avoid
    extra sounds: the timbre reads a non-native language, or the speaker is `stable`.
    """
    meta = get_speaker(speaker)
    engine_speaker = meta.engine_speaker or meta.id
    if language == "Auto" and meta.force_language:
        language = _auto_language(text, meta.force_language)
    engine_native = get_speaker(engine_speaker).native_language
    cross_lingual = language != "Auto" and not engine_native.startswith(language)

    user_instruct = (instruct or "").strip()
    final_instruct = user_instruct if user_instruct else (meta.default_instruct or "")

    return GenerationPlan(
        speaker=engine_speaker,
        language=language,
        instruct=final_instruct,
        tight=cross_lingual or meta.stable,
        max_temperature=meta.max_temperature,
        subtalker_temperature=meta.subtalker_temperature,
    )


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
