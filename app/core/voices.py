from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Speaker:
    id: str
    description: str
    native_language: str
    # CustomVoice timbre actually used; None means `id` itself.
    engine_speaker: str | None = None
    force_language: str | None = None
    style_instruct: str | None = None
    # Always use the tight sampling caps, even in the native language.
    stable: bool = False
    # Per-speaker caps below the global tight ones.
    max_temperature: float | None = None
    subtalker_temperature: float | None = None


@dataclass(frozen=True)
class GenerationPlan:
    speaker: str
    language: str
    instruct: str
    tight: bool
    max_temperature: float | None
    subtalker_temperature: float | None


# CustomVoice 1.7B has only Ono_Anna as a native Japanese preset (and it
# tends to add breath on short text, hence `stable`). The other Japanese
# names are other premium timbres speaking Japanese, each with a distinct style.
JAPANESE_CLEAN_INSTRUCT = (
    "Read the text exactly once in standard Japanese at a steady, natural pace, "
    "like a calm narrator, and end right after the last word."
)


def _jp(style: str) -> str:
    return f"{style} {JAPANESE_CLEAN_INSTRUCT}"


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
        style_instruct=_jp("A warm, gentle young Japanese woman."),
    ),
    Speaker(
        "Yuki",
        "Nữ Nhật, sáng, hơi sắc (Vivian).",
        "Japanese",
        engine_speaker="Vivian",
        force_language="Japanese",
        style_instruct=_jp("A bright, clear young Japanese woman."),
    ),
    Speaker(
        "Aoi",
        "Nữ Nhật, ấm, mềm (Sohee).",
        "Japanese",
        engine_speaker="Sohee",
        force_language="Japanese",
        style_instruct=_jp("A warm, soft young Japanese woman."),
    ),
    Speaker(
        "Ken",
        "Nam Nhật trẻ, rõ ràng (Ryan).",
        "Japanese",
        engine_speaker="Ryan",
        force_language="Japanese",
        style_instruct=_jp("A calm young Japanese man with a clear voice."),
    ),
    Speaker(
        "Ryo",
        "Nam Nhật, trong, trung âm (Aiden).",
        "Japanese",
        engine_speaker="Aiden",
        force_language="Japanese",
        style_instruct=_jp("A clear, sunny young Japanese man."),
    ),
    Speaker(
        "Hiro",
        "Nam Nhật trưởng thành, trầm, ấm (Uncle_Fu).",
        "Japanese",
        engine_speaker="Uncle_Fu",
        force_language="Japanese",
        style_instruct=_jp("A mature Japanese man with a low, calm voice."),
    ),
    Speaker(
        "Sora",
        "Nam Nhật trẻ, tự nhiên (Dylan).",
        "Japanese",
        engine_speaker="Dylan",
        force_language="Japanese",
        style_instruct=_jp("A youthful Japanese man with a natural, clear voice."),
    ),
    Speaker(
        "Ono_Anna",
        "Nữ Nhật bản xứ, nhẹ (preset gốc, đã chặn tiếng thở).",
        "Japanese",
        force_language="Japanese",
        style_instruct=(
            "A light, calm young Japanese woman reading like a news narrator. "
            "Speak smoothly with no audible breathing, sighs, or laughter. "
            "Read the text exactly once and end right after the last word."
        ),
        stable=True,
        max_temperature=0.45,
        subtalker_temperature=0.45,
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
) -> GenerationPlan:
    """Map public speaker id → CustomVoice speaker / language / instruct / caps.

    `tight` is True when the run needs tighter sampling to avoid extra sounds:
    the timbre reads a non-native language, or the speaker is `stable`.
    """
    meta = get_speaker(speaker)
    engine_speaker = meta.engine_speaker or meta.id
    if language == "Auto" and meta.force_language:
        language = meta.force_language
    cleaned = (instruct or "").strip()
    if meta.style_instruct and not cleaned:
        cleaned = meta.style_instruct
    engine_native = get_speaker(engine_speaker).native_language
    cross_lingual = language != "Auto" and not engine_native.startswith(language)
    return GenerationPlan(
        speaker=engine_speaker,
        language=language,
        instruct=cleaned,
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
