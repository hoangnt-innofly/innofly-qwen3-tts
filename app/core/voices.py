from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Speaker:
    id: str
    description: str
    native_language: str
    engine_speaker: str | None = None
    force_language: str | None = None
    stable: bool = False
    max_temperature: float | None = None
    subtalker_temperature: float | None = None
    default_instruct: str | None = None


@dataclass(frozen=True)
class GenerationPlan:
    speaker: str
    language: str
    instruct: str
    tight: bool
    max_temperature: float | None
    subtalker_temperature: float | None


OMNIVOICE_SPEAKERS: tuple[Speaker, ...] = (
    # --- Multilingual & Vietnamese Presets ---
    Speaker("Default", "Giọng chuẩn tự nhiên đa ngôn ngữ (OmniVoice base)", "Multilingual"),
    Speaker("Female_Warm", "Giọng nữ ấm áp, nhẹ nhàng, truyền cảm", "Multilingual", default_instruct="A warm, gentle, and expressive female voice."),
    Speaker("Female_Clear", "Giọng nữ trong trẻo, tự nhiên, phong cách MC / đọc sách", "Multilingual", default_instruct="A clear, crisp, and articulate female narrator voice."),
    Speaker("Male_Deep", "Giọng nam trầm ấm, rõ ràng, phong cách dẫn chuyện", "Multilingual", default_instruct="A deep, resonant, and calm male voice with clear enunciation."),
    Speaker("Male_Energetic", "Giọng nam trẻ trung, năng động, đĩnh đạc", "Multilingual", default_instruct="An energetic, upbeat, and modern young male voice."),
    Speaker("Vietnamese_Female", "Giọng nữ tiếng Việt chuẩn, tự nhiên, truyền cảm", "Vietnamese", default_instruct="Giọng nữ Việt Nam chuẩn giọng Bắc/Nam thanh thoát, tự nhiên, biểu cảm ấm áp."),
    Speaker("Vietnamese_Male", "Giọng nam tiếng Việt trầm ấm, rõ ràng", "Vietnamese", default_instruct="Giọng nam Việt Nam trầm ấm, rõ chữ, ngữ điệu tự nhiên và tự tin."),

    # --- Japanese Diverse Presets (日本語 音声バリエーション) ---
    Speaker(
        "Japanese_Anime_Female",
        "Nữ Nhật ngọt ngào, tươi vui, phong cách Anime/Kawaii (アニメ声・可愛い)",
        "Japanese",
        force_language="Japanese",
        default_instruct="A cute, sweet, and lively young Japanese female voice with an anime heroine tone, bright pitch and cheerful intonation.",
    ),
    Speaker(
        "Japanese_Warm_Female",
        "Nữ Nhật dịu dàng, ấm áp, phong cách Onee-san / ASMR (お姉さん・優しい)",
        "Japanese",
        force_language="Japanese",
        default_instruct="A gentle, warm, and soothing mature Japanese female voice (Onee-san style), soft-spoken, calm and relaxing pace.",
    ),
    Speaker(
        "Japanese_News_Female",
        "Nữ Nhật phát thanh viên NHK, chuẩn xác, trang trọng (アナウンサー・女性)",
        "Japanese",
        force_language="Japanese",
        default_instruct="A formal and professional Japanese female news anchor with clear, articulate diction, steady pace and studio recording quality.",
    ),
    Speaker(
        "Japanese_Story_Female",
        "Nữ Nhật kể chuyện / Radio podcast, diễn cảm, sâu lắng (朗読・ラジオ)",
        "Japanese",
        force_language="Japanese",
        default_instruct="An emotional, clear, and articulate Japanese female storyteller with natural studio pacing and expressive storytelling nuance, no extra vocal noises.",
    ),
    Speaker(
        "Japanese_Young_Male",
        "Nam Nhật trẻ tuổi, năng động, phong cách Shonen Anime (少年・青年・元気)",
        "Japanese",
        force_language="Japanese",
        default_instruct="An energetic, enthusiastic young Japanese male voice with a shonen anime protagonist tone, clear and bright.",
    ),
    Speaker(
        "Japanese_Ikemen_Male",
        "Nam Nhật trầm ấm, điềm tĩnh, phong cách lãng tử (イケメン・低音)",
        "Japanese",
        force_language="Japanese",
        default_instruct="A smooth, deep, attractive and calm Japanese male voice (Ikemen style), steady rhythm and charismatic tone.",
    ),
    Speaker(
        "Japanese_News_Male",
        "Nam Nhật phát thanh viên, đĩnh đạc, chuẩn xác (アナウンサー・男性)",
        "Japanese",
        force_language="Japanese",
        default_instruct="A formal, authoritative and articulate Japanese male broadcaster with crisp pronunciation and neutral cadence.",
    ),
    Speaker(
        "Japanese_Narrator_Male",
        "Nam Nhật trung niên, trầm sâu, dẫn chuyện tài liệu / lịch sử (ナレーション・重厚)",
        "Japanese",
        force_language="Japanese",
        default_instruct="A mature, deep, seasoned Japanese male documentary narrator voice with rich low-frequencies and solemn gravitas.",
    ),

    # --- Other Major Languages Presets ---
    Speaker("English_Female", "Natural English Female voice", "English", default_instruct="A natural, fluent American English female voice."),
    Speaker("English_Male", "Natural English Male voice", "English", default_instruct="A natural, confident American English male voice."),
    Speaker("Chinese_Female", "中文 女性 (Ấm áp, chuẩn phổ thông)", "Chinese", default_instruct="A natural, clear Standard Mandarin female voice."),
    Speaker("Chinese_Male", "中文 男性 (Trầm ấm, phát âm chuẩn)", "Chinese", default_instruct="A calm, resonant Standard Mandarin male voice."),
    Speaker("Korean_Female", "한국어 여성 (Nhẹ nhàng, tình cảm)", "Korean", default_instruct="A soft, expressive Korean female voice with natural intonation."),
)

QWEN3_SPEAKERS: tuple[Speaker, ...] = (
    Speaker("Vivian", "Giọng nữ trẻ, sáng, hơi sắc.", "Chinese"),
    Speaker("Serena", "Giọng nữ trẻ, ấm, dịu.", "Chinese"),
    Speaker("Uncle_Fu", "Giọng nam trưởng thành, trầm, ấm.", "Chinese"),
    Speaker("Dylan", "Giọng nam Bắc Kinh, trẻ, tự nhiên.", "Chinese (Beijing)"),
    Speaker("Eric", "Giọng nam Thành Đô, sống động, hơi khàn.", "Chinese (Sichuan)"),
    Speaker("Ryan", "Giọng nam năng động, nhịp điệu rõ.", "English"),
    Speaker("Aiden", "Giọng nam Mỹ, trong, trung âm.", "English"),
    Speaker("Sohee", "Giọng nữ Hàn, ấm, giàu cảm xúc.", "Korean"),
    Speaker("Hana", "Nữ Nhật, ấm, dịu (Serena).", "Japanese", engine_speaker="Serena", force_language="Japanese"),
    Speaker("Yuki", "Nữ Nhật, sáng, hơi sắc (Vivian).", "Japanese", engine_speaker="Vivian", force_language="Japanese"),
    Speaker("Aoi", "Nữ Nhật, ấm, mềm (Sohee).", "Japanese", engine_speaker="Sohee", force_language="Japanese"),
    Speaker("Ken", "Nam Nhật trẻ, rõ ràng (Ryan).", "Japanese", engine_speaker="Ryan", force_language="Japanese"),
    Speaker("Ryo", "Nam Nhật, trong, trung âm (Aiden).", "Japanese", engine_speaker="Aiden", force_language="Japanese"),
    Speaker("Hiro", "Nam Nhật trưởng thành, trầm, ấm (Uncle_Fu).", "Japanese", engine_speaker="Uncle_Fu", force_language="Japanese"),
    Speaker("Sora", "Nam Nhật trẻ, tự nhiên (Dylan).", "Japanese", engine_speaker="Dylan", force_language="Japanese"),
    Speaker(
        "Ono_Anna",
        "Nữ Nhật bản xứ, nhẹ (preset gốc, phong cách phát thanh viên rõ ràng).",
        "Japanese",
        force_language="Japanese",
        stable=True,
        max_temperature=0.60,
        subtalker_temperature=0.40,
        default_instruct="A professional Japanese female news anchor with a clear, articulate, and serious tone. Moderate speaking pace, steady, formal, and studio quality.",
    ),
)

SPEAKERS = OMNIVOICE_SPEAKERS

LANGUAGES: tuple[str, ...] = (
    "Auto",
    "Vietnamese",
    "Japanese",
    "English",
    "Chinese",
    "Korean",
    "French",
    "German",
    "Spanish",
    "Portuguese",
    "Russian",
    "Italian",
    "Thai",
    "Indonesian",
    "Hindi",
    "Arabic",
)

_SPEAKER_BY_KEY = {s.id.lower(): s.id for s in SPEAKERS}
_LANGUAGE_BY_KEY = {lang.lower(): lang for lang in LANGUAGES}

DEFAULT_SPEAKER = "Default"
DEFAULT_LANGUAGE = "Auto"

_KANA_RE = re.compile(r"[\u3041-\u3096\u30a1-\u30fa\u30fc]")
_HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_HANGUL_RE = re.compile(r"[\uac00-\ud7af]")
_ASCII_LETTER_RE = re.compile(r"[A-Za-z]")


def canonicalize_speaker(value: str | None, engine: str = "omnivoice") -> str:
    speakers_list = OMNIVOICE_SPEAKERS if engine == "omnivoice" else QWEN3_SPEAKERS
    default = "Default" if engine == "omnivoice" else "Vivian"
    if value is None or not str(value).strip():
        return default
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    speaker_map = {s.id.lower(): s.id for s in speakers_list}
    if key in speaker_map:
        return speaker_map[key]
    return default


def canonicalize_language(value: str | None) -> str:
    if value is None or not str(value).strip():
        return DEFAULT_LANGUAGE
    key = str(value).strip().lower()
    aliases = {
        "vi": "vietnamese",
        "vn": "vietnamese",
        "vietnamese": "vietnamese",
        "tiếng việt": "vietnamese",
        "ja": "japanese",
        "jp": "japanese",
        "japanese": "japanese",
        "tiếng nhật": "japanese",
        "zh": "chinese",
        "zh-cn": "chinese",
        "en": "english",
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
        return "Auto"
    return _LANGUAGE_BY_KEY[key]


def get_speaker(speaker_id: str, engine: str = "omnivoice") -> Speaker:
    speakers_list = OMNIVOICE_SPEAKERS if engine == "omnivoice" else QWEN3_SPEAKERS
    for speaker in speakers_list:
        if speaker.id.lower() == speaker_id.lower():
            return speaker
    return speakers_list[0]


def _auto_language(text: str, force_language: str) -> str:
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
    engine: str = "omnivoice",
) -> GenerationPlan:
    meta = get_speaker(speaker, engine=engine)
    engine_speaker = meta.engine_speaker or meta.id
    if language == "Auto" and meta.force_language:
        language = _auto_language(text, meta.force_language)
    engine_native = meta.native_language
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


def voices_payload(engine: str = "omnivoice") -> dict:
    if engine == "omnivoice":
        return {
            "engine": "omnivoice",
            "model": "k2-fsa/OmniVoice (VoiceStudio 646 Languages)",
            "mode": "t2s",
            "languages": list(LANGUAGES),
            "speakers": [
                {
                    "id": s.id,
                    "description": s.description,
                    "native_language": s.native_language,
                }
                for s in OMNIVOICE_SPEAKERS
            ],
        }
    return {
        "engine": "qwen3",
        "model": "Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "mode": "t2s",
        "languages": list(LANGUAGES),
        "speakers": [
            {
                "id": s.id,
                "description": s.description,
                "native_language": s.native_language,
            }
            for s in QWEN3_SPEAKERS
        ],
    }
