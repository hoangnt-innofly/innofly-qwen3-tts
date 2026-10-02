from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.voices import DEFAULT_LANGUAGE, DEFAULT_SPEAKER

ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_host: str = "0.0.0.0"
    app_port: int = 8000
    public_base_url: str = "http://127.0.0.1:8000"
    secret_api_key: str = ""

    # Engine selection: "omnivoice" (VoiceStudio default, 646 languages) or "qwen3"
    tts_engine: str = "omnivoice"

    tts_mock: bool = False
    # Model ID: "k2-fsa/OmniVoice" or "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice" or local path
    tts_model_id: str = "k2-fsa/OmniVoice"
    tts_device_map: str = "auto"
    tts_dtype: str = "float16"
    tts_attn_implementation: str = "sdpa"

    # After each job, park weights off GPU so VRAM is free for other models.
    tts_free_vram: bool = True

    # OmniVoice specific defaults
    tts_num_step: int = 16
    tts_speed: float = 1.0
    tts_guidance_scale: float = 2.0

    # Common & Qwen3 defaults
    tts_default_language: str = DEFAULT_LANGUAGE
    tts_default_speaker: str = DEFAULT_SPEAKER
    tts_default_instruct: str = ""
    tts_default_temperature: float = 0.9
    tts_default_top_k: int = 50
    tts_default_top_p: float = 1.0
    tts_default_repetition_penalty: float = 1.05
    tts_default_max_new_tokens: int = 2048
    tts_default_do_sample: bool = True
    tts_default_seed: int = 42

    tts_subtalker_temperature: float = 0.6
    tts_subtalker_top_k: int = 50
    tts_subtalker_top_p: float = 0.9
    tts_stable_temperature: float = 0.6
    tts_stable_top_p: float = 0.85
    tts_stable_repetition_penalty: float = 1.1
    tts_max_attempts: int = 3

    max_text_chars: int = 8000
    job_ttl_seconds: int = 86400

    @property
    def audio_dir(self) -> Path:
        return ROOT_DIR / "storage" / "audio"

    @property
    def uploads_dir(self) -> Path:
        return ROOT_DIR / "storage" / "uploads"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.audio_dir.mkdir(parents=True, exist_ok=True)
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    return settings
