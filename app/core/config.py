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

    tts_mock: bool = False
    tts_model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    tts_device_map: str = "auto"
    tts_dtype: str = "bfloat16"
    tts_attn_implementation: str = "sdpa"
    # After each job, park weights in RAM so the 12GB card is free for LTX/Comfy.
    # Generate still runs on GPU. Set 0 only if TTS owns the GPU.
    tts_free_vram: bool = True

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

    max_text_chars: int = 4000
    job_ttl_seconds: int = 86400

    @property
    def audio_dir(self) -> Path:
        return ROOT_DIR / "storage" / "audio"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.audio_dir.mkdir(parents=True, exist_ok=True)
    return settings
