from typing import Any

from pydantic import BaseModel, Field


class JobResponse(BaseModel):
    job_id: str
    status: str
    text: str
    language: str
    speaker: str
    instruct: str = ""
    mode: str = "t2s"
    sample_rate: int | None = None
    duration_seconds: float | None = None
    audio_url: str | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
    mock: bool
    cuda_available: bool
    device_name: str | None = None
    pipeline_ready: bool
    defaults: dict[str, Any] = Field(default_factory=dict)


class VoicesResponse(BaseModel):
    model: str
    mode: str = "t2s"
    languages: list[str]
    speakers: list[dict[str, str]]
