from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, Optional
from urllib.parse import unquote, urlparse

from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request
from fastapi.responses import FileResponse

from app.core.auth import require_api_key
from app.core.config import get_settings
from app.core.voices import canonicalize_language, canonicalize_speaker, voices_payload
from app.models.schemas import DeleteAudioResponse, HealthResponse, JobResponse, VoicesResponse
from app.services.jobs import JobService

router = APIRouter()
api = APIRouter(prefix="/api/v1", dependencies=[Depends(require_api_key)])
settings = get_settings()
jobs = JobService(settings)


def public_base(request: Request) -> str:
    configured = settings.public_base_url.rstrip("/")
    if configured:
        return configured
    return str(request.base_url).rstrip("/")


def to_response(request: Request, job) -> JobResponse:
    return JobResponse.model_validate(job.to_public(public_base(request)))


async def _await_job(job, timeout: float = 600):
    try:
        job = await jobs.wait(job.id, timeout=timeout)
    except (TimeoutError, asyncio.TimeoutError) as exc:
        raise HTTPException(status_code=504, detail="Generation timed out") from exc
    if job.status == "failed":
        raise HTTPException(status_code=500, detail=job.error or "Generation failed")
    return job


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    cuda_available, device_name = jobs.cuda_info()
    return HealthResponse(
        mock=settings.tts_mock,
        cuda_available=cuda_available,
        device_name=device_name,
        pipeline_ready=jobs.pipeline_ready(),
        defaults={
            "model": settings.tts_model_id,
            "mode": "t2s",
            "language": settings.tts_default_language,
            "speaker": settings.tts_default_speaker,
            "instruct": settings.tts_default_instruct,
            "temperature": settings.tts_default_temperature,
            "top_k": settings.tts_default_top_k,
            "top_p": settings.tts_default_top_p,
            "repetition_penalty": settings.tts_default_repetition_penalty,
            "max_new_tokens": settings.tts_default_max_new_tokens,
            "do_sample": settings.tts_default_do_sample,
            "seed": settings.tts_default_seed,
            "device_map": settings.tts_device_map,
            "dtype": settings.tts_dtype,
            "attn_implementation": settings.tts_attn_implementation,
            "free_vram": settings.tts_free_vram,
        },
    )


@api.get("/voices", response_model=VoicesResponse)
def list_voices() -> VoicesResponse:
    return VoicesResponse.model_validate(voices_payload())


@api.post("/free-memory")
def free_memory() -> dict:
    """Release TTS VRAM after use so other 12GB-card apps (Comfy, LTX) can run."""
    return jobs.free_vram()


@api.post("/generate", response_model=JobResponse)
async def generate(
    request: Request,
    text: Annotated[str, Form(min_length=1)],
    language: Annotated[Optional[str], Form()] = None,
    speaker: Annotated[Optional[str], Form()] = None,
    instruct: Annotated[Optional[str], Form()] = None,
    temperature: Annotated[Optional[float], Form()] = None,
    top_k: Annotated[Optional[int], Form()] = None,
    top_p: Annotated[Optional[float], Form()] = None,
    repetition_penalty: Annotated[Optional[float], Form()] = None,
    max_new_tokens: Annotated[Optional[int], Form()] = None,
    do_sample: Annotated[Optional[bool], Form()] = None,
    seed: Annotated[Optional[int], Form()] = None,
    wait: Annotated[bool, Form()] = True,
) -> JobResponse:
    """Text-to-speech with Qwen3-TTS 1.7B CustomVoice. Waits and returns audio_url."""
    job = await _enqueue(
        text,
        language,
        speaker,
        instruct,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        max_new_tokens,
        do_sample,
        seed,
    )
    if wait:
        job = await _await_job(job)
    return to_response(request, job)


@api.post("/jobs", response_model=JobResponse)
async def create_job(
    request: Request,
    text: Annotated[str, Form(min_length=1)],
    language: Annotated[Optional[str], Form()] = None,
    speaker: Annotated[Optional[str], Form()] = None,
    instruct: Annotated[Optional[str], Form()] = None,
    temperature: Annotated[Optional[float], Form()] = None,
    top_k: Annotated[Optional[int], Form()] = None,
    top_p: Annotated[Optional[float], Form()] = None,
    repetition_penalty: Annotated[Optional[float], Form()] = None,
    max_new_tokens: Annotated[Optional[int], Form()] = None,
    do_sample: Annotated[Optional[bool], Form()] = None,
    seed: Annotated[Optional[int], Form()] = None,
) -> JobResponse:
    """Same as /generate: wait until WAV is ready, then return audio_url."""
    job = await _enqueue(
        text,
        language,
        speaker,
        instruct,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        max_new_tokens,
        do_sample,
        seed,
    )
    job = await _await_job(job)
    return to_response(request, job)


@api.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str, request: Request) -> JobResponse:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return to_response(request, job)


@router.get("/media/audio/{filename}")
def get_audio(filename: str):
    path = settings.audio_dir / Path(filename).name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(path, media_type="audio/wav", filename=path.name)


def _audio_file_from_path(raw: str) -> Path:
    value = unquote((raw or "").strip())
    if not value:
        raise HTTPException(status_code=400, detail="path is required")

    parsed = urlparse(value if "://" in value else f"file:///{value.lstrip('/')}")
    parts = [p for p in parsed.path.replace("\\", "/").split("/") if p]
    if any(part in {".", ".."} for part in parts):
        raise HTTPException(status_code=400, detail="invalid path")

    filename = Path(parts[-1]).name
    if filename.lower() in {"", ".gitkeep"} or not filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="only generated .wav files can be deleted")

    audio_root = settings.audio_dir.resolve()
    path = (audio_root / filename).resolve()
    if path.parent != audio_root:
        raise HTTPException(status_code=400, detail="invalid path")
    return path


def _delete_audio(raw_path: str) -> DeleteAudioResponse:
    path = _audio_file_from_path(raw_path)
    public_path = f"media/audio/{path.name}"
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Audio not found: {public_path}")
    path.unlink()
    return DeleteAudioResponse(deleted=True, path=public_path, filename=path.name)


@api.delete("/media", response_model=DeleteAudioResponse)
def delete_audio(path: Annotated[str, Query(description="media/audio/<id>.wav or audio_url")]):
    """Delete a generated WAV so storage does not grow."""
    return _delete_audio(path)


@api.post("/media/delete", response_model=DeleteAudioResponse)
def delete_audio_form(path: Annotated[str, Form()]):
    """Same as DELETE /media, for form clients."""
    return _delete_audio(path)


async def _enqueue(
    text: str,
    language: str | None,
    speaker: str | None,
    instruct: str | None,
    temperature: float | None,
    top_k: int | None,
    top_p: float | None,
    repetition_penalty: float | None,
    max_new_tokens: int | None,
    do_sample: bool | None,
    seed: int | None,
):
    cleaned = (text or "").strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="text is required")
    if len(cleaned) > settings.max_text_chars:
        raise HTTPException(
            status_code=400,
            detail=f"text exceeds {settings.max_text_chars} characters",
        )

    try:
        language = canonicalize_language(language)
        speaker = canonicalize_speaker(speaker)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if temperature is not None and not (0.0 <= temperature <= 2.0):
        raise HTTPException(status_code=400, detail="temperature must be between 0 and 2")
    if top_p is not None and not (0.0 < top_p <= 1.0):
        raise HTTPException(status_code=400, detail="top_p must be in (0, 1]")
    if top_k is not None and top_k < 0:
        raise HTTPException(status_code=400, detail="top_k must be >= 0")
    if repetition_penalty is not None and not (0.5 <= repetition_penalty <= 2.0):
        raise HTTPException(status_code=400, detail="repetition_penalty must be between 0.5 and 2")
    if max_new_tokens is not None and not (64 <= max_new_tokens <= 8192):
        raise HTTPException(status_code=400, detail="max_new_tokens must be between 64 and 8192")

    return jobs.create_job(
        text=cleaned,
        language=language,
        speaker=speaker,
        instruct=instruct,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        seed=seed,
    )
