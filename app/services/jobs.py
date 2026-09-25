from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from app.core.config import Settings
from app.core.voices import canonicalize_language, canonicalize_speaker, resolve_generation
from app.services.audio import budget_max_new_tokens
from app.services.mock_engine import generate_placeholder_wav

logger = logging.getLogger("qwen3-tts-api")

JobStatus = Literal["queued", "running", "succeeded", "failed"]


@dataclass
class Job:
    id: str
    text: str
    language: str
    speaker: str
    instruct: str
    audio_path: Path
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float
    max_new_tokens: int
    do_sample: bool
    seed: int
    status: JobStatus = "queued"
    sample_rate: int | None = None
    duration_seconds: float | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    _done: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    def to_public(self, base_url: str) -> dict:
        audio_url = None
        if self.status == "succeeded":
            audio_url = f"{base_url.rstrip('/')}/media/audio/{self.audio_path.name}"
        return {
            "job_id": self.id,
            "status": self.status,
            "text": self.text,
            "language": self.language,
            "speaker": self.speaker,
            "instruct": self.instruct,
            "mode": "t2s",
            "sample_rate": self.sample_rate,
            "duration_seconds": self.duration_seconds,
            "audio_url": audio_url,
            "error": self.error,
        }


class JobService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.jobs: dict[str, Job] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._engine = None
        self._engine_lock = threading.Lock()
        self._worker_task: asyncio.Task | None = None

    def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker_loop(), name="qwen3-tts-worker")

    async def stop(self) -> None:
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    def create_job(
        self,
        *,
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
    ) -> Job:
        job_id = uuid.uuid4().hex
        job = Job(
            id=job_id,
            text=text.strip(),
            language=canonicalize_language(language or self.settings.tts_default_language),
            speaker=canonicalize_speaker(speaker or self.settings.tts_default_speaker),
            instruct=(instruct or self.settings.tts_default_instruct).strip(),
            audio_path=self.settings.audio_dir / f"{job_id}.wav",
            temperature=self.settings.tts_default_temperature if temperature is None else temperature,
            top_k=self.settings.tts_default_top_k if top_k is None else top_k,
            top_p=self.settings.tts_default_top_p if top_p is None else top_p,
            repetition_penalty=(
                self.settings.tts_default_repetition_penalty
                if repetition_penalty is None
                else repetition_penalty
            ),
            max_new_tokens=budget_max_new_tokens(
                text.strip(),
                ceiling=(
                    self.settings.tts_default_max_new_tokens
                    if max_new_tokens is None
                    else max_new_tokens
                ),
            ),
            do_sample=self.settings.tts_default_do_sample if do_sample is None else do_sample,
            seed=self.settings.tts_default_seed if seed is None else seed,
        )
        self.jobs[job_id] = job
        self._queue.put_nowait(job_id)
        return job

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    async def wait(self, job_id: str, timeout: float = 600) -> Job:
        job = self.jobs[job_id]
        await asyncio.wait_for(job._done.wait(), timeout=timeout)
        return job

    def pipeline_ready(self) -> bool:
        if self.settings.tts_mock:
            return True
        return bool(self._engine and getattr(self._engine, "ready", False))

    @staticmethod
    def _free_cuda() -> None:
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def free_vram(self) -> dict:
        """Park TTS weights off GPU so Comfy/LTX can use the 12GB card."""
        with self._engine_lock:
            if self._engine is not None:
                self._engine.release_vram()
            else:
                self._free_cuda()
        return {"freed": True}

    def cuda_info(self) -> tuple[bool, str | None]:
        try:
            import torch

            if torch.cuda.is_available():
                return True, torch.cuda.get_device_name(0)
        except Exception:
            pass
        return False, None

    async def _worker_loop(self) -> None:
        while True:
            job_id = await self._queue.get()
            job = self.jobs[job_id]
            job.status = "running"
            try:
                await asyncio.to_thread(self._run_job, job)
                job.status = "succeeded"
            except Exception as exc:
                logger.exception("Job %s failed", job_id)
                job.status = "failed"
                job.error = str(exc)
                if exc.__cause__ and str(exc.__cause__) not in job.error:
                    job.error = f"{exc} | {exc.__cause__}"
                self._free_cuda()
            finally:
                if self.settings.tts_free_vram:
                    self.free_vram()
                job.finished_at = datetime.now(timezone.utc)
                job._done.set()
                self._queue.task_done()

    def _run_job(self, job: Job) -> None:
        if self.settings.tts_mock:
            _, sample_rate, duration = generate_placeholder_wav(
                job.audio_path,
                text=job.text,
                speaker=job.speaker,
            )
            job.sample_rate = sample_rate
            job.duration_seconds = duration
            return

        engine = self._get_engine()
        engine_speaker, language, instruct = resolve_generation(
            job.speaker, job.language, job.instruct
        )
        if engine_speaker != job.speaker:
            logger.info(
                "Speaker %s → %s (language=%s)",
                job.speaker,
                engine_speaker,
                language,
            )
        _, sample_rate, duration = engine.generate(
            text=job.text,
            language=language,
            speaker=engine_speaker,
            output_path=job.audio_path,
            instruct=instruct,
            temperature=job.temperature,
            top_k=job.top_k,
            top_p=job.top_p,
            repetition_penalty=job.repetition_penalty,
            max_new_tokens=job.max_new_tokens,
            do_sample=job.do_sample,
            seed=job.seed,
        )
        job.sample_rate = sample_rate
        job.duration_seconds = duration

    def _get_engine(self):
        with self._engine_lock:
            if self._engine is None:
                from app.services.tts_engine import TTSEngine

                self._engine = TTSEngine(
                    self.settings.tts_model_id,
                    device_map=self.settings.tts_device_map,
                    dtype=self.settings.tts_dtype,
                    attn_implementation=self.settings.tts_attn_implementation,
                    free_vram=self.settings.tts_free_vram,
                )
                try:
                    self._engine.load()
                except Exception:
                    self._engine = None
                    raise
            return self._engine
