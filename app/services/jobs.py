from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from app.core.config import Settings
from app.core.voices import canonicalize_language, canonicalize_speaker, resolve_generation
from app.services.audio import audible_items, drill_text, split_list_items
from app.services.mock_engine import generate_placeholder_wav

logger = logging.getLogger("tts-job-service")

JobStatus = Literal["queued", "running", "succeeded", "failed"]


@dataclass
class Job:
    id: str
    text: str
    engine: str
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
    num_step: int = 16
    speed: float = 1.0
    guidance_scale: float = 2.0
    ref_audio: str | None = None
    ref_text: str | None = None
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
            "engine": self.engine,
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
        self._engines: dict[str, Any] = {}
        self._engine_lock = threading.Lock()
        self._worker_task: asyncio.Task | None = None

    def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker_loop(), name="tts-worker")

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
        engine: str | None = None,
        language: str | None = None,
        speaker: str | None = None,
        instruct: str | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        max_new_tokens: int | None = None,
        do_sample: bool | None = None,
        seed: int | None = None,
        num_step: int | None = None,
        speed: float | None = None,
        guidance_scale: float | None = None,
        ref_audio: str | None = None,
        ref_text: str | None = None,
    ) -> Job:
        job_id = uuid.uuid4().hex
        chosen_engine = (engine or self.settings.tts_engine or "omnivoice").strip().lower()
        if chosen_engine not in {"omnivoice", "qwen3"}:
            chosen_engine = "omnivoice"

        job = Job(
            id=job_id,
            text=text.strip(),
            engine=chosen_engine,
            language=canonicalize_language(language or self.settings.tts_default_language),
            speaker=canonicalize_speaker(speaker or self.settings.tts_default_speaker, engine=chosen_engine),
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
            max_new_tokens=(
                self.settings.tts_default_max_new_tokens if max_new_tokens is None else max_new_tokens
            ),
            do_sample=self.settings.tts_default_do_sample if do_sample is None else do_sample,
            seed=self.settings.tts_default_seed if seed is None else seed,
            num_step=self.settings.tts_num_step if num_step is None else num_step,
            speed=self.settings.tts_speed if speed is None else speed,
            guidance_scale=self.settings.tts_guidance_scale if guidance_scale is None else guidance_scale,
            ref_audio=ref_audio,
            ref_text=ref_text,
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
        engine_name = self.settings.tts_engine.lower()
        engine = self._engines.get(engine_name)
        return bool(engine and getattr(engine, "ready", False))

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
        with self._engine_lock:
            for name, eng in list(self._engines.items()):
                if eng is not None:
                    try:
                        eng.release_vram()
                    except Exception:
                        pass
            self._engines.clear()
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

        engine_name = job.engine.lower()
        if engine_name == "omnivoice":
            self._run_omnivoice_job(job)
        else:
            self._run_qwen3_job(job)

    def _run_omnivoice_job(self, job: Job) -> None:
        engine = self._get_omnivoice_engine()
        plan = resolve_generation(job.speaker, job.language, job.instruct, text=job.text, engine="omnivoice")
        logger.info(
            "OmniVoice Job %s: Speaker=%s (Native=%s), Instruct=%r",
            job.id,
            job.speaker,
            plan.language,
            plan.instruct,
        )

        _, sample_rate, duration = engine.generate(
            text=job.text,
            output_path=job.audio_path,
            ref_audio=job.ref_audio,
            ref_text=job.ref_text,
            num_step=job.num_step,
            speed=job.speed,
            guidance_scale=job.guidance_scale,
            seed=job.seed,
        )
        job.sample_rate = sample_rate
        job.duration_seconds = duration

    def _run_qwen3_job(self, job: Job) -> None:
        engine = self._get_qwen3_engine()
        s = self.settings
        plan = resolve_generation(job.speaker, job.language, job.instruct, text=job.text, engine="qwen3")
        items = split_list_items(job.text)
        temperature, top_p, repetition_penalty = job.temperature, job.top_p, job.repetition_penalty
        subtalker_temperature = s.tts_subtalker_temperature
        if plan.tight or items:
            temperature = min(temperature, s.tts_stable_temperature)
            top_p = min(top_p, s.tts_stable_top_p)
            repetition_penalty = max(repetition_penalty, s.tts_stable_repetition_penalty)
        if plan.max_temperature is not None:
            temperature = min(temperature, plan.max_temperature)
        if plan.subtalker_temperature is not None:
            subtalker_temperature = min(subtalker_temperature, plan.subtalker_temperature)

        text = job.text
        drill_units = None
        language = plan.language
        if items:
            text = drill_text(items)
            drill_units = audible_items(items)
            if language == "Auto":
                language = "Japanese"

        _, sample_rate, duration = engine.generate(
            text=text,
            drill_units=drill_units,
            language=language,
            speaker=plan.speaker,
            output_path=job.audio_path,
            instruct=plan.instruct,
            temperature=temperature,
            top_k=job.top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_new_tokens=job.max_new_tokens,
            do_sample=job.do_sample,
            seed=job.seed,
            subtalker_temperature=subtalker_temperature,
            subtalker_top_k=s.tts_subtalker_top_k,
            subtalker_top_p=s.tts_subtalker_top_p,
            max_attempts=s.tts_max_attempts,
        )
        job.sample_rate = sample_rate
        job.duration_seconds = duration

    def _get_omnivoice_engine(self):
        with self._engine_lock:
            if "omnivoice" not in self._engines or self._engines["omnivoice"] is None:
                from app.services.omnivoice_engine import OmniVoiceEngine

                model_id = self.settings.tts_model_id
                if "qwen" in model_id.lower():
                    model_id = "k2-fsa/OmniVoice"

                engine = OmniVoiceEngine(
                    model_id=model_id,
                    device_map=self.settings.tts_device_map,
                    dtype=self.settings.tts_dtype,
                    free_vram=self.settings.tts_free_vram,
                )
                engine.load()
                self._engines["omnivoice"] = engine
            return self._engines["omnivoice"]

    def _get_qwen3_engine(self):
        with self._engine_lock:
            if "qwen3" not in self._engines or self._engines["qwen3"] is None:
                from app.services.tts_engine import TTSEngine

                model_id = self.settings.tts_model_id
                if "omnivoice" in model_id.lower():
                    model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

                engine = TTSEngine(
                    model_id,
                    device_map=self.settings.tts_device_map,
                    dtype=self.settings.tts_dtype,
                    attn_implementation=self.settings.tts_attn_implementation,
                    free_vram=self.settings.tts_free_vram,
                )
                engine.load()
                self._engines["qwen3"] = engine
            return self._engines["qwen3"]
