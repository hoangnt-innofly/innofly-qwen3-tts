import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import jobs, router
from app.core.config import ROOT_DIR, get_settings

settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    jobs.start()
    yield
    await jobs.stop()


app = FastAPI(
    title="Qwen3-TTS 1.7B API",
    description="Demo backend: multilingual text-to-speech with Qwen3-TTS-12Hz-1.7B-CustomVoice.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

static_dir = ROOT_DIR / "static"
if static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", include_in_schema=False)
def demo_page():
    from fastapi.responses import FileResponse

    index = static_dir / "index.html"
    if not index.is_file():
        from fastapi.responses import RedirectResponse

        return RedirectResponse(url="/docs")
    return FileResponse(index)
