"""Download Qwen3-TTS 1.7B CustomVoice weights into ./models."""

from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
REPO_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"


def main() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(MODELS / "Qwen3-TTS-12Hz-1.7B-CustomVoice"),
    )
    print(path)


if __name__ == "__main__":
    main()
