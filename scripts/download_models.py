"""Download OmniVoice (VoiceStudio) or Qwen3-TTS weights into ./models."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

REPOS = {
    "omnivoice": "k2-fsa/OmniVoice",
    "qwen3": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
}


def download_model(model_key: str) -> None:
    repo_id = REPOS.get(model_key, model_key)
    target_dir = MODELS / repo_id.split("/")[-1]
    print(f"Downloading {repo_id} -> {target_dir} ...")
    MODELS.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
    )
    print(f"Downloaded successfully: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download TTS models")
    parser.add_argument(
        "--model",
        default="omnivoice",
        choices=["omnivoice", "qwen3", "all"],
        help="Which model to download (default: omnivoice)",
    )
    args = parser.parse_args()

    if args.model == "all":
        download_model("omnivoice")
        download_model("qwen3")
    else:
        download_model(args.model)


if __name__ == "__main__":
    main()
