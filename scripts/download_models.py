"""One-time download of the model weights local-coach needs at runtime.

app.py forces HF_HUB_OFFLINE=1 on every launch, so the Kokoro snapshot must
already be in the HuggingFace cache before the first run. This script fetches:

  - Whisper tiny.en            (~75 MB,  ~/.cache/whisper)
  - Kokoro-82M + all voices    (~380 MB, ~/.cache/huggingface)
  - the Ollama "coach" model   (~5 GB,   built from ./Modelfile; needs Ollama)

Run with: make setup   (or: uv run python scripts/download_models.py)
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

KOKORO_REPO = "prince-canuma/Kokoro-82M"
WHISPER_MODEL = "tiny.en"
REPO_ROOT = Path(__file__).resolve().parent.parent


def download_whisper() -> None:
    import whisper

    print(f"==> Whisper {WHISPER_MODEL}")
    whisper.load_model(WHISPER_MODEL, device="cpu")


def download_kokoro() -> None:
    from huggingface_hub import snapshot_download
    from mlx_audio.tts.utils import get_model_path

    print(f"==> Kokoro ({KOKORO_REPO})")
    # Model weights + config, using the same file filter mlx_audio applies.
    get_model_path(KOKORO_REPO)
    # mlx_audio's pipeline loads voices as voices/<name>.pt, which the snapshot
    # above skips. Fetch them all so any --kokoro-voice works offline.
    snapshot_download(KOKORO_REPO, allow_patterns=["voices/*.pt"])


def build_ollama_model() -> None:
    print("==> Ollama model 'coach'", flush=True)
    if shutil.which("ollama") is None:
        print("    ollama not found; install it (brew install ollama), start it, then re-run.")
        return
    # `ollama create` pulls the FROM base model if it isn't present yet.
    result = subprocess.run(
        ["ollama", "create", "coach", "-f", str(REPO_ROOT / "Modelfile")],
        check=False,
    )
    if result.returncode != 0:
        print("    ollama create failed; is the Ollama server running? (ollama serve)")


def main() -> None:
    download_whisper()
    download_kokoro()
    build_ollama_model()
    print("Done.")


if __name__ == "__main__":
    main()
