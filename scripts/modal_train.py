"""Run training jobs on Modal (H100/A100 GPU cloud).

Usage:
    # Baseline SAC, seed 0
    uv run modal run --detach scripts/modal_train.py::train_sac -- \
        --config configs/pick_place_baseline.yaml --seed 0 \
        --run-name sac_pp_baseline_s0 --wandb

    # Decoupled correction, seed 0
    uv run modal run --detach scripts/modal_train.py::train_decoupled -- \
        --seed 0 --run-name sac_pp_decoupled_s0 --wandb

    # Download experiments from volume
    uv run modal volume get latent-recovery-rl-volume /experiments .
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal


APP_NAME = "latent-recovery-rl"
NETRC_PATH = Path("~/.netrc").expanduser()
PROJECT_DIR = "/root/project"
VOLUME_PATH = "/vol"
DEFAULT_GPU = "A100"
DEFAULT_CPU = 8.0
DEFAULT_MEMORY_MB = 32768
DEFAULT_TIMEOUT_SECONDS = 60 * 60 * 24
DEFAULT_VOLUME_COMMIT_INTERVAL_SECONDS = 300

volume = modal.Volume.from_name("latent-recovery-rl-volume", create_if_missing=True)


def load_gitignore_patterns() -> list[str]:
    """Translate .gitignore entries into Modal ignore globs."""
    if not modal.is_local():
        return []
    root = Path(__file__).resolve().parents[1]
    gitignore_path = root / ".gitignore"
    if not gitignore_path.is_file():
        return []
    patterns: list[str] = []
    for line in gitignore_path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#") or entry.startswith("!"):
            continue
        entry = entry.lstrip("/")
        if entry.endswith("/"):
            entry = entry.rstrip("/")
            patterns.append(f"**/{entry}/**")
        else:
            patterns.append(f"**/{entry}")
    return patterns


def _run_subprocess_with_periodic_volume_commits(cmd: list[str]) -> None:
    """Run cmd, committing the Modal volume every 5 minutes so checkpoints persist."""
    proc = subprocess.Popen(cmd, cwd=PROJECT_DIR)
    returncode: int | None = None
    try:
        while returncode is None:
            try:
                returncode = proc.wait(timeout=DEFAULT_VOLUME_COMMIT_INTERVAL_SECONDS)
            except subprocess.TimeoutExpired:
                volume.commit()
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        volume.commit()

    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)


def _setup_experiments_symlink() -> None:
    """Redirect experiments/ inside the project to the persistent volume.

    The training scripts write to experiments/<run_name>/ relative to the
    working directory (PROJECT_DIR).  We create a symlink so that all
    checkpoint files land on the Modal volume and survive after the
    container exits.
    """
    import os
    vol_exp = Path(VOLUME_PATH) / "experiments"
    vol_exp.mkdir(parents=True, exist_ok=True)

    proj_exp = Path(PROJECT_DIR) / "experiments"
    if proj_exp.is_symlink():
        proj_exp.unlink()
    elif proj_exp.exists():
        # Shouldn't happen (gitignored), but handle gracefully.
        import shutil
        shutil.rmtree(proj_exp)
    proj_exp.symlink_to(vol_exp)
    print(f"[modal] experiments/ → {vol_exp}")


# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        # MuJoCo headless rendering
        "libosmesa6-dev",
        "libgl1-mesa-glx",
        "libglfw3",
        "patchelf",
        "libglew-dev",
    )
    .uv_sync(extras=["remote"])
    # CUDA-enabled torch (overrides the CPU-only wheel from uv_sync)
    .run_commands(
        "uv pip install --system "
        "--index-url https://download.pytorch.org/whl/cu124 "
        "'torch>=2.0,<2.7'"
    )
    # MetaWorld 3.0 (not on PyPI with v3 support — must install from git).
    # uv detects the active venv from PATH (set by uv_sync) and installs there.
    .run_commands(
        "uv pip install 'git+https://github.com/Farama-Foundation/Metaworld.git'"
    )
)

if NETRC_PATH.is_file():
    image = image.add_local_file(
        NETRC_PATH,
        remote_path="/root/.netrc",
        copy=True,
    )

image = image.add_local_dir(
    ".",
    remote_path=PROJECT_DIR,
    ignore=load_gitignore_patterns(),
)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = modal.App(APP_NAME)

function_secrets = []
if os.environ.get("WANDB_API_KEY"):
    function_secrets.append(
        modal.Secret.from_dict({"WANDB_API_KEY": os.environ["WANDB_API_KEY"]})
    )

env = {
    "PYTHONPATH": PROJECT_DIR,
    "PYTHONUNBUFFERED": "1",
    # Headless MuJoCo rendering
    "MUJOCO_GL": "osmesa",
    # Cache HuggingFace / wandb artefacts on the volume
    "WANDB_DIR": f"{VOLUME_PATH}/wandb",
}


# ---------------------------------------------------------------------------
# Remote functions
# ---------------------------------------------------------------------------

@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=DEFAULT_TIMEOUT_SECONDS,
    env=env,
    image=image,
    secrets=function_secrets,
    gpu=DEFAULT_GPU,
    cpu=DEFAULT_CPU,
    memory=DEFAULT_MEMORY_MB,
)
def train_sac(*args: str) -> None:
    """Run scripts/train_sac.py with the given args."""
    _setup_experiments_symlink()
    cmd = ["python", "-u", "scripts/train_sac.py", *args]
    print(f"[modal] running: {' '.join(cmd)}")
    _run_subprocess_with_periodic_volume_commits(cmd)


@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=DEFAULT_TIMEOUT_SECONDS,
    env=env,
    image=image,
    secrets=function_secrets,
    gpu=DEFAULT_GPU,
    cpu=DEFAULT_CPU,
    memory=DEFAULT_MEMORY_MB,
)
def train_decoupled(*args: str) -> None:
    """Run scripts/train_decoupled.py with the given args."""
    _setup_experiments_symlink()
    cmd = ["python", "-u", "scripts/train_decoupled.py", *args]
    print(f"[modal] running: {' '.join(cmd)}")
    _run_subprocess_with_periodic_volume_commits(cmd)


# ---------------------------------------------------------------------------
# Local entrypoint (default: train_sac)
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(*args: str) -> None:
    """Default entrypoint: forward args to train_sac.

    For train_decoupled use:
        modal run scripts/modal_train.py::train_decoupled -- [args]
    """
    train_sac.remote(*args)
