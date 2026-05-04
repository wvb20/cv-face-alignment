"""Helpers for bootstrapping notebooks locally and in Google Colab."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
import sys
from pathlib import Path


@dataclass(frozen=True)
class NotebookRuntime:
    """Resolved notebook runtime settings."""
    in_colab: bool
    repo_root: Path
    workspace_dir: Path


def is_colab() -> bool:
    """Return True when running inside Google Colab."""
    try:
        import google.colab  # type: ignore
        return True
    except ImportError:
        return False


def find_repo_root(start: Path | None = None) -> Path:
    """Walk upward until we find the repository root containing src/config.py."""
    current = (start or Path.cwd()).expanduser().resolve()
    for candidate in (current, *current.parents):
        if (candidate / 'src' / 'config.py').exists():
            return candidate
    raise RuntimeError(
        'Could not find the repo root. Start Jupyter from the repository or '
        'set CV_FACE_ALIGNMENT_REPO_ROOT.'
    )


def configure_notebook_runtime(
    repo_root: str | Path | None = None,
    workspace_dir: str | Path | None = None,
    mount_drive: bool = True,
) -> NotebookRuntime:
    """
    Configure environment variables and workspace paths for notebook sessions.

    Call this after the repo is available locally (either already present or
    cloned into the Colab VM) and before importing modules that depend on
    ``src.config``.
    """
    repo_root_path = Path(repo_root).expanduser().resolve() if repo_root else find_repo_root()
    if str(repo_root_path) not in sys.path:
        sys.path.insert(0, str(repo_root_path))

    in_colab = is_colab()
    if in_colab and mount_drive:
        from google.colab import drive  # type: ignore
        drive.mount('/content/drive')

    default_workspace = (
        Path('/content/drive/MyDrive/cv-face-alignment')
        if in_colab else repo_root_path
    )
    workspace_path = Path(
        workspace_dir
        if workspace_dir is not None
        else os.environ.get('CV_FACE_ALIGNMENT_WORKSPACE_DIR', default_workspace)
    ).expanduser().resolve()

    os.environ['CV_FACE_ALIGNMENT_REPO_ROOT'] = str(repo_root_path)
    os.environ['CV_FACE_ALIGNMENT_WORKSPACE_DIR'] = str(workspace_path)

    from . import config
    importlib.reload(config)
    config.ensure_workspace_dirs()

    return NotebookRuntime(
        in_colab=in_colab,
        repo_root=repo_root_path,
        workspace_dir=Path(config.PROJECT_DIR),
    )
