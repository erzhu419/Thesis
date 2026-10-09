"""Reproducible experiment, evaluation, and artifact protocol."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def seed_everything(seed: int, deterministic_torch: bool = True) -> None:
    """Seed all RNGs used by the training pipeline."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic_torch, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic_torch


@dataclass(frozen=True)
class EvaluationProtocol:
    """Disjoint seeds for adaptive validation and one-shot final testing."""

    validation_seed_start: int = 10_000
    test_seed_start: int = 20_000
    validation_episodes: int = 5
    test_episodes: int = 20

    def __post_init__(self) -> None:
        if self.validation_episodes <= 0 or self.test_episodes <= 0:
            raise ValueError("evaluation episode counts must be positive")
        val = set(self.validation_seeds)
        test = set(self.test_seeds)
        if val & test:
            raise ValueError("validation and test seeds overlap")

    @property
    def validation_seeds(self) -> tuple[int, ...]:
        return tuple(range(self.validation_seed_start,
                           self.validation_seed_start + self.validation_episodes))

    @property
    def test_seeds(self) -> tuple[int, ...]:
        return tuple(range(self.test_seed_start,
                           self.test_seed_start + self.test_episodes))


@dataclass
class InteractionLedger:
    """Count target-domain access separately by purpose."""

    sim_training_steps: int = 0
    real_training_steps: int = 0
    real_validation_steps: int = 0
    real_test_steps: int = 0

    @property
    def real_total_steps(self) -> int:
        return self.real_training_steps + self.real_validation_steps + self.real_test_steps

    def to_dict(self) -> dict[str, int]:
        result = asdict(self)
        result["real_total_steps"] = self.real_total_steps
        return result


@dataclass
class TrainingWindowTracker:
    """Aggregate online-training episodes without consulting evaluation data."""

    current_return: float = 0.0
    current_violations: int = 0
    completed_returns: list[float] = field(default_factory=list)
    completed_violations: list[float] = field(default_factory=list)

    def observe(self, reward: float, episode_done: bool,
                violation: bool = False) -> None:
        self.current_return += float(reward)
        self.current_violations += int(bool(violation))
        if episode_done:
            self.completed_returns.append(self.current_return)
            self.completed_violations.append(float(self.current_violations))
            self.current_return = 0.0
            self.current_violations = 0

    def snapshot(self) -> dict[str, float | int]:
        """Return one reporting window and retain any unfinished episode."""
        completed = len(self.completed_returns)
        if completed:
            return_mean = float(np.mean(self.completed_returns))
            violations_mean = float(np.mean(self.completed_violations))
        else:
            return_mean = float(self.current_return)
            violations_mean = float(self.current_violations)
        self.completed_returns.clear()
        self.completed_violations.clear()
        return {
            "return_mean": return_mean,
            "violations_per_episode": violations_mean,
            "completed_episodes": completed,
        }


def _run_git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo_root, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    return result.stdout.strip()


def _dependency_versions(names: Iterable[str]) -> dict[str, str]:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "missing"
    return versions


def runtime_metadata() -> dict[str, Any]:
    """Return stable runtime and accelerator metadata for run provenance."""
    metadata: dict[str, Any] = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        metadata["accelerator"] = {
            "name": torch.cuda.get_device_name(device),
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": [int(properties.major), int(properties.minor)],
        }
    else:
        metadata["accelerator"] = None
    return metadata


def repository_status(repo_root: str | os.PathLike) -> str:
    """Return porcelain Git status for provenance and clean-run gates."""
    return _run_git(Path(repo_root), "status", "--porcelain=v1", "--untracked-files=all")


def require_clean_repository(repo_root: str | os.PathLike) -> None:
    """Reject confirmatory runs whose source cannot be named by one commit."""
    status = repository_status(repo_root)
    if status:
        preview = "\n".join(status.splitlines()[:20])
        suffix = "\n..." if len(status.splitlines()) > 20 else ""
        raise RuntimeError(
            "confirmatory runs require a clean Git worktree; commit or stash "
            f"the following changes first:\n{preview}{suffix}"
        )


def source_tree_sha256(repo_root: str | os.PathLike) -> str:
    """Hash executable project sources, including relevant untracked files."""
    root = Path(repo_root)
    include = (root / "mc_wm", root / "experiments", root / "requirements.txt",
               root / "requirements-lock.txt", root / "pytest.ini")
    allowed_suffixes = {".py", ".sh", ".txt", ".ini", ".toml", ".yaml", ".yml"}
    files: list[Path] = []
    for candidate in include:
        if candidate.is_file():
            files.append(candidate)
        elif candidate.is_dir():
            files.extend(
                path for path in candidate.rglob("*")
                if path.is_file()
                and path.suffix in allowed_suffixes
                and "__pycache__" not in path.parts
            )
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def completed_run_matches_source(run_dir: str | os.PathLike,
                                 repo_root: str | os.PathLike) -> bool:
    """Return True only for a clean, complete run from the current source tree."""
    path = Path(run_dir)
    try:
        with (path / "manifest.json").open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        with (path / "summary.json").open(encoding="utf-8") as handle:
            summary = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        isinstance(summary, dict)
        and int(manifest.get("schema_version", 0)) >= 2
        and not manifest.get("git", {}).get("dirty", True)
        and manifest.get("git", {}).get("source_tree_sha256")
        == source_tree_sha256(repo_root)
    )


class RunArtifacts:
    """Versioned, machine-readable outputs for one training seed."""

    def __init__(self, run_dir: str | os.PathLike, overwrite: bool = False):
        self.run_dir = Path(run_dir)
        if self.run_dir.exists() and any(self.run_dir.iterdir()):
            if not overwrite:
                raise FileExistsError(
                    f"run directory is not empty: {self.run_dir}; "
                    "use --overwrite_run explicitly"
                )
            archive_dir = self.run_dir.parent / ".archive"
            archive_dir.mkdir(parents=True, exist_ok=True)
            backup = archive_dir / (
                f"{self.run_dir.name}.{time.strftime('%Y%m%d-%H%M%S')}."
                f"{os.getpid()}.{time.time_ns()}")
            if backup.exists():
                raise FileExistsError(f"run backup already exists: {backup}")
            os.replace(self.run_dir, backup)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"

    def write_manifest(self, args: dict[str, Any], repo_root: str | os.PathLike,
                       evaluation: EvaluationProtocol,
                       extra: dict[str, Any] | None = None) -> None:
        root = Path(repo_root)
        diff = _run_git(root, "diff", "--binary", "HEAD")
        status = repository_status(root)
        manifest = {
            "schema_version": 2,
            "arguments": args,
            "evaluation_protocol": asdict(evaluation),
            "git": {
                "commit": _run_git(root, "rev-parse", "HEAD"),
                "dirty": bool(status),
                "status_porcelain": status.splitlines(),
                "diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
                "source_tree_sha256": source_tree_sha256(root),
            },
            "dependencies": _dependency_versions(
                ("numpy", "torch", "pysindy", "scikit-learn", "gymnasium",
                 "mujoco", "scipy", "statsmodels", "asteval", "matplotlib")
            ),
            "runtime": runtime_metadata(),
        }
        if extra:
            manifest["extra"] = extra
        self._write_json(self.run_dir / "manifest.json", manifest)

    def append_metric(self, record: dict[str, Any]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    def write_summary(self, summary: dict[str, Any]) -> None:
        self._write_json(self.run_dir / "summary.json", summary)

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
