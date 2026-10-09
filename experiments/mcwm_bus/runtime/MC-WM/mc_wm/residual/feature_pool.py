"""
Cross-environment feature pool for self-hypothesis discovery.

Persists accepted SINDy features across runs and environments so that
features that worked once can be retried as candidates in later runs
(possibly in different envs).  Each entry tracks per-env acceptance
counts and average reward gain so the candidate-injection step can rank
by global priority.

JSON storage is opt-in and must be scoped by the experiment runner.  Format::

    {
      "schema_version": 1,
      "features": {
        "sin_3_x2": {
          "expr": "np.sin(3 * s[:, 2])",
          "envs": {
            "gravity_cheetah": {"accept": 5, "reject": 2, "avg_reward_gain": 0.012}
          },
          "accept_total": 5,
          "reject_total": 2,
          "global_avg_reward_gain": 0.012,
          "last_used_iso": "2026-04-26T..."
        }
      }
    }

Constructing ``FeaturePool()`` creates an in-memory, run-local pool.  A
persistent pool requires an explicit path.  This prevents hidden cross-run
state from contaminating seeds and ablations.  Deliberate transfer is still
possible by passing a versioned snapshot path from the experiment manifest.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable

SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


class FeaturePool:
    """
    Run-local or explicitly persisted feature pool.

    Typical lifecycle in a training run:

    1. ``pool = FeaturePool(path=run_dir / "feature_pool.json")``
    2. ``cands = pool.query_candidates(env="gravity_cheetah", top_k=10)``
       — returns expressions to add to the candidate library
    3. After training, for each accepted feature with measured reward gain::

           pool.record(name, expr, env, reward_gain=0.012, was_accepted=True)
           pool.save()

    Reward gain is the optional Stage-D mini-rollout score; without it, a
    feature still counts as "accepted" but with ``reward_gain=0`` so its
    global priority only reflects acceptance frequency.
    """

    def __init__(self, path: str | os.PathLike | None = None,
                 priority_floor: float = 0.0, read_only: bool = False):
        self.path = Path(path) if path is not None else None
        # When `query_candidates` ranks features, anything with global
        # priority below this threshold is dropped from the result.
        self.priority_floor = priority_floor
        self.read_only = bool(read_only)
        self._data = self._load()
        self._pending: list[dict] = []

    # ─── Persistence ─────────────────────────────────────────────────
    def _load(self) -> dict:
        if self.path is None:
            return {"schema_version": SCHEMA_VERSION, "features": {}}
        if not self.path.exists():
            return {"schema_version": SCHEMA_VERSION, "features": {}}
        try:
            with self.path.open("r") as f:
                d = json.load(f)
        except (json.JSONDecodeError, OSError):
            # Corrupt or unreadable — start fresh, but back up the bad file.
            backup = self.path.with_suffix(".bad")
            try:
                self.path.rename(backup)
            except OSError:
                pass
            return {"schema_version": SCHEMA_VERSION, "features": {}}
        if d.get("schema_version") != SCHEMA_VERSION:
            # Future-proof: in v2+ migrate fields here. For now reset.
            return {"schema_version": SCHEMA_VERSION, "features": {}}
        return d

    def save(self) -> None:
        """Merge pending records under a process-safe lock and save atomically."""
        if self.path is None or self.read_only or not self._pending:
            return

        import fcntl

        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        with lock_path.open("a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            disk_data = self._load()
            for event in self._pending:
                self._apply_record(disk_data, **event)
            tmp = self.path.with_suffix(self.path.suffix + f".{os.getpid()}.tmp")
            with tmp.open("w") as f:
                json.dump(disk_data, f, indent=2, sort_keys=True)
            os.replace(tmp, self.path)
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        self._data = disk_data
        self._pending.clear()

    # ─── Read API ────────────────────────────────────────────────────
    def all_features(self) -> dict:
        """Return raw dict {name: meta}.  Caller must not mutate."""
        return self._data["features"]

    def query_candidates(self, env: str | None = None,
                         top_k: int = 10) -> list[tuple[str, str, float]]:
        """
        Return up to ``top_k`` candidate features ranked by global priority.

        Priority = (accept_total / max(1, accept_total + reject_total))
                 * (1 + reward_gain_clipped),  reward_gain in [0, 1]

        Features that have never appeared in ``env`` are ranked using the
        global priority alone; features already accepted in the same env
        get a small bonus so they re-appear preferentially.

        Returns:
            list of (name, expr, priority) tuples, sorted desc by priority.
        """
        feats = self._data["features"]
        ranked: list[tuple[str, str, float]] = []
        for name, meta in feats.items():
            acc = meta.get("accept_total", 0)
            rej = meta.get("reject_total", 0)
            denom = max(1, acc + rej)
            base = acc / denom
            gain = max(0.0, min(1.0, meta.get("global_avg_reward_gain", 0.0)))
            priority = base * (1.0 + gain)
            if env is not None and env in meta.get("envs", {}):
                priority *= 1.1  # mild bonus for in-env reuse
            if priority < self.priority_floor:
                continue
            ranked.append((name, meta["expr"], priority))
        ranked.sort(key=lambda t: -t[2])
        return ranked[:top_k]

    # ─── Write API ───────────────────────────────────────────────────
    def record(self, name: str, expr: str, env: str,
               reward_gain: float | None = None,
               was_accepted: bool = True) -> None:
        """
        Update the pool with one feature outcome from a training run.

        Args:
          name: feature name (e.g. "sin_3_x2")
          expr: Python expression (e.g. "np.sin(3 * s[:, 2])")
          env: env identifier (e.g. "gravity_soft_ceiling")
          reward_gain: per-feature reward improvement from mini-rollout.
                       ``None`` is allowed if Stage D wasn't run; treated
                       as 0.0 in the priority computation.
          was_accepted: True if the feature was accepted into the basis
                        (e.g. orthogonality + correlation passed).
        """
        if self.read_only:
            raise RuntimeError("cannot record into a read-only feature pool")
        event = {
            "name": name, "expr": expr, "env": env,
            "reward_gain": reward_gain, "was_accepted": was_accepted,
        }
        self._apply_record(self._data, **event)
        if self.path is not None:
            self._pending.append(event)

    @staticmethod
    def _apply_record(data: dict, name: str, expr: str, env: str,
                      reward_gain: float | None = None,
                      was_accepted: bool = True) -> None:
        feats = data["features"]
        entry = feats.setdefault(name, {
            "expr": expr,
            "envs": {},
            "accept_total": 0,
            "reject_total": 0,
            "global_avg_reward_gain": 0.0,
        })
        entry["expr"] = expr
        entry["last_used_iso"] = _utc_now()
        env_stats = entry["envs"].setdefault(
            env, {"accept": 0, "reject": 0, "avg_reward_gain": 0.0}
        )
        n_prev_global = entry["accept_total"]
        if was_accepted:
            entry["accept_total"] += 1
            env_stats["accept"] += 1
        else:
            entry["reject_total"] += 1
            env_stats["reject"] += 1
        if was_accepted and reward_gain is not None:
            env_acc = env_stats["accept"]
            env_stats["avg_reward_gain"] = (
                (env_stats["avg_reward_gain"] * (env_acc - 1) + reward_gain)
                / max(1, env_acc)
            )
            entry["global_avg_reward_gain"] = (
                (entry["global_avg_reward_gain"] * n_prev_global + reward_gain)
                / max(1, entry["accept_total"])
            )

    def record_batch(self, items: Iterable[dict]) -> None:
        """Convenience wrapper to record many outcomes at once."""
        for it in items:
            self.record(**it)


def evaluate_expression(expr: str, s: "np.ndarray", a: "np.ndarray"):
    """
    Safely evaluate a feature expression against numpy state-action arrays.

    Mirrors the safe-eval used in `sindy_nau_adapter._eval_llm_features`:
    expressions can reference ``s`` (shape (N, obs_dim)) and ``a`` (shape
    (N, act_dim)) and ``np``.  Anything else raises NameError.

    Returns:
        (N,) numpy array, or None on failure / non-finite values.
    """
    from mc_wm.self_audit.safe_expression import evaluate_array_expression
    # Historical expressions use either ``s[:, i]`` or ``s[i]``. Evaluate
    # row-major first, then dimension-major, and accept only an (N,) result.
    for transpose in (False, True):
        arr = evaluate_array_expression(expr, s, a, transpose=transpose)
        if arr is not None:
            return arr
    return None
