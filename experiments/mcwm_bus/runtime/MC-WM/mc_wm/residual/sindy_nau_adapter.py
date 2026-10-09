"""Sparse additive residual adapter.

Architecture:
  1. SINDy discovers symbolic features Θ(s,a) from paired data (poly2 + STLSQ)
  2. A small masked neural head learns only the SINDy remainder
  3. Online refit updates sparse coefficients and the remainder head

Advantages over MLP δ:
  - Interpretable: symbolic terms show WHAT the gap is (e.g., -0.3*v for gravity)
  - Sparse: STLSQ keeps only active terms → less overfitting on small data
  - Auditable: feature selection and reporting use disjoint validation/calibration data

Same interface as ResidualAdapter:
  - fit(states, actions, sim_ns, sim_r, real_ns, real_r)
  - predict_correction(states, actions) → (Δs_correction, Δr_correction)
  - _trained flag
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pysindy as ps
from sklearn.linear_model import Ridge
from copy import deepcopy

from mc_wm.networks.residual_head import SparseNeuralRemainderHead
from mc_wm.self_audit.diagnosis import DiagnosisBattery
from mc_wm.self_audit.auto_expand import AutoExpander
from mc_wm.self_audit.llm_oracle import LLMOracle
from mc_wm.self_audit.orthogonal_expand import OrthogonalExpander
from mc_wm.self_audit.hypothesis import (
    Hypothesis, HypothesisLog, leave_one_feature_out_mse,
)
from mc_wm.self_audit.safe_expression import evaluate_array_expression


class SparseAdditiveResidualAdapter:
    """
    SINDy feature discovery + masked neural remainder correction.

    Phase 1 (SINDy): Discover symbolic features from paired data
      Θ(s,a) = poly2 features, sparsified by STLSQ
      Per-dim SINDy coefficients give initial linear prediction

    Phase 2 (additive remainder):
      correction = sparse_SINDy(Θ) + neural_remainder(masked Θ)
      The neural branch is zero-initialized and sees only active SINDy terms.
    """

    def __init__(self, obs_dim, act_dim, sindy_threshold=0.05, sindy_alpha=0.05,
                 nau_hidden=32, lr=1e-3, device="cpu",
                 max_rounds=3, eps_threshold=0.05, diagnosis_alpha=0.05,
                 log_fn=None, env_type=None, claude_oracle=None,
                 env_description_for_llm: str | None = None,
                 # ── RAHD knobs (all default-off: legacy behavior preserved)
                 # A. cross-env feature pool: when set, query/update this pool
                 #    so candidates are seeded from prior runs and accepted
                 #    features are written back.
                 feature_pool=None,
                 feature_pool_top_k: int = 10,
                 # B. ΔL_eff stability gate on orthogonal expansion.
                 max_delta_beta_inf: float | None = None,
                 # C. policy-aware fit weights.  Pass a PolicyDensity instance
                 #    or None.  When set, the SINDy fit becomes a weighted LS.
                 policy_density=None,
                 # D. reward-aware acceptance.  Pass a RewardValidator-like
                 #    callable, or None.  When set, every accepted feature
                 #    is run through ``reward_validator`` and rejected if
                 #    its delta is below ``reward_validator_min_delta``.
                 reward_validator=None,
                 reward_validator_min_delta: float = 0.0,
                 # ── E3: feature-library baseline selector ─────────────
                 # "poly2"        — default, current behavior (poly2 + orth + LLM)
                 # "poly2_only"   — poly2, no orthogonal expansion, no LLM
                 # "poly3_only"   — poly3, no orthogonal expansion, no LLM
                 # "trig_only"    — sin/cos library, no orth, no LLM
                 # "random_K"     — random Fourier features, count=K
                 feature_library: str = "poly2",
                 random_feature_count: int = 100,
                 feature_budget: int | None = None,
                 disable_orthogonal_expansion: bool = False,
                 random_state: int = 0,
                 validation_fraction: float = 0.1,
                 calibration_fraction: float = 0.1,
                 min_feature_val_gain: float = 1e-5,
                 residual_mode: str = "additive",
                 # ── HypothesisLog: evidence chain for every accepted/rejected
                 # feature/constraint/HP.  Default = on; pass run_id="" to
                 # disable persistence.
                 hypothesis_log_run_id: str | None = None,
                 hypothesis_log_root=None):
        self._log = log_fn or (lambda msg: print(msg))
        self._env_type = env_type  # if set, use LLM oracle instead of auto-expand
        # RAHD module hooks (any can be None → legacy behavior).
        self._feature_pool = feature_pool
        self._feature_pool_top_k = feature_pool_top_k
        self._max_delta_beta_inf = max_delta_beta_inf
        self._policy_density = policy_density
        self._reward_validator = reward_validator
        self._reward_validator_min_delta = reward_validator_min_delta
        # Pool-side tag — independent from ``_env_type`` so we don't change
        # the semantics of the legacy LLMOracle gating.  Defaults to a stable
        # string; callers may override to a fine-grained env name.
        self._pool_env_tag = env_type or "unknown_env"
        # Track feature provenance so the pool can be updated correctly
        # at end-of-fit: name → "pool" | "orth" | "llm".
        self._feat_provenance: dict[str, str] = {}
        # Hypothesis-evidence log (one JSONL per run).  When ``hypothesis_log_run_id``
        # is None we synthesise one; pass "" to disable.
        if hypothesis_log_run_id == "":
            self._hyp_log = None
        else:
            import time, os
            rid = hypothesis_log_run_id or f"{self._pool_env_tag}-{os.getpid()}-{int(time.time())}"
            self._hyp_log = (HypothesisLog(rid, root=hypothesis_log_root)
                             if hypothesis_log_root is not None else HypothesisLog(rid))
        # Role #2: Claude oracle for feature hypotheses in the self-hypothesis loop.
        # When set, augments orthogonal expansion with LLM-suggested physics features.
        self._claude_oracle = claude_oracle
        self._env_desc_llm = env_description_for_llm or ""
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.input_dim = obs_dim + act_dim
        self.device = device
        self.sindy_threshold = sindy_threshold
        self.sindy_alpha = sindy_alpha
        self.random_state = int(random_state)
        self._rng = np.random.default_rng(self.random_state)
        self.validation_fraction = float(validation_fraction)
        self.calibration_fraction = float(calibration_fraction)
        self.min_feature_val_gain = float(min_feature_val_gain)
        if residual_mode not in (
                "additive", "sindy_only", "neural_only", "dense_neural"):
            raise ValueError(f"unknown residual_mode: {residual_mode}")
        self.residual_mode = residual_mode
        if not (0.0 < self.validation_fraction < 0.5):
            raise ValueError("validation_fraction must be in (0, 0.5)")
        if not (0.0 < self.calibration_fraction < 0.5):
            raise ValueError("calibration_fraction must be in (0, 0.5)")
        if self.validation_fraction + self.calibration_fraction >= 0.8:
            raise ValueError("validation and calibration fractions leave too little fit data")

        # Self-hypothesis loop config
        self.max_rounds = max_rounds
        self.eps_threshold = eps_threshold

        # SINDy components (initialized on first fit).  E3-style baselines:
        self._feature_library_kind = feature_library
        self._random_feature_count = random_feature_count
        if feature_budget is not None and feature_budget < 1:
            raise ValueError("feature_budget must be positive or None")
        self.feature_budget = feature_budget
        self._base_feature_indices = None
        self._disable_orthogonal_expansion = disable_orthogonal_expansion
        self._library = self._build_library(feature_library, random_feature_count)
        self._sindy_coefs = None
        self._n_features = None
        self._feature_names = None
        self._active_features = None
        self._extra_columns_fn = None  # function to compute extra columns for new data

        # Diagnosis battery + auto-expander
        self._battery = DiagnosisBattery(alpha=diagnosis_alpha)
        self._expander = AutoExpander(obs_dim=obs_dim, act_dim=act_dim)

        # Masked neural remainder head. ``_nau_head`` remains as a compatibility
        # alias for older scripts; it no longer represents an NAU/NMU module.
        self._nau_head = None
        self._nau_optimizer = None
        self._nau_lr = lr

        # A genuinely non-symbolic baseline. It is initialized lazily so the
        # established sparse modes retain their exact random-number stream.
        self._dense_state_net = None
        self._dense_state_optimizer = None
        self._dense_input_mean = None
        self._dense_input_std = None
        self._dense_target_mean = None
        self._dense_target_std = None

        # Separate reward correction (simple MLP, not shared with state SINDy)
        import torch.nn as tnn
        self._reward_net = tnn.Sequential(
            tnn.Linear(obs_dim + act_dim, 64), tnn.ReLU(),
            tnn.Linear(64, 1),
        ).to(device)
        self._reward_opt = optim.Adam(self._reward_net.parameters(), lr=lr, weight_decay=1e-4)
        self._reward_input_mean = None
        self._reward_input_std = None
        self._reward_target_mean = None
        self._reward_target_std = None

        # Termination prediction head: P(done | s, a, s') → full-tuple residual.
        # Binary classifier via BCE loss on real done labels.
        self._done_net = tnn.Sequential(
            tnn.Linear(obs_dim + act_dim + obs_dim, 64), tnn.ReLU(),
            tnn.Linear(64, 1),
        ).to(device)
        self._done_opt = optim.Adam(self._done_net.parameters(), lr=lr, weight_decay=1e-4)
        self._done_input_mean = None
        self._done_input_std = None

        # SGD-based coefficient layer for smooth online refit
        # Initialized after first fit with discovered sparsity pattern
        self._coef_layer = None  # nn.Linear(n_features, n_outputs, bias=False)
        self._coef_optimizer = None
        self._active_mask = None  # boolean mask from STLSQ sparsity

        # Training state
        self._trained = False
        self._train_center = None
        self._theta_scale = None
        self._support_reference = None
        self._calibration_error_q95 = None
        self._fit_errors = None
        self._hypothesis_logs = []
        self._loop_done = False
        # LLM (Role #2) tracking counters — all zero unless self._claude_oracle set.
        self._llm_feat_proposed = 0
        self._llm_feat_accepted = 0
        self._llm_feat_rejected_unsafe = 0
        self._llm_feat_rejected_eval = 0
        self._llm_feat_rejected_shape = 0
        self._llm_feat_rejected_nonfinite = 0
        self._llm_feat_rejected_validation = 0
        self._llm_rounds = 0
        # Decision history: each accepted/rejected LLM feature with final coef
        # outcome.  Fed back to future Role #2 calls so LLM sees how its past
        # proposals actually performed.
        self._llm_feat_history: list[dict] = []
        # Name → expression for LLM-added features, used by `_build_full_theta`
        # to re-evaluate on new (s, a) points at predict time.
        self._llm_feat_exprs: dict = {}

    def _eval_llm_features(self, SA, llm_feats):
        """
        Safely evaluate LLM-proposed expressions against the (s, a) stack.

        SA has shape (N, obs_dim + act_dim).  `llm_feats` is a list of dicts
        `{"expr": "np.sin(3*s[1])", "name": "sin_3theta", "why": "..."}`.

        Each expression is evaluated per-row with `s` = SA[i, :obs_dim],
        `a` = SA[i, obs_dim:].  We restrict globals to {"np": np} and forbid
        attribute access beyond numpy module.  Malformed / unsafe exprs
        silently drop the feature.
        """
        import numpy as np
        cols, names = [], []
        n = SA.shape[0]
        self._llm_rounds += 1
        for feat in llm_feats[:8]:
            self._llm_feat_proposed += 1
            expr = feat.get("expr", "")
            name = feat.get("name", expr[:20])
            if not expr:
                self._llm_feat_rejected_unsafe += 1
                self._log(f"    ✗ LLM expr rejected (unsafe): {expr[:50]}")
                continue
            try:
                s_arr = SA[:, :self.obs_dim]
                a_arr = SA[:, self.obs_dim:]
                col = evaluate_array_expression(
                    expr, s_arr, a_arr, transpose=True)
                if col is None:
                    self._llm_feat_rejected_shape += 1
                    self._log(f"    ✗ LLM expr rejected by safe evaluator: {name}")
                    continue
                std = col.std() + 1e-8
                cols.append(col / std)
                feat_name = f"llm_{name}"
                names.append(feat_name)
                # Store expr + std for predict-time re-evaluation
                self._llm_feat_exprs[feat_name] = {"expr": expr, "std": float(std)}
                self._llm_feat_history.append({
                    "round": self._llm_rounds,
                    "name": feat_name,
                    "expr": expr,
                    "why": feat.get("why", "")[:80],
                    "outcome": {"syntax_valid": True, "heldout_accepted": None},
                })
                self._log(f"      + {name}: {expr[:40]} (std={std:.3f})")
            except Exception as e:
                self._llm_feat_rejected_eval += 1
                self._log(f"    ✗ LLM expr eval failed: {name}  ({str(e)[:60]})")
                continue
        if not cols:
            return None, []
        return np.column_stack(cols), names

    def _update_llm_history_outcomes(self, feature_names: list, coefs: np.ndarray,
                                       threshold: float = 0.02) -> None:
        """Fill `outcome` field for LLM features based on final SINDy coefs."""
        if not self._llm_feat_history:
            return
        # Build name → (max_abs_coef, n_dims_active)
        name_to_stat = {}
        for i, n in enumerate(feature_names):
            if isinstance(n, str) and n.startswith("llm_"):
                col = coefs[:, i] if coefs.ndim == 2 else np.array([coefs[i]])
                name_to_stat[n] = {
                    "max_abs_coef": float(np.abs(col).max()),
                    "n_dims_active": int((np.abs(col) > threshold).sum()),
                }
        for entry in self._llm_feat_history:
            if entry["outcome"] is None and entry["name"] in name_to_stat:
                entry["outcome"] = name_to_stat[entry["name"]]

    def prune_llm_features(self, feature_names: list, coefs: np.ndarray,
                           threshold: float = 0.02,
                           theta_val: np.ndarray = None,
                           target_val: np.ndarray = None) -> list:
        """
        Role #4: review LLM-added feature contributions and recommend names
        to drop.  Now reports a real ``val_mse_delta`` (positive ⇒ feature
        is helping) computed via leave-one-feature-out on a validation slice
        when ``theta_val`` / ``target_val`` are supplied; falls back to a
        coefficient-magnitude heuristic only when they are not.

        Args:
            feature_names: full feature name list (same order as coefs cols).
            coefs: (n_dims, n_features) SINDy coefficient matrix after fit.
            threshold: |coef| below this counts as "inactive" for that dim.
            theta_val: optional (N_val, F) validation feature matrix.
            target_val: optional (N_val, D) validation target.

        Returns:
            list of LLM feature names to drop (caller rebuilds without them).
        """
        from mc_wm.self_audit.hypothesis import leave_one_feature_out_mse

        if self._claude_oracle is None:
            return []
        llm_indices = [i for i, n in enumerate(feature_names)
                       if isinstance(n, str) and n.startswith("llm_")]
        if not llm_indices:
            return []

        stats = []
        for i in llm_indices:
            col = coefs[:, i] if coefs.ndim == 2 else np.array([coefs[i]])
            max_abs = float(np.abs(col).max())
            n_active = int((np.abs(col) > threshold).sum())

            # Real per-feature val MSE drop (positive ⇒ feature helps).
            if theta_val is not None and target_val is not None and coefs.ndim == 2:
                mse_drop = leave_one_feature_out_mse(theta_val, coefs, target_val, i)
            else:
                mse_drop = 0.0  # not measurable here; caller may override later

            stats.append({
                "name": feature_names[i],
                "expr": feature_names[i][4:],  # strip "llm_" prefix for display
                "max_abs_coef": round(max_abs, 5),
                "n_dims_active": n_active,
                "val_mse_delta": round(mse_drop, 6),
            })
        to_drop = self._claude_oracle.role4_prune_features(
            self._env_desc_llm, stats)
        if to_drop:
            self._log(f"  [LLM Role #4] drops {len(to_drop)} LLM features: {to_drop[:4]}")
        else:
            self._log(f"  [LLM Role #4] reviewed {len(stats)} LLM features, kept all")
        self._llm_role4_dropped_features = getattr(
            self, "_llm_role4_dropped_features", 0) + len(to_drop)
        return to_drop

    def get_llm_summary(self) -> dict:
        """Return LLM Role #2 activity summary for end-of-training reporting."""
        oracle_stats = {}
        if self._claude_oracle is not None and hasattr(self._claude_oracle, "stats"):
            oracle_stats = self._claude_oracle.stats()
        return {
            "rounds": self._llm_rounds,
            "proposed": self._llm_feat_proposed,
            "accepted": self._llm_feat_accepted,
            "rejected_unsafe": self._llm_feat_rejected_unsafe,
            "rejected_eval": self._llm_feat_rejected_eval,
            "rejected_shape": self._llm_feat_rejected_shape,
            "rejected_nonfinite": self._llm_feat_rejected_nonfinite,
            "rejected_validation": self._llm_feat_rejected_validation,
            "role4_dropped_features": getattr(self, "_llm_role4_dropped_features", 0),
            "oracle_calls": oracle_stats.get("calls", 0),
            "oracle_cache_hits": oracle_stats.get("cache_hits", 0),
            "oracle_errors": oracle_stats.get("errors", 0),
        }

    def log_llm_summary(self):
        """Pretty-print the LLM summary via self._log.  Called at end of training."""
        if self._claude_oracle is None:
            return
        s = self.get_llm_summary()
        self._log("")
        self._log("┌─ [LLM Role #2 Summary] ─────────────────")
        self._log(f"│ rounds invoked      : {s['rounds']}")
        self._log(f"│ features proposed   : {s['proposed']}")
        self._log(f"│ features accepted   : {s['accepted']}")
        self._log(f"│ rejected (unsafe)   : {s['rejected_unsafe']}")
        self._log(f"│ rejected (eval err) : {s['rejected_eval']}")
        self._log(f"│ rejected (shape)    : {s['rejected_shape']}")
        self._log(f"│ rejected (non-fin)  : {s['rejected_nonfinite']}")
        self._log(f"│ rejected (held-out) : {s['rejected_validation']}")
        self._log(f"│ Role#4 pruned       : {s['role4_dropped_features']}")
        self._log(f"│ Claude CLI calls    : {s['oracle_calls']}")
        self._log(f"│ Claude cache hits   : {s['oracle_cache_hits']}")
        self._log(f"│ Claude errors       : {s['oracle_errors']}")
        if s['proposed'] > 0:
            accept_rate = 100.0 * s['accepted'] / s['proposed']
            self._log(f"│ acceptance rate     : {accept_rate:.1f}%")
        self._log("└──────────────────────────────────────────")

    def _build_library(self, kind, K):
        """E3 baseline feature libraries.

        ``kind``:
          * "poly2"          : poly2 (legacy default, with bias)
          * "poly2_only"     : poly2 with bias; orthogonal expansion is
                               disabled at fit-time, LLM is silenced
                               separately by ``claude_oracle=None``.
          * "poly3_only"     : poly3 with bias; same fit-time rules.
          * "trig_only"      : pysindy FourierLibrary (sin/cos basis).
          * "random_K"       : random-Fourier features fixed at construction.

        Setting any of the ``_only`` kinds also forces
        ``disable_orthogonal_expansion=True`` so the library is the only
        source of nonlinearity; this is the matched-parameter baseline
        the reviewer asked for in (E3).
        """
        kind = kind.lower()
        if kind == "poly2":
            return ps.PolynomialLibrary(degree=2, include_bias=True)
        if kind == "poly2_only":
            self._disable_orthogonal_expansion = True
            return ps.PolynomialLibrary(degree=2, include_bias=True)
        if kind == "poly3_only":
            self._disable_orthogonal_expansion = True
            return ps.PolynomialLibrary(degree=3, include_bias=True)
        if kind == "trig_only":
            self._disable_orthogonal_expansion = True
            return ps.FourierLibrary(n_frequencies=3, include_sin=True, include_cos=True)
        if kind.startswith("random"):
            self._disable_orthogonal_expansion = True
            parts = kind.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                K = int(parts[1])
            # Random Fourier features approximating an RBF kernel.
            class _RandomFourierLib:
                def __init__(self, K, dim, seed):
                    rng = np.random.default_rng(seed)
                    self._K = K
                    self._W = rng.standard_normal((dim, K))
                    self._b = rng.uniform(0, 2*np.pi, size=K)
                def fit(self, X): return self
                def transform(self, X):
                    proj = X @ self._W + self._b
                    return np.sqrt(2.0 / self._K) * np.cos(proj)
                def get_feature_names_out(self, input_features=None):
                    return [f"rff_{i}" for i in range(self._K)]
            return _RandomFourierLib(K=K, dim=self.input_dim,
                                     seed=self.random_state)
        raise ValueError(f"unknown feature_library '{kind}'")

    def _feature_names_from_library(self) -> list[str]:
        if hasattr(self._library, "get_feature_names_out"):
            return list(self._library.get_feature_names_out())
        return list(self._library.get_feature_names())

    def _select_base_feature_indices(self, names: list[str]) -> np.ndarray:
        """Choose a deterministic, target-independent base library subset."""
        total = len(names)
        if self.feature_budget is None or total <= self.feature_budget:
            return np.arange(total, dtype=int)
        has_expansion = (not self._disable_orthogonal_expansion
                         or self._claude_oracle is not None)
        reserve = min(8, self.feature_budget - 1) if has_expansion else 0
        base_budget = self.feature_budget - reserve
        essential = [
            index for index, name in enumerate(names)
            if name == "1" or (name.startswith("x") and name[1:].isdigit())
        ]
        essential = essential[:base_budget]
        remaining = [index for index in range(total) if index not in set(essential)]
        selector_rng = np.random.default_rng(self.random_state + 91_827)
        selector_rng.shuffle(remaining)
        selected = essential + remaining[:max(0, base_budget - len(essential))]
        return np.asarray(sorted(selected), dtype=int)

    def _split_indices(self, n_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return disjoint fit, feature-selection, and calibration indices."""
        if n_samples < 30:
            raise ValueError("at least 30 transitions are required for a three-way split")
        perm = self._rng.permutation(n_samples)
        n_val = max(5, int(round(n_samples * self.validation_fraction)))
        n_cal = max(5, int(round(n_samples * self.calibration_fraction)))
        if n_val + n_cal >= n_samples - 5:
            raise ValueError("not enough fit samples after validation/calibration split")
        return perm[n_val + n_cal:], perm[:n_val], perm[n_val:n_val + n_cal]

    def fit_done_head(self, states, actions, next_states, real_dones,
                      train_indices=None, validation_indices=None,
                      n_epochs=80, patience=15):
        """Fit or update P(done | s, a, s') without refitting state/reward heads."""
        states = np.asarray(states)
        actions = np.asarray(actions)
        next_states = np.asarray(next_states)
        dones = np.asarray(real_dones, dtype=np.float32).reshape(-1, 1)
        n_samples = len(states)
        if not (len(actions) == len(next_states) == len(dones) == n_samples):
            raise ValueError("termination-head arrays must have equal lengths")
        if n_samples < 30:
            raise ValueError("termination-head fit requires at least 30 transitions")
        if (train_indices is None) != (validation_indices is None):
            raise ValueError(
                "train_indices and validation_indices must be provided together")
        if train_indices is None:
            fit_idx, val_idx, _ = self._split_indices(n_samples)
        else:
            fit_idx = np.asarray(train_indices, dtype=np.int64)
            val_idx = np.asarray(validation_indices, dtype=np.int64)
            if len(fit_idx) == 0 or len(val_idx) == 0:
                raise ValueError("termination-head splits must be nonempty")
            if (fit_idx.min() < 0 or fit_idx.max() >= n_samples
                    or val_idx.min() < 0 or val_idx.max() >= n_samples):
                raise ValueError("termination-head split index is out of range")
            if np.intersect1d(fit_idx, val_idx).size:
                raise ValueError("termination-head splits must be disjoint")

        sas = np.concatenate(
            [states, actions, next_states], axis=-1).astype(np.float32)
        if self._done_input_mean is None:
            self._done_input_mean = sas[fit_idx].mean(axis=0, keepdims=True)
            self._done_input_std = sas[fit_idx].std(axis=0, keepdims=True)
            self._done_input_std = np.where(
                self._done_input_std > 1e-6, self._done_input_std, 1.0)
        sas = (sas - self._done_input_mean) / self._done_input_std
        sas_t = torch.as_tensor(sas, dtype=torch.float32, device=self.device)
        done_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        best_val = float("inf")
        best_epoch = 0
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in self._done_net.state_dict().items()
        }
        for epoch in range(n_epochs):
            self._done_net.train()
            permutation = self._rng.permutation(fit_idx)
            for start in range(0, len(permutation), 256):
                batch = permutation[start:start + 256]
                logits = self._done_net(sas_t[batch])
                loss = nn.BCEWithLogitsLoss()(logits, done_t[batch])
                self._done_opt.zero_grad()
                loss.backward()
                self._done_opt.step()
            self._done_net.eval()
            with torch.no_grad():
                logits = self._done_net(sas_t[val_idx])
                val_loss = float(nn.BCEWithLogitsLoss()(logits, done_t[val_idx]))
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self._done_net.state_dict().items()
                }
            if epoch - best_epoch >= patience:
                break
        self._done_net.load_state_dict(best_state)
        self._done_trained = True
        self._done_fit_calls = getattr(self, "_done_fit_calls", 0) + 1
        self._done_last_train_rows = int(n_samples)
        n_positive = int(dones.sum())
        self._log(
            f"  Done MLP: val BCE={best_val:.5f} "
            f"(pos/total={n_positive}/{n_samples}, "
            f"{100*n_positive/n_samples:.1f}%, fit={self._done_fit_calls})")
        return best_val

    @staticmethod
    def _column_scale(theta_fit: np.ndarray) -> np.ndarray:
        scale = np.sqrt(np.mean(np.square(theta_fit), axis=0))
        return np.where(scale > 1e-8, scale, 1.0)

    def _validation_mse(self, theta_fit: np.ndarray, targets_fit: np.ndarray,
                        theta_val: np.ndarray, targets_val: np.ndarray,
                        sample_weights=None) -> float:
        scale = self._column_scale(theta_fit)
        coefs, _, _ = self._sindy_fit_one_round(
            theta_fit / scale, targets_fit, sample_weights=sample_weights)
        pred = (theta_val / scale) @ coefs.T
        return float(np.mean((pred - targets_val) ** 2))

    def _gate_candidate_columns(self, theta_base: np.ndarray,
                                candidate_columns: np.ndarray,
                                candidate_names: list[str], targets: np.ndarray,
                                fit_idx: np.ndarray, val_idx: np.ndarray,
                                sample_weights=None) -> tuple[np.ndarray, list[str], list[dict]]:
        """Accept candidates only when a train-refit improves held-out MSE."""
        if candidate_columns is None or candidate_columns.shape[1] == 0:
            return np.zeros((len(theta_base), 0)), [], []

        accepted_cols: list[np.ndarray] = []
        accepted_names: list[str] = []
        decisions: list[dict] = []
        current = theta_base
        fit_weights = sample_weights[fit_idx] if sample_weights is not None else None
        baseline = self._validation_mse(
            current[fit_idx], targets[fit_idx], current[val_idx], targets[val_idx],
            sample_weights=fit_weights)

        for j, name in enumerate(candidate_names):
            col = np.asarray(candidate_columns[:, j], dtype=np.float64)
            scale = float(np.sqrt(np.mean(col[fit_idx] ** 2)))
            if not np.isfinite(scale) or scale <= 1e-8:
                decisions.append({"name": name, "accepted": False,
                                  "reason": "constant_or_nonfinite"})
                continue
            trial = np.column_stack([current, col])
            trial_mse = self._validation_mse(
                trial[fit_idx], targets[fit_idx], trial[val_idx], targets[val_idx],
                sample_weights=fit_weights)
            gain = baseline - trial_mse
            accepted = bool(np.isfinite(gain) and gain > self.min_feature_val_gain)
            decisions.append({"name": name, "accepted": accepted,
                              "baseline_mse": baseline, "trial_mse": trial_mse,
                              "val_mse_gain": gain})
            if accepted:
                current = trial
                baseline = trial_mse
                accepted_cols.append(col)
                accepted_names.append(name)
            else:
                self._log(f"      - {name}: held-out gain={gain:.3e} (rejected)")

        cols = (np.column_stack(accepted_cols) if accepted_cols
                else np.zeros((len(theta_base), 0)))
        return cols, accepted_names, decisions

    def _update_feature_pool(self, Theta_full, targets, extra_names):
        """Record refit-based leave-one-feature-out calibration evidence."""
        try:
            import numpy as _np
            n_extra = len(extra_names)
            if n_extra == 0:
                return
            n_poly = Theta_full.shape[1] - n_extra
            fit_idx = self._fit_idx
            cal_idx = self._cal_idx
            theta_scaled = Theta_full / self._theta_scale
            pred_full = theta_scaled[cal_idx] @ self._sindy_coefs.T
            mse_with = float(_np.mean((pred_full - targets[cal_idx]) ** 2))
            sample_weights = self._compute_policy_weights(
                getattr(self, "_last_fit_sa", _np.zeros((len(Theta_full), self.input_dim))))

            for j_idx, name in enumerate(extra_names):
                col = n_poly + j_idx
                mask = _np.ones(Theta_full.shape[1], dtype=bool)
                mask[col] = False
                theta_drop = Theta_full[:, mask]
                drop_scale = self._column_scale(theta_drop[fit_idx])
                beta_drop, _, _ = self._sindy_fit_one_round(
                    theta_drop[fit_idx] / drop_scale, targets[fit_idx],
                    sample_weights=(sample_weights[fit_idx]
                                    if sample_weights is not None else None))
                pred_drop = (theta_drop[cal_idx] / drop_scale) @ beta_drop.T
                mse_without = float(_np.mean((pred_drop - targets[cal_idx]) ** 2))
                improvement = mse_without - mse_with
                accepted = improvement > 0.0
                gain = max(0.0, min(1.0, improvement / max(mse_with, 1e-6)))
                env = self._pool_env_tag
                expr_for_pool = self._llm_feat_exprs.get(name, {}).get("expr") \
                                if name in self._llm_feat_exprs else None
                if expr_for_pool:
                    self._feature_pool.record(
                        name=name, expr=expr_for_pool, env=env,
                        reward_gain=gain, was_accepted=accepted,
                    )
                # Persist a Hypothesis record so the evidence chain survives
                # the run. Positive improvement means the feature helped.
                if self._hyp_log is not None:
                    src = self._feat_provenance.get(name, "unknown")
                    h = Hypothesis(
                        claim=f"feature {name} reduces residual val MSE on {env}",
                        source={"orth": "orth_expander", "llm": "llm_role2",
                                "pool": "feature_pool"}.get(src, src),
                        kind="feature",
                        expr=expr_for_pool,
                        env=env,
                        round=getattr(self, "_current_round", 0),
                        expected_metric="calibration_mse_improvement",
                        expected_direction="increase",
                        expected_min_delta=0.0,
                    )
                    h.record_outcome(
                        evidence={
                            "calibration_mse_with": mse_with,
                            "calibration_mse_without": mse_without,
                            "calibration_mse_improvement": improvement,
                            "reward_gain_proxy": gain,
                        },
                        decision="accepted" if accepted else "rejected",
                        failure_reason=("no_calibration_gain" if not accepted else None),
                    )
                    self._hyp_log.append(h)
            # Save once after the round to avoid IO churn.
            self._feature_pool.save()
            self._log(f"  RAHD-A: calibration evidence computed for {n_extra} entries")
        except Exception as e:
            self._log(f"  RAHD-A: pool update failed ({e}); ignoring")

    def record_feature_reward_gain(self, name: str, env: str,
                                   reward_gain: float) -> None:
        """RAHD-D hook: external caller (e.g., step2 mini-rollout) can
        override the validation-MSE proxy with a real reward delta."""
        if self._feature_pool is None:
            return
        # Re-record with the same expr but the better gain estimate.  The
        # pool's running-mean update will weight in proportion to call count.
        expr = self._llm_feat_exprs.get(name, {}).get("expr", f"<orth:{name}>")
        self._feature_pool.record(name=name, expr=expr, env=env,
                                  reward_gain=reward_gain, was_accepted=True)

    def _compute_policy_weights(self, SA):
        """RAHD-C: per-sample weights for SINDy fit.

        Returns None when no PolicyDensity was configured (legacy path).
        Otherwise returns (N,) numpy array in [w_min, 1].
        """
        if self._policy_density is None:
            return None
        try:
            return self._policy_density.weights(SA)
        except Exception as e:
            self._log(f"  RAHD-C: policy_density.weights failed ({e}); falling back to uniform")
            return None

    @staticmethod
    def _weighted_torch_mse(prediction: torch.Tensor, target: torch.Tensor,
                            weights: torch.Tensor | None) -> torch.Tensor:
        """Mean squared error with one non-negative weight per transition."""
        per_sample = torch.mean((prediction - target) ** 2, dim=-1)
        if weights is None:
            return per_sample.mean()
        weights = weights.reshape(-1).clamp_min(1e-8)
        return torch.sum(per_sample * weights) / torch.sum(weights)

    def _fit_dense_state(self, SA, targets, fit_idx, val_idx, cal_idx,
                         n_epochs, batch_size, patience, weight_t):
        """Fit a dense raw-(s,a) state residual without symbolic features."""
        if self._dense_state_net is None:
            self._dense_state_net = nn.Sequential(
                nn.Linear(self.input_dim, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, self.obs_dim),
            ).to(self.device)
            self._dense_state_optimizer = optim.Adam(
                self._dense_state_net.parameters(), lr=self._nau_lr,
                weight_decay=1e-4)
        if self._dense_input_mean is None:
            self._dense_input_mean = SA[fit_idx].mean(axis=0, keepdims=True)
            self._dense_input_std = SA[fit_idx].std(axis=0, keepdims=True)
            self._dense_input_std = np.where(
                self._dense_input_std > 1e-6, self._dense_input_std, 1.0)
            self._dense_target_mean = targets[fit_idx].mean(
                axis=0, keepdims=True)
            self._dense_target_std = targets[fit_idx].std(
                axis=0, keepdims=True)
            self._dense_target_std = np.where(
                self._dense_target_std > 1e-6, self._dense_target_std, 1.0)

        inputs = (SA - self._dense_input_mean) / self._dense_input_std
        outputs = (targets - self._dense_target_mean) / self._dense_target_std
        inputs_t = torch.as_tensor(
            inputs, dtype=torch.float32, device=self.device)
        outputs_t = torch.as_tensor(
            outputs, dtype=torch.float32, device=self.device)
        best_val = float("inf")
        best_epoch = 0
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in self._dense_state_net.state_dict().items()
        }
        for epoch in range(n_epochs):
            self._dense_state_net.train()
            permutation = self._rng.permutation(fit_idx)
            for start in range(0, len(permutation), batch_size):
                batch = permutation[start:start + batch_size]
                prediction = self._dense_state_net(inputs_t[batch])
                loss = self._weighted_torch_mse(
                    prediction, outputs_t[batch],
                    weight_t[batch] if weight_t is not None else None)
                self._dense_state_optimizer.zero_grad()
                loss.backward()
                self._dense_state_optimizer.step()
            self._dense_state_net.eval()
            with torch.no_grad():
                prediction = self._dense_state_net(inputs_t[val_idx])
                val_loss = float(self._weighted_torch_mse(
                    prediction, outputs_t[val_idx],
                    weight_t[val_idx] if weight_t is not None else None))
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self._dense_state_net.state_dict().items()
                }
            if epoch - best_epoch >= patience:
                break
        self._dense_state_net.load_state_dict(best_state)
        self._dense_state_net.eval()

        with torch.no_grad():
            prediction = self._dense_state_net(inputs_t[cal_idx]).cpu().numpy()
        prediction = (
            prediction * self._dense_target_std + self._dense_target_mean)
        errors = prediction - targets[cal_idx]
        dense_calibration_mse = float(np.mean(errors ** 2))
        self._fit_errors = np.mean(errors ** 2, axis=0)
        self._calibration_error_q95 = float(np.quantile(
            np.linalg.norm(errors, axis=1), 0.95))
        self._sindy_val_mse = None
        self._nau_val_mse = dense_calibration_mse
        self._additive_cal_mse = dense_calibration_mse
        self._n_features = self.input_dim
        self._feature_names = []
        self._active_features = []
        self._sa_scale = np.std(SA[fit_idx], axis=0)
        self._sa_scale = np.where(self._sa_scale > 1e-6, self._sa_scale, 1.0)
        support_n = min(2048, len(fit_idx))
        support_idx = self._rng.choice(fit_idx, support_n, replace=False)
        self._support_reference = (
            (SA[support_idx] - self._train_center) / self._sa_scale)
        self._trained = True
        self._loop_done = True
        self._log(
            "  Dense neural state residual: "
            f"selection MSE={best_val:.5f}, "
            f"physical calibration MSE={dense_calibration_mse:.5f}")

    def _sindy_fit_one_round(self, Theta, targets, sample_weights=None):
        """Fit SINDy coefficients (Ridge + STLSQ) on feature matrix Theta.

        Args:
            Theta: (N, F) feature matrix.
            targets: (N, D) per-dim residual targets.
            sample_weights: optional (N,) per-sample weights for RAHD-C.
                Implemented via row-scaling of (Theta, y) so sklearn's
                Ridge sees a weighted least-squares problem; this is
                equivalent to passing ``sample_weight`` to Ridge.fit but
                keeps the existing STLSQ-mask code path unchanged.
        """
        n_outputs = targets.shape[1]
        coefs = np.zeros((n_outputs, Theta.shape[1]))
        active = []
        errors = np.zeros(n_outputs)

        # Pre-scale rows by sqrt(w) — equivalent to weighted LS in Ridge.
        if sample_weights is not None:
            w = np.clip(np.asarray(sample_weights, dtype=np.float64), 1e-8, None)
            sw = np.sqrt(w)
            Theta_w = Theta * sw[:, None]
        else:
            Theta_w = Theta

        for dim in range(n_outputs):
            y = targets[:, dim]
            y_w = y * sw if sample_weights is not None else y
            reg = Ridge(alpha=self.sindy_alpha, fit_intercept=False)
            reg.fit(Theta_w, y_w)
            coef = reg.coef_.copy()
            mask = np.abs(coef) > self.sindy_threshold
            for _ in range(10):
                if not np.any(mask): break
                reg2 = Ridge(alpha=self.sindy_alpha, fit_intercept=False)
                reg2.fit(Theta_w[:, mask], y_w)
                coef_new = np.zeros_like(coef)
                coef_new[mask] = reg2.coef_
                coef = coef_new
                mask = np.abs(coef) > self.sindy_threshold
            coefs[dim] = coef
            active.append(mask)
            # Report the *unweighted* residual MSE for backward compatibility
            # with downstream calibration diagnostics.
            y_pred = Theta @ coef
            errors[dim] = float(np.mean((y - y_pred) ** 2))

        return coefs, active, errors

    def fit(self, states, actions, sim_next_states, sim_rewards,
            real_next_states, real_rewards, n_epochs=100, batch_size=256, patience=20,
            real_dones=None):
        """
        First call discovers sparse features and fits an additive remainder.
        Subsequent calls reuse the support and warm-start both additive terms.

        `real_dones`: optional (N,) float array of done flags from real env.
        When provided, trains the termination prediction head P(done|s,a,s').
        """
        sim_deltas = sim_next_states - states
        real_deltas = real_next_states - states
        delta_correction_s = real_deltas - sim_deltas
        delta_correction_r = (real_rewards - sim_rewards).reshape(-1, 1)
        # The sparse additive model fits state correction (obs_dim outputs).
        # Reward correction uses separate MLP
        targets = delta_correction_s  # (N, obs_dim) — state only

        SA = np.concatenate([states, actions], axis=-1).astype(np.float32)
        self._last_fit_sa = SA
        N = len(SA)
        n_outputs = self.obs_dim  # state only, reward is separate
        fit_idx, val_idx, cal_idx = self._split_indices(N)
        self._fit_idx, self._val_idx, self._cal_idx = fit_idx, val_idx, cal_idx
        self._train_center = SA[fit_idx].mean(axis=0)
        sample_weights = self._compute_policy_weights(SA)
        weight_t = (torch.as_tensor(sample_weights, dtype=torch.float32,
                                    device=self.device)
                    if sample_weights is not None else None)
        self._last_policy_weight_summary = (
            None if sample_weights is None else {
                "min": float(np.min(sample_weights)),
                "mean": float(np.mean(sample_weights)),
                "max": float(np.max(sample_weights)),
            }
        )

        # ── Train reward correction MLP separately
        if self._reward_input_mean is None:
            self._reward_input_mean = SA[fit_idx].mean(axis=0, keepdims=True)
            self._reward_input_std = SA[fit_idx].std(axis=0, keepdims=True)
            self._reward_input_std = np.where(
                self._reward_input_std > 1e-6, self._reward_input_std, 1.0)
            self._reward_target_mean = delta_correction_r[fit_idx].mean(
                axis=0, keepdims=True)
            self._reward_target_std = delta_correction_r[fit_idx].std(
                axis=0, keepdims=True)
            self._reward_target_std = np.where(
                self._reward_target_std > 1e-6, self._reward_target_std, 1.0)
        SA_reward = (SA - self._reward_input_mean) / self._reward_input_std
        dr_reward = ((delta_correction_r - self._reward_target_mean)
                     / self._reward_target_std)
        SA_t = torch.as_tensor(SA_reward, dtype=torch.float32, device=self.device)
        dr_t = torch.as_tensor(dr_reward, dtype=torch.float32, device=self.device)
        train_r, val_r = fit_idx, val_idx
        best_r_val = float('inf'); best_r_state = {k: v.detach().cpu().clone() for k, v in self._reward_net.state_dict().items()}
        best_r_ep = 0
        for ep in range(100):
            self._reward_net.train()
            perm_r = self._rng.permutation(train_r)
            for i in range(0, len(perm_r), 256):
                bi = perm_r[i:i + 256]
                if len(bi) == 0: break
                pred_r = self._reward_net(SA_t[bi])
                loss_r = nn.MSELoss()(pred_r, dr_t[bi])
                self._reward_opt.zero_grad(); loss_r.backward(); self._reward_opt.step()
            self._reward_net.eval()
            with torch.no_grad():
                vl = float(nn.MSELoss()(self._reward_net(SA_t[val_r]), dr_t[val_r]))
                if vl < best_r_val: best_r_val = vl; best_r_ep = ep; best_r_state = {k: v.detach().cpu().clone() for k, v in self._reward_net.state_dict().items()}
            if ep - best_r_ep >= 20: break
        self._reward_net.load_state_dict(best_r_state)
        with torch.no_grad():
            reward_cal_pred = self._reward_net(SA_t[cal_idx]).cpu().numpy()
        reward_cal_pred = (reward_cal_pred * self._reward_target_std
                           + self._reward_target_mean)
        reward_cal_mse = float(np.mean(
            (reward_cal_pred - delta_correction_r[cal_idx]) ** 2))
        self._reward_cal_mse = reward_cal_mse
        self._log(f"  Reward MLP: standardized selection MSE={best_r_val:.5f}, "
                  f"physical calibration MSE={reward_cal_mse:.5f}")

        # ── Train termination (done) prediction head: P(done | s, a, s')
        if real_dones is not None:
            self.fit_done_head(
                states, actions, real_next_states, real_dones,
                train_indices=fit_idx, validation_indices=val_idx)
        else:
            self._done_trained = getattr(self, "_done_trained", False)

        if self.residual_mode == "dense_neural":
            self._fit_dense_state(
                SA, targets, fit_idx, val_idx, cal_idx,
                n_epochs=n_epochs, batch_size=batch_size, patience=patience,
                weight_t=weight_t)
            return

        # ── REFIT path: update coefficients, then fit only their remainder.
        # The sparsity pattern is fixed after discovery so online updates cannot
        # silently change the reported symbolic support.
        if self._loop_done:
            Theta_full = self._build_full_theta(SA)
            Theta_t = torch.FloatTensor(Theta_full).to(self.device)
            tgt_t = torch.FloatTensor(targets).to(self.device)

            if self.residual_mode != "neural_only":
                for epoch in range(min(n_epochs, 30)):  # fewer epochs for refit
                    self._coef_layer.train()
                    perm_t = self._rng.permutation(fit_idx)
                    for i in range(0, len(perm_t), batch_size):
                        bi = perm_t[i:i+batch_size]
                        pred = self._coef_layer(Theta_t[bi])
                        loss = self._weighted_torch_mse(
                            pred, tgt_t[bi],
                            weight_t[bi] if weight_t is not None else None)
                        self._coef_optimizer.zero_grad()
                        loss.backward()
                        with torch.no_grad():
                            self._coef_layer.weight.grad *= self._active_mask_t
                        self._coef_optimizer.step()
                    self._coef_layer.eval()

            # Update SINDy coefs from the trained layer (for logging/diagnostics)
            with torch.no_grad():
                if self.residual_mode != "neural_only":
                    self._sindy_coefs = self._coef_layer.weight.cpu().numpy()
                    self._discovered_sindy_coefs = self._sindy_coefs.copy()
                base_cal = self._coef_layer(Theta_t[cal_idx])

            # Warm-start the neural branch on the unexplained remainder only.
            best_val = float('inf'); best_epoch = 0
            best_state = {k: v.detach().cpu().clone() for k, v in self._nau_head.state_dict().items()}
            head_epochs = 0 if self.residual_mode == "sindy_only" else min(n_epochs, 30)
            for epoch in range(head_epochs):
                self._nau_head.train()
                perm_t = self._rng.permutation(fit_idx)
                for i in range(0, len(perm_t), batch_size):
                    bi = perm_t[i:i+batch_size]
                    with torch.no_grad():
                        remainder_target = tgt_t[bi] - self._coef_layer(Theta_t[bi])
                    pred_remainder = self._nau_head(Theta_t[bi])
                    loss = (self._weighted_torch_mse(
                                pred_remainder, remainder_target,
                                weight_t[bi] if weight_t is not None else None)
                            + 1e-4 * self._nau_head.regularization_loss())
                    self._nau_optimizer.zero_grad(); loss.backward(); self._nau_optimizer.step()
                self._nau_head.eval()
                with torch.no_grad():
                    pred = (self._coef_layer(Theta_t[val_idx])
                            + self._nau_head(Theta_t[val_idx]))
                    vl = float(self._weighted_torch_mse(
                        pred, tgt_t[val_idx],
                        weight_t[val_idx] if weight_t is not None else None))
                    if vl < best_val: best_val = vl; best_epoch = epoch; best_state = {k: v.detach().cpu().clone() for k, v in self._nau_head.state_dict().items()}
                if epoch - best_epoch >= 10: break
            self._nau_head.load_state_dict(best_state)
            with torch.no_grad():
                pred_cal = base_cal + self._nau_head(Theta_t[cal_idx])
                cal_errors = (pred_cal.cpu().numpy() - targets[cal_idx]) ** 2
            self._fit_errors = np.mean(cal_errors, axis=0)
            self._calibration_error_q95 = float(np.quantile(
                np.linalg.norm(pred_cal.cpu().numpy() - targets[cal_idx], axis=1), 0.95))
            self._sindy_val_mse = float(nn.MSELoss()(base_cal, tgt_t[cal_idx]))
            self._nau_val_mse = float(np.mean(cal_errors))
            self._additive_cal_mse = self._nau_val_mse
            self._trained = True
            return

        # ── FIRST FIT
        self._library.fit(SA[fit_idx])
        Theta = np.asarray(self._library.transform(SA))
        all_base_names = self._feature_names_from_library()
        self._base_feature_indices = self._select_base_feature_indices(all_base_names)
        Theta = Theta[:, self._base_feature_indices]
        self._feature_names = [all_base_names[index]
                               for index in self._base_feature_indices]
        if len(self._base_feature_indices) < len(all_base_names):
            self._log(f"  Feature budget: retained {len(self._base_feature_indices)}/"
                      f"{len(all_base_names)} target-independent base terms")

        extra_cols = None
        extra_names = []
        self._hypothesis_logs = []
        # If LLM oracle is available: use physics-informed features instead of loop
        if self._env_type is not None:
            oracle = LLMOracle(env_type=self._env_type, log_fn=self._log)
            oracle_cols, oracle_names, _ = oracle.suggest_features(
                SA, obs_dim=self.obs_dim, act_dim=self.act_dim)
            if oracle_cols.shape[1] > 0:
                gated_cols, gated_names, _ = self._gate_candidate_columns(
                    Theta, oracle_cols, oracle_names, targets,
                    fit_idx, val_idx, sample_weights=sample_weights)
                if gated_cols.shape[1] == 0:
                    self._log("  [Physics oracle] no candidate improved held-out MSE")
                else:
                    extra_cols = gated_cols
                    extra_names = gated_names
                    self._oracle_feature_names = list(gated_names)
                Theta_full = (np.hstack([Theta, extra_cols])
                              if extra_cols is not None else Theta)
                round_scale = self._column_scale(Theta_full[fit_idx])
                coefs, active, errors = self._sindy_fit_one_round(
                    Theta_full[fit_idx] / round_scale, targets[fit_idx],
                    sample_weights=(sample_weights[fit_idx]
                                    if sample_weights is not None else None))
                self._log(f"  [LLM Oracle] {Theta_full.shape[1]} features "
                          f"({Theta.shape[1]} base + {len(extra_names)} accepted physics), "
                          f"eps_max={errors.max():.5f}")
                # Skip diagnosis (too slow on 50k data, and LLM oracle doesn't need it)
                self._hypothesis_logs.append({
                    "round": 1, "method": "llm_oracle",
                    "n_features": Theta_full.shape[1], "eps_max": float(errors.max()),
                })
        # Run hypothesis loop only if LLM oracle was NOT used
        if extra_cols is None:
          for round_num in range(1, self.max_rounds + 1):
            # Build full feature matrix
            if extra_cols is not None and extra_cols.shape[1] > 0:
                Theta_full = np.hstack([Theta, extra_cols])
            else:
                Theta_full = Theta

            # Fit only on the fit partition. Validation chooses features;
            # calibration data remains untouched until final reporting.
            round_scale = self._column_scale(Theta_full[fit_idx])
            coefs, active, errors = self._sindy_fit_one_round(
                Theta_full[fit_idx] / round_scale, targets[fit_idx],
                sample_weights=(sample_weights[fit_idx]
                                if sample_weights is not None else None))
            eps_max = float(errors.max())

            # Quality gate
            quality_passed = eps_max < self.eps_threshold

            # Diagnose remainder (subsample for speed — diagnosis is O(N²))
            sindy_pred_s = (Theta_full / round_scale) @ coefs[:self.obs_dim].T
            remainder_s = delta_correction_s - sindy_pred_s
            diag_n = min(2000, len(fit_idx))
            diag_idx = self._rng.choice(fit_idx, diag_n, replace=False)
            diagnoses = self._battery.run(remainder_s[diag_idx], SA[diag_idx])
            any_structure = any(d.any_fired() for d in diagnoses)

            # Count active
            n_active = sum(m.sum() for m in active)
            n_fired = sum(1 for d in diagnoses if d.any_fired())
            diag_summary = [d.summary() for d in diagnoses if d.any_fired()]

            log_entry = {
                "round": round_num, "eps_max": eps_max, "n_features": Theta_full.shape[1],
                "n_active": int(n_active), "quality_passed": quality_passed,
                "n_diagnosis_fired": n_fired, "reason": "",
            }

            self._log(f"  [Round {round_num}] features={Theta_full.shape[1]} active={n_active} "
                  f"eps_max={eps_max:.5f} diagnosed={n_fired}/{self.obs_dim}")

            if quality_passed:
                log_entry["reason"] = "quality_gate"
                self._hypothesis_logs.append(log_entry)
                self._log(f"    ✓ Quality gate PASSED (eps_max={eps_max:.5f} < {self.eps_threshold})")
                break

            if not any_structure:
                log_entry["reason"] = "white_noise"
                self._hypothesis_logs.append(log_entry)
                self._log(f"    ✓ Remainder is white noise (no structure detected)")
                break

            if round_num == self.max_rounds:
                log_entry["reason"] = "max_rounds"
                self._hypothesis_logs.append(log_entry)
                self._log(f"    ⊘ Max rounds reached")
                break

            # Orthogonal expansion: find features in orthogonal complement of Θ.
            # Skipped when the user requested a fixed-library baseline (E3).
            if self._disable_orthogonal_expansion:
                self._log(f"    → Orthogonal expansion DISABLED (feature_library={self._feature_library_kind})")
                new_extra = np.zeros((SA.shape[0], 0))
                new_names = []
                orth_diag = {"n_candidates": 0, "n_accepted": 0}
            else:
                self._log(f"    → Orthogonal feature discovery ({n_fired} dims with structure)...")
                for ds in diag_summary[:3]:
                    self._log(f"      {ds}")
                orth_expander = OrthogonalExpander(
                    self.obs_dim, self.act_dim,
                    max_delta_beta_inf=self._max_delta_beta_inf,
                )
                _, new_names, orth_diag = orth_expander.expand(
                    SA[fit_idx], Theta_full[fit_idx], remainder_s[fit_idx],
                    log_fn=self._log)
                all_candidates, all_candidate_names = \
                    orth_expander._generate_candidates(SA, len(SA))
                candidate_map = dict(zip(all_candidate_names, all_candidates))
                new_extra = (np.column_stack([candidate_map[n] for n in new_names])
                             if new_names else np.zeros((len(SA), 0)))
            # Track provenance for end-of-fit feature_pool update.
            for n in new_names:
                self._feat_provenance.setdefault(n, "orth")

            # RAHD-A: seed candidates from cross-env feature pool.  These
            # features were accepted in prior runs (any env) — re-evaluate
            # them on the current SA and orthogonalise against Θ; if they
            # still pass the same correlation gate they get added.
            if self._feature_pool is not None:
                from mc_wm.residual.feature_pool import evaluate_expression
                pool_cands = self._feature_pool.query_candidates(
                    env=self._pool_env_tag, top_k=self._feature_pool_top_k)
                if pool_cands:
                    self._log(f"    RAHD-A: trying {len(pool_cands)} pool candidates")
                pool_added = 0
                for pname, pexpr, _prio in pool_cands:
                    if pname in extra_names or pname in new_names:
                        continue  # already in basis
                    col = evaluate_expression(pexpr, SA[:, :self.obs_dim],
                                              SA[:, self.obs_dim:])
                    if col is None:
                        continue
                    # Quick orthogonality + correlation check (mirrors
                    # OrthogonalExpander internals).
                    theta_fit = Theta_full[fit_idx]
                    col_fit = col[fit_idx]
                    Q, _ = np.linalg.qr(theta_fit, mode="reduced")
                    proj = Q @ (Q.T @ col_fit)
                    c_orth = col_fit - proj
                    if np.linalg.norm(c_orth) / (np.linalg.norm(col_fit) + 1e-12) < 0.1:
                        continue  # mostly redundant
                    corrs = []
                    for d in range(remainder_s.shape[1]):
                        cc = np.corrcoef(c_orth, remainder_s[fit_idx, d])[0, 1]
                        if not np.isnan(cc):
                            corrs.append(abs(float(cc)))
                    if not corrs or max(corrs) <= 0.05:
                        continue
                    # Pool features bypass auto-name conflict by prefixing.
                    pname2 = f"pool_{pname}"
                    if new_extra.shape[1] > 0:
                        new_extra = np.hstack([new_extra, col[:, None]])
                    else:
                        new_extra = col[:, None]
                    new_names.append(pname2)
                    self._feat_provenance[pname2] = "pool"
                    pool_added += 1
                if pool_added > 0:
                    self._log(f"    RAHD-A: {pool_added} pool features added")

            # Role #2: ask Claude for physics-informed features (optional).
            # Augments orthogonal expansion; LLM's suggestions are evaluated
            # as safe numpy expressions over s[·], a[·] with access to np.
            if self._claude_oracle is not None:
                # Collect previously-accepted LLM features (across earlier rounds).
                prev_accepted_llm = [n for n in extra_names
                                     if isinstance(n, str) and n.startswith("llm_")]
                # Per-dim residual std for LLM prioritisation.
                try:
                    resid_std = list(np.std(remainder_s, axis=0))
                except Exception:
                    resid_std = None
                # Current val MSE estimate (approximate — use in-sample mse as proxy).
                cur_val_mse = float(np.mean(remainder_s ** 2))
                nau_L = float(self._nau_head.L_eff) if self._nau_head is not None else None
                _nau_L_str = f"{nau_L:.3f}" if nau_L is not None else "N/A"
                self._log(f"    [LLM Role #2] ctx: round={round_num}, "
                          f"prev_accepted={len(prev_accepted_llm)}, "
                          f"val_mse={cur_val_mse:.5f}, "
                          f"L_eff={_nau_L_str}, "
                          f"history={len(self._llm_feat_history)} entries")
                llm_feats = self._claude_oracle.role2_feature_hypotheses(
                    env_description=self._env_desc_llm,
                    current_basis=extra_names if extra_names else [],
                    diagnosis_summary="; ".join(diag_summary[:5]),
                    obs_dim=self.obs_dim, act_dim=self.act_dim,
                    round_num=round_num,
                    prev_accepted=prev_accepted_llm,
                    current_val_mse=cur_val_mse,
                    nau_L_eff=nau_L,
                    residual_per_dim_std=resid_std,
                    feature_history=self._llm_feat_history[-10:],
                )
                if llm_feats:
                    self._log(f"    + LLM Role#2: {len(llm_feats)} feature suggestions")
                    llm_cols, llm_names = self._eval_llm_features(SA, llm_feats)
                    if llm_cols is not None and llm_cols.shape[1] > 0:
                        new_extra = (np.hstack([new_extra, llm_cols])
                                     if new_extra.shape[1] > 0 else llm_cols)
                        new_names = new_names + llm_names
                        # Tag provenance so HypothesisLog records the right source.
                        for n in llm_names:
                            self._feat_provenance[n] = "llm"

            if len(new_names) > 0:
                proposed_names = list(new_names)
                new_extra, new_names, gate_decisions = self._gate_candidate_columns(
                    Theta_full, new_extra, new_names, targets,
                    fit_idx, val_idx, sample_weights=sample_weights)
                if self.feature_budget is not None:
                    remaining_budget = max(0, self.feature_budget - Theta_full.shape[1])
                    if len(new_names) > remaining_budget:
                        self._log(f"    Feature budget: retaining first "
                                  f"{remaining_budget}/{len(new_names)} accepted candidates")
                        new_names = new_names[:remaining_budget]
                        new_extra = new_extra[:, :remaining_budget]
                accepted_set = set(new_names)
                for decision in gate_decisions:
                    name = decision["name"]
                    if isinstance(name, str) and name.startswith("llm_"):
                        if name in accepted_set:
                            self._llm_feat_accepted += 1
                        else:
                            self._llm_feat_rejected_validation += 1
                        for history in reversed(self._llm_feat_history):
                            if history["name"] == name:
                                history["outcome"] = {
                                    "syntax_valid": True,
                                    "heldout_accepted": name in accepted_set,
                                    "val_mse_gain": decision.get("val_mse_gain"),
                                }
                                break
                rejected_count = len(proposed_names) - len(new_names)
                self._log(f"    Held-out gate: accepted {len(new_names)}/"
                          f"{len(proposed_names)} candidates; rejected {rejected_count}")

            if len(new_names) > 0:
                if extra_cols is not None:
                    extra_cols = np.hstack([extra_cols, new_extra])
                else:
                    extra_cols = new_extra
                extra_names.extend(new_names)
            else:
                log_entry["reason"] = "no_orthogonal_features"
                self._hypothesis_logs.append(log_entry)
                self._log(f"    No orthogonal features found, stopping")
                break

            log_entry["reason"] = "expanded"
            self._hypothesis_logs.append(log_entry)

        # Final sparse fit uses only the fit partition. Feature selection has
        # seen ``val_idx``; ``cal_idx`` remains untouched for honest reporting.
        self._theta_scale = self._column_scale(Theta_full[fit_idx])
        theta_scaled = Theta_full / self._theta_scale
        coefs, active, train_errors = self._sindy_fit_one_round(
            theta_scaled[fit_idx], targets[fit_idx],
            sample_weights=(sample_weights[fit_idx]
                            if sample_weights is not None else None))
        cal_pred_sindy = theta_scaled[cal_idx] @ coefs.T
        errors = np.mean((cal_pred_sindy - targets[cal_idx]) ** 2, axis=0)

        # Save final SINDy state
        self._sindy_coefs = coefs
        self._discovered_sindy_coefs = coefs.copy()
        self._active_features = active
        self._fit_errors = errors
        self._n_features = Theta_full.shape[1]
        if self.feature_budget is not None and self._n_features > self.feature_budget:
            raise RuntimeError(
                f"deployed feature count {self._n_features} exceeds budget "
                f"{self.feature_budget}")
        # Update feature names to include extras
        all_names = list(self._feature_names) + extra_names
        self._feature_names = all_names
        # Save expand specs for predict-time reconstruction.
        # CRITICAL: must preserve fit-time ORDER across orthogonal / pool /
        # LLM features so the column count at predict time matches the remainder
        # head's expected input.  Pool features (RAHD-A) carry their expr
        # in feature_pool, so we look it up by name when stashing.
        self._expand_specs = []
        if self._env_type is not None and getattr(self, "_oracle_feature_names", []):
            self._expand_specs.append(
                {"type": "llm_oracle", "env_type": self._env_type,
                 "names": list(getattr(self, "_oracle_feature_names", []))})
        else:
            # Walk extra_names IN ORDER and assign each to one of the three
            # reconstruction modes.  When two consecutive features share a
            # mode they are batched into the same spec to minimise spec
            # count; otherwise each switch creates a new spec entry.  This
            # keeps the column order at predict time identical to fit time.
            current_mode = None
            current_items: list = []
            def _flush():
                nonlocal current_mode, current_items
                if current_mode is None or not current_items:
                    current_items = []; current_mode = None; return
                if current_mode == "orthogonal":
                    self._expand_specs.append(
                        {"type": "orthogonal", "names": list(current_items)})
                elif current_mode == "pool":
                    self._expand_specs.append({
                        "type": "pool_features",
                        "items": list(current_items),
                    })
                elif current_mode == "llm_features":
                    self._expand_specs.append({
                        "type": "llm_features",
                        "items": list(current_items),
                    })
                current_items = []; current_mode = None

            for n in extra_names:
                if isinstance(n, str) and n.startswith("llm_"):
                    mode = "llm_features"
                    item = (n, self._llm_feat_exprs.get(n, {}).get("expr", ""),
                            self._llm_feat_exprs.get(n, {}).get("std", 1.0))
                elif isinstance(n, str) and n.startswith("pool_"):
                    mode = "pool_features"
                    # Look up expr from the pool by stripping the prefix.
                    pool_name = n[len("pool_"):]
                    pool_meta = self._feature_pool.all_features().get(pool_name, {}) \
                                if self._feature_pool is not None else {}
                    item = (n, pool_meta.get("expr", ""))
                else:
                    mode = "orthogonal"
                    item = n
                if mode != current_mode:
                    _flush()
                    current_mode = mode
                current_items.append(item)
            _flush()
            self._expand_obs_dim = self.obs_dim
            self._expand_act_dim = self.act_dim

        n_active_total = sum(m.sum() for m in active)
        self._log(f"  SINDy final: {self._n_features} features, {n_active_total} active, "
              f"calibration MSE={errors.mean():.5f}, rounds={len(self._hypothesis_logs)}")

        # RAHD-A write-back: record each non-poly2 feature's outcome to the
        # cross-env pool.  For each accepted feature we estimate a per-feature
        # validation-MSE drop via leave-one-out on the held-out tail.  This
        # is a cheap proxy for the full mini-rollout reward gain handled by
        # RAHD-D in step2_mbrl_residual.py; both are useful — the pool prefers
        # whichever is supplied last.
        if self._feature_pool is not None and len(extra_names) > 0:
            self._update_feature_pool(Theta_full, targets, extra_names)

        # ── Phase 2: sparse base + masked neural remainder.
        n_out = self._sindy_coefs.shape[0]
        n_feat = self._sindy_coefs.shape[1]
        self._coef_layer = nn.Linear(n_feat, n_out, bias=False).to(self.device)
        with torch.no_grad():
            if self.residual_mode == "neural_only":
                self._coef_layer.weight.zero_()
            else:
                self._coef_layer.weight.copy_(torch.FloatTensor(self._sindy_coefs))
        self._coef_optimizer = optim.Adam(self._coef_layer.parameters(), lr=1e-3, weight_decay=1e-4)
        # Build active mask: only features that STLSQ kept as non-zero
        mask = np.zeros_like(self._sindy_coefs)
        for dim, m in enumerate(self._active_features):
            mask[dim] = m.astype(float)
        self._active_mask_t = torch.FloatTensor(mask).to(self.device)
        feature_union = torch.FloatTensor((mask.sum(axis=0) > 0).astype(np.float32)).to(self.device)

        self._nau_head = SparseNeuralRemainderHead(
            self._n_features, n_outputs, hidden_dim=32).to(self.device)
        self._nau_head.set_feature_mask(feature_union)
        self._nau_optimizer = optim.Adam(
            self._nau_head.parameters(), lr=self._nau_lr, weight_decay=1e-4)

        Theta_t = torch.FloatTensor(theta_scaled).to(self.device)
        tgt_t = torch.FloatTensor(targets).to(self.device)
        with torch.no_grad():
            base_all = self._coef_layer(Theta_t)
            remainder_targets = tgt_t - base_all

        best_val = float('inf'); best_epoch = 0
        best_state = {k: v.detach().cpu().clone()
                      for k, v in self._nau_head.state_dict().items()}
        head_epochs = 0 if self.residual_mode == "sindy_only" else n_epochs
        for epoch in range(head_epochs):
            self._nau_head.train()
            perm_train = self._rng.permutation(fit_idx)
            for i in range(0, len(perm_train), batch_size):
                bi = perm_train[i:i + batch_size]
                pred_remainder = self._nau_head(Theta_t[bi])
                loss = (self._weighted_torch_mse(
                            pred_remainder, remainder_targets[bi],
                            weight_t[bi] if weight_t is not None else None)
                        + 1e-4 * self._nau_head.regularization_loss())
                self._nau_optimizer.zero_grad(); loss.backward(); self._nau_optimizer.step()
            self._nau_head.eval()
            with torch.no_grad():
                val_pred = base_all[val_idx] + self._nau_head(Theta_t[val_idx])
                val_loss = float(self._weighted_torch_mse(
                    val_pred, tgt_t[val_idx],
                    weight_t[val_idx] if weight_t is not None else None))
                if val_loss < best_val:
                    best_val = val_loss; best_epoch = epoch
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in self._nau_head.state_dict().items()}
            if (epoch + 1) % 20 == 0:
                self._log(f"    remainder epoch {epoch+1} | selection={val_loss:.5f} "
                          f"best={best_val:.5f}")
            if epoch - best_epoch >= patience:
                self._log(f"    remainder early stop at epoch {epoch+1} "
                          f"(best={best_epoch+1})")
                break

        self._nau_head.load_state_dict(best_state)
        self._trained = True
        with torch.no_grad():
            sindy_weight = torch.as_tensor(
                self._discovered_sindy_coefs, dtype=torch.float32, device=self.device)
            sindy_cal = torch.nn.functional.linear(Theta_t[cal_idx], sindy_weight)
            additive_cal = (base_all[cal_idx]
                            + self._nau_head(Theta_t[cal_idx]))
            sindy_mse = float(nn.MSELoss()(sindy_cal, tgt_t[cal_idx]))
            additive_mse = float(nn.MSELoss()(additive_cal, tgt_t[cal_idx]))
            cal_residual_norm = torch.linalg.vector_norm(
                additive_cal - tgt_t[cal_idx], dim=1).cpu().numpy()
        self._sindy_val_mse = sindy_mse
        self._nau_val_mse = additive_mse  # compatibility alias
        self._additive_cal_mse = additive_mse
        self._fit_errors = np.mean(
            (additive_cal.cpu().numpy() - targets[cal_idx]) ** 2, axis=0)
        self._calibration_error_q95 = float(np.quantile(cal_residual_norm, 0.95))
        self._sa_scale = np.std(SA[fit_idx], axis=0)
        self._sa_scale = np.where(self._sa_scale > 1e-6, self._sa_scale, 1.0)
        support_n = min(2048, len(fit_idx))
        support_idx = self._rng.choice(fit_idx, support_n, replace=False)
        self._support_reference = (SA[support_idx] - self._train_center) / self._sa_scale

        n_active_total = int(mask.sum())
        self._log(f"  SGD coef layer initialized: {n_active_total} active coefficients (locked sparsity)")
        self._log(f"  Deployed residual mode:    {self.residual_mode}")
        self._log(f"  SINDy calibration MSE:   {sindy_mse:.5f}")
        self._log(f"  Additive calibration MSE:{additive_mse:.5f}")
        self._log(f"  Calibration error q95:   {self._calibration_error_q95:.5f}")
        self._loop_done = True

    def _build_full_theta(self, SA):
        """Build full feature matrix including expanded columns."""
        Theta = np.asarray(self._library.transform(SA))
        if self._base_feature_indices is not None:
            Theta = Theta[:, self._base_feature_indices]
        if hasattr(self, '_expand_specs') and self._expand_specs:
            for spec in self._expand_specs:
                if spec['type'] == 'llm_oracle':
                    oracle = LLMOracle(env_type=spec['env_type'])
                    cols, names, _ = oracle.suggest_features(
                        SA, obs_dim=self.obs_dim, act_dim=self.act_dim)
                    name_to_col = {n: cols[:, i] for i, n in enumerate(names)}
                    selected = [name_to_col[n] for n in spec.get('names', names)
                                if n in name_to_col]
                    if selected:
                        Theta = np.hstack([Theta, np.column_stack(selected)])
                elif spec['type'] == 'orthogonal':
                    oe = OrthogonalExpander(self.obs_dim, self.act_dim)
                    all_cands, all_names = oe._generate_candidates(SA, len(SA))
                    name_to_col = {n: c for c, n in zip(all_cands, all_names)}
                    extra = []
                    for n in spec['names']:  # preserve original order
                        if n in name_to_col:
                            extra.append(name_to_col[n])
                    if extra:
                        Theta = np.hstack([Theta, np.column_stack(extra)])
                elif spec['type'] == 'pool_features':
                    # Re-evaluate pool-feature expressions on new SA.  Pool
                    # exprs use the same sandboxed namespace as LLM, with
                    # ``s`` and ``a`` referring to (N, dim) arrays (NOT the
                    # transposed convention that LLM features use).
                    from mc_wm.residual.feature_pool import evaluate_expression
                    s_arr = SA[:, :self.obs_dim]
                    a_arr = SA[:, self.obs_dim:]
                    extra = []
                    for _name, expr in spec['items']:
                        col = evaluate_expression(expr, s_arr, a_arr)
                        if col is None:
                            extra.append(np.zeros(len(SA)))
                        else:
                            extra.append(col)
                    if extra:
                        Theta = np.hstack([Theta, np.column_stack(extra)])
                elif spec['type'] == 'llm_features':
                    # Re-evaluate LLM-proposed expressions on new SA points.
                    # Uses same sandboxed namespace as _eval_llm_features.
                    s_arr = SA[:, :self.obs_dim]
                    a_arr = SA[:, self.obs_dim:]
                    extra = []
                    for _name, expr, std in spec['items']:
                        col = evaluate_array_expression(
                            expr, s_arr, a_arr, transpose=True)
                        if col is None:
                            extra.append(np.zeros(len(SA)))
                        else:
                            extra.append(col / max(std, 1e-8))
                    if extra:
                        Theta = np.hstack([Theta, np.column_stack(extra)])
        if self._theta_scale is not None:
            if Theta.shape[1] != len(self._theta_scale):
                raise RuntimeError(
                    f"feature reconstruction produced {Theta.shape[1]} columns; "
                    f"fit used {len(self._theta_scale)}"
                )
            Theta = Theta / self._theta_scale
        return Theta

    def predict_correction(self, states, actions):
        """Return additive sparse-plus-neural state and reward corrections."""
        SA = np.concatenate([states, actions], axis=-1).astype(np.float32)

        if self.residual_mode == "dense_neural":
            dense_inputs = (
                (SA - self._dense_input_mean) / self._dense_input_std)
            dense_inputs_t = torch.as_tensor(
                dense_inputs, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                ds = self._dense_state_net(dense_inputs_t).cpu().numpy()
            ds = ds * self._dense_target_std + self._dense_target_mean
        else:
            # State correction: deployed sparse SINDy term plus masked remainder.
            Theta = self._build_full_theta(SA)
            Theta_t = torch.FloatTensor(Theta).to(self.device)
            with torch.no_grad():
                ds = (self._coef_layer(Theta_t)
                      + self._nau_head(Theta_t)).cpu().numpy()

        # Reward correction: separate MLP
        SA_reward = (SA - self._reward_input_mean) / self._reward_input_std
        SA_t = torch.as_tensor(SA_reward, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            dr = self._reward_net(SA_t).cpu().numpy().squeeze(-1)  # (N,)
        dr = dr * float(self._reward_target_std.squeeze()) \
             + float(self._reward_target_mean.squeeze())

        return ds, dr

    def predict_done(self, states, actions, next_states):
        """
        P(done | s, a, s') ∈ [0, 1]  — full-tuple termination prediction.

        Returns (N,) float array of done probabilities.  Caller typically
        thresholds at 0.5 or samples Bernoulli.
        """
        if not getattr(self, "_done_trained", False):
            return np.zeros(len(states))
        SAS = np.concatenate([states, actions, next_states],
                             axis=-1).astype(np.float32)
        SAS = (SAS - self._done_input_mean) / self._done_input_std
        SAS_t = torch.as_tensor(SAS, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self._done_net(SAS_t).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    def get_support_distance(self, states, actions):
        """Nearest-reference distance in standardized state-action space."""
        SA = np.concatenate([states, actions], axis=-1).astype(np.float32)
        if self._support_reference is None:
            return np.full(len(SA), np.inf)
        query = (SA - self._train_center) / self._sa_scale
        out = np.empty(len(query), dtype=np.float64)
        for start in range(0, len(query), 256):
            q = query[start:start + 256]
            sq = np.sum((q[:, None, :] - self._support_reference[None, :, :]) ** 2,
                        axis=2)
            out[start:start + len(q)] = np.sqrt(np.min(sq, axis=1))
        return out

    def get_empirical_error_score(self, states, actions):
        """Calibration error scaled by support distance; not a formal bound."""
        base = (self._calibration_error_q95
                if self._calibration_error_q95 is not None else np.inf)
        return base * (1.0 + self.get_support_distance(states, actions))

    def get_ood_bound(self, states, actions):
        """Compatibility alias for the empirical error score."""
        return self.get_empirical_error_score(states, actions)

    def get_active_terms(self):
        """Return discovered symbolic terms per dimension."""
        if self.residual_mode == "dense_neural":
            return {}
        result = {}
        n_dims = len(self._active_features) if self._active_features else 0
        for dim in range(n_dims):
            name = f"Δs_{dim}" if dim < self.obs_dim else "Δr"
            active = [(self._feature_names[j],
                       float(self._discovered_sindy_coefs[dim, j]
                             / self._theta_scale[j]))
                      for j in range(self._n_features)
                      if self._active_features[dim][j]]
            if active:
                result[name] = active
        return result


# Backward-compatible import name for old experiment scripts and checkpoints.
SINDyNAUAdapter = SparseAdditiveResidualAdapter
