"""Bus MC-WM adaptation using only target outcomes already released by the stream."""

from collections import deque
from dataclasses import asdict
from pathlib import Path
import os
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BUS = HERE.parent / "mcwm_bus"
H2O = Path(os.environ.get("THESIS_H2O_ROOT", "/home/erzhu419/mine_code/sumo-rl/H2Oplus"))
os.environ["THESIS_H2O_ROOT"] = str(H2O)
os.environ["THESIS_MCWM_ROOT"] = str(BUS / "runtime/MC-WM")
for directory in (BUS, H2O, BUS / "runtime/MC-WM"):
    sys.path.insert(0, str(directory))

from model import BusModel, Normalizer, losses, targets
from control_model import ModelAugmentation
from data import within_batch
from mc_wm.policy.qdelta_bellman import QDeltaBellman
from SimpleSAC.transit_v7.trainer import TransitH2OTrainerV7, TransitTrainerConfig


def load_table(path, limit=4096):
    """Read a bounded local pilot snapshot; no full dataset/checkpoint copy."""
    with np.load(path) as archive:
        table = {key: archive[key][:limit].copy() for key in ("s", "a", "ns", "r", "dt")}
        table["terminal"] = (archive["terminal"][:limit].copy() if "terminal" in archive
                             else np.zeros(len(table["s"]), np.float32))
    if len(table["s"]) < 2:
        raise ValueError("At least two history rows are needed")
    return table


def row_table(rows):
    return {key: np.asarray([getattr(row, key) for row in rows])
            for key in ("s", "a", "ns", "r", "dt", "terminal")}


def concatenate(tables):
    tables = [table for table in tables if len(table["s"])]
    return {key: np.concatenate([table[key] for table in tables]) for key in tables[0]}


def learned_rows(table):
    keep = (table["dt"].reshape(-1) > 0) & ~table["terminal"].reshape(-1).astype(bool)
    keep &= ~within_batch(table["s"])
    return {key: value[keep] for key, value in table.items()}


def tensor_batch(table, size, rng, device):
    ids = rng.integers(len(table["s"]), size=size)
    mapping = dict(observations="s", actions="a", next_observations="ns", rewards="r",
                   delta_t_seconds="dt", terminals="terminal")
    batch = {key: torch.as_tensor(table[value][ids], dtype=torch.float32, device=device)
             for key, value in mapping.items()}
    batch["actions"] = batch["actions"].reshape(-1, 1)
    for key in ("rewards", "delta_t_seconds", "terminals"):
        batch[key] = batch[key].reshape(-1)
    batch["discounts"] = torch.ones(size, device=device)
    return batch


class DomainReplay:
    """Keep historical target, newly available target, and source rows distinct."""

    def __init__(self, history, source, contract, stations, device="cpu", window=1024):
        self.history, self.source = history, source
        self.recent = deque(maxlen=window)
        self.contract_config, self.station_max_by_line = asdict(contract), stations
        self.device = device
        self.target_received = 0

    @property
    def online_size(self):
        return len(self.source["s"])

    def add_target(self, rows):
        self.recent.extend(rows)
        self.target_received += len(rows)

    def add_source(self, rows):
        if rows:
            joined = concatenate([self.source, row_table(rows)])
            self.source = {key: value[-4096:] for key, value in joined.items()}

    def sample(self, size, *, scope="offline", rng=None, **unused):
        if scope == "online":
            return tensor_batch(self.source, size, rng, self.device)
        if not self.recent:
            return tensor_batch(self.history, size, rng, self.device)
        # Half of the target batch is recent, half historical; source has its own batch.
        recent_size = size // 2
        batches = [tensor_batch(self.history, size-recent_size, rng, self.device),
                   tensor_batch(row_table(self.recent), recent_size, rng, self.device)]
        return {key: torch.cat([batch[key] for batch in batches]) for key in batches[0]}

    def sample_mixed(self, size, online_fraction, *, rng):
        count = int(round(size * online_fraction))
        return (self.sample(size-count, rng=rng) if count < size else None,
                self.sample(count, scope="online", rng=rng) if count else None)


class OnlineWorld:
    """Frozen source prior, direct/residual models, and optional fast residual context."""

    def __init__(self, replay, *, seed=6201, updates=20, context=False):
        self.device, self.replay = replay.device, replay
        self.seed = seed
        self.stations = torch.as_tensor(replay.station_max_by_line, device=self.device)
        self.rng = np.random.default_rng(seed)
        self.generator = torch.Generator(device=self.device).manual_seed(seed+30_000)
        source = targets(learned_rows(replay.source), self.stations, self.device)
        self.norm = Normalizer(source).to(self.device)
        self.source = BusModel().to(self.device)
        self._fit(self.source, source, updates)
        self.source.requires_grad_(False).eval()
        self.models = {"direct": BusModel().to(self.device),
                       "residual": BusModel(self.source).to(self.device)}
        self.context_enabled = context
        self.context = {family: torch.zeros(4, device=self.device) for family in self.models}
        self.errors = {family: deque(maxlen=128) for family in self.models}
        self.pending, self.scored = {}, []
        self.version = self.refits = self.model_updates = 0
        self.trust = QDeltaBellman(229, 1, gamma=.9, device=self.device)
        history = learned_rows(replay.history)
        boundary = max(1, int(.8*len(history["s"])))
        fit = targets({key: value[:boundary] for key, value in history.items()}, self.stations, self.device)
        stop = targets({key: value[boundary:] for key, value in history.items()}, self.stations, self.device)
        self.initial_scores = {}
        for family, model in self.models.items():
            self._fit(model, fit, updates)
            with torch.no_grad():
                self.initial_scores[family] = float(losses(model(stop["s"], stop["a"]), stop, self.norm).mean())
        self.family = min(self.initial_scores, key=self.initial_scores.get)

    def _fit(self, model, data, updates):
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-4)
        for _ in range(updates):
            ids = torch.as_tensor(self.rng.integers(len(data["s"]), size=min(64, len(data["s"]))), device=self.device)
            batch = {key: value[ids] for key, value in data.items()}
            loss = losses(model(batch["s"], batch["a"]), batch, self.norm).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite world-model loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
        model.eval()

    @torch.no_grad()
    def predict(self, s, a, family=None):
        family = family or self.family
        output = self.models[family](s, a).clone()
        if self.context_enabled:
            output[:, :4] += self.context[family]
        return output

    @torch.no_grad()
    def register(self, issued):
        """Called at action issuance, using the action actually sent to the environment."""
        if within_batch(issued["s"][None])[0]:
            return
        s = torch.as_tensor(issued["s"][None], device=self.device)
        a = torch.as_tensor(issued["a"].reshape(1, 1), device=self.device)
        self.pending[issued["sequence"]] = dict(version=self.version,
            issued_at=issued["started_at"],
            outputs={family: self.predict(s, a, family).detach().clone() for family in self.models})

    @torch.no_grad()
    def observe(self, rows, now):
        """Score saved predictions first; the caller adds rows to replay afterwards."""
        logs = []
        for row in rows:
            if row.available_at > now or row.completed_at > now:
                raise ValueError("Target label was released before it was available")
            prediction = self.pending.pop(row.sequence, None)
            if prediction is None or row.terminal or row.dt <= 0:
                continue
            y = targets(row_table([row]), self.stations, self.device)
            scores = {family: float(losses(output, y, self.norm).mean())
                      for family, output in prediction["outputs"].items()}
            for family, score in scores.items():
                self.errors[family].append(score)
                if self.context_enabled:
                    # A delayed label may outlive a refit. Score the old saved
                    # prediction above, but calibrate the CURRENT model's context.
                    raw = self.models[family](y["s"], y["a"])
                    error = ((y["g"]-self.norm.g_mean)/self.norm.g_std - raw[:, :4])[0]
                    self.context[family].lerp_(error, .1)
            self.family = min(self.models, key=lambda family: np.mean(self.errors[family]))
            log = dict(sequence=row.sequence, model_version=prediction["version"],
                issued_at=prediction["issued_at"], completed_at=row.completed_at,
                available_at=row.available_at, scored_at=now, scores=scores, selected=self.family)
            logs.append(log)
        self.scored.extend(logs)
        return logs

    def refit(self, updates):
        recent = learned_rows(row_table(self.replay.recent))
        if not len(recent["s"]):
            return False
        old = learned_rows(self.replay.history)
        ids = self.rng.integers(len(old["s"]), size=len(recent["s"]))
        data = targets(concatenate([{key: value[ids] for key, value in old.items()}, recent]),
                       self.stations, self.device)
        for family, model in self.models.items():
            self._fit(model, data, updates)
            self.context[family].zero_()
        self.version += 1
        self.refits += 1
        self.model_updates += 2*updates
        return True

    def update_trust(self, actor, size=64):
        rows = learned_rows(row_table(self.replay.recent))
        if not len(rows["s"]):
            return
        y = targets({key: rows[key] for key in ("s", "a", "ns", "r", "dt")}, self.stations, self.device)
        # These are training errors for trust fitting, separate from the saved prequential log.
        with torch.no_grad():
            error = losses(self.predict(y["s"], y["a"]), y, self.norm).mean(1)
        ids = self.rng.integers(len(error), size=size)
        s, a = y["s"][ids], y["a"][ids]
        ns = torch.as_tensor(rows["ns"][ids], device=self.device)
        return self.trust.update(s, a, ns, torch.zeros(size, device=self.device), error[ids],
                                 lambda states: actor(states, deterministic=True)[0], n_iters=1)

    @torch.no_grad()
    def generate(self, actor, size):
        rows = learned_rows(row_table(self.replay.recent))
        # Only states with room for a nonterminal step may start imagination.
        eligible = np.flatnonzero(rows["s"][:, 0] < .8)
        if not len(eligible):
            return None
        ids = self.rng.choice(eligible, size=size)
        s = torch.as_tensor(rows["s"][ids], device=self.device)
        a = actor(s, deterministic=True)[0]
        ns, reward, dt = ModelAugmentation.decode(self, s, self.predict(s, a))
        return dict(observations=s, actions=a, next_observations=ns, rewards=reward,
                    delta_t_seconds=dt, discounts=torch.ones(size, device=self.device),
                    terminals=torch.zeros(size, device=self.device)), self.trust.weight(s, a)

    def summary(self):
        return dict(version=self.version, target_refits=self.refits, gradient_updates=self.model_updates,
            selected_family=self.family, prequential_rows=len(self.scored),
            rolling_prequential_loss={key: float(np.mean(value)) if value else None
                                     for key, value in self.errors.items()},
            context={key: value.tolist() for key, value in self.context.items()})


class ClosedLoopTrainer(TransitH2OTrainerV7):
    world = None
    model_loss = None
    model_fraction = .25

    def _ensemble_td_loss(self, predictions, target, weights=None):
        real_loss = super()._ensemble_td_loss(predictions, target, weights)
        return real_loss if self.model_loss is None else (
            (1-self.model_fraction)*real_loss + self.model_fraction*self.model_loss)

    def continue_step(self):
        self.model_loss = None
        metrics = {}
        if self.world is not None and self.replay_buffer.recent:
            devices = [self.device.index or 0] if self.device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(self.world.seed+30_000+self.total_updates)
                self.world.update_trust(self.actor, self.config.batch_size)
                generated = self.world.generate(self.actor, self.config.batch_size)
                if generated is not None:
                    batch, weights = generated
                    error = self._elementwise_td_loss(
                        self.critic(batch["observations"], batch["actions"]),
                        self._target(batch, backup="h2oplus"))
                    # Preserve the absolute confidence: small trust reduces model influence.
                    self.model_loss = (error*weights[None]).mean()
                    metrics = dict(model_td_loss=float(self.model_loss.detach()),
                                   model_trust_mean=float(weights.mean()))
        try:
            result = super().update(online=True)
            return dict(result, **metrics)
        finally:
            self.model_loss = None
