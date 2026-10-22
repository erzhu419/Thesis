"""Passive initial/online world comparison on one executed target trajectory."""
from __future__ import annotations

import copy
from collections import deque
from types import SimpleNamespace

import numpy as np
import torch

from experiments.mcwm_closed_loop_action_local_v3_20261020.local_world import phase
from experiments.mcwm_closed_loop_context_shadow_v5_20261020.context import _score_outputs
from experiments.mcwm_closed_loop_context_shadow_v5_20261020.shadow import (
    FrozenBellmanEvaluator, restore_rng, rng_state, transition_batch,
)
from experiments.mcwm_closed_loop_head_diagnostic_v4_20261020.probes import ProbedWorld
from experiments.mcwm_closed_loop_v1.learning import (
    ModelAugmentation, learned_rows, losses, row_table, targets,
)


ISSUE_BRANCHES = ("online", "frozen_own", "frozen_common")


def _moments(values):
    values = np.asarray(values, dtype=float)
    return dict(signed_mean=values.mean(0).tolist(),
                mae=np.abs(values).mean(0).tolist(), mse=np.square(values).mean(0).tolist())


def aggregate(rows):
    result = dict(n=len(rows))
    if not rows:
        return result
    result["branches"] = {}
    for branch in rows[0]["branches"]:
        get = lambda key: np.asarray([r["branches"][branch][key] for r in rows], dtype=float)
        record = dict(six_loss_mean=get("six_losses").mean(0).tolist(),
            g4_mse=np.square(get("g4_signed_error")).mean(0).tolist(),
            physical_mae=np.abs(get("physical_signed_error")).mean(0).tolist(),
            physical_signed_mean=get("physical_signed_error").mean(0).tolist(),
            decoded_physical_mae=np.abs(get("decoded_physical_signed_error")).mean(0).tolist(),
            decode_saturation_fraction=get("decode_saturated").mean(0).tolist())
        for key in ("bellman_target", "scaled_reward", "bootstrap"):
            if key+"_signed_error" in rows[0]["branches"][branch]:
                record[key+"_error"] = _moments(get(key+"_signed_error"))
        result["branches"][branch] = record
    return result


class PairedWorld(ProbedWorld):
    """Keep action_model live learning unchanged; twin models never learn.

    The twin QDelta starts from the SAME initialization and follows every live
    trust update on the same released rows and actor actions. It therefore
    measures adapting confidence in the initial model, not an untrained net.
    """
    def __init__(self, *args, **kwargs):
        if kwargs.get("enabled", False):
            raise ValueError("v6 paired diagnostics require the action_model arm")
        kwargs["enabled"] = False
        super().__init__(*args, **kwargs)
        self.device = torch.device(self.device)
        self._initialize_paired()

    def _initialize_paired(self):
        self.diagnostic_enabled = True
        self.frozen_models = copy.deepcopy(self.models)
        for model in self.frozen_models.values():
            model.requires_grad_(False).eval()
        self.frozen_norm = copy.deepcopy(self.norm)
        self.frozen_family = self.family
        self.frozen_errors = {name: deque(maxlen=128) for name in self.models}
        self.twin_trust = copy.deepcopy(self.trust)
        self.paired_probe_deadlines = [8400., 10200., 13800., 17100.]
        self.paired_issue_records, self.paired_future = [], []
        self.paired_anchor = None
        self.trust_records, self.trust_anchor_steps = [], []
        self.trust_updates = 0
        self.paired_processed_at = None
        self._trust_packet = None
        self._trust_record = None
        # Direct tensor comparison at finalization; these copies stay server-side.
        self._initial_parameters = {family: copy.deepcopy(model.state_dict())
                                    for family, model in self.frozen_models.items()}

    @torch.no_grad()
    def register(self, issued):
        super().register(issued)
        pending = self.pending.get(issued["sequence"])
        if pending is None or not self.diagnostic_enabled:
            return
        s = torch.as_tensor(issued["s"][None], device=self.device)
        a = torch.as_tensor(issued["a"].reshape(1, 1), device=self.device)
        pending["paired_world"] = dict(
            online_family=self.family, frozen_family=self.frozen_family,
            frozen_outputs={name: model(s, a).detach().clone()
                            for name, model in self.frozen_models.items()})

    @torch.no_grad()
    def observe(self, rows, now):
        if not self.diagnostic_enabled:
            return super().observe(rows, now)
        if any(row.available_at > now or row.completed_at > now for row in rows):
            raise ValueError("A paired world label was supplied before availability")
        self.paired_processed_at = float(now)
        selected = [(row, self.pending[row.sequence]) for row in rows
            if row.sequence in self.pending and not row.terminal and row.dt > 0]
        details = {}
        for row, pending in selected:
            saved = pending["paired_world"]
            live, own = saved["online_family"], saved["frozen_family"]
            y = targets(row_table([row]), self.stations, self.device)
            outputs = dict(online=pending["outputs"][live],
                frozen_own=saved["frozen_outputs"][own],
                frozen_common=saved["frozen_outputs"][live])
            scores = _score_outputs(outputs, y, self.norm, [row])[0]
            model_families = {name: _score_outputs(dict(
                online=pending["outputs"][name], frozen=saved["frozen_outputs"][name]),
                y, self.norm, [row])[0] for name in self.models}
            family_scores = {name: float(losses(output, y, self.frozen_norm).mean())
                             for name, output in saved["frozen_outputs"].items()}
            for name, error in family_scores.items():
                self.frozen_errors[name].append(error)
            self.frozen_family = min(self.frozen_models,
                key=lambda name: float(np.mean(self.frozen_errors[name])))
            record = dict(sequence=int(row.sequence), issued_at=float(row.started_at),
                completed_at=float(row.completed_at), available_at=float(row.available_at),
                scored_at=float(now), issue_phase=phase(row.started_at, self.physical.shock_at),
                line_id=str(row.line_id), online_family=live, frozen_family=own,
                same_family=live == own, branches=scores,
                model_families=model_families,
                frozen_family_six_loss=family_scores,
                scored_before_next_packet_training=True)
            self.paired_issue_records.append(record)
            details[row.sequence] = record
        if self.paired_anchor is not None and selected:
            self._resolve_anchor([row for row, _ in selected], now)
        logs = super().observe(rows, now)
        for log in logs:
            log["paired_world"] = details[log["sequence"]]
        return logs

    @torch.no_grad()
    def diagnose_packet(self, learner, packet, deadline, now):
        """Capture after the live refit and before this packet's eight updates."""
        self.paired_processed_at = float(now)
        if not self.diagnostic_enabled or float(deadline) not in self.paired_probe_deadlines:
            return None
        if self.paired_anchor is not None:
            raise RuntimeError("Previous paired anchor has no released next packet")
        anchor = SimpleNamespace(models=dict(online=copy.deepcopy(self.models),
                frozen=copy.deepcopy(self.frozen_models)),
            norm=copy.deepcopy(self.norm), stations=self.stations.clone(), device=self.device,
            generator=torch.Generator(device=self.device),
            family=self.family, frozen_family=self.frozen_family,
            deadline=float(deadline), now=float(now), version=int(self.version),
            evaluator=FrozenBellmanEvaluator(learner),
            decoder_seed=self.seed+170000+int(deadline))
        self.paired_anchor = anchor
        self._trust_packet = dict(deadline=float(deadline), anchor_at=float(now), steps=[])
        self.trust_anchor_steps.append(self._trust_packet)
        return dict(deadline=float(deadline), common_family=self.family,
                    frozen_family=self.frozen_family, captured_before_eight_updates=True)

    @torch.no_grad()
    def _resolve_anchor(self, rows, now):
        anchor = self.paired_anchor
        y = targets(row_table(rows), self.stations, self.device)
        outputs = dict(online=anchor.models["online"][anchor.family](y["s"], y["a"]),
            frozen_common=anchor.models["frozen"][anchor.family](y["s"], y["a"]),
            frozen_own=anchor.models["frozen"][anchor.frozen_family](y["s"], y["a"]))
        scores = _score_outputs(outputs, y, anchor.norm, rows)
        truth_batch = transition_batch(rows, self.device)
        truth = anchor.evaluator.target(truth_batch)
        reward_scale = anchor.evaluator.learner.config.reward_scale
        for branch, output in outputs.items():
            anchor.generator.manual_seed(anchor.decoder_seed)
            ns, reward, dt = ModelAugmentation.decode(anchor, y["s"], output)
            generated = dict(truth_batch, next_observations=ns, rewards=reward, delta_t_seconds=dt)
            target = anchor.evaluator.target(generated)
            reward_error = reward_scale*(reward-truth_batch["rewards"])
            error = target-truth
            for i, detail in enumerate(scores):
                detail[branch].update(bellman_target_signed_error=float(error[i]),
                    scaled_reward_signed_error=float(reward_error[i]),
                    bootstrap_signed_error=float(error[i]-reward_error[i]))
        records = [dict(sequence=int(row.sequence),
                       issue_phase=phase(row.started_at, self.physical.shock_at), branches=score)
                   for row, score in zip(rows, scores)]
        self.paired_future.append(dict(deadline=anchor.deadline, anchor_at=anchor.now,
            scored_at=float(now), n=len(rows), common_family=anchor.family,
            frozen_family=anchor.frozen_family, world_version=anchor.version,
            evaluator_fixed_at_anchor=True, identical_decoder_and_target_draws=True,
            scored_before_next_packet_training=True, overall=aggregate(records),
            by_issue_phase={name: aggregate([r for r in records if r["issue_phase"] == name])
                            for name in sorted({r["issue_phase"] for r in records})}))
        self.paired_anchor = None

    def _trust_update(self, trust, s, a, ns, error, actor, *, next_actions=None):
        """Observe the real QDelta update; extra reads do not affect its RNG."""
        trace = {}
        def next_action(states):
            value = (self.action_transform(states, actor(states, deterministic=True)[0])
                     if next_actions is None else next_actions)
            with torch.no_grad():
                trace["next_actions"] = value.detach().clone()
                trace["bootstrap"] = (trust.gamma*trust.target(states, value)).reshape(-1).clone()
                trace["prediction_before"] = trust.net(s, a).reshape(-1).clone()
            return value
        result = trust.update(s, a, ns, torch.zeros(len(s), device=self.device), error,
                             next_action, n_iters=1)
        with torch.no_grad():
            c = trust._c_step(error).clamp_min(1e-8)
            log_c = c.log()
            target = log_c+trace["bootstrap"]
            prediction = trust.net(s, a).reshape(-1)
            trace["summary"] = dict(tau=float(trust._tau), training_error_mean=float(error.mean()),
                training_error_median=float(error.median()), c_step_mean=float(c.mean()),
                log_c_step_mean=float(log_c.mean()),
                bootstrap_target_mean=float(trace["bootstrap"].mean()),
                bellman_target_mean=float(target.mean()),
                prediction_before_mean=float(trace["prediction_before"].mean()),
                prediction_after_mean=float(prediction.mean()),
                td_mse_before=float((trace["prediction_before"]-target).square().mean()),
                td_mse_after=float((prediction-target).square().mean()),
                update_td_loss=float(result["td_loss"]),
                observed_action_weight_mean=float(trust.weight(s, a).mean()))
        return result, trace

    def update_trust(self, actor, size=64):
        if not self.diagnostic_enabled:
            return super().update_trust(actor, size)
        rows = learned_rows(row_table(self.replay.recent))
        if not len(rows["s"]):
            return
        y = targets({key: rows[key] for key in ("s", "a", "ns", "r", "dt")},
                    self.stations, self.device)
        with torch.no_grad():
            live_six = losses(self.predict(y["s"], y["a"]), y, self.norm)
            frozen_six = losses(self.frozen_models[self.family](y["s"], y["a"]), y, self.frozen_norm)
        ids = self.rng.integers(len(live_six), size=size)
        s, a = y["s"][ids], y["a"][ids]
        ns = torch.as_tensor(rows["ns"][ids], device=self.device)
        result, live = self._trust_update(self.trust, s, a, ns, live_six.mean(1)[ids], actor)
        after_live_rng = rng_state()
        try:
            _, twin = self._trust_update(self.twin_trust, s, a, ns, frozen_six.mean(1)[ids],
                                       actor, next_actions=live["next_actions"])
        finally:
            restore_rng(after_live_rng)
        self.trust_updates += 1
        at = (self.paired_processed_at if self.paired_processed_at is not None
              else max(float(row.available_at) for row in self.replay.recent))
        record = dict(update=self.trust_updates, phase=phase(at, self.physical.shock_at),
            processed_at=float(at),
            released_rows=len(rows["s"]), common_family=self.family,
            identical_training_indices_and_next_actions=True,
            online=dict(live["summary"], six_loss_mean=live_six[ids].mean(0).tolist()),
            frozen=dict(twin["summary"], six_loss_mean=frozen_six[ids].mean(0).tolist()))
        self.trust_records.append(record)
        self._trust_record = record
        if self._trust_packet is not None:
            self._trust_packet["steps"].append(record)
            if len(self._trust_packet["steps"]) == 8:
                self._trust_packet = None
        return result

    @torch.no_grad()
    def generate(self, actor, size):
        result = super().generate(actor, size)
        if not self.diagnostic_enabled or result is None or self._trust_record is None:
            return result
        batch, weight = result
        self._trust_record["online"]["actor_action_weight_mean"] = float(weight.mean())
        self._trust_record["frozen"]["actor_action_weight_mean"] = float(
            self.twin_trust.weight(batch["observations"], batch["actions"]).mean())
        return result

    def summary(self):
        result = super().summary()
        unchanged = all(torch.equal(value, self._initial_parameters[family][name])
            for family, model in self.frozen_models.items() for name, value in model.state_dict().items())
        def trust_mean(records):
            value = dict(n=len(records))
            if records:
                for branch in ("online", "frozen"):
                    value[branch] = {key: np.mean([r[branch][key] for r in records], axis=0).tolist()
                                     for key in records[0][branch]}
            return value
        issue = self.paired_issue_records
        result["paired_world_diagnostics"] = dict(enabled=self.diagnostic_enabled,
            local_adapter_enabled=self.local_enabled, selected_rows=len(issue),
            issue_time=dict(overall=aggregate(issue),
                by_issue_phase={name: aggregate([r for r in issue if r["issue_phase"] == name])
                                for name in sorted({r["issue_phase"] for r in issue})},
                same_family_rows=sum(r["same_family"] for r in issue),
                by_model_family={family: aggregate([dict(branches=r["model_families"][family])
                    for r in issue]) for family in self.models},
                online_own_equals_online_common=True),
            next_packet_frozen_anchor=self.paired_future,
            pending_anchor=None if self.paired_anchor is None else self.paired_anchor.deadline,
            trust_updates=self.trust_updates, trust_anchor_steps=self.trust_anchor_steps,
            trust_period_means={name: trust_mean([r for r in self.trust_records if r["phase"] == name])
                                for name in sorted({r["phase"] for r in self.trust_records})},
            frozen_parameters_unchanged=unchanged,
            twin_trust_semantics="initial QDelta clone; trained every live update using same released rows, live common family, sampled indices and executed next actor actions; own adaptive tau and bootstrap",
            issue_semantics="paired saved issue-time predictions; own-family and common online issue-family; labels scored before fitting",
            future_semantics="both world snapshots, norm, families, common Bellman evaluator and decoder draws frozen after anchor refit; next released packet scored before training")
        return result
