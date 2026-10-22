"""Passive context pairing and frozen-anchor next-packet prediction diagnostics."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import torch

from experiments.mcwm_closed_loop_action_local_v3_20261020.local_world import (
    AFFECTED_LINES, FEATURE_NAMES, adapter_features, phase,
)
from experiments.mcwm_closed_loop_head_diagnostic_v4_20261020.probes import ProbedWorld
from experiments.mcwm_closed_loop_v1.learning import ModelAugmentation, losses, row_table, targets


BRANCHES = ("base", "saved", "current")
SIX_HEADS = ("global", "local", "count", "line", "station", "neighbor")
GLOBAL_HEADS = ("logdt", "logcost", "dWaiting", "dOnboard")


def _predict(models, readouts, numeric_lines, s, a, family, branch, deltas):
    output = models[family](s, a).clone()
    if branch != "base":
        output[:, :4] += adapter_features(s, a, deltas, numeric_lines) @ readouts[family]
    return output


def _score_outputs(outputs, y, norm, rows):
    """No sampling: raw heads and exactly the decoder's global projection."""
    truth = y["g"]
    standardized_truth = (truth-norm.g_mean)/norm.g_std
    physical_truth = truth.clone()
    physical_truth[:, 0] = torch.as_tensor([r.dt for r in rows], device=truth.device)
    physical_truth[:, 1] = torch.as_tensor([-r.r for r in rows], device=truth.device)
    result = []
    scores = {}
    for name, output in outputs.items():
        g = output[:, :4]*norm.g_std+norm.g_mean
        raw = g.clone()
        raw[:, :2] = torch.expm1(g[:, :2])
        duration = torch.expm1(g[:, 0].clamp(0, 10)).clamp_min(1.)
        capped_duration = torch.minimum(duration, (1-y["s"][:, 0])*24000-1)
        cost = torch.expm1(g[:, 1].clamp(0, 20))
        counts_before_clip = y["s"][:, 3:5]+g[:, 2:4]/20000
        increments = (counts_before_clip.clamp(0, 1.25)-y["s"][:, 3:5])*20000
        decoded = torch.cat((capped_duration[:, None], cost[:, None], increments), 1)
        saturation = torch.cat((((g[:, 0] < 0) | (g[:, 0] > 10)
            | (torch.expm1(g[:, 0].clamp(0, 10)) < 1)
            | (duration > capped_duration))[:, None],
            ((g[:, 1] < 0) | (g[:, 1] > 20))[:, None],
            (counts_before_clip < 0) | (counts_before_clip > 1.25)), 1)
        scores[name] = dict(six_losses=losses(output, y, norm).cpu().tolist(),
            g4_signed_error=(output[:, :4]-standardized_truth).cpu().tolist(),
            physical_signed_error=(raw-physical_truth).cpu().tolist(),
            decoded_physical_signed_error=(decoded-physical_truth).cpu().tolist(),
            decode_saturated=saturation.cpu().tolist(),
            prediction_standardized=output[:, :4].cpu().tolist())
    for i in range(len(rows)):
        result.append({name: {key: values[i] for key, values in branch.items()}
                       for name, branch in scores.items()})
    return result


def _aggregate(rows, compact=False):
    result = dict(n=len(rows))
    if not rows:
        return result
    branch_scores = {}
    for branch in BRANCHES:
        get = lambda key: np.asarray([r["branches"][branch][key] for r in rows], dtype=float)
        standardized, physical = get("g4_signed_error"), get("physical_signed_error")
        score = dict(six_loss_mean=get("six_losses").mean(0).tolist(),
                     g4_mse=np.square(standardized).mean(0).tolist(),
                     physical_mae=np.abs(physical).mean(0).tolist())
        if not compact:
            decoded = get("decoded_physical_signed_error")
            score.update(g4_signed_mean=standardized.mean(0).tolist(),
                physical_signed_mean=physical.mean(0).tolist(),
                physical_rmse=np.sqrt(np.square(physical).mean(0)).tolist(),
                decoded_physical_mae=np.abs(decoded).mean(0).tolist(),
                decoded_physical_signed_mean=decoded.mean(0).tolist(),
                decode_saturation_fraction=get("decode_saturated").mean(0).tolist())
            if "bellman_target_signed_error" in rows[0]["branches"][branch]:
                errors = get("bellman_target_signed_error")
                score["bellman_target_error"] = dict(signed_mean=float(errors.mean()),
                    mae=float(np.abs(errors).mean()), rmse=float(np.square(errors).mean()**.5))
        branch_scores[branch] = score
    result["branches"] = branch_scores
    if not compact:
        saved, current = branch_scores["saved"], branch_scores["current"]
        prediction_delta = np.asarray([np.subtract(r["branches"]["current"]["prediction_standardized"],
            r["branches"]["saved"]["prediction_standardized"]) for r in rows])
        result["current_minus_saved"] = dict(
            six_loss_mean=(np.asarray(current["six_loss_mean"])-saved["six_loss_mean"]).tolist(),
            g4_mse=(np.asarray(current["g4_mse"])-saved["g4_mse"]).tolist(),
            physical_mae=(np.asarray(current["physical_mae"])-saved["physical_mae"]).tolist(),
            prediction_standardized_signed_mean=prediction_delta.mean(0).tolist(),
            prediction_standardized_absolute_mean=np.abs(prediction_delta).mean(0).tolist())
        result["delta_context_l2_seconds_mean"] = float(np.mean([r["delta_z_l2_seconds"] for r in rows]))
    return result


def _grouped(rows):
    """Groups overlap; labels, rather than groups, are the paired observations."""
    def group(key):
        return {str(value): _aggregate([row for row in rows if row[key] == value], compact=True)
                for value in sorted({row[key] for row in rows})}
    return dict(overall=_aggregate(rows), by_issue_phase=group("issue_phase"),
        by_age=group("age_group"), by_line=group("line_id"),
        by_delta_z=group("delta_z_group"), by_phase_crossing=group("phase_crossing"))


def feature_identifiability(states, actions, saved_deltas, current_deltas, numeric_lines):
    """Rank of the observed design; singular/constant columns are reported as such."""
    result = dict(n=len(states), feature_names=list(FEATURE_NAMES), branches={})
    for name, deltas in (("saved", saved_deltas), ("current", current_deltas)):
        x = adapter_features(states, actions, deltas, numeric_lines).double().cpu().numpy()
        singular = np.linalg.svd(x, compute_uv=False)
        rank = int(np.linalg.matrix_rank(x))
        nonzero = singular[:rank]
        residuals, correlations = [], []
        for i in range(3):
            column = 6+i
            others = np.delete(x, column, axis=1)
            norm = np.linalg.norm(x[:, column])
            residual = x[:, column]-others@np.linalg.lstsq(others, x[:, column], rcond=None)[0]
            residuals.append(float(np.linalg.norm(residual)/norm) if norm > 0 else None)
            route, hold = x[:, 3+i], x[:, column]
            correlations.append(float(np.corrcoef(route, hold)[0, 1])
                if np.std(route) > 0 and np.std(hold) > 0 else None)
        result["branches"][name] = dict(rank=rank, singular_values=singular.tolist(),
            condition_number_full=float(singular[0]/singular[-1]) if rank == 9 else None,
            condition_number_nonzero=float(nonzero[0]/nonzero[-1]) if rank else None,
            hold_column_residual_norm_fraction_after_other_columns=residuals,
            route_hold_column_correlation=correlations)
    fractions = ((actions.reshape(-1)+1)/2).cpu().numpy()
    result["executed_hold_fraction"] = dict(mean=float(fractions.mean()), std=float(fractions.std()),
        minimum=float(fractions.min()), maximum=float(fractions.max()))
    return result


class ContextWorld(ProbedWorld):
    """All inherited training/prediction paths are unchanged; diagnostics are explicit."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device(self.device)
        self.context_probe_deadlines = [8400, 10200, 13800, 17100]
        self.context_selected = []
        self.context_historical = []
        self.context_future = []
        self.context_anchor = None

    @torch.no_grad()
    def predict_context(self, s, a, deltas=None, branch="base", family=None):
        if branch not in BRANCHES:
            raise ValueError(branch)
        if branch == "saved" and deltas is None:
            raise ValueError("saved context needs issue-time deltas")
        if branch == "current":
            deltas = self.physical.context_delta_seconds() if deltas is None else deltas
        return _predict(self.models, self.local_readouts, self.numeric_lines, s, a,
                        family or self.family, branch, deltas)

    @torch.no_grad()
    def register(self, issued):
        super().register(issued)
        pending = self.pending.get(issued["sequence"])
        if pending is not None:
            pending["context_diagnostic"] = dict(sequence=int(issued["sequence"]),
                started_at=float(issued["started_at"]), line_id=str(issued["line_id"]),
                saved_delta=np.asarray(pending["local_covariates"]["delta_seconds"]).copy())

    def _score(self, rows, metadata, *, now, current_delta, family, snapshot=None):
        y = targets(row_table(rows), self.stations, self.device)
        saved = np.stack([item["saved_delta"] for item in metadata])
        models = self.models if snapshot is None else snapshot.models
        readouts = self.local_readouts if snapshot is None else snapshot.local_readouts
        norm = self.norm if snapshot is None else snapshot.norm
        outputs = {branch: _predict(models, readouts, self.numeric_lines, y["s"], y["a"], family,
            branch, saved if branch == "saved" else current_delta) for branch in BRANCHES}
        values = _score_outputs(outputs, y, norm, rows)
        records = []
        for row, item, branches in zip(rows, metadata, values):
            delta = np.asarray(current_delta)-item["saved_delta"]
            delta_norm = float(np.linalg.norm(delta))
            issue_phase = phase(item["started_at"], self.physical.shock_at)
            complete_phase = phase(row.completed_at, self.physical.shock_at)
            records.append(dict(sequence=int(row.sequence), line_id=item["line_id"],
                started_at=item["started_at"], completed_at=float(row.completed_at),
                available_at=float(row.available_at), scored_at=float(now),
                issue_phase=issue_phase, completed_phase=complete_phase,
                age_seconds=float(now-item["started_at"]),
                age_group="at_most_900s" if now-item["started_at"] <= 900 else "older_than_900s",
                phase_crossing="same_issue_completion_phase" if issue_phase == complete_phase
                    else "crosses_issue_completion_phase",
                evaluation_phase_crossing=issue_phase != phase(now, self.physical.shock_at),
                saved_context_seconds=item["saved_delta"].tolist(),
                current_context_seconds=np.asarray(current_delta).tolist(),
                delta_z_seconds=delta.tolist(), delta_z_l2_seconds=delta_norm,
                delta_z_group="zero" if delta_norm <= 1e-8 else
                    ("nonzero_at_most_one_feature_unit" if delta_norm <= 60 else "above_one_feature_unit"),
                branches=branches))
        return records, outputs, y

    @torch.no_grad()
    def observe(self, rows, now):
        selected = []
        for row in rows:
            if row.available_at > now or row.completed_at > now:
                raise ValueError("A context diagnostic label was supplied before availability")
            pending = self.pending.get(row.sequence)
            if pending is not None and not row.terminal and row.dt > 0:
                selected.append((row, copy.deepcopy(pending["context_diagnostic"])))
        future_details = {}
        if self.context_anchor is not None and selected:
            anchor = self.context_anchor
            devices = [self.device.index or 0] if self.device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                rows_for_score, metadata = zip(*selected)
                details, outputs, y = self._score(rows_for_score, metadata, now=now,
                    current_delta=anchor.delta, family=anchor.family, snapshot=anchor)
                if anchor.evaluator is not None:
                    from experiments.mcwm_closed_loop_context_shadow_v5_20261020.shadow import transition_batch
                    truth_target = anchor.evaluator.target(transition_batch(rows_for_score, self.device))
                    for branch, output in outputs.items():
                        anchor.generator.manual_seed(anchor.decoder_seed)
                        ns, rewards, dt = ModelAugmentation.decode(anchor, y["s"], output)
                        batch = dict(observations=y["s"], actions=y["a"], next_observations=ns,
                            rewards=rewards, delta_t_seconds=dt, discounts=torch.ones(len(rows_for_score), device=self.device),
                            terminals=torch.zeros(len(rows_for_score), device=self.device))
                        target = anchor.evaluator.target(batch)
                        for detail, prediction, truth in zip(details, target.tolist(), truth_target.tolist()):
                            detail["branches"][branch]["bellman_target_prediction"] = prediction
                            detail["branches"][branch]["bellman_target_truth"] = truth
                            detail["branches"][branch]["bellman_target_signed_error"] = prediction-truth
                packet_summary = dict(deadline=anchor.deadline, anchor_at=anchor.now,
                    scored_at=float(now), family=anchor.family, world_version=anchor.version,
                    rows=len(details), n=len(details), scored_before_next_packet_training=True,
                    current_context_fixed_at_anchor=True, evaluator_fixed_at_anchor=anchor.evaluator is not None,
                    **_grouped(details))
                self.context_future.append(packet_summary)
                future_details = {row["sequence"]: dict(anchor_deadline=anchor.deadline, **row)
                                  for row in details}
                self.context_anchor = None
        logs = super().observe(rows, now)
        logged = {log["sequence"] for log in logs}
        self.context_selected.extend((row, metadata) for row, metadata in selected if row.sequence in logged)
        for log in logs:
            if log["sequence"] in future_details:
                log["context_next_packet"] = future_details[log["sequence"]]
        return logs

    @torch.no_grad()
    def diagnose_context(self, packet, now, deadline, learner=None):
        if deadline not in self.context_probe_deadlines:
            return None
        if self.context_anchor is not None:
            raise RuntimeError("Previous context anchor has no next-packet outcome")
        if any(item["deadline"] == deadline for item in self.context_historical):
            raise RuntimeError("Context anchor was evaluated twice")
        if not self.context_selected:
            raise RuntimeError("Context anchor has no selected replay labels")
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            rows, metadata = zip(*self.context_selected)
            delta = self.physical.context_delta_seconds().copy()
            details, _, y = self._score(rows, metadata, now=now, current_delta=delta, family=self.family)
            summary = dict(deadline=float(deadline), now=float(now), family=self.family,
                world_version=self.version, rows=len(details), n=len(details), **_grouped(details),
                identifiability=feature_identifiability(y["s"], y["a"],
                    np.stack([item["saved_delta"] for item in metadata]), delta, self.numeric_lines))
            self.context_historical.append(summary)
            evaluator = None
            if learner is not None:
                from experiments.mcwm_closed_loop_context_shadow_v5_20261020.shadow import FrozenBellmanEvaluator
                evaluator = FrozenBellmanEvaluator(learner, seed=self.seed+int(deadline)+70000)
            self.context_anchor = SimpleNamespace(models=copy.deepcopy(self.models),
                local_readouts=copy.deepcopy(self.local_readouts), norm=copy.deepcopy(self.norm),
                delta=delta, family=self.family, evaluator=evaluator,
                deadline=float(deadline), now=float(now), version=self.version,
                device=self.device, stations=self.stations.clone(),
                decoder_seed=self.seed+int(deadline)+80000,
                generator=torch.Generator(device=self.device))
        return dict(**summary, row_records=details)

    def summary(self):
        result = super().summary()
        result["context_diagnostics"] = dict(branches=list(BRANCHES), six_heads=list(SIX_HEADS),
            global_heads=list(GLOBAL_HEADS), physical_units=["seconds", "passenger_seconds", "persons", "persons"],
            expected_anchor_deadlines=list(self.context_probe_deadlines),
            historical_anchors=len(self.context_historical), resolved_future_packets=len(self.context_future),
            future_rows=sum(item["rows"] for item in self.context_future),
            selected_rows=len(self.context_selected),
            pending_anchor_deadline=self.context_anchor.deadline if self.context_anchor is not None else None,
            historical=self.context_historical, next_packet_frozen_anchor=self.context_future,
            semantics="All three branches use one current family and identical world parameters/readout. Historical saved means the action-issue context; current means anchor-time context. Future world parameters, readout, family, current context and optional common Bellman evaluator are fixed before the next packet. Next-packet scoring occurs before its labels enter replay or fitting. Saved context is a causal estimate, not a true operating-regime label.",
            error_sign="prediction-minus-observation; paired differences current-minus-saved",
            delta_z_group_scale_seconds=60.,
            group_semantics="Overlapping descriptive groups; age at scoring, issue/completion phase crossing, L2 context difference. Historical anchors reuse cumulative selected labels and are not independent samples.",
            diagnostics_change_training=False)
        return result
