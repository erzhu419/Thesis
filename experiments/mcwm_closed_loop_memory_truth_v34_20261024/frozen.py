"""Exact resident reconstruction and read-only scoring of frozen v33 learners.

Reconstruction reuses only v32's admitted factual285 rows and v33's completed
packet records. It happens before the new native prefix; its optimizer work is
an explicitly separate diagnostic cost. New prefix/suffix truth never enters
either learner. Full learner checkpoints remain in this version's server dir.
"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch

from experiments.mcwm_closed_loop_critic_shadow_v33_20261024.learning import (
    ARMS, critic_states, make_learner)
from experiments.mcwm_closed_loop_critic_shadow_v33_20261024.facts import load_facts
from experiments.mcwm_closed_loop_critic_shadow_v33_20261024.probes import probe, window
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _equal, _rng, _restore


V33 = Path(__file__).resolve().parents[1]/"mcwm_closed_loop_critic_shadow_v33_20261024"
REPO = V33.parents[1]


def plain(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def optimizer_steps(learner):
    result = {}
    for name in ("actor_optimizer", "critic_optimizer", "value_optimizer", "alpha_optimizer"):
        steps = [float(state["step"]) for state in getattr(learner, name).state.values() if "step" in state]
        result[name] = dict(states=len(steps), min=min(steps, default=None), max=max(steps, default=None))
    return result


def _rebuild_arm(learner, packets, reference, *, steps_per_packet):
    """Reproduce each saved first-step diagnostic and all sampler/RNG records."""
    if len(packets) != len(reference):
        raise RuntimeError("The original packet record count differs")
    initial = learner.total_updates
    labels = 0
    for packet, expected in zip(packets, reference):
        index, receipt = packet["index"], packet["receipt"]
        if expected["packet"] != index or expected["receipt"] != receipt:
            raise RuntimeError(f"Original receipt differs at packet {index}")
        learner.replay_buffer.add_target(packet["rows"])
        labels += len(packet["rows"])
        sampling = []
        for step in range(steps_per_packet):
            metrics = learner.continue_step(collect_diagnostics=step == 0)
            sampling.append(deepcopy(learner.replay_buffer.last_sampling))
            if step == 0:
                actual = dict(packet=index, processed_at=receipt["processed_at"], **metrics)
                if plain(actual) != expected["metrics"]:
                    changed = [key for key in actual if plain(actual[key]) != expected["metrics"].get(key)]
                    raise RuntimeError(f"Rebuilt first-step metrics differ at packet {index}: {changed}")
        rng = dict(global_state=plain(_rng()), sampler=plain(learner.rng.bit_generator.state),
                   diagnostic=plain(learner.diagnostic_rng.bit_generator.state))
        checks = dict(sampling=sampling == expected["sampling"], rng=rng == expected["rng"],
            optimizer_steps=optimizer_steps(learner) == expected["optimizer_steps"],
            updates=learner.total_updates-initial == expected["updates"],
            feedback=learner.replay_buffer.target_received == labels)
        if not all(checks.values()):
            raise RuntimeError(f"Rebuilt packet {index} differs: {[key for key,value in checks.items() if not value]}")
        if index % 10 == 0 or index == len(packets):
            print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase='frozen_v33_rebuild',
                arm=getattr(learner, 'memory_arm', 'test'), current=index,
                total=len(packets), unit='packets')), flush=True)
    return dict(all_pass=True, packets=len(packets), feedback_slots=labels,
        reconstruction_optimizer_updates=learner.total_updates-initial,
        saved_first_step_metrics_exact=True, all_sampling_draws_exact=True,
        packet_global_and_private_rng_exact=True, saved_optimizer_steps_exact=True)


def _final_predictions(learner, packets, saved_chunks, protocol):
    seen, bank = set(), []
    for packet in packets:
        for row in packet["rows"]:
            key = row.line_id, window(row.started_at)
            if row.line_id in protocol["focus_lines"] and key not in seen:
                seen.add(key)
                bank.append(row)
    count = 0
    for phase, rows in (("packet_after", packets[-1]["rows"]), ("bank_after", bank)):
        expected = [chunk["rows"] for chunk in saved_chunks
                    if chunk["packet"] == packets[-1]["index"] and chunk["phase"] == phase]
        if len(expected) != 1:
            raise RuntimeError("The original final prediction chunk is missing or duplicated")
        actual = probe(learner, rows, seed=protocol["diagnostic_seed"])
        if plain(actual) != expected[0]:
            raise RuntimeError(f"Rebuilt complete old {phase} predictions differ")
        count += len(rows)
    return dict(all_pass=True, old_final_prediction_rows=count, old_prediction_chunks=2,
        complete_old_q_advantage_actor_F_and_bellman_exact=True,
        old_admitted_bank_only=True, new_prefix_context_read=False)


def freeze(learner):
    for name in ("actor", "critic", "target_critic", "value", "ratio"):
        module = getattr(learner, name, None)
        if module is not None:
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
                parameter.grad = None
    learner.log_alpha.requires_grad_(False)
    learner.log_alpha.grad = None
    return learner


def _jsonlines(path):
    with path.open() as source:
        return [json.loads(line) for line in source]


def build_frozen(seed, scenario, outdir):
    """Build once per scenario, then load this version's resident exact cache.

    Returns ``learners, contract, stations, proof``. Global RNG is restored even
    on failed reconstruction, so this cannot reseed the new native prefix.
    """
    saved_rng = _rng()
    outdir = Path(outdir)
    checkpoint = outdir/"final_models.pt"
    try:
        if checkpoint.exists():
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if payload["seed"] != seed or payload["scenario"] != scenario or not payload["proof"]["all_pass"]:
                raise RuntimeError("Frozen cache does not match the declared scenario")
            return payload["learners"], payload["contract"], payload["stations"], payload["proof"]
        outdir.mkdir(parents=True, exist_ok=True)
        protocol = json.loads((V33/"protocol.json").read_text())
        if seed not in protocol["training_seeds"] or scenario not in protocol["scenarios"]:
            raise ValueError("Frozen model outside the completed v33 development cells")
        learners, arms, facts, contract, stations = {}, {}, None, None, None
        for arm in ARMS:
            original = V33/"runs"/f"seed{seed}"/scenario/arm
            summary = json.loads((original/"summary.json").read_text())
            if not summary["integrity"]["all_pass"]:
                raise RuntimeError("Reconstruction requires a completed, verified v33 learner")
            learner, _, contract, stations, initializer = make_learner(seed, arm=arm)
            if learner.total_updates != protocol["initializer_updates"]:
                raise RuntimeError("Frozen reconstruction changed the shared initialization")
            if facts is None:
                before_facts = _rng()
                facts = load_facts(REPO/protocol["original_runs"][scenario], contract, protocol)
                if not facts["proof"]["all_pass"] or not _equal(before_facts, _rng()):
                    raise RuntimeError("Ordinary factual reconstruction failed or changed RNG")
            replay_proof = _rebuild_arm(learner, facts["packets"], _jsonlines(original/"packets.jsonl"),
                steps_per_packet=protocol["policy_updates_per_packet"])
            prediction_proof = _final_predictions(learner, facts["packets"],
                _jsonlines(original/"scores.jsonl"), protocol)
            if (replay_proof["packets"] != 57 or replay_proof["feedback_slots"] != 285
                    or replay_proof["reconstruction_optimizer_updates"] != 456
                    or prediction_proof["old_final_prediction_rows"] != 14):
                raise RuntimeError("The completed v33 reconstruction budget or old bank differs")
            learners[arm] = freeze(learner)
            arms[arm] = dict(replay=replay_proof, prediction=prediction_proof,
                initializer=str(initializer), original_v33_gate=summary["integrity"],
                reused_zero_actor_trace_gate=summary["integrity"]["checks"]["zero_full_actor_trace_exact"])
        proof = dict(all_pass=True, seed=seed, scenario=scenario, arms=arms,
            budget=dict(reconstruction_optimizer_updates=912, new_training_labels=0,
                new_prefix_optimizer_updates=0, suffix_optimizer_updates=0,
                original_feedback_slots_per_arm=285, original_updates_per_arm=456),
            truth_used_for_training=False, native_prefix_started_during_rebuild=False,
            resident_checkpoint=str(checkpoint))
        payload = dict(seed=seed, scenario=scenario, learners=learners, contract=contract,
                       stations=stations, proof=proof)
        with checkpoint.open("xb") as handle:
            torch.save(payload, handle)
        with (outdir/"rebuild_proof.json").open("x") as handle:
            json.dump(proof, handle, separators=(",", ":"), allow_nan=False)
            handle.write("\n")
        return learners, contract, stations, proof
    finally:
        _restore(saved_rng)


@torch.no_grad()
def score_frozen(learners, state, memory, executed_actions):
    """Score two already executed actions; never transform those actions twice."""
    saved_rng = _rng()
    private = {arm: {name: deepcopy(getattr(learner, name).bit_generator.state)
                    for name in ("rng", "diagnostic_rng")} for arm, learner in learners.items()}
    try:
        result = {}
        for arm, learner in learners.items():
            states = torch.as_tensor(np.asarray(state), dtype=torch.float32, device=learner.device).reshape(1, 229)
            context = torch.as_tensor(np.asarray(memory), dtype=torch.float32, device=learner.device).reshape(1, 84)
            actions = torch.as_tensor(executed_actions, dtype=torch.float32, device=learner.device).reshape(2, 1)
            explicit = critic_states(learner, states.expand(2, -1), context.expand(2, -1))
            values = learner.critic(explicit, actions)
            minimum, mean = values.min(0).values, values.mean(0)
            proposal = learner.actor(states, deterministic=True)[0]
            executed = learner.action_transform(states, proposal)
            result[arm] = dict(q_min=minimum.tolist(), q_mean=mean.tolist(),
                advantage_min=(minimum-minimum[0]).tolist(), advantage_mean=(mean-mean[0]).tolist(),
                predicted_choice=int(minimum.argmax()), executed_actions=actions[:, 0].tolist(),
                holding_seconds=learner.action_transform.hold_seconds(actions[:, 0]).tolist(),
                actor_proposal_action=float(proposal[0, 0]),
                actor_proposal_holding_seconds=float(learner.action_transform.hold_seconds(proposal[0, 0])),
                actor_executed_action=float(executed[0, 0]),
                actor_holding_seconds=float(learner.action_transform.hold_seconds(executed[0, 0])))
        if result["memory"]["holding_seconds"] != result["zero_memory"]["holding_seconds"]:
            raise RuntimeError("The two frozen critics interpreted different physical actions")
        return result
    finally:
        for arm, learner in learners.items():
            for name, state in private[arm].items():
                getattr(learner, name).bit_generator.state = state
        _restore(saved_rng)
