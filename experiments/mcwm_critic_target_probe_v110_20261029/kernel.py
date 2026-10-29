"""Frozen endpoint target/gradient readout; no optimizer or native calls."""

import numpy as np
import torch


def distribution(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Probe statistics require finite nonempty values")
    return dict(n=int(values.size), mean=float(values.mean()), std=float(values.std()),
                min=float(values.min()), max=float(values.max()),
                negative=int((values < 0).sum()), zero=int((values == 0).sum()),
                positive=int((values > 0).sum()))


def _stats(tensor):
    return distribution(tensor.detach().cpu().numpy())


def _gradient(loss, parameters, *, retain_graph=False):
    return torch.cat([part.reshape(-1) for part in torch.autograd.grad(
        loss, parameters, retain_graph=retain_graph)]).detach()


def _responses(gradient, directions):
    return {name: float(-torch.dot(gradient.double(), direction.double()))
            for name, direction in directions.items()}


def _target_parts(learner, batch):
    """Capture the actual SAC sample used by the production target method."""
    original = learner._sac_bootstrap
    owned = "_sac_bootstrap" in learner.__dict__
    captured = []

    def capture(values):
        value = original(values)
        captured.append(value)
        return value

    learner._sac_bootstrap = capture
    try:
        target = learner._target(batch, backup=learner.config.online_backup)
    finally:
        if owned:
            learner._sac_bootstrap = original
        else:
            del learner._sac_bootstrap
    if len(captured) != 1:
        raise RuntimeError("Expected one SAC bootstrap for the full mixed batch")
    with torch.no_grad():
        discount = batch["discounts"].reshape(-1) * (1-batch["terminals"].reshape(-1))
        mixing = learner.config.mixed_backup_lambda
        parts = dict(reward=learner.config.reward_scale * batch["rewards"].reshape(-1),
                     value=discount * mixing * learner.value(batch["next_observations"]),
                     sac=discount * (1-mixing) * captured[0], target=target)
    return parts


def probe(learner, original_offline, source, fresh_offline, root_rows, contract):
    from SimpleSAC.transit_v3.trainer import TransitH2OTrainer
    from SimpleSAC.evaluate_transit_v19_completion_safety_reserve import completion_safety_action

    config = learner.config
    if config.bootstrap_probability != 1.0:
        return dict(status="HOLD", reason="bootstrap_probability_requires_full_masked_loss")
    if config.actor_q_reduction not in ("min-all", "mean-all"):
        return dict(status="HOLD", reason="random_actor_q_reduction_not_in_fixed_probe")
    if (config.online_backup != "h2oplus" or config.online_fraction != .5
            or config.use_ratio_correction or config.online_cql_alpha != 0
            or learner.protocol["real_td_coefficient"] != .75 or learner.coefficient != .25):
        return dict(status="HOLD", reason="restored_critic_contract_differs")
    if any(len(batch["observations"]) != 128 for batch in (original_offline, source, fresh_offline)):
        raise ValueError("Expected original128, source128 and fresh-replaced128")
    for key in original_offline:
        if not torch.equal(original_offline[key][:64], fresh_offline[key][:64]):
            raise ValueError("The retained offline prefix changed")
    parameters = tuple(p for p in learner.critic.parameters() if p.requires_grad)
    devices = [torch.cuda.current_device()] if learner.device.type == "cuda" else []
    suffix = slice(64, 128)
    scale = .75 * .5 * .5
    with torch.random.fork_rng(devices=devices):
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state() if devices else None
        parts, gradients, summaries = {}, {}, {}
        for name, offline in (("removed_old", original_offline), ("fresh", fresh_offline)):
            torch.set_rng_state(cpu_rng)
            if devices:
                torch.cuda.set_rng_state(cuda_rng)
            batch = learner._concatenate_batches(offline, source)
            part = _target_parts(learner, batch)
            parts[name] = part
            q = learner.critic(offline["observations"][suffix], offline["actions"][suffix])
            target = part["target"][suffix]
            loss = TransitH2OTrainer._ensemble_td_loss(learner, q, target)
            gradients[name] = scale * _gradient(loss, parameters)
            summaries[name] = dict(components={key: _stats(value[suffix]) for key, value in part.items()},
                td_residual=_stats(q.detach()-target.unsqueeze(0)), td_loss=float(loss.detach()),
                weighted_td_loss=float(scale*loss.detach()),
                weighted_gradient_norm=float(torch.linalg.vector_norm(gradients[name])))
        unchanged = torch.cat((torch.arange(64, device=learner.device),
                               torch.arange(128, 256, device=learner.device)))
        if not torch.equal(parts["removed_old"]["target"][unchanged], parts["fresh"]["target"][unchanged]):
            raise RuntimeError("Common full-batch targets did not remain identical")
        gradients["delta"] = gradients["fresh"]-gradients["removed_old"]
        rows = []
        for row in root_rows:
            states = torch.as_tensor(row["s"], device=learner.device,
                                     dtype=original_offline["observations"].dtype).reshape(1, -1)
            with torch.no_grad():
                components = learner.actor.soft_action_components(states, deterministic_positive=True)
                proposed = torch.cat((components["zero_action"], components["positive_action"]), dim=0)
            raw = row["raw_s"]
            if isinstance(raw, torch.Tensor):
                raw = raw.detach().cpu().numpy()
            executed = torch.as_tensor(np.stack([completion_safety_action(raw, a, "quarter-load-sixty", contract)
                for a in proposed.detach().cpu().numpy()]), device=learner.device, dtype=proposed.dtype)
            record = dict(sequence=int(row["sequence"]), feedback_selected=bool(row.get("feedback_selected", False)),
                hold_probability=float(components["hold_probability"][0]),
                proposal_hold_seconds=((proposed[:, 0]+1)*.5*contract.max_hold_seconds).cpu().tolist(),
                executed_hold_seconds=((executed[:, 0]+1)*.5*contract.max_hold_seconds).cpu().tolist())
            for kind, actions in (("proposal", proposed), ("executed", executed)):
                q = learner._reduce_actor_q(learner.critic(states.expand(2, -1), actions.detach()))
                difference, level = q[1]-q[0], (q[1]+q[0])*.5
                difference_gradient = _gradient(difference, parameters, retain_graph=True)
                level_gradient = _gradient(level, parameters)
                record[kind] = dict(difference=float(difference.detach()),
                    response=_responses(difference_gradient, gradients), level=float(level.detach()),
                    level_response=_responses(level_gradient, gradients))
            rows.append(record)
    aggregate = {}
    for kind in ("proposal", "executed"):
        aggregate[kind] = dict(difference=distribution([r[kind]["difference"] for r in rows]),
            level=distribution([r[kind]["level"] for r in rows]),
            response={name: distribution([r[kind]["response"][name] for r in rows]) for name in gradients},
            level_response={name: distribution([r[kind]["level_response"][name] for r in rows]) for name in gradients})
    selected = [{key: value for key, value in row.items() if key != "feedback_selected"}
                for row in rows if row["feedback_selected"]]
    for row in selected:
        for kind in ("proposal", "executed"):
            row[kind].pop("level")
            row[kind].pop("level_response")
    result = dict(status="complete", roots=len(rows), selected_roots=selected, aggregate=aggregate,
        targets=summaries, nominal_fresh_td_coefficient=scale,
        delta_gradient_norm=float(torch.linalg.vector_norm(gradients["delta"])),
        gradient_inner_product=float(torch.dot(gradients["fresh"].double(), gradients["removed_old"].double())),
        collapsed_executed_roots=sum(r["executed_hold_seconds"][0] == r["executed_hold_seconds"][1] for r in rows),
        common_targets_exact=True, optimizer_steps=0, native_seconds=0,
        config=dict(critic_loss=config.critic_loss, huber_delta=config.huber_delta,
            bootstrap_probability=config.bootstrap_probability, actor_q_reduction=config.actor_q_reduction,
            mixed_backup_lambda=config.mixed_backup_lambda, reward_scale=config.reward_scale),
        interpretation="Frozen-endpoint SGD infinitesimal sensitivity, not an Adam update, policy change or service benefit; common model/source terms cancel only at this shared endpoint.")
    return result
