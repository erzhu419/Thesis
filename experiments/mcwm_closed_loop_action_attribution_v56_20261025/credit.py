"""Frozen v53-chain target decomposition; never an optimizer-path attribution.

The v55 issues have no fabricated successor here. Only the resident 271-row
physical_batch supplies rewards, next states, executed actions and coefficients.
"""
from __future__ import annotations

import numpy as np
import torch

ETA = 9. / 270.
DEFAULT_SEEDS = tuple(tuple(566202 + 8*b+j for j in range(8)) for b in range(4))
ATOM_NAMES = ('reward', 'value', 'sac_base', 'sac_level', 'sac_interaction',
              'entropy', 'minus_q_base', 'minus_q_level', 'minus_q_interaction')


def _get(row, field):
    return row[field] if isinstance(row, dict) else getattr(row, field)


def _stats(values):
    a = np.asarray(values, dtype=np.float64)
    return dict(n=int(a.size), mean=float(a.mean()), mean_abs=float(np.abs(a).mean()),
                min=float(a.min()), max=float(a.max()))


def _compact(tensor):
    return tensor.detach().double().cpu().numpy()


@torch.no_grad()
def _components(critic, states, actions):
    """Return the original full head Q and its additive components."""
    canonical, memory = states[:, :229], states[:, 229:]
    base = critic.base(canonical, actions)
    level = (memory @ critic.level_weight.T).T
    if level.shape[0] == 1:
        level = level.expand(base.shape[0], -1)
    interaction = ((actions[:, 0]+1.)*.5)[None, :]*(memory @ critic.action_weight.T).T
    # Keep official native addition order for the full minimum/head selection.
    full = critic(states, actions)
    reconstructed = base + (level + interaction)
    if not torch.equal(full, reconstructed):
        raise RuntimeError('Target critic decomposition no longer matches native addition order')
    return dict(full=full, base=base, level=level, interaction=interaction)


def _selected(parts):
    heads = parts['full'].argmin(0)
    indices = torch.arange(len(heads), device=heads.device)
    return {name: value[heads, indices] for name, value in parts.items()}, heads


def future_correction(atoms, coefficients, *, eta=ETA):
    """Root i receives successor k>i atoms only, on the real adjacent chain."""
    result = torch.zeros_like(atoms)
    for i in range(len(atoms)-2, -1, -1):
        result[i] = coefficients[i] * (atoms[i+1]+result[i+1])
    return result * float(eta)


@torch.no_grad()
def frozen_block(learner, seeds):
    """One CRN M8 draw block, preserving global and learner private streams.

    Head subset is the complete original two-of-two ensemble, so no subset
    randomness is consumed. Extra draws retain first-draw zero/gate terms.
    """
    if len(seeds) != 8:
        raise ValueError('Each predeclared block contains exactly eight samples')
    batch = learner.physical_batch
    if len(batch['rewards']) != 271 or len(learner.physical_coefficients) != 270:
        raise ValueError('Only the original real 271-row chain is supported')
    config = learner.config
    if (config.ensemble_size != 2 or config.target_subset_size != 2
            or config.target_subset_replacement or config.mixed_backup_lambda != .1):
        raise ValueError('The original two-of-two .1V/.9SAC target is required')
    states, following, actions = batch['observations'], batch['next_observations'], batch['actions']
    reward = config.reward_scale * batch['rewards']
    discount = batch['discounts']*(1.-batch['terminals'])
    current, current_heads = _selected(_components(learner.target_critic, states, actions))
    value = .1 * discount * learner.value(following[:, :229])
    zero, positive, first = None, [], None
    zero_heads, positive_heads = None, []
    reference_positive, reference_zero = [], None
    features = learner.target_critic.encode_states(following)
    repeated_features = torch.cat((features, features))
    subset = torch.tensor([0, 1], dtype=torch.long, device=states.device)
    rng = torch.get_rng_state().clone()
    try:
        for seed in seeds:
            torch.manual_seed(int(seed))
            parts = learner.sync_actor.soft_action_components(following[:, :229])
            a0 = learner.sync_transform(following[:, :229], parts['zero_action'])
            ap = learner.sync_transform(following[:, :229], parts['positive_action'])
            scored = _components(learner.target_critic, torch.cat((following, following)),
                                  torch.cat((a0, ap)))
            # Independent official features/subset recipe. Components above
            # follow the full forward; this path checks native M8 assembly.
            native_q = learner.target_critic.forward_features_subset(repeated_features,
                torch.cat((a0, ap)), subset)
            native_zero, native_positive = native_q.split(len(states), dim=1)
            native_zero, native_positive = native_zero.min(0).values, native_positive.min(0).values
            if config.backup_entropy:
                native_zero = native_zero-learner.alpha.detach()*parts['zero_log_prob']
                native_positive = native_positive-learner.alpha.detach()*parts['positive_log_prob']
            chosen, heads = _selected(scored)
            selected_zero = {k: v[:len(states)] for k, v in chosen.items()}
            selected_positive = {k: v[len(states):] for k, v in chosen.items()}
            selected_zero['entropy'] = (-learner.alpha.detach()*parts['zero_log_prob']
                                       if config.backup_entropy else torch.zeros_like(reward))
            selected_positive['entropy'] = (-learner.alpha.detach()*parts['positive_log_prob']
                                           if config.backup_entropy else torch.zeros_like(reward))
            if first is None:
                first, zero = parts, selected_zero
                zero_heads = heads[:len(states)]
                reference_zero = native_zero
            positive.append(selected_positive)
            positive_heads.append(heads[len(states):])
            reference_positive.append(native_positive)
    finally:
        torch.set_rng_state(rng)
    sac = {k: .9*discount*(first['zero_probability']*zero[k]
            + first['hold_probability']*torch.stack([p[k] for p in positive]).mean(0))
           for k in ('base', 'level', 'interaction', 'entropy', 'full')}
    atoms = dict(reward=reward, value=value, sac_base=sac['base'], sac_level=sac['level'],
                 sac_interaction=sac['interaction'], entropy=sac['entropy'],
                 minus_q_base=-current['base'], minus_q_level=-current['level'],
                 minus_q_interaction=-current['interaction'])
    # Float64 sums are an accounting decomposition, not the native float32 target.
    raw = torch.stack([atoms[k].double() for k in ATOM_NAMES], dim=1)
    corrections = future_correction(raw, learner.physical_coefficients.double())
    native = reward + .1*discount*learner.value(following[:, :229]) + sac['full'] + sac['entropy']
    reference_sac = first['zero_probability']*reference_zero + first['hold_probability']*torch.stack(reference_positive).mean(0)
    reference_native = reward + discount*(.1*learner.value(following[:, :229])+.9*reference_sac)
    reference_error = float((native-reference_native).abs().max())
    reference_bound = 16.*torch.finfo(native.dtype).eps*max(1.,float(reference_native.abs().max()))
    if reference_error > reference_bound:
        raise RuntimeError('Frozen component assembly differs from official M8 mixed target')
    delta = reward + .1*discount*learner.value(following[:, :229]) + sac['full'] + sac['entropy'] - current['full']
    return dict(atoms=raw, corrections=corrections, native=native, residual=delta,
        current_heads=current_heads, zero_heads=zero_heads,
        positive_heads=torch.stack(positive_heads), discount=discount,
        native_M8_reference_max_abs=reference_error, native_M8_float32_bound=reference_bound,
        decomposition_max_abs=float((raw.sum(1)-delta.double()).abs().max()))


def _group(blocks, indices):
    arrays = np.stack([_compact(block['corrections'])[indices] for block in blocks])
    reward = arrays[:, :, 0]
    bq = arrays[:, :, 1:].sum(-1)
    return dict(rows=len(indices), reward=_stats(reward), bootstrap_minus_q=_stats(bq),
        total=_stats(reward+bq), components={name:dict(
            mean=float(arrays[:,:,i].mean()), mean_abs=float(np.abs(arrays[:,:,i]).mean()))
            for i,name in enumerate(ATOM_NAMES)},
        positive_total=int(np.count_nonzero((reward+bq).mean(0)>0)),
        negative_total=int(np.count_nonzero((reward+bq).mean(0)<0)))


def _offset_checks(blocks, coefficients):
    """Analytic fixed-shift sensitivities, with no parameter mutation.

    Q and V jointly +1 leave each nonterminal B-Q atom invariant. Q alone +1
    leaves the entropy and reward unchanged and contributes (.9*d-1).
    """
    d = blocks[0]['discount'].double()
    joint_atoms = d-1.
    q_atoms = .9*d-1.
    joint = future_correction(joint_atoms, coefficients.double())
    qonly = future_correction(q_atoms, coefficients.double())
    return dict(joint_Q_V_offset=1., joint_B_minus_Q_atom_max_abs=float(joint_atoms.abs().max()),
        joint_trace_correction_max_abs=float(joint.abs().max()),
        Q_only_offset=1., Q_only_atom=_stats(_compact(q_atoms)),
        Q_only_trace_correction=_stats(_compact(qonly)),
        interpretation='Frozen algebraic sensitivity; not a historical optimization attribution')


def compare_targets(learners, chainrows, seeds=None):
    """Return compact endpoint component summaries on the admitted v53 chain.

    Seeds are a sequence of four (or explicitly supplied) eight-seed blocks.
    No v55 successor states or held-out outcomes are admitted or reconstructed.
    """
    seeds = tuple(tuple(int(s) for s in block) for block in (seeds or DEFAULT_SEEDS))
    if not seeds or any(len(b)!=8 for b in seeds) or len(chainrows)!=271:
        raise ValueError('Need a real271-row chain and nonempty M8 blocks')
    for learner in learners.values():
        batch = learner.physical_batch
        for i,row in enumerate(chainrows):
            for field,key in (('s','observations'),('ns','next_observations')):
                expected = np.asarray(_get(row,field),np.float32)
                if not np.array_equal(batch[key][i,:229].cpu().numpy(),expected):
                    raise RuntimeError('The target probe does not use the same admitted chain')
            if not np.array_equal(batch['actions'][i].cpu().numpy(),np.asarray(_get(row,'a'),np.float32)):
                raise RuntimeError('The target probe executed action differs')
    blocks = {arm:[frozen_block(learner,b) for b in seeds] for arm,learner in learners.items()}
    lines = sorted(set(str(_get(row,'line_id')) for row in chainrows))
    groups = {'all':list(range(271))}
    groups.update({line:[i for i,r in enumerate(chainrows) if str(_get(r,'line_id'))==line] for line in lines})
    result = dict(seeds=[list(b) for b in seeds], eta=ETA, arms={},
        chain_rows=271, source='v53 admitted real physical_batch; not v55 held-out suffixes',
        probe_lambda=1., installed_lambda={arm:float(getattr(learner,'lambda_trace',0.))
            for arm,learner in learners.items()},
        scope='Unit-lambda frozen post456 operator on both endpoints; single-step arm did not receive these trace corrections during training. Endpoint component differences are not cumulative training causality.',
        new_optimizer_updates=0,new_feedback_slots=0,new_native_branches=0)
    for arm,rows in blocks.items():
        result['arms'][arm] = dict(groups={name:_group(rows,indices) for name,indices in groups.items()},
            head_usage=dict(current=[int((rows[0]['current_heads']==h).sum()) for h in range(2)],
                zero=[int((rows[0]['zero_heads']==h).sum()) for h in range(2)],
                positive=[int(sum((r['positive_heads']==h).sum() for r in rows)) for h in range(2)]),
            decomposition_max_abs=max(row['decomposition_max_abs'] for row in rows),
            native_M8_reference_max_abs=max(row['native_M8_reference_max_abs'] for row in rows),
            native_M8_float32_bound=max(row['native_M8_float32_bound'] for row in rows),
            offsets=_offset_checks(rows,learners[arm].physical_coefficients))
    if 'one_step' in blocks and 'trace' in blocks:
        differences=[]
        for single,trace in zip(blocks['one_step'],blocks['trace']):
            differences.append(dict(corrections=trace['corrections']-single['corrections']))
        result['trace_minus_single']={name:_group(differences,indices) for name,indices in groups.items()}
    return result
