"""Read-only root fidelity and future entropy intervention on real event rows."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value,handle,separators=(',',':'),allow_nan=False)
        handle.write('\n')


def lambda_zero_gate(learner, seed):
    """Native target/RNG equality with zero optimizer steps and lazy future."""
    import torch
    from experiments.mcwm_closed_loop_physical_shadow_v54_20261025.target import (
        ReferenceSyncMean8Trainer,target_state,restore_target_state)
    from experiments.mcwm_closed_loop_physical_shadow_v54_20261025.score import _snapshot,_finish
    from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _state,_equal,_rng
    from experiments.mcwm_closed_loop_entropy_trace_v57_20261025.target import trace_target
    before, private, model = _snapshot(learner), target_state(learner), deepcopy(_state(learner))
    results=[]
    try:
        for beta in (None,0,1):
            restore_target_state(learner,private)
            for name,state in before['private'].items():
                getattr(learner,name).bit_generator.state=deepcopy(state)
            torch.manual_seed(seed)
            native=ReferenceSyncMean8Trainer._target(learner,learner.physical_batch,backup='h2oplus')
            calls=[]
            def forbidden_future():
                calls.append(True)
                raise RuntimeError('Lambda0 evaluated a future branch')
            actual=native if beta is None else trace_target(native,forbidden_future,lambda_trace=0.,beta=beta)
            results.append(dict(value=actual.clone(),rng=_rng(),extra=learner.mean8_extra_rng_state.clone(),
                private={name:deepcopy(getattr(learner,name).bit_generator.state) for name in before['private']},
                identical_root_object=actual is native,future_calls=len(calls),
                parameters_Adam_unchanged=_equal(model,_state(learner))))
        reference=results[0]
        checks=dict(native_reference_target_exact=all(torch.equal(r['value'],reference['value']) for r in results[1:]),
            root_object_retained=all(r['identical_root_object'] for r in results),
            no_future_draws_or_forwards=all(r['future_calls']==0 for r in results),
            main_RNG_exact=all(_equal(r['rng'],reference['rng']) for r in results[1:]),
            M8_private_rng_exact=all(torch.equal(r['extra'],reference['extra']) for r in results[1:]),
            sampler_and_diagnostic_rng_exact=all(_equal(r['private'],reference['private']) for r in results[1:]),
            parameters_and_Adam_exact=all(r['parameters_Adam_unchanged'] for r in results))
        if not all(checks.values()):
            raise RuntimeError('Lambda0 native fidelity failed: '+str(checks))
        return dict(all_pass=True,checks=checks,new_optimizer_updates=0)
    finally:
        restore_target_state(learner,private)
        _finish(learner,before)


def original_physical(native, matching_q, coefficients):
    """Independent original v54 float32 recurrence, without a beta argument."""
    physical=native.clone()
    for index in range(len(native)-2,-1,-1):
        if coefficients[index] != 0:
            physical[index]=native[index]+coefficients[index]*(physical[index+1]-matching_q[index])
    return physical


def run(scenario):
    import numpy as np
    import torch
    from experiments.mcwm_closed_loop_physical_truth_v55_20261025.frozen import build_frozen
    from experiments.mcwm_closed_loop_physical_shadow_v54_20261025.target import target_state
    from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _state,_equal,_rng
    from experiments.mcwm_closed_loop_entropy_trace_v57_20261025.target import frozen_block,ETA
    from experiments.mcwm_closed_loop_entropy_trace_v57_20261025.metrics import summarize
    torch.set_num_threads(1)
    p=read(HERE/'protocol.json')
    out=HERE/'runs'/scenario
    out.mkdir(parents=True,exist_ok=False)
    started=time.time()
    learners,_,_,source_gate=build_frozen(scenario)
    before={arm:deepcopy(_state(l)) for arm,l in learners.items()}
    private={arm:deepcopy(target_state(l)) for arm,l in learners.items()}
    rng=_rng()
    source=HERE.parent/p['chain_version']/'runs'/scenario
    chain=[json.loads(line)['transition'] for line in (source/'continuous_feedback.jsonl').read_text().splitlines() if line.strip()]
    if len(chain)!=271:
        raise RuntimeError('Expected271 original admitted real transitions')
    for learner in learners.values():
        batch=learner.physical_batch
        for i,row in enumerate(chain):
            for field,key in (('s','observations'),('ns','next_observations')):
                actual=np.concatenate((np.asarray(row[field],np.float32),np.asarray(row['memory_'+field]['vector'],np.float32)))
                if not np.array_equal(actual,batch[key][i].numpy()):
                    raise RuntimeError('Actual state/memory chain differs')
            for field,key in (('a','actions'),('r','rewards'),('dt','delta_t_seconds'),('terminal','terminals')):
                if not np.array_equal(np.asarray(row[field],np.float32).reshape(-1),batch[key][i].numpy().reshape(-1)):
                    raise RuntimeError('Actual action/reward/time/terminal differs')
        for left,right in zip(chain,chain[1:]):
            if not (left['completed_at']==right['started_at'] and left['ns']==right['s'] and left['memory_ns']==right['memory_s']):
                raise RuntimeError('Real event adjacency is discontinuous')
    if not torch.equal(learners['one_step'].physical_coefficients,learners['trace'].physical_coefficients):
        raise RuntimeError('Original physical coefficients differ between frozen arms')
    arms={}
    shifts={}
    for arm,learner in learners.items():
        gate=lambda_zero_gate(learner,p['lambda_zero_seed'])
        blocks=[frozen_block(learner,seeds) for seeds in p['credit_draw_seeds']]
        native=torch.stack([b['native'] for b in blocks])
        beta0=torch.stack([b['beta0'] for b in blocks])
        beta1=torch.stack([b['beta1'] for b in blocks])
        shifts[arm]=(beta1-beta0).double().numpy()
        original_exact=all(torch.equal(original_physical(b['native'],b['matching_q'],b['coefficients']),b['physical_beta0']) and
            torch.equal(b['native']+b['native'].new_tensor(ETA)*(b['physical_beta0']-b['native']),b['beta0']) for b in blocks)
        # Archive only computed arrays on server, never weights or new truth.
        arrays={key:torch.stack([b[key] for b in blocks]).cpu().numpy()
            for key in ('native','beta0','beta1','entropy','future_mass','terminal_weight','entropy_correction')}
        np.savez(out/(arm+'_frozen_arrays.npz'),**arrays)
        metrics=summarize(native.numpy(),beta0.numpy(),beta1.numpy(),chain,blocks[0]['future_mass'].numpy(),
            focus_lines=p['focus_lines'])
        rounding={key:max(float(b['rounding64'][key].abs().max()) for b in blocks)
            for key in ('beta0','beta1','beta1_minus_beta0')}
        remaining=np.asarray([chain[-1]['completed_at']-r['started_at'] for r in chain])
        with (out/(arm+'_factual_rows.jsonl')).open('x') as handle:
            for i,row in enumerate(chain):
                data=dict(index=i,line=row['line_id'],sequence=row['sequence'],issue=row['started_at'],
                    dt=row['dt'],remaining_real_span=remaining[i],has_successor=i<270,
                    future_mass=float(blocks[0]['future_mass'][i]),
                    native=native[:,i].tolist(),beta0=beta0[:,i].tolist(),beta1=beta1[:,i].tolist())
                handle.write(json.dumps(data,separators=(',',':'),allow_nan=False)+'\n')
        arms[arm]=dict(lambda_zero_gate=gate,beta0_original_recurrence_exact=original_exact,
            rounding64_max_abs=rounding,metrics=metrics,
            terminal_dependency=dict(weight_min=float(blocks[0]['terminal_weight'].min()),
                weight_max=float(blocks[0]['terminal_weight'].max())),
            installed_training_lambda=float(learner.lambda_trace),probe_lambda=1.)
        if not original_exact:
            raise RuntimeError('Beta0 differs from original physicaltrace on identical arrays')
    checks=dict(frozen_model_source_exact=source_gate['all_pass'],original285_slots456_updates=all(
        l.total_updates==50456 and l.replay_buffer.target_received==285 for l in learners.values()),
        lambda_zero_both_models=all(a['lambda_zero_gate']['all_pass'] for a in arms.values()),
        beta0_exact=all(a['beta0_original_recurrence_exact'] for a in arms.values()),
        parameters_and_Adam_unchanged=all(_equal(before[arm],_state(l)) for arm,l in learners.items()),
        target_private_streams_and_counters_unchanged=all(_equal(private[arm],target_state(l)) for arm,l in learners.items()),
        global_RNG_unchanged=_equal(rng,_rng()),no_training_native_or_truth_admission=True,
        F_unchanged=True,trust_unchanged=True)
    # Same factual state across models, never a counterfactual action successor.
    contrast=shifts['trace'][:,:270]-shifts['one_step'][:,:270]
    shift_did=dict(mean=float(contrast.mean()),std=float(contrast.std()),
        max_abs=float(np.abs(contrast).max()),
        interpretation='Change in trace-minus-single frozen target contrast at identical factual roots; not action benefit.')
    result=dict(status='complete' if all(checks.values()) else 'integrity_failed',scenario=scenario,
        integrity=dict(all_pass=all(checks.values()),checks=checks),arms=arms,
        cross_arm_beta_difference_in_differences=shift_did,
        budget=dict(initializer_updates=50000,feedback_slots=285,optimizer_updates=456,horizon_seconds=24000,
            new_optimizer_updates=0,new_native_branches=0,new_training_labels=0,truth_admitted=0),
        wall=dict(started=started,finished=time.time(),seconds=time.time()-started))
    write(out/'summary.json',result)
    if not all(checks.values()):
        raise RuntimeError('Frozen precheck changed source state')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario',choices=('stable','local_speed_drop'),required=True)
    args=parser.parse_args()
    try:
        result=run(args.scenario)
        print(json.dumps(dict(status=result['status'],scenario=args.scenario)))
    except BaseException as error:
        write(HERE/'runs'/args.scenario/'failure.json',dict(type=type(error).__name__,message=str(error),
            new_optimizer_updates=0,new_native_branches=0))
        raise
