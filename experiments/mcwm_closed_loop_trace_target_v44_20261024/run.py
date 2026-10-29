"""Score existing v43 snapshots and real closed chains without an update."""
import argparse
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.mcwm_closed_loop_trace_target_v44_20261024.summary import (
    PREFIX_COLUMNS, REPLICATE_COLUMNS, compact, summarize_checkpoint,
)


def write(path, value, max_bytes=None):
    encoded = json.dumps(value, separators=(',', ':'), allow_nan=False)+'\n'
    if max_bytes is not None and len(encoded.encode()) > max_bytes:
        raise RuntimeError('Reduce the resident summary before any local pull')
    with path.open('x') as handle:
        handle.write(encoded)
    return len(encoded.encode())


def run(scenario):
    from experiments.mcwm_closed_loop_trace_target_v44_20261024.probe import score_scenario
    protocol = json.loads((HERE/'protocol.json').read_text())
    if scenario not in protocol['scenarios']:
        raise ValueError('Scenario outside the declared two-day precheck')
    out = HERE/'runs'/scenario
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    source = HERE.parent/protocol['source_version']/'runs'/scenario
    print(json.dumps(dict(phase='resident_target_precheck', scenario=scenario,
        checkpoints=protocol['checkpoints'], diagnostic_seeds=protocol['diagnostic_seeds'],
        new_training_updates=0, new_native_branches=0)), flush=True)
    try:
        probe = score_scenario(source, out=out, checkpoints=tuple(protocol['checkpoints']),
                               diagnostic_seeds=tuple(protocol['diagnostic_seeds']))
        write(out/'probe_detail.json', probe)
        checkpoints, detail, checks = {}, {}, {'resident_probe': probe['integrity']['all_pass']}
        for name, source_checkpoint in probe['checkpoints'].items():
            pairs, full = summarize_checkpoint(source_checkpoint['seeds'], protocol['diagnostic_seeds'])
            detail[name] = full
            traces = [value for bank in full.values() for value in bank.values()]
            checks[name+'_lambda0_exact'] = all(value['zero']['lambda_zero_native_exact'] for value in traces)
            checks[name+'_all285_native_exact'] = source_checkpoint['all285_lambda0']['all_pass']
            checks[name+'_six_chains_per_seed'] = all(len(bank) == 6 for bank in full.values())
            checks[name+'_all_real45'] = all(value['full']['event_count'] == 45 for value in traces)
            checks[name+'_decomposition_closure'] = all(
                abs(value['full']['root']['reconstruction_residual']) < 1e-8 for value in traces)
            checkpoints[name] = dict(proof=source_checkpoint['proof'],
                all285_lambda0=source_checkpoint['all285_lambda0'], pairs=pairs)
        write(out/'trace_detail.json', detail)
        all_pass = all(checks.values())
        result = dict(status='complete' if all_pass else 'failed', scenario=scenario, priority='high',
            integrity=dict(all_pass=all_pass, checks=checks), source_budget=probe['source_budget'],
            source_contract=probe['source_contract'], checkpoints=checkpoints,
            schema=dict(replicate_columns=REPLICATE_COLUMNS, prefix_columns=PREFIX_COLUMNS,
                action_difference='high1.875_minus_low0; higher target favors high action',
                primary_seed=protocol['diagnostic_seeds'][0],
                prefix_attribution='event-indexed partial sums of the fixed full45 target, not alternate truncated targets or matched-clock causal costs',
                reward_weight_split='Delta(w*r)=mean(w)*Delta(r)+mean(r)*Delta(w)',
                terminal_dependency='unrolled algebraic coefficient; not an SGD gradient',
                exactness='native frozen-snapshot _target parity under diagnostic CRN; historic sampled training targets not replayed'),
            diagnostic_budget=dict(new_training_updates=0, new_feedback_slots=0,
                new_native_branches=0, new_native_seconds=0, checkpoints_downloaded=0,
                resident_checkpoints_loaded=2),
            scientific_scope=dict(physical_horizon='existing45 actual closures only, about50seconds',
                cost300_direction=None, regret300=None, deployed_control_gain=None),
            wall_seconds=time.time()-started,
            decision=dict(shadow_training_started=False,
                next_gate='Review whether actual rewards yield a reproducible root action difference after bootstrap/Qbar cancellation, attenuation and terminal dependence. No automatic training or new collection.'))
        size = write(out/'summary.json', compact(result), protocol['summary_max_bytes_per_case'])
        print(json.dumps(dict(status=result['status'], integrity=result['integrity'],
                              bytes=size, wall_seconds=result['wall_seconds'])), flush=True)
        if not all_pass:
            raise RuntimeError('The resident target precheck failed its implementation gate')
    except Exception as error:
        write(out/'failure.json', dict(status='failed', type=type(error).__name__, message=str(error),
            new_training_updates=0, new_native_branches=0, elapsed_seconds=time.time()-started))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', required=True)
    run(parser.parse_args().scenario)
