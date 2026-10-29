"""Reuse resident arrays and freeze models for missing actor/head factors."""
import argparse
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, max_bytes=None):
    encoded = json.dumps(value, separators=(',', ':'), allow_nan=False)+'\n'
    if max_bytes is not None and len(encoded.encode()) > max_bytes:
        raise RuntimeError('Reduce the resident summary before a bounded local pull')
    with Path(path).open('x') as handle:
        handle.write(encoded)
    return len(encoded.encode())


def run(scenario):
    protocol = read(HERE/'protocol.json')
    if scenario not in protocol['scenarios']:
        raise ValueError('Undeclared scenario')
    out = HERE/'runs'/scenario
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    try:
        import torch
        from experiments.mcwm_closed_loop_trace_components_v45_20261024.components import (
            COMPONENTS, decompose_components, compare_pair,
        )
        from experiments.mcwm_closed_loop_trace_components_v45_20261024.factor import factorize_row
        from experiments.mcwm_closed_loop_trace_components_v45_20261024.summary import (
            CACHE_COLUMNS, CELL_COLUMNS, EVENT_COLUMNS, checkpoint_small, compact,
        )
        from experiments.mcwm_closed_loop_trace_target_v44_20261024.probe import (
            _loaded, jsonlines, optimizer_versions,
        )
        from experiments.mcwm_closed_loop_multievent_credit_v38_20261024.frozen import frozen_state, unchanged
        from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _rng, _equal
        torch.set_num_threads(1)
        seeds = protocol['diagnostic_seeds']
        arrays_root = HERE.parent/protocol['array_source_version']/'runs'/scenario
        model_root = HERE.parent/protocol['model_source_version']/'runs'/scenario
        source_summary, source = read(arrays_root/'summary.json'), read(arrays_root/'probe_detail.json')
        if not (source_summary['integrity']['all_pass'] and source['integrity']['all_pass']
                and source['source_budget']['feedback_slots'] == 285
                and source['source_budget']['updates'] == 456
                and source['source_contract']['initialization_seed'] == 6202
                and source['source_contract']['native_external_seed'] == 106243):
            raise RuntimeError('Source v44 budget or native parity gate is not intact')
        receipt = read(model_root/'response_receipt.json')
        pair = next(item for item in receipt['pairs'] if item['line_id'] == '705X')
        actual_rows = {arm: [value['transition'] for value in jsonlines(pair['branches'][arm]['events_path'])]
                       for arm in ('low', 'high')}
        if any(len(rows) != 45 for rows in actual_rows.values()):
            raise RuntimeError('The reused705X chains are not complete45-event chains')
        details, checkpoints, checks = {}, {}, {'source_v44_complete': True}
        for name in protocol['checkpoints']:
            old = source['checkpoints'][name]
            bank = {seed: {(chain['line_id'], chain['arm']): chain
                           for chain in old['seeds'][str(seed)]['chains']} for seed in seeds}
            cached_pairs = {}
            for line in protocol['cached_context_lines']:
                cached_pairs[line] = {}
                for seed in seeds:
                    low, high = [decompose_components(bank[seed][(line, arm)]['rows'],
                        bank[seed][(line, arm)]['c']) for arm in ('low', 'high')]
                    cached_pairs[line][str(seed)] = dict(low=low, high=high, pair=compare_pair(low, high))
            path = (read(model_root/'behavior_snapshot.json')['resident_checkpoint'] if name == 'actual_post24'
                    else model_root/'learning/post456_model.pt')
            learner, payload = _loaded(path)
            update, slots = (24, 15) if name == 'actual_post24' else (456, 285)
            if learner.total_updates != 50000+update or learner.replay_buffer.target_received != slots:
                raise RuntimeError('Frozen source checkpoint budget differs')
            if not (learner.config.ensemble_size == 2 and learner.config.target_subset_size == 2
                    and not learner.config.target_subset_replacement):
                raise RuntimeError('The declared two-of-two head-order control does not match the resident configuration')
            before, rng, adam = frozen_state(learner), _rng(), optimizer_versions(learner)
            factor_rows = {f'{actor}/{head}': {arm: [] for arm in ('low', 'high')}
                           for actor in seeds for head in seeds}
            target_config = None
            for event in range(45):
                for arm in ('low', 'high'):
                    rows = actual_rows[arm]
                    diagonal = {str(seed): bank[seed][('705X', arm)]['rows'][event] for seed in seeds}
                    value = factorize_row(learner, rows[event], rows[event+1] if event < 44 else None,
                                          seeds=seeds, cached_diagonal=diagonal)
                    if not value['integrity']['all_pass'] or not value['diagonal_exact']:
                        raise RuntimeError('Read-only factor/native parity failed')
                    if target_config is None:
                        target_config = value['target_config']
                    elif target_config != value['target_config']:
                        raise RuntimeError('Target factor configuration differs across the fixed event stream')
                    for key, row in value['cells'].items():
                        factor_rows[key][arm].append(row)
                if event in (0, 8, 16, 24, 44):
                    print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase='v45_frozen_factor_scoring',
                        scenario=scenario, checkpoint=name, event=event, total=45, cells=9)), flush=True)
            factor_pairs = {}
            for key, by_arm in factor_rows.items():
                low, high = [decompose_components(by_arm[arm], bank[seeds[0]][('705X', arm)]['c'])
                             for arm in ('low', 'high')]
                factor_pairs[key] = dict(low=low, high=high, pair=compare_pair(low, high))
            proof = unchanged(learner, before, rng)
            if not _equal(adam, optimizer_versions(learner)):
                raise RuntimeError('Read-only factor scoring changed Adam tensors')
            small = checkpoint_small(cached_pairs, factor_pairs, seeds)
            residuals = [abs(item[arm]['reconstruction_residual']) for item in factor_pairs.values()
                         for arm in ('low', 'high')]
            checks[name+'_cached_diagonals_exact'] = small['native_diagonal_pair_deltas_exact']
            checks[name+'_component_closure'] = max(residuals) < 1e-8
            checks[name+'_models_adam_rng_unchanged'] = proof['all_pass']
            details[name] = dict(cached_pairs=cached_pairs, factor_rows=factor_rows, factor_pairs=factor_pairs)
            checkpoints[name] = dict(proof=proof, admitted_rows=slots, source_updates=update,
                replay_modified=False, target_config=target_config,
                max_component_residual=max(residuals), **small)
            del learner, payload
        write(out/'component_detail.json', details)
        all_pass = all(checks.values())
        value = dict(status='complete' if all_pass else 'failed', scenario=scenario, priority='high',
            integrity=dict(all_pass=all_pass, checks=checks), source_budget=source['source_budget'],
            source_contract=source['source_contract'], checkpoints=checkpoints,
            schema=dict(components=list(COMPONENTS), cached_context_columns=CACHE_COLUMNS,
                crossed_cell_columns=CELL_COLUMNS, event_columns=EVENT_COLUMNS,
                bin_columns=['name', 'events', 'total', *COMPONENTS],
                high_minus_low='1.875s minus0s; positive target favors1.875s',
                head_factor='native two-of-two member set is fixed; order-only numerical control',
                actor_factor='continuous positive rsample plus its entropy; gate mixture analytic',
                factor_scope='3x3 declared scoring factors, not independent states or population variance',
                event_scope='fixed full45 real events; ordinal attribution, not common-clock/300s cost'),
            diagnostic_budget=dict(new_training_updates=0, new_feedback_slots=0,
                new_native_branches=0, new_native_seconds=0, checkpoints_downloaded=0,
                resident_checkpoints_loaded=2),
            wall_seconds=time.time()-started,
            decision=dict(training_started=False, target_change_installed=False,
                next_gate='Use component and factor attribution to name one target-change candidate; no automatic training or collection.'))
        size = write(out/'summary.json', compact(value), protocol['summary_max_bytes_per_case'])
        print(json.dumps(dict(status=value['status'], integrity=value['integrity'], bytes=size,
                              wall_seconds=value['wall_seconds'])), flush=True)
        if not all_pass:
            raise RuntimeError('Component diagnostic failed its implementation gate')
    except Exception as error:
        write(out/'failure.json', dict(status='failed', type=type(error).__name__, message=str(error),
            new_training_updates=0, new_native_branches=0, elapsed_seconds=time.time()-started))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', required=True)
    run(parser.parse_args().scenario)
