"""Join existing native issue logs by record identity, never by state value."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from experiments.mcwm_closed_loop_v1.stream import (
    EI, TransitMDPConfigV7, transition_reward_v7)


def read_jsonl(path):
    with Path(path).open() as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def context_vector(context):
    if not context['observed']:
        return np.array([np.nan, 0.], np.float32)
    value = float(context['z'])
    if not np.isfinite(value):
        raise ValueError('Observed native context is not finite')
    return np.array([value, 1.], np.float32)


def _check_binding(issue, context):
    binding, clocks = context['binding'], context['clocks']
    for key in ('sequence','episode_id','trip_id','vehicle_id','line_id','station_id'):
        if binding[key] != issue[key]:
            raise ValueError('Native context identity differs at '+key)
    now = float(issue['started_at'])
    if not (binding['issue_clock'] == clocks['issue'] == clocks['obs_start'] ==
            clocks['obs_end'] == issue['issue_ledger']['time_seconds'] == now):
        raise ValueError('Context and original issue clocks differ')
    if clocks['event'] != float(issue['event'][EI['sim_time_seconds']]):
        raise ValueError('Context event clock differs from original native event')
    if clocks['event'] > now:
        raise ValueError('Native event occurs after its issue context')


def chronological_outcome(issue, next_issue, selected, contract, *, delay_seconds=30.):
    """Recover an actual nonterminal transition from adjacent native records."""
    if (next_issue['sequence'] != issue['sequence']+1 or
            next_issue['episode_id'] != issue['episode_id']):
        raise ValueError('A factual successor must be the actual next native issue')
    if selected['ctx_s'] != issue['ctx_s'] or selected['ctx_ns'] != next_issue['ctx_s']:
        raise ValueError('Selected endpoint context differs from its same-record issue')
    end = float(next_issue['started_at'])
    dt = end-float(issue['started_at'])
    if dt < 0:
        raise ValueError('Native issue clock moves backwards')
    return dict(issue, ns=next_issue['s'].copy(), raw_ns=next_issue['raw_s'].copy(),
        r=float(transition_reward_v7(issue['issue_ledger'],next_issue['issue_ledger'],contract)),
        dt=dt, terminal=False, completed_at=end, available_at=end+delay_seconds,
        next_vehicle_id=next_issue['vehicle_id'], next_sequence=next_issue['sequence'],
        ctx_ns=deepcopy(selected['ctx_ns']), next_context=context_vector(selected['ctx_ns']))


def load_run(run_directory, contract, *, expected_packets=57, records_per_packet=5,
             delay_seconds=30.):
    """Read resident logs; return issues,285 covariates and fresh285 outcomes.

    Issue arrays are copies of the logged native fields. The factual successor
    is sequence+1, even when it is another vehicle or occurs at the same tick.
    The source experiment excludes terminal outcomes; no endpoint is invented.
    """
    root = Path(run_directory)
    summary = json.loads((root/'summary.json').read_text())
    cell = summary['cell']
    contexts = {}
    for entry in read_jsonl(root/'context_issues.jsonl'):
        seq = entry['sequence']
        if seq in contexts:
            raise ValueError('Duplicate native context sequence')
        contexts[seq] = entry['ctx_s']
    issues, by_sequence = [], {}
    for entry in read_jsonl(root/'prefix_trace.jsonl'):
        seq = entry['sequence']
        if seq != len(issues):
            raise ValueError('Native prefix has a missing or duplicate issue sequence')
        context = contexts.pop(seq)
        _check_binding(entry,context)
        issue = dict(entry)
        for key in ('s','raw_s','event','a','proposed_a'):
            issue[key] = np.asarray(entry[key],np.float32)
        if issue['s'].shape != (229,) or issue['a'].shape != (1,):
            raise ValueError('Native canonical state/action dimensions changed')
        issue.update(ctx_s=context,context=context_vector(context),
            arm=cell['arm'],seed=cell['seed'],scenario=cell['scenario'])
        issues.append(issue)
        by_sequence[seq] = issue
    if contexts:
        raise ValueError('Context log contains unpaired native issue records')
    covariates, outcomes, packet_count = [], [], 0
    receipts = summary['receipts']
    for entry in read_jsonl(root/'context_selected.jsonl'):
        packet_count += 1
        if entry['packet'] != packet_count:
            raise ValueError('Selected packet order changed')
        receipt = receipts[packet_count-1]
        cov = entry['covariates']
        expected_outcomes = records_per_packet if cell['arm'] == 'fresh' else 0
        if len(cov) != records_per_packet or len(entry['outcomes']) != expected_outcomes:
            raise ValueError('Selected information budget changed')
        if [row['sequence'] for row in cov] != receipt['covariate_sequence_ids']:
            raise ValueError('Covariate receipt identity differs from selected log')
        if cell['arm'] == 'fresh' and [row['sequence'] for row in entry['outcomes']] != receipt['outcome_sequence_ids']:
            raise ValueError('Outcome receipt identity differs from selected log')
        for selected in cov:
            issue = by_sequence[selected['sequence']]
            if selected['ctx_s'] != issue['ctx_s']:
                raise ValueError('Covariate context is not from its own issue')
            covariates.append(dict(issue,packet=packet_count,processed_at=entry['processed_at']))
        for selected in entry['outcomes']:
            seq = selected['sequence']
            row = chronological_outcome(by_sequence[seq],by_sequence[seq+1],selected,
                contract,delay_seconds=delay_seconds)
            if not (row['dt'] > 0 and receipt['deadline']-300 < row['available_at'] <=
                    receipt['deadline'] <= entry['processed_at']):
                raise ValueError('Selected outcome violates original physical-time availability')
            row.update(packet=packet_count,processed_at=entry['processed_at'])
            outcomes.append(row)
    expected = expected_packets*records_per_packet
    if packet_count != expected_packets or len(covariates) != expected or len({r['sequence'] for r in covariates}) != expected:
        raise ValueError('Covariate packets do not reproduce the original information budget')
    if len(outcomes) != (expected if cell['arm'] == 'fresh' else 0) or len({r['sequence'] for r in outcomes}) != len(outcomes):
        raise ValueError('Complete outcomes do not reproduce the original information budget')
    return dict(cell=cell,summary=summary,issues=issues,covariates=covariates,outcomes=outcomes,
        checks=dict(native_context_identity_exact=True,selected_contexts_exact=True,
            real_chronological_successors=True,record_budgets_exact=True,
            reward_reconstructed_from_original_ledger=True,no_terminal_successor_fabricated=True),
        counts=dict(issues=len(issues),covariates=len(covariates),outcomes=len(outcomes),
            cross_vehicle_outcomes=sum(r['vehicle_id'] != r['next_vehicle_id'] for r in outcomes),
            zero_duration_outcomes=sum(r['dt'] == 0 for r in outcomes)))


def exact_outcome_exposures(outcomes, seed, *, expected_packets=57,
                           updates_per_packet=8,draws_per_update=64):
    """Replay only v73's independent fifth sampler, without learner RNG access.

    The original code appends five outcomes then draws64 indices eight times
    from the latest min(30,admitted) records, using default_rng(seed+600000).
    Weights are exact realized draw counts, not estimated sampling rates.
    """
    if len(outcomes) != expected_packets*5:
        raise ValueError('Exposure reconstruction requires every admitted complete outcome')
    rng = np.random.default_rng(seed+600000)
    weights = np.zeros(len(outcomes),np.int64)
    calls = 0
    for packet in range(1,expected_packets+1):
        start,stop = (packet-1)*5,packet*5
        if any(row['packet'] != packet for row in outcomes[start:stop]):
            raise ValueError('Complete outcome admission order differs from the actual packets')
        window = min(30,stop)
        for _ in range(updates_per_packet):
            ids = rng.integers(window,size=draws_per_update)
            np.add.at(weights,stop-window+ids,1)
            calls += 1
    return dict(weights=weights, sequence_ids=[row['sequence'] for row in outcomes],
        sampler_seed=seed+600000,calls=calls,total_draws=int(weights.sum()),
        expected_draws=expected_packets*updates_per_packet*draws_per_update,
        final_rng_state=deepcopy(rng.bit_generator.state),
        scope='Exact independent private draw counts; not historical TD gradients')
