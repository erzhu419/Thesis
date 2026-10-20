"""Add the prespecified post-update cost window to one resident pilot summary.

Declared before reading any full-day outcomes. Final feedback deadline18000
can be processed a few seconds later; use18300 and verify the final update
precedes that endpoint. This observes actual ledger increments through24000. It neither
reruns training nor overwrites the first finalizer result.
"""
import json
from pathlib import Path
from analysis import CONTRASTS, difference, paired

HERE = Path(__file__).resolve().parent


def window_cost(rows, start=18300., stop=24000.):
    costs = {float(row['time_seconds']): float(row['accrued_generalized_passenger_seconds'])
             for row in rows}
    missing = [time for time in (start, stop) if time not in costs]
    if missing:
        return dict(status='missing_endpoint', missing_times=missing,
                    generalized_passenger_seconds=None)
    value = costs[stop]-costs[start]
    if value < 0:
        return dict(status='invalid_ledger_increment', generalized_passenger_seconds=None)
    return dict(status='complete', start_seconds=start, stop_seconds=stop,
                generalized_passenger_seconds=value)


def supplement(root=HERE):
    root = Path(root)
    output = root/'pilot_summary_with_holdout.json'
    if output.exists():
        raise FileExistsError(output)
    result = json.loads((root/'pilot_summary.json').read_text())
    protocol = json.loads((root/'protocol.json').read_text())
    values, cells, missing = {}, [], []
    for cell in result['control_cells']:
        key = cell['seed'], cell['scenario'], cell['arm']
        path = root/'runs/control'/cell['scenario']/cell['arm']/f"seed{cell['seed']}"/'metrics.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        run = json.loads((path.parent/'summary.json').read_text())
        last_update = max(receipt['processed_at'] for receipt in run['feedback_receipts'])
        cost = window_cost(rows)
        if last_update > 18300:
            cost = dict(status='updates_overlap_window', generalized_passenger_seconds=None)
        seconds = (cost['generalized_passenger_seconds']/cell['departed']
                   if cost['status'] == 'complete' else None)
        values[key] = seconds
        cells.append(dict(seed=key[0], scenario=key[1], arm=key[2], last_policy_update_at_seconds=last_update,
            seconds_per_full_day_departed_passenger=seconds, **cost))
        if cost['status'] != 'complete':
            missing.append(dict(seed=key[0], scenario=key[1], arm=key[2], **cost))
    effects = [dict(candidate=candidate, reference=reference, scenario=scenario,
        **paired([difference(values.get((seed, scenario, candidate)),
                             values.get((seed, scenario, reference)))
                  for seed in protocol['training_seeds']]))
        for candidate, reference in CONTRASTS for scenario in protocol['scenarios']]
    result['post_training_holdout'] = dict(
        status='complete' if len(cells) == 16 and not missing else 'incomplete',
        window_seconds=[18300,24000], cells=cells, paired_effects=effects, missing=missing,
        semantics='Actual accrued2wait+ride increment afterpolicyupdatesstop. Normalizeonlybyfull-daydepartedpassengercohort; includesunfinishedpassengers. Twopaireddevelopmentseeds, noformalCI.')
    output.write_text(json.dumps(result,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
    return dict(status=result['status'],holdout_status=result['post_training_holdout']['status'],
                bytes=output.stat().st_size, path=str(output))


if __name__ == '__main__':
    print(json.dumps(supplement()))
