"""Server-only closure: compare the common prefix and return a bounded receipt."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))


def read_records(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def collect():
    final = HERE/'summary.json'
    if final.exists():
        return json.loads(final.read_text())
    protocol = json.loads((HERE/'protocol.json').read_text())
    scenes = [c['scenario'] for c in protocol['cases']]
    cases, progress = {}, {}
    for scene in scenes:
        base = HERE/'runs'/scene
        if (base/'summary.json').exists():
            cases[scene] = json.loads((base/'summary.json').read_text())
        elif (base/'progress.json').exists():
            progress[scene] = json.loads((base/'progress.json').read_text())
    if len(cases) != len(scenes):
        return dict(all_summaries_present=False, finished=len(cases),
                    statuses={s:cases[s]['status'] if s in cases else None for s in scenes},
                    progress=progress)
    technical = all(c['status']=='complete' and c['integrity']['all_pass'] for c in cases.values())
    shared = None
    if technical:
        from analyze import compare_prefixes
        records = {s:read_records(ROOT/protocol['selected_root']/s/'selected_records.jsonl') for s in scenes}
        shared = compare_prefixes(records['stable'], records['local_speed_drop'],
            HERE/'runs/stable/captures', HERE/'runs/local_speed_drop/captures')
        technical = shared['all_pass']
    values = [c['budget']['native_seconds'] for c in cases.values()]
    result = dict(all_summaries_present=True, finished=len(cases),
        status='complete' if technical else 'technical_hold', technical_pass=technical,
        cases=cases, shared_prefix=shared,
        budget=dict(native_seconds=sum(values) if all(v is not None for v in values) else None,
            native_seconds_is_lower_bound=any(c['budget']['native_seconds_is_lower_bound'] for c in cases.values()),
            optimizer_updates=0, new_predictor_fits=0, new_policy_actions=0, feedback_admitted=0),
        automatic_followup=False, performance_evidence=False)
    observations = [c.get('observation', {}) for c in cases.values()]
    reasons = []
    if not technical:
        reasons.append('replay_or_shared_prefix_not_qualified')
    for key in ('field_coverage_complete', 'labels_complete', 'prefix_has_both_classes', 'passivity_all_pass'):
        if not all(o.get(key) is True for o in observations):
            reasons.append(key)
    result['learning_admission'] = dict(status='READY_FOR_SEPARATE_FACTUAL_STUDY' if not reasons else 'HOLD',
        reasons=reasons, fitting_launched=False)
    walls = [c['execution_wall'] for c in cases.values()]
    if all(w['native_started_wall'] is not None and w['native_finished_wall'] is not None for w in walls):
        result['execution'] = dict(actual_overlap_seconds=max(0,
            min(w['native_finished_wall'] for w in walls)-max(w['native_started_wall'] for w in walls)),
            total_native_wall_span_seconds=max(w['native_finished_wall'] for w in walls)-
                min(w['native_started_wall'] for w in walls))
    final.write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__ == '__main__':
    raw = json.dumps(collect(),separators=(',', ':'),allow_nan=False)
    if len(raw.encode())+1 > 16000:
        raise ValueError('Summary exceeds the fixed16000-byte receipt bound')
    print(raw)
