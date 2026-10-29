"""Descriptive feedback timing on resident derived rows; no estimation."""
from collections import Counter
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def audit(rows):
    suffix = sorted((r for r in rows if r['selection']['window_start'] > 9000),
                    key=lambda r: (r['selection']['window_start'], r['selection']['sequence']))
    events = {
        'first_positive_feedback': lambda r: r['target']['value'] == 1,
        'first_nonzero_count_feedback': lambda r: r['inputs'][0] > 0,
        'first_nonzero_positive_feedback': lambda r: r['inputs'][0] > 0 and r['target']['value'] == 1,
    }
    milestones = {}
    for name, predicate in events.items():
        matching = [r for r in rows if predicate(r)]
        deadline = min((r['selection']['packet_deadline'] for r in matching), default=None)
        later = [r for r in suffix if deadline is not None and deadline < r['selection']['window_start']]
        milestones[name] = dict(packet_deadline=deadline,
            first_usable_issue=None if not later else later[0]['selection']['window_start'],
            later_roots=len(later), later_nonzero_count=sum(r['inputs'][0] > 0 for r in later),
            later_positives=sum(r['target']['value'] == 1 for r in later),
            later_nonzero_positives=sum(r['inputs'][0] > 0 and r['target']['value'] == 1 for r in later))
    trace = []
    for r in suffix:
        issue = r['selection']['window_start']
        past = [a for a in rows if a['selection']['packet_deadline'] < issue]
        four = Counter(('n>0' if a['inputs'][0] > 0 else 'n=0') + ',Y=' + str(a['target']['value']) for a in past)
        trace.append(dict(sequence=r['selection']['sequence'], issue=issue,
            available_count=len(past), available_cells=dict(four),
            current_n=r['inputs'][0], current_y=r['target']['value']))
    last_issue = max(r['selection']['window_start'] for r in suffix)
    unused = [r for r in suffix if r['selection']['packet_deadline'] >= last_issue]
    result = dict(milestones=milestones, suffix_roots=len(suffix), last_issue=last_issue,
        roots_before_any_positive_feedback=sum(not any(k.endswith('Y=1') and v for k,v in t['available_cells'].items()) for t in trace),
        unused_feedback_slots=len(unused), unused_positive_labels=sum(r['target']['value'] == 1 for r in unused),
        unused_nonzero_positive_labels=sum(r['inputs'][0] > 0 and r['target']['value'] == 1 for r in unused))
    return result, trace


if __name__ == '__main__':
    summaries = {}
    for scene in ('stable', 'local_speed_drop'):
        base = HERE/'runs'/scene
        with (base/'derived_records.jsonl').open() as stream:
            rows = [json.loads(line) for line in stream]
        summaries[scene], trace = audit(rows)
        with (base/'feedback_support_trace.jsonl').open('x') as stream:
            for row in trace:
                stream.write(json.dumps(row,separators=(',', ':'))+'\n')
    result = dict(cases=summaries, new_native_seconds=0, fits=0,
        interpretation='Later true classes are retrospective opportunity counts only; never inputs to a predictor or a selection rule. v126 remains HOLD.')
    (HERE/'support_timing.json').write_text(json.dumps(result,indent=2)+'\n')
    raw = json.dumps(result,separators=(',', ':'))
    assert len(raw.encode())+1 <= 5000
    print(raw)
