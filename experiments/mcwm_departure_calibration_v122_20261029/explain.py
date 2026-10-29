"""Descriptive decomposition of completed v122 errors; no new fit or selection."""
import argparse
import json
from pathlib import Path
from statistics import mean


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def explain(source, runs):
    result = {}
    for scene in ('stable', 'local_speed_drop'):
        records = {r['selection']['sequence']:r for r in rows(source/scene/'selected_records.jsonl')}
        errors = rows(runs/scene/'errors.jsonl')
        groups, exceptions, positives = {}, [], []
        for name, exact in (('clock_exact', True), ('clock_not_exact', False)):
            subset = [r for r in errors if (r['absolute_errors']['clock_reference']==0)==exact]
            groups[name] = dict(records=len(subset),mae_seconds={key:mean(r['absolute_errors'][key] for r in subset)
                for key in ('fresh','frozen','clock_reference')},
                mean_fresh_minus_frozen=mean(r['absolute_errors']['fresh']-r['absolute_errors']['frozen'] for r in subset))
        for row in errors:
            original = records[row['sequence']]
            labels = original['labels']
            item = dict(sequence=row['sequence'],issue_time=row['issue_time'],
                actual_minus_clock=-row['signed_errors']['clock_reference'],
                hold_seconds=original['action']['hold_seconds'],
                post_issue_boarded=labels['receiver_post_issue_boarded_count']['value'],
                cohort_boarded=labels['cohort_boarded_count']['value'])
            if row['absolute_errors']['clock_reference'] != 0:
                exceptions.append(item)
            if item['cohort_boarded'] > 0:
                positives.append(item)
        result[scene] = dict(groups=groups,exceptions=exceptions,connection_positives=positives,
            exception_extra_seconds=sum(r['actual_minus_clock'] for r in exceptions),
            exception_with_boarding=sum(r['post_issue_boarded']>0 for r in exceptions),
            explanation_scope='Association on the fixed evaluation roots; not causal service-delay attribution.')
    return dict(cases=result,new_native_seconds=0,optimizer_updates=0,additional_calibration=0)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--runs',type=Path,required=True)
    args=parser.parse_args()
    result=explain(args.source,args.runs)
    payload=json.dumps(result,ensure_ascii=False,separators=(',',':'),allow_nan=False)
    if len(payload.encode())>6000:
        raise ValueError('Decomposition summary exceeds6KB')
    with (args.runs/'explanation.json').open('x') as handle:
        handle.write(payload+'\n')
    print(payload)
