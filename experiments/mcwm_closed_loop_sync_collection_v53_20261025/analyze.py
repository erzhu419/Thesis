"""Reduce native results on the server; leave all trajectories and weights there."""
from pathlib import Path
import json
import math

HERE=Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def common_prefix():
    paths=[HERE/'runs'/case/'prefix_trace.jsonl' for case in ('stable','local_speed_drop')]
    if not all(path.exists() for path in paths):
        return dict(status='unavailable',exact=None)
    count=0
    with paths[0].open() as left,paths[1].open() as right:
        while True:
            a,b=left.readline(),right.readline()
            if not a or not b:
                return dict(status='short',exact=False,rows=count)
            a,b=json.loads(a),json.loads(b)
            if a['started_at']>=9000 or b['started_at']>=9000:
                return dict(status='exact' if a['started_at']==b['started_at'] else 'different_boundary',
                    exact=a['started_at']==b['started_at'],rows=count)
            if a!=b:
                return dict(status='different',exact=False,rows=count,sequence=a['sequence'])
            count+=1


def persisted_logs(root):
    def rows(name):
        with (root/name).open() as handle:
            return [json.loads(line) for line in handle if line.strip()]
    feedback=rows('continuous_feedback.jsonl')
    support=rows('issue_support.jsonl')
    applications=rows('native_applications.jsonl')
    checks=dict(feedback271=len(feedback)==271,
        unique_feedback=len({row['transition']['feedback_id'] for row in feedback})==271,
        support272_including_real_endpoint=len(support)==272,
        all_actual_issue_checks=all(row['all_pass'] for row in support),
        native_applications271=len(applications)==271,
        all_native_application_checks=all(row['all_pass'] for row in applications))
    return dict(all_pass=all(checks.values()),checks=checks,
        feedback_rows=len(feedback),issue_rows=len(support),applications=len(applications),
        c_product=math.prod(row['ratio']['c'] for row in support[:271]),
        maximum_hold_error=max(abs(row['actual_hold']-row['expected_hold']) for row in applications))


def main():
    cases={}
    for case in ('stable','local_speed_drop'):
        root=HERE/'runs'/case
        if (root/'summary.json').exists():
            result=read(root/'summary.json')
            cases[case]={key:result[key] for key in ('status','integrity','initializer_seed',
                'initializer_updates','external_seed','private_action_seed','snapshot_clock',
                'scenario_actual_shock_at','coverage','overlap','budget','execution_wall')}
            post24=result['post24_preflight'];first=result['native_preflight']
            cases[case]['preflight']=dict(reused=first['reused_in_collection'],
                duplicate_prefixes=first['duplicate_prefixes'],all_pass=first['all_pass'],
                post24_checks=post24['checks'],post24_wiring=post24['wiring'],
                post24_M1=post24['m1'],post24_M8=post24['m8'],
                first_real_row_checks=first['real_first_response']['checks'])
            cases[case]['persisted_logs']=persisted_logs(root)
        elif (root/'failure.json').exists():
            cases[case]=read(root/'failure.json')
        else:
            cases[case]=dict(status='not_finished')
    complete=all(row['status']=='complete' and row['persisted_logs']['all_pass'] for row in cases.values())
    parallel=None
    if complete:
        intervals=[row['execution_wall'] for row in cases.values()]
        overlap=max(0.,min(row['native_finished_wall'] for row in intervals)-
            max(row['native_started_wall'] for row in intervals))
        parallel=dict(actual_native_overlap_seconds=overlap,actual_parallel=overlap>0)
    result=dict(status='complete' if complete else 'incomplete_or_failed',cases=cases,
        execution_parallel=parallel,
        common_before_shock=common_prefix() if complete else dict(status='not_compared',exact=None),
        source_contract=dict(initializer_seed=6202,initializer_updates=50000,feedback_slots=285,
            optimizer_updates_contract=456,actual_updates_this_stage=24,
            unused_response_updates=432,horizon_seconds=24000,F_unchanged=True,trust_unchanged=True),
        scientific_scope='Native synchronized-evaluator support and acquisition only; no response learning or control improvement claim')
    encoded=json.dumps(result,separators=(',',':'),allow_nan=False)+'\n'
    if len(encoded.encode())>24000: raise RuntimeError('Reduce result before downloading')
    with (HERE/'summary.json').open('x') as handle: handle.write(encoded)
    print(json.dumps(dict(status=result['status'],bytes=len(encoded.encode()))),flush=True)


if __name__=='__main__': main()
