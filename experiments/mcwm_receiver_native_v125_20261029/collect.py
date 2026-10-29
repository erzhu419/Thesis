"""Wait for the authorized prefixes; collect only their compact summaries."""
import json
import time
from infra import HERE, poll, write


def collect():
    downloaded=0
    polls=0
    previous=None
    while True:
        value,size=poll()
        downloaded+=size
        polls+=1
        if value['all_summaries_present']:
            cases=value['cases']
            technical=all(r['status']=='complete' and r['integrity']['all_pass'] for r in cases.values())
            amounts=[r['budget']['native_seconds'] for r in cases.values()]
            result=dict(status='complete' if technical else 'technical_hold',all_pass=technical,cases=cases,
                budget=dict(native_seconds=sum(amounts) if all(v is not None for v in amounts) else None,
                    known_native_seconds=sum(v for v in amounts if v is not None),
                    native_seconds_is_lower_bound=any(r['budget']['native_seconds_is_lower_bound'] for r in cases.values()),
                    optimizer_updates=0,new_predictor_fits=0,new_policy_actions=0,feedback_admitted=0),
                performance_evidence=False,automatic_followup=False)
            walls=[r['execution_wall'] for r in cases.values()]
            if all(w['native_started_wall'] is not None and w['native_finished_wall'] is not None for w in walls):
                result['execution']=dict(actual_overlap_seconds=max(0,min(w['native_finished_wall'] for w in walls)
                    -max(w['native_started_wall'] for w in walls)),
                    total_native_wall_span_seconds=max(w['native_finished_wall'] for w in walls)
                    -min(w['native_started_wall'] for w in walls))
            write(HERE/'summary.json',result)
            write(HERE/'collection_receipt.json',dict(downloaded_stdout_bytes=downloaded,polls=polls,
                full_logs_downloaded=False,checkpoints_downloaded=False))
            print(json.dumps(dict(status=result['status'],budget=result['budget'],
                execution=result.get('execution'),summary=str(HERE/'summary.json'))),flush=True)
            return
        if value['statuses']!=previous:
            print(json.dumps(dict(finished=value['finished'],statuses=value['statuses'])),flush=True)
            previous=value['statuses']
        time.sleep(35)


if __name__=='__main__':
    try:
        collect()
    except Exception as error:
        write(HERE/'collection_error.json',dict(type=type(error).__name__,message=str(error)))
        raise
