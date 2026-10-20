"""One-off continuation of this queued v4 batch; run locally after review.

Only resident metadata crosses SSH until the terminal compact result is ready.
An existing submission receipt resumes that phase; this script never resubmits
the preflights, changes training, retries failed runs or starts a formal batch.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import time

import infra
import launch

HERE = Path(__file__).resolve().parent
RESULT_NAME = "head_diagnostic_summary.json"
MAX_RESULT_BYTES = 1_000_000


def log(stage, **values):
    print(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(), stage=stage, **values),
                     ensure_ascii=False, allow_nan=False), flush=True)


def resident_metadata(*, preflights=True):
    # All labels and logs stay resident. Only gate booleans, counts and final
    # result metadata are printed; no passenger cost number is returned here.
    script = r'''
import json, math
from pathlib import Path
here=Path(__HERE__)
base=here.parent
rows=[]
paired_metrics=('mean_delta','mean_absolute_delta','max_absolute_delta','mean','mean_absolute')
variants={'full_base','omit_logdt','omit_logcost','omit_dWaiting','omit_dOnboard'}
for arm,seed in ([('action_local',6201),('action_model',6203)] if __PREFLIGHTS__ else []):
 p=here/'preflight/control/local_speed_drop'/arm/('seed%d'%seed)
 row=dict(arm=arm,seed=seed,directory_exists=p.is_dir(),summary=(p/'summary.json').is_file(),failure=(p/'failure.json').is_file())
 if row['failure']:
  row['failure_detail']=json.loads((p/'failure.json').read_text())
 if row['summary']:
  s=json.loads((p/'summary.json').read_text())
  rp=json.loads((p/'protocol.json').read_text());pp=rp.get('protocol',{})
  world=s.get('worlds',{}).get(arm,{})
  heads=world.get('head_diagnostics',{})
  receipts=s.get('feedback_receipts',[])
  seq=[value for receipt in receipts for value in receipt.get('sequence_ids',[])]
  logs=[json.loads(line) for line in (p/'td_diagnostics.jsonl').read_text().splitlines() if line.strip()]
  probes=[value.get('head_counterfactual') or value.get('diagnostic',{}).get('head_counterfactual') for value in logs]
  probes_complete=len(probes)==10 and all(isinstance(probe,dict) and set(probe.get('variants',{}))==variants and probe.get('rows',0)>0 for probe in probes)
  applied=0;max_delta=0.
  if probes_complete:
   for probe in probes:
    for variant in probe['variants'].values():
     applied+=variant.get('applied_rows',0)
     for metric in variant.values():
      if isinstance(metric,dict):
       for key in paired_metrics:
        if key in metric:max_delta=max(max_delta,abs(float(metric[key])))
  gates=dict(
   completed=s.get('status')=='complete' and s.get('preflight') is True,
   native_runtime=s.get('sumo_version')=='SUMO 1.25.0',
   identity=s.get('seed')==seed and s.get('scenario')=='local_speed_drop' and s.get('arm')==arm,
   horizon=4200<=s.get('target_time_seconds',-1)<24000 and s.get('full_horizon_completed') is False,
   budgets=s.get('target_online_training_events')==50 and s.get('packets')==10 and s.get('policy_updates')==10,
   initializer=rp.get('initial_updates')==50000 and rp.get('history_rows')==32768 and rp.get('source_rows')==4096,
   preflight_fit=pp.get('model_init_updates')==2 and pp.get('model_refit_updates')==1 and pp.get('policy_updates_per_packet')==1,
   shock=s.get('scenario_metadata',{}).get('actual_shock_at_seconds')==1350,
   global_feedback=[r.get('deadline') for r in receipts]==list(range(1200,3901,300)) and len(seq)==len(set(seq))==50 and all(r.get('records')==5 and r.get('max_available_at',math.inf)<=r.get('deadline',-1)<=r.get('processed_at',-1) for r in receipts),
   head_labels=heads.get('head_order')==['logdt','logcost','dWaiting','dOnboard'] and heads.get('overall',{}).get('n')==50 and sum(value.get('n',0) for value in heads.get('by_phase',{}).values())==50,
   local_entry=s.get('physical_learning_enters_policy')==(arm=='action_local') and world.get('local_adapter',{}).get('enabled')==(arm=='action_local'),
   matched_probes=probes_complete,
   probe_effect=(applied>0 and max_delta>0) if arm=='action_local' else (applied==0 and max_delta==0),
  )
  if arm=='action_local':
   old=base/'mcwm_closed_loop_action_local_v3_20261020/preflight/control/local_speed_drop/action_local/seed6201'
   gates['v3_reference_exists']=(old/'summary.json').is_file() and (old/'protocol.json').is_file()
   if gates['v3_reference_exists']:
    v=json.loads((old/'summary.json').read_text());vp=json.loads((old/'protocol.json').read_text());op=vp.get('protocol',{})
    keys=('seed','scenario','arm','target_time_seconds','target_online_training_events','policy_updates','packets')
    pkeys=('target_seed_offset','model_init_updates','model_refit_updates','policy_updates_per_packet','feedback_first_deadline','feedback_last_deadline','feedback_interval_seconds','feedback_records_per_packet','feedback_delay_seconds','shock_at_seconds','speed_factor','model_td_fraction')
    rkeys=('initial_updates','history_rows','source_rows','initializer')
    gates['v3_same_configuration']=all(k in s and k in v and s[k]==v[k] for k in keys) and all(k in pp and k in op and pp[k]==op[k] for k in pkeys) and all(k in rp and k in vp and rp[k]==vp[k] for k in rkeys)
    gates['v3_exact_cost_reproduction']=s.get('generalized_passenger_seconds')==v.get('generalized_passenger_seconds')
  row.update(gates=gates,head_labels=heads.get('overall',{}).get('n'),probe_packets=len(probes),
    probe_applied_rows=applied,probe_any_delta_nonzero=max_delta>0,all_gates_pass=all(gates.values()))
 rows.append(row)
result=here/'head_diagnostic_summary.json'
out=dict(preflights=rows,result_exists=result.is_file())
if result.is_file():
 value=json.loads(result.read_text());out['result_metadata']=dict(status=value.get('status'),bytes=result.stat().st_size,
  errors=len(value.get('integrity',{}).get('errors',[])),complete_new_runs=value.get('integrity',{}).get('complete_new_runs'))
print(json.dumps(out,allow_nan=False))
'''.replace("__HERE__", repr(infra.remote(HERE))).replace("__PREFLIGHTS__", repr(preflights))
    _, output, _ = infra.scheduler.run_on("node001", shlex.join([infra.PYTHON, "-c", script]), timeout=45, check=True)
    return json.loads(output)


def wait_preflights():
    if not (HERE / "preflight_submission.json").is_file():
        raise RuntimeError("The two native preflights have not been submitted")
    previous = None
    while True:
        metadata = resident_metadata()
        states = metadata["preflights"]
        if states != previous:
            log("preflight_metadata", runs=states)
            previous = states
        if any(row["failure"] for row in states):
            raise RuntimeError("Native preflight failed; no continuation submitted")
        if all(row["summary"] for row in states):
            infra.write(HERE / "native_preflight_gate.json", metadata)
            if not all(row.get("all_gates_pass") for row in states):
                raise RuntimeError("Native preflight gate mismatch; inspect native_preflight_gate.json")
            return
        time.sleep(60)


def wait_result():
    while True:
        metadata = resident_metadata(preflights=False)
        if metadata["result_exists"]:
            result = metadata["result_metadata"]
            log("resident_final_result", **result)
            if result["status"] != "complete" or result["errors"] or result["complete_new_runs"] != 20:
                raise RuntimeError("Resident diagnosis is incomplete; no result pulled")
            if result["bytes"] > MAX_RESULT_BYTES:
                raise RuntimeError("Compact result exceeds1MB; reduce resident summary before collection")
            local = HERE / RESULT_NAME
            if local.exists():
                log("result_already_collected", bytes=local.stat().st_size)
            else:
                infra.collect_json(RESULT_NAME)
                log("compact_result_collected", bytes=local.stat().st_size)
            return
        time.sleep(60)


def main():
    if not (HERE / "campaign_submission.json").exists():
        wait_preflights()
        result = launch.submit_campaign()
        log("campaign_submitted", ids=[row["id"] for row in result["submitted"]])
    else:
        log("campaign_receipt_exists")
    if not (HERE / "finalize_submission.json").exists():
        result = launch.submit_finalizer()
        log("finalizer_submitted", ids=[row["id"] for row in result["submitted"]])
    else:
        log("finalizer_receipt_exists")
    wait_result()
    log("one_off_continuation_complete")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        log("stopped", error_type=type(error).__name__, error=str(error))
        raise
