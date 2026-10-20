"""Launch the isolated matched campaign and collect only compact server summaries."""

import argparse
import json
from pathlib import Path
import shlex

import infra

HERE=Path(__file__).resolve().parent


def protocol():
    return json.loads((HERE/"protocol.json").read_text())


def make_job(mode,arm,seed,scenario,*,preflight=False):
    p=protocol()
    label=f"{mode}/{scenario}/{arm}/seed{seed}"
    output=HERE/("preflight" if preflight else "runs")/label
    argv=["--mode",mode,"--arm",arm,"--seed",str(seed),"--scenario",scenario,
          "--out",infra.remote(output)]
    if preflight:
        argv.append("--preflight")
    return infra.task(p,("preflight/" if preflight else "")+label,argv)


def submit_preflight():
    prepared=json.loads((HERE/"inputs/prepared.json").read_text())
    if prepared["status"]!="complete":
        raise RuntimeError("Shared inputs are not ready")
    tasks=[make_job("control","context",6201,"local_speed_drop",preflight=True),
           make_job("prediction","common",6201,"local_speed_drop",preflight=True)]
    for row,node in zip(tasks,("node001","node004")):
        row["preferred_node"]=node
    return infra.submit(tasks,"preflight")


def submit_campaign():
    for mode,arm in (("control","context"),("prediction","common")):
        path=HERE/"preflight"/mode/"local_speed_drop"/arm/"seed6201/summary.json"
        summary=json.loads(path.read_text())
        if summary["status"]!="complete" or summary["target_online_training_events"]!=15:
            raise RuntimeError("Native SUMO preflight did not close its data budget")
        if summary["scenario_metadata"]["actual_shock_at_seconds"]!=1350:
            raise RuntimeError("Native clock intervention did not execute")
        if summary["sumo_version"]!="SUMO 1.25.0":
            raise RuntimeError("Preflight used a different target runtime")
    tasks=[]
    p=protocol()
    for seed in p["training_seeds"]:
        for scenario in p["scenarios"]:
            for arm in p["arms"]:
                tasks.append(make_job("control",arm,seed,scenario))
            tasks.append(make_job("prediction","common",seed,scenario))
    return infra.submit(tasks,"campaign")


def collect():
    command=shlex.join([infra.PYTHON,infra.remote(HERE/"analysis.py"),infra.remote(HERE/"runs"),
        "--expected-seeds","5","--out",infra.remote(HERE/"campaign_analysis.json")])
    result=infra.scheduler.run_on("node001",command,timeout=60,check=True)
    analysis=infra.collect_json("campaign_analysis.json")
    return {key:analysis[key] for key in ("status","complete_control_cells","complete_prediction_cells")}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation",choices=("stage","preflight","submit","status","collect"))
    parser.add_argument("--batch",choices=("prepare","preflight","campaign","finalize"),default="campaign")
    args=parser.parse_args()
    if args.operation=="stage":
        files=[path for path in HERE.iterdir() if path.suffix in (".py",".json")
               and not path.name.startswith(("test_","prepare_","preflight_","campaign_"))]
        result=infra.stage(files)
    elif args.operation=="preflight":
        result=submit_preflight()
    elif args.operation=="submit":
        result=submit_campaign()
    elif args.operation=="status":
        submitted=json.loads((HERE/f"{args.batch}_submission.json").read_text())
        result=infra.status([row["id"] for row in submitted["submitted"]])
    else:
        result=collect()
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
