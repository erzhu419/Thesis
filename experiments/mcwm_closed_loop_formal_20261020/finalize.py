"""Wait for this finite 40-run campaign, then produce its compact comparison on the server."""

import json
from pathlib import Path
import time

from analysis import analyze_campaign

HERE=Path(__file__).resolve().parent


def main():
    p=json.loads((HERE/"protocol.json").read_text())
    expected=[]
    for seed in p["training_seeds"]:
        for scenario in p["scenarios"]:
            expected.extend(HERE/"runs/control"/scenario/arm/f"seed{seed}" for arm in p["arms"])
            expected.append(HERE/"runs/prediction"/scenario/"common"/f"seed{seed}")
    started=time.monotonic()
    while True:
        complete=sum((path/"summary.json").exists() for path in expected)
        failed=sum((path/"failure.json").exists() for path in expected)
        terminal=sum((path/"summary.json").exists() or (path/"failure.json").exists() for path in expected)
        print("SCHEDULEURM_PROGRESS "+json.dumps(dict(current=terminal,total=len(expected),unit="runs",
            elapsed_seconds=time.monotonic()-started,complete=complete,failed=failed)),flush=True)
        if terminal==len(expected) or time.monotonic()-started>=48*3600:
            result=analyze_campaign(HERE/"runs",expected_seeds=len(p["training_seeds"]))
            (HERE/"campaign_analysis.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
            print(json.dumps(dict(stage="comparison_written",status=result["status"],
                complete_control_cells=result["complete_control_cells"],
                complete_prediction_cells=result["complete_prediction_cells"])),flush=True)
            return
        time.sleep(60)


if __name__=="__main__":
    main()
