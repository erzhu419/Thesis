"""Export already-trained policies before moving their evaluation to CPUs."""

import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
names = ["blocks2_calibrated_offline", "blocks2_calibrated_h2o",
         "blocks2_calibrated_mcwm", "blocks6_calibrated_offline"]
for name in names:
    out = HERE / "runs/campaign_v3" / name
    summary = json.loads((out / "results/training_summary.json").read_text())
    assert summary["snapshots"][-1]["matched_update_budget_events"] == 16384
    protocol = json.loads((out / "results/protocol.json").read_text())
    policies = out / "results/policies"
    policies.mkdir(exist_ok=True)
    for method in protocol["methods"]:
        shutil.copy2(out / f"{method}.pt", policies / f"{method}.pt")
    print(name, "training complete; exported", len(protocol["methods"]), "policies")
