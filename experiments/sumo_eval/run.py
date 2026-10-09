"""Evaluate a completed training cell on a CPU node."""

import argparse
import json
import os
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ["THESIS_H2O_ROOT"] = str(HERE / "runtime/H2Oplus")
os.environ["THESIS_SUMO_ROOT"] = str(HERE / "assets/SUMO_ruiguang")
os.environ["THESIS_DATASET"] = str(HERE / "contract.h5")
os.environ.setdefault("SUMO_HOME", str(HERE / "assets/SUMO_ruiguang"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    import libsumo
    if libsumo.getVersion()[1] != "SUMO 1.25.0":
        raise RuntimeError(f"Expected paired SUMO 1.25.0, got {libsumo.getVersion()}")
    from evaluate_control import main as evaluate
    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.input / "protocol.json", args.out / "protocol.json")
    protocol = json.loads((args.out / "protocol.json").read_text())
    for method in protocol["methods"]:
        shutil.copy2(args.input / "policies" / f"{method}.pt", args.out / f"{method}.pt")
    print(json.dumps({"stage": "sumo-evaluation", "methods": protocol["methods"],
                      "sumo_version": libsumo.getVersion()}), flush=True)
    evaluate(args.out.resolve(), protocol["methods"], args.workers)
    results = args.out / "results"
    shutil.copytree(args.out / "evaluation", results / "evaluation", dirs_exist_ok=True)
    (results / "completed.json").write_text(json.dumps({"methods": protocol["methods"],
                                                         "evaluation_complete": True})+"\n")


if __name__ == "__main__":
    main()
