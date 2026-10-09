"""Keep historical actors separate from the matched-width seed extension."""

import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/resac_archive"
EXTENSION = ROOT.parent / "RE-SAC/results/bus_seed_extension/logs"


def summary(values):
    return dict(values=values, mean=mean(values), sd=stdev(values))


def main():
    manifest = json.loads((EXPERIMENT / "archive_manifest.json").read_text())
    files = list((EXPERIMENT / "runs").glob("*/*.json"))
    records = [json.loads(path.read_text()) for path in files]
    reference = [row for row in records if row["variant"] == "reference_K10_seed32"]
    assert len(reference) == 1, "Missing or duplicated compatibility rollout"
    expected = manifest["reference_episode"]
    differences = {key: reference[0][key] - value for key, value in expected.items()}
    compatible = all(np.isclose(reference[0][key], value, rtol=1e-6, atol=1e-5)
                     for key, value in expected.items())
    assert compatible, differences
    output = dict(archive_id=8, extension_seeds=[32, 40], episodes_per_actor=10,
                  dispersion="sample standard deviation across two training repeats",
                  compatibility_differences=differences, rows=[])
    for actor in manifest["actors"]:
        name = actor["variant"]
        archived = sorted([row for row in records if row["variant"] == name],
                          key=lambda row: row["episode"])
        assert [row["episode"] for row in archived] == list(range(10)), name
        assert all(row["evaluation_seed"] == 100008 + row["episode"] for row in archived), name
        trained, evaluated = [], []
        for seed in [32, 40]:
            folder = EXTENSION / f"{name}_seed{seed}"
            reward = np.load(folder / "rewards.npy")
            assert len(reward) == 500
            trained.append(float(reward[-100:].mean()))
            data = json.loads((folder / "operational_metrics.json").read_text())
            assert data["seed"] == seed and data["eval_episodes"] == 10
            assert data["deterministic"] and data["route_sigma"] == 1.5
            evaluated.append(data["mean"])
        keys = evaluated[0].keys()
        row = dict(variant=name, archive_hidden_width=actor["hidden_width"],
                   extension_hidden_width=64, archive_train=actor["train_last100"],
                   extension_train=summary(trained),
                   archive_operational={key: mean(r[key] for r in archived) for key in keys},
                   extension_operational={key: summary([r[key] for r in evaluated]) for key in keys},
                   archive_episodes=archived)
        output["rows"].append(row)
        print(name, "archive train/eval", round(row["archive_train"] / 1000, 2),
              round(row["archive_operational"]["reward"] / 1000, 2),
              "extension train/eval",
              {k: round(row["extension_train"][k] / 1000, 2) for k in ["mean", "sd"]},
              {k: round(row["extension_operational"]["reward"][k] / 1000, 2) for k in ["mean", "sd"]})
    assert len(records) == 81
    destination = ROOT / "manuscript/figures/source_data/resac_archive_evaluation.json"
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print("Compatibility differences:", differences)


if __name__ == "__main__":
    main()
