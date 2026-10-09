"""Describe the existing paired block effects without changing the original test."""

import json
from pathlib import Path

import numpy as np
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "manuscript/figures/source_data/h2o_confirmatory_means.json"


def main():
    blocks = json.loads(SOURCE.read_text())["blocks"]
    result = {
        "analysis": "posthoc descriptive paired t intervals",
        "unit": "environment block averaged across five fixed actors and three demands",
        "original_decision": "unchanged paired sign tests",
        "n_blocks": len(blocks),
        "effects": {},
    }
    for key, scale, unit in (
        ("primary_change_seconds", 1, "seconds per departed passenger"),
        ("completion_change", 100, "percentage points"),
    ):
        values = np.array([row[key] for row in blocks.values()]) * scale
        sd = float(values.std(ddof=1))
        half = float(t.ppf(0.975, len(values) - 1) * sd / np.sqrt(len(values)))
        mean = float(values.mean())
        result["effects"][key] = {
            "unit": unit, "mean": mean, "sample_sd": sd,
            "ci95": [mean - half, mean + half],
        }
    destination = ROOT / "workpapers/completion_effect_intervals.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
