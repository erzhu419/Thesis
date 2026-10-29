"""Read-only full-day equality against the retained native reference logs."""
import json
from pathlib import Path

from experiments.mcwm_connection_full_day_v118_20261029.validation import (
    CUMULATIVE, OBSERVER_FIELDS, SEGMENTED, compare_segments,
)


def compare_full_day(observed_dir, reference_dir):
    """Compare eight complete JSONL streams and their native accounting receipts."""
    observed, reference = map(Path, (observed_dir, reference_dir))
    files = {name + ".jsonl": compare_segments(
        [observed / (name + ".jsonl")], [reference / (name + ".jsonl")])
        for name in (*SEGMENTED, *CUMULATIVE)}
    ledger = None
    try:
        summaries = [json.loads((directory / "observer_summary.json").read_text())
                     for directory in (observed, reference)]
        actual, original = summaries
        equal = {key: key in actual and key in original and actual[key] == original[key]
                 for key in OBSERVER_FIELDS}
        complete = {
            side: row.get("status") == "complete" and row.get("ledger_exact") is True
                and bool(row.get("checks"))
                and all(value is True for value in row["checks"].values())
                and row.get("start") == 0 and row.get("end") == 24000
                and row.get("ledger", {}).get("time_seconds") == 24000
            for side, row in zip(("observed", "reference"), summaries)}
        count_fields = (("actions", "issues"), ("commands", "commands"),
                        ("ticks", "ticks"), ("passengers", "passengers"), ("visits", "visits"))
        row_counts = {
            side: {name: files[name + ".jsonl"]["observed_rows"][index] == row.get(field)
                   for name, field in count_fields}
            for index, (side, row) in enumerate(zip(("observed", "reference"), summaries))}
        observer = dict(all_pass=all(complete.values()) and all(equal.values())
            and all(all(counts.values()) for counts in row_counts.values()),
            fields_equal=equal, complete_horizon=complete, row_counts=row_counts)
        ledger = actual.get("ledger")
    except (OSError, ValueError) as error:
        observer = dict(all_pass=False, error=str(error))
    return dict(all_pass=all(row["all_pass"] for row in files.values()) and observer["all_pass"],
        files=files, observer=observer, final_ledger=ledger,
        observed_root=str(observed), reference_root=str(reference),
        semantics="Eight single-directory native JSONL streams compare field-for-field, "
                  "including complete passenger/visit accounts and both 0-to-24000 receipts. "
                  "No learner, optimizer or policy-equivalence claim.")
