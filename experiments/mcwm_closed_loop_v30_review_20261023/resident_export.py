"""Read-only v30 recovery: compact resident summaries without replay or old writes."""
from pathlib import Path
import json
import re
import shlex
import sys


SOURCE = "mcwm_closed_loop_physical_execution_v30_20261023"
ARMS = ("low", "mid", "high")


def encoded(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def compact_audit(value):
    checks = value.get("checks", {})
    return {"all_pass": value.get("all_pass"), "count": len(checks),
            "failed": [key for key, passed in checks.items() if not passed]}


def project(root):
    root = Path(root)
    sys.path.insert(0, str(root.parents[1]))
    from experiments.mcwm_closed_loop_physical_execution_v30_20261023 import analyze
    full = analyze.analyze(root)
    projected = {key: full[key] for key in ("status", "integrity", "pending", "failures", "budget")}
    projected["export"] = dict(original_compact_bytes=len(encoded(full).encode()),
        original_section_bytes={key: len(encoded(value).encode()) for key, value in full.items()},
        original_case_bytes={case["scenario"]: len(encoded(case).encode()) for case in full["cases"]},
        old_files_written=0, resident_replay_or_training=0)
    projected["tick_columns"] = ["native_clock_seconds", "ledger_clock_seconds", "action_step_length_seconds",
        "last_action_time_seconds", "is_at_bus_stop", "stop_state", "speed_mps", "lane_position_m",
        "registered_stop_duration_seconds"]
    projected["event_columns"] = ["status", "station_id", "clock_seconds", "time_source"]
    projected["apply_after_columns"] = ["native_clock_seconds", "ledger_clock_seconds", "action_step_length_seconds",
        "last_action_time_seconds", "registered_stop_duration_seconds", "is_at_bus_stop", "stop_state"]
    projected["cases"] = []
    for case in full["cases"]:
        path = root/"runs"/"seed6202"/case["scenario"]
        main = analyze.read(path/"summary.json")
        if main is None:
            projected["cases"].append(case)
            continue
        item = {key: case[key] for key in ("scenario", "status", "selected", "external_seed", "integrity", "execution")}
        item["endpoints"] = {}
        for name, endpoint in case["endpoints"].items():
            item["endpoints"][name] = {key: endpoint[key] for key in (
                "integrity", "zero_duration_arms", "missing_arms", "truncated_arms")}
            item["endpoints"][name]["columns"] = ["dt", "wait", "ride", "cost"]
            item["endpoints"][name]["arms"] = {arm: [endpoint["arms"][arm].get(key)
                for key in ("dt", "wait", "ride", "cost")] for arm in ARMS}
        item["predictions"] = {}
        for model, prediction in case["predictions"].items():
            item["predictions"][model] = {key: prediction[key] for key in (
                "supported", "min_q", "rank_high_to_low", "fixed300")}
        main_checks = main["integrity"]["checks"]
        if "main_check_columns" not in projected:
            projected["main_check_columns"] = list(main_checks)
        if set(main_checks) != set(projected["main_check_columns"]):
            raise RuntimeError("The two completed parents expose different integrity schemas")
        item["main_checks"] = [main_checks[key] for key in projected["main_check_columns"]]
        item["main_failed"] = [key for key, passed in main_checks.items() if not passed]
        item["main"] = {key: main[key] for key in ("wall_seconds", "execution_wall",
            "post_budget_counts", "post_budget_shadow_counts")}
        parallel = case["parallel"]
        item["parallel"] = {key: parallel[key] for key in ("integrity", "common_native_overlap_seconds", "child_exitcodes")}
        item["parallel"]["clock_columns"] = ["pid", "native_started_wall", "native_finished_wall"]
        item["parallel"]["children"] = {arm: [row.get(key) for key in item["parallel"]["clock_columns"]]
            for arm, row in parallel["children"].items()}
        physical = case["physical"]
        item["physical"] = {key: physical[key] for key in ("issue_state", "contrasts",
            "three_distinct_observed_responses", "insufficient_contrasts", "missing_departure_arms",
            "missing_next_arrival_arms")}
        item["physical"]["contrasts"] = {key: dict(status=value["status"],
            separation_sources=value["separation_sources"], first_difference=value["trajectory"]["first_difference"])
            for key, value in physical["contrasts"].items()}
        item["physical"]["arms"] = {}
        item["branches"] = {}
        for arm in ARMS:
            branch = analyze.read(path/"branches"/arm/"summary.json")
            raw = branch["physical"]
            arm_physical = physical["arms"][arm]
            item["physical"]["arms"][arm] = {key: arm_physical[key] for key in (
                "registered_hold_minus_command_seconds", "short_tick_coverage", "coverage", "missing")}
            item["physical"]["arms"][arm]["events"] = {key: [arm_physical[key].get(field) for field in
                ("status", "station_id", "clock_seconds", "time_source")] for key in (
                "departure", "native_history_departure", "next_same_vehicle_arrival", "native_history_next_arrival")}
            item["physical"]["arms"][arm]["apply_after"] = [raw["apply_after"].get(key)
                for key in projected["apply_after_columns"]]
            item["physical"]["arms"][arm]["first_four_ticks"] = [
                [tick.get(key) for key in projected["tick_columns"]] for tick in raw["short_tick_samples"][:4]]
            native = branch["execution"]["native_application"]
            item["branches"][arm] = dict(integrity=compact_audit(branch["integrity"]),
                counter_delta=branch["counters"]["delta"],
                native_application={key: native.get(key) for key in ("time", "issue_ledger_time", "real_clock_before",
                    "real_clock_after", "predicted_departure_time_seconds")})
            log = path/"branches"/(arm+".log")
            matching = []
            count = 0
            with log.open(errors="replace") as stream:
                for line in stream:
                    if re.search(r"WARN|Error|Traceback", line):
                        count += 1
                        if len(matching) < 2:
                            matching.append(line.rstrip()[:300])
            item["branches"][arm]["log"] = dict(bytes=log.stat().st_size,
                warning_error_lines=count, first_lines=matching)
        projected["cases"].append(item)
    data = encoded(projected)+"\n"
    if len(data.encode()) > 20000:
        # Columnize repeated field names only; no observations or failed checks
        # disappear when the complete direct representation exceeds the cap.
        branch_keys = list(projected["cases"][0]["branches"]["low"]["native_application"])
        projected["native_application_columns"] = branch_keys
        for case in projected["cases"]:
            for branch in case.get("branches", {}).values():
                branch["native_application"] = [branch["native_application"][key] for key in branch_keys]
        data = encoded(projected)+"\n"
    if len(data.encode()) > 20000:
        sizes = {key: len(encoded(value).encode()) for key, value in projected.items()}
        raise RuntimeError(f"Compact resident projection still exceeds 20KB: {len(data.encode())}; sections {sizes}")
    return data


def collect():
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parents[1]))
    from experiments.mcwm_closed_loop_physical_execution_v30_20261023 import infra
    root = here.parent/SOURCE
    code = "__name__='resident_projection'\n"+Path(__file__).read_text()+"\nprint(project("+repr(infra.remote(root))+"),end='')\n"
    rc, output, error = infra.scheduler.run_on("node001", shlex.join([infra.PYTHON, "-c", code]), timeout=45, check=False)
    if rc:
        raise RuntimeError(error[-3000:]+output[-1000:])
    value = json.loads(output)
    destination = here/"resident_summary.json"
    with destination.open("x") as stream:
        stream.write(encoded(value)+"\n")
    print(json.dumps(dict(downloaded_bytes=len(output.encode()), local_bytes=destination.stat().st_size)))


if __name__ == "__main__":
    collect()
