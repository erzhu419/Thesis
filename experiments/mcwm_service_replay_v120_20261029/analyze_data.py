"""Small descriptive readout of the already selected service records; no fitting."""
import argparse
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path


SCENES = ("stable", "local_speed_drop")
PACKET_COLUMNS = ("deadline", "selected", "valid_mature", "root_cohort_positive",
                  "root_cohort_boarding_positive", "records_with_errors")


def read_rows(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle]


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def marks(row):
    label = row.get("labels") or {}
    departure = label.get("receiver_departure", {})
    cohort = label.get("cohort_boarded_count", {})
    visit = label.get("receiver_post_issue_boarded_count", {})
    return dict(valid_mature=departure.get("value") is not None and cohort.get("value") is not None
                    and row.get("cohort_definition_complete") is True,
        root_cohort_positive=any(feeder["cohort_ids"] for feeder in row["context"]["feeders"]),
        root_cohort_boarding_positive=cohort.get("observed_count_lower_bound", 0) > 0,
        any_boarding_positive=visit.get("observed_count_lower_bound", 0) > 0,
        departure_mature=departure.get("value") is not None,
        visit_count_mature=visit.get("value") is not None,
        complete_cohort_count_mature=cohort.get("value") is not None
                    and row.get("cohort_definition_complete") is True,
        departure_censored=departure.get("right_censored") is True,
        cohort_count_censored=cohort.get("right_censored") is True,
        label_missing=row.get("labels") is None, records_with_errors=bool(row.get("errors")))


def describe_scene(folder):
    summary = json.loads((folder/"summary.json").read_text())
    records = read_rows(folder/"selected_records.jsonl")
    packets = read_rows(folder/"packets.jsonl")
    by_sequence = {row["selection"]["sequence"]: row for row in records}
    totals, missing = Counter(), Counter()
    holdings, gaps, packet_rows, discrepancies = [], [], [], []
    for row in records:
        context, selection = row["context"], row["selection"]
        totals.update({key: int(value) for key, value in marks(row).items()})
        missing.update(row.get("missing_inputs", {}))
        totals["unknown_input_records"] += bool(row.get("missing_inputs"))
        totals["incomplete_capture_records"] += context.get("capture", {}).get("status") != "complete"
        totals["incomplete_cohort_records"] += row.get("cohort_definition_complete") is not True
        action = row.get("action")
        if action is not None:
            holdings.append(action["hold_seconds"])
        immediate, gap_support, immediate_gap_support = set(), False, False
        immediate_unknown = context.get("capture", {}).get("checks", {}).get("feeder_selection_complete") is not True
        for feeder in context["feeders"]:
            x = feeder["x"]
            is_immediate = x.get("feeder_next_stop_matches") is True
            if is_immediate:
                immediate.update(feeder["cohort_ids"])
                immediate_unknown |= x.get("current_leg_dest_count") is None
            supported = x.get("same_lane") is True and finite(x.get("gap_m"))
            if supported:
                gaps.append(x["gap_m"])
                gap_support = True
                immediate_gap_support |= is_immediate and bool(feeder["cohort_ids"])
        totals["immediate_cohort_positive_records"] += bool(immediate)
        totals["observed_immediate_cohort_members_sum"] += len(immediate)
        totals["immediate_cohort_count_unknown_records"] += immediate_unknown
        totals["same_lane_gap_supported_records"] += gap_support
        totals["immediate_positive_cohort_gap_supported_records"] += immediate_gap_support
        labels = row.get("labels") or {}
        feeder_censored = False
        for feeder in labels.get("feeders", []):
            arrival = feeder["arrival"]
            totals["feeder_arrival_targets"] += 1
            totals["feeder_arrival_right_censored_targets"] += arrival.get("right_censored") is True
            totals["feeder_arrival_left_censored_targets"] += arrival.get("left_censored") is True
            feeder_censored |= arrival.get("right_censored") is True
        totals["records_with_right_censored_feeder_arrival"] += feeder_censored
        if (context["meta"]["sequence"] != selection["sequence"]
                or selection["window_end"] > 24000
                or selection["feedback_available_at"] > selection["packet_deadline"]
                or (row.get("labels") is not None and labels["available_at"] != selection["window_end"])):
            discrepancies.append(dict(sequence=selection["sequence"], reason="stored_temporal_or_identity_mismatch"))
    listed = []
    for packet in packets:
        sequences = packet["selected_sequences"]
        subset = [by_sequence[seq] for seq in sequences if seq in by_sequence]
        counts = Counter()
        for row in subset:
            counts.update({key: int(value) for key, value in marks(row).items()})
        packet_rows.append([packet["deadline"], len(subset), *(counts[key] for key in PACKET_COLUMNS[2:])])
        listed.extend(sequences)
        if (len(subset) != packet["selected_records"]
                or any(row["selection"]["packet_deadline"] != packet["deadline"] for row in subset)):
            discrepancies.append(dict(deadline=packet["deadline"], reason="packet_record_mismatch"))
    if len(records) != len(by_sequence) or len(listed) != len(set(listed)) or set(listed) != set(by_sequence):
        discrepancies.append(dict(reason="selected_and_packet_identity_sets_differ"))
    old = summary.get("service_records", {})
    if old.get("selected_records") != len(records) or old.get("packet_slots") != len(packets):
        discrepancies.append(dict(reason="saved_selection_summary_count_mismatch"))
    if old.get("statistics", {}).get("valid_mature_records") != totals["valid_mature"]:
        discrepancies.append(dict(reason="saved_valid_mature_count_mismatch"))
    denominator = len(records)
    ratios = {key: dict(count=totals[key], selected=denominator,
        fraction=totals[key]/denominator if denominator else None) for key in
        ("valid_mature", "departure_mature", "visit_count_mature", "complete_cohort_count_mature",
         "root_cohort_positive", "root_cohort_boarding_positive")}
    result = dict(run_status=summary.get("status"), run_integrity=summary.get("integrity", {}).get("all_pass"),
        selected_records=denominator, packet_slots=len(packets), counts=dict(totals), ratios=ratios,
        packet_columns=list(PACKET_COLUMNS), packet_rows=packet_rows,
        unknown_input_field_counts=dict(missing),
        action_seconds=dict(observed=len(holdings), missing=denominator-len(holdings),
            zero=sum(value == 0 for value in holdings), nonzero=sum(value != 0 for value in holdings),
            minimum=min(holdings) if holdings else None, maximum=max(holdings) if holdings else None),
        same_lane_gap_m=dict(feeder_entries=len(gaps), minimum=min(gaps) if gaps else None,
            maximum=max(gaps) if gaps else None, negative=sum(value<0 for value in gaps),
            zero=sum(value==0 for value in gaps), positive=sum(value>0 for value in gaps)),
        mismatch_count=len(discrepancies), first_mismatches=discrepancies[:3],
        selected_error_count=sum(len(row.get("errors", [])) for row in records),
        first_selected_errors=[dict(sequence=row["selection"]["sequence"], errors=row["errors"])
                               for row in records if row.get("errors")][:3])
    return result, records


def normalized_context(row):
    context = deepcopy(row["context"])
    context["meta"].pop("source_group", None)
    return context


def analyze_data(runs):
    runs = Path(runs)
    scene_results, records = {}, {}
    for scene in SCENES:
        scene_results[scene], records[scene] = describe_scene(runs/scene)
    before = {scene: {(row["context"]["meta"]["sequence"], row["context"]["meta"]["observed_at"]): row
        for row in records[scene] if row["context"]["meta"]["observed_at"] < 9000} for scene in SCENES}
    stable, shock = before["stable"], before["local_speed_drop"]
    common = sorted(stable.keys() & shock.keys())
    matched, examples = Counter(), []
    for key in common:
        left, right = stable[key], shock[key]
        same_context = normalized_context(left) == normalized_context(right)
        same_action, same_labels = left.get("action") == right.get("action"), left.get("labels") == right.get("labels")
        before_shock = max(left["selection"]["window_end"], right["selection"]["window_end"]) <= 9000
        matched.update(dict(context_equal=int(same_context), action_equal=int(same_action),
            labels_equal=int(same_labels), context_action_equal=int(same_context and same_action),
            complete_payload_equal=int(same_context and same_action and same_labels),
            both_label_windows_end_before_or_at_shock=int(before_shock)))
        if not (same_context and same_action and same_labels) and len(examples) < 3:
            examples.append(dict(sequence=key[0], observed_at=key[1], context_equal=same_context,
                action_equal=same_action, labels_equal=same_labels, both_windows_pre_shock=before_shock))
    slots = sum(len(value) for value in records.values())
    data_match = all(row["mismatch_count"] == 0 for row in scene_results.values())
    return dict(status="complete" if data_match else "stored_data_mismatch",
        data_reconciliation_passed=data_match, scenes=scene_results,
        common_prefix=dict(shock_clock=9000, paired_by=["sequence", "observed_at"],
            removed_context_field_only="meta.source_group", pre_shock_records_per_scene={s:len(v) for s,v in before.items()},
            matched_root_identifiers=len(common), unpaired_pre_shock_per_scene={
                "stable":len(stable.keys()-shock.keys()), "local_speed_drop":len(shock.keys()-stable.keys())},
            comparison_counts=dict(matched), first_payload_differences=examples,
            total_record_slots=slots, root_units_after_common_prefix_identifier_dedup=slots-len(common),
            payload_records_after_exact_pre_shock_dedup=slots-matched["complete_payload_equal"]),
        interpretation="Record slots are not independent observations. The scenes share the pre-shock trajectory; even deduplicated root units remain temporally dependent within the same fixed external seed. Feeder and passenger target counts do not enlarge the root-action sample budget. Missing inputs, censoring, positive boarding and feature support describe this selected data only, not predictor accuracy or controller benefit.",
        budget=dict(new_native_seconds=0, optimizer_updates=0, new_selected_records=0, fitted_models=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_data(args.runs)
    payload = json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False)+"\n"
    if len(payload.encode()) > 16000:
        raise ValueError("Cross-scene data report exceeds the fixed 16000-byte bound")
    with args.out.open("x") as handle:
        handle.write(payload)
    print(payload, end="")
