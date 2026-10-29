"""v84's issue-visible reference and real remaining-day cost contract."""
from __future__ import annotations


HORIZON = 24000
DELAY = 30
PACKETS = 37
RECORDS = 185
UPDATES = 296
DEADLINE = 12000
WINDOW_END = 12600
COUNTS = dict(critic=UPDATES, value=UPDATES, actor=UPDATES, alpha=UPDATES)


def protocol():
    return dict(
        name="thesis-mcwm-supervision-v84-20261027", priority="high",
        stage="native_preflight_only", horizon_seconds=HORIZON,
        feedback_delay_seconds=DELAY, label_available_at=HORIZON + DELAY,
        reference=dict(source="v75", arm="fresh", packets=PACKETS,
                       covariates=RECORDS, complete_outcomes=RECORDS,
                       counts=dict(COUNTS), data_deadline=DEADLINE,
                       deterministic=True, actor_frozen=True),
        preflight=dict(seed=6207, scenario="stable", line="102X",
                       issue_window=[DEADLINE, WINDOW_END], pairs=1,
                       suffixes=2, maximum_native_seconds=36000),
        proposed_collection=dict(seed=6207, scenarios=["stable", "local_speed_drop"],
                                 lines=["7X", "102X", "705X"], pairs=6,
                                 preflight_pair_reused=True, authorized=False),
        support="F(s,-1),F(s,+1); gap>=2*max(simulation_step,action_step)",
        intervention="Change only the root action; frozen reference actor thereafter",
        suffix_training_updates=0, suffix_admitted_labels=0,
        new_initializer_updates=0, diagnostic_labels_fed_back=0,
        original_v75_contract_unchanged=True,
        downloads="Bounded JSON and necessary log tails; native data stay on server")


def source_snapshot_metadata(*, seed, scenario, published_clock, max_available_at):
    """Describe the packet37 model; publishing must not use a later packet."""
    if not (0 <= max_available_at <= DEADLINE <= published_clock):
        raise ValueError("Reference must use only feedback available by12000")
    return dict(source="v75", source_arm="fresh", seed=seed, scenario=scenario,
                packets=PACKETS, covariates=RECORDS, complete_outcomes=RECORDS,
                counts=dict(COUNTS), data_deadline=DEADLINE,
                max_available_at=float(max_available_at),
                published_clock=float(published_clock),
                deterministic=True, actor_frozen=True, future_training_used=False)


def remaining_cost(initial, final):
    """Subtract the actual issue ledger, never the nominal12000 endpoint."""
    if final["time_seconds"] != HORIZON or not initial["time_seconds"] < HORIZON:
        raise ValueError("Remaining-day label requires an actual24000 endpoint")
    wait = final["accrued_wait_seconds"] - initial["accrued_wait_seconds"]
    ride = final["accrued_in_vehicle_seconds"] - initial["accrued_in_vehicle_seconds"]
    if wait < 0 or ride < 0:
        raise ValueError("Cumulative passenger ledgers must not decrease")
    return dict(dt=HORIZON-initial["time_seconds"], wait=wait, ride=ride,
                cost=2*wait+ride, observed_at=HORIZON, available_at=HORIZON+DELAY)


def label_available(available_at, consumer_clock):
    return bool(available_at >= HORIZON+DELAY and consumer_clock >= available_at)


def _reference_valid(reference, root):
    return bool(reference["source"] == "v75" and reference["source_arm"] == "fresh"
                and reference["packets"] == PACKETS
                and reference["covariates"] == RECORDS
                and reference["complete_outcomes"] == RECORDS
                and reference["counts"] == COUNTS
                and reference["data_deadline"] == DEADLINE
                and 0 <= reference["max_available_at"] <= DEADLINE
                <= reference["published_clock"] <= root["started_at"]
                and reference["deterministic"] is True
                and reference["actor_frozen"] is True
                and reference["future_training_used"] is False)


def pair_gate(low, high):
    """Command support and observed physical separation are distinct gates.

    ``root`` is the plain full issue/ledger record; ``reference`` is the frozen
    snapshot metadata. Costs do not participate in this technical gate.
    """
    tick = max(low["sim_step_seconds"], high["sim_step_seconds"])
    control_tick = max(tick, low["action_step_seconds"], high["action_step_seconds"])
    low_departure, high_departure = low.get("departure_clock"), high.get("departure_clock")
    observed = low_departure is not None and high_departure is not None
    checks = dict(
        same_complete_root=low["root"] == high["root"],
        same_reference=low["reference"] == high["reference"],
        issue_visible_reference=all(_reference_valid(row["reference"], row["root"])
                                    for row in (low, high)),
        same_external_seed=low["external_seed"] == high["external_seed"],
        real_predeclared_window=all(DEADLINE <= row["root"]["started_at"] < WINDOW_END
                                   for row in (low, high)),
        command_support=high["executed_hold_seconds"]-low["executed_hold_seconds"] >= 2*control_tick,
        actual_departures_observed=observed,
        physical_departure_separation=bool(observed and high_departure-low_departure >= tick),
        real_terminal_endpoint=all(row["final_clock"] == HORIZON for row in (low, high)),
        suffix_zero_training=all(row["suffix_updates"] == 0 for row in (low, high)),
        suffix_zero_admission=all(row["suffix_admitted_labels"] == 0 for row in (low, high)),
    )
    return dict(all_pass=all(checks.values()), checks=checks)
