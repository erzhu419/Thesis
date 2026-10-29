"""Full-day v113 accounting retained across the v118 native fork."""
import json
from pathlib import Path

from experiments.mcwm_behavior_replay_v113_20261029.observer import Observer


def costs(wait, ride):
    return dict(W=float(wait), R=float(ride), G=float(2*wait+ride))


def subtract(full, prefix):
    return {key: full[key]-prefix[key] for key in ("W", "R", "G")}


def sum_costs(rows):
    rows = list(rows)
    return costs(sum(row["W"] for row in rows), sum(row["R"] for row in rows))


class Accounting(Observer):
    """Install before reset, before the scenario wrapper, as in v113/v114.

    Native tick sampling and ledger integration are inherited unchanged. A
    child detaches only file writers; every cumulative account remains alive.
    """
    def __init__(self, bridge, out):
        super().__init__(bridge, out)
        self.root_binding = None
        self.log_manifest = None

    def bind_root(self, root_capture, *, diagnostic_person_id):
        if self.root_binding is not None:
            raise RuntimeError("The fixed root can only be bound once")
        issue = root_capture["summary"]["issue"]
        receiver = next(vehicle for vehicle in root_capture["vehicles"] if vehicle["receiver"])
        riders = sorted(person["person_id"] for person in receiver["people"])
        target = str(diagnostic_person_id)
        all_onboard = [person["person_id"] for vehicle in root_capture["vehicles"] for person in vehicle["people"]]
        selected = riders+[target]
        checks = dict(capture_passed=root_capture["summary"]["all_pass"],
            accounting_from_zero=self.start == 0., root_clock_exact=self.end == issue["observed_clock"],
            accounting_ledger_exact=self.checks["ledger_exact"],
            receiver_group_unique=len(riders) == len(set(riders)),
            target_distinct=target not in riders, target_observed_once=all_onboard.count(target) == 1,
            selected_prefix_accounts_present=all(pid in self.people for pid in selected))
        if not all(checks.values()):
            raise RuntimeError("Root accounting binding failed: "+str([k for k, v in checks.items() if not v]))
        prefix = {pid: costs(self.people[pid]["wait"], self.people[pid]["ride"]) for pid in selected}
        whole = costs(self.total["wait"], self.total["ride"])
        observed_total = costs(sum(row["wait"] for row in self.people.values()),
                               sum(row["ride"] for row in self.people.values()))
        if observed_total != whole:
            raise RuntimeError("Root cumulative individual accounts do not equal the ledger account")
        self.root_binding = dict(all_pass=True, checks=checks, clock=self.end,
            receiver=receiver["trip_id"], receiver_rider_ids=riders, diagnostic_person_id=target,
            whole_network=whole, selected_prefix=prefix, prefix_ticks=self.ticks,
            prefix_issues=self.issues, prefix_commands=self.commands,
            source_path=str((self.out/"accounting_root.json").resolve()))
        self._save("accounting_root.json", self.root_binding)
        return self.root_binding

    def _save(self, name, value):
        with (self.out/name).open("x") as handle:
            json.dump(value, handle, separators=(",", ":"), allow_nan=False)

    def flush(self):
        for writer in self.writers.values():
            writer.flush()

    def detach(self, child_out):
        """Call in the fork child after parent flush and v34 route detachment."""
        if self.root_binding is None:
            raise RuntimeError("Bind the common root before detaching the child logs")
        child_out = Path(child_out)
        child_out.mkdir(parents=True, exist_ok=True)
        prefix = {name: str(Path(writer.name).resolve()) for name, writer in self.writers.items()}
        manifest = dict(parent_prefix=prefix,
            child_suffix={name: str((child_out/(name+".jsonl")).resolve()) for name in prefix},
            prefix_ticks=self.ticks, prefix_issues=self.issues, prefix_commands=self.commands,
            prefix_clock=self.end, root_binding=self.root_binding["source_path"],
            cumulative_observer_preserved=True)
        for writer in self.writers.values():
            writer.close()
        self.out = child_out
        self.writers = {name: (child_out/(name+".jsonl")).open("x") for name in prefix}
        self.log_manifest = manifest
        self._save("log_manifest.json", manifest)
        return manifest

    def finish(self):
        if self.root_binding is None:
            raise RuntimeError("Full-day comparison requires the fixed root binding")
        observer = super().finish()
        binding = self.root_binding
        departed, completed = self.bridge.departed_passenger_times, self.bridge.completed_passenger_times
        waiting = set(self.previous_waiting)  # Original final=True endpoint native getters.
        onboard = set().union(*self.previous_onboard.values())
        self._save("native_endpoint.json", dict(clock=self.end, waiting_ids=sorted(waiting),
            onboard_ids=sorted(onboard), ledger=observer["ledger"]))

        def person(pid):
            row = self.people[pid]
            full = costs(row["wait"], row["ride"])
            prefix = binding["selected_prefix"][pid]
            return dict(person_id=pid, departure_time=departed.get(pid), completion_time=completed.get(pid),
                completed=pid in completed, full_day=full, before_root=prefix,
                after_root=subtract(full, prefix), native_waiting_at_end=pid in waiting,
                native_onboard_at_end=pid in onboard)

        riders = [person(pid) for pid in binding["receiver_rider_ids"]]
        target = person(binding["diagnostic_person_id"])
        selected_ids = set(binding["receiver_rider_ids"]) | {binding["diagnostic_person_id"]}
        ids = dict(root_receiver_riders=set(binding["receiver_rider_ids"]),
            diagnostic_target={binding["diagnostic_person_id"]}, other_passengers=set(self.people)-selected_ids)
        prefix_groups = dict(root_receiver_riders=sum_costs(row["before_root"] for row in riders),
                             diagnostic_target=target["before_root"])
        prefix_groups["other_passengers"] = subtract(binding["whole_network"], sum_costs(prefix_groups.values()))
        groups = {}
        for name, members in ids.items():
            full = costs(sum(self.people[pid]["wait"] for pid in members),
                         sum(self.people[pid]["ride"] for pid in members))
            groups[name] = dict(passengers=len(members), completed=sum(pid in completed for pid in members),
                native_waiting_at_end=len(members & waiting), native_onboard_at_end=len(members & onboard),
                incomplete_other_state=sum(pid not in completed and pid not in waiting and pid not in onboard for pid in members),
                full_day=full, before_root=prefix_groups[name], after_root=subtract(full, prefix_groups[name]))
        full = costs(self.total["wait"], self.total["ride"])
        whole = dict(full_day=full, before_root=binding["whole_network"],
            after_root=subtract(full, binding["whole_network"]), departed=len(departed), completed=len(completed),
            native_waiting_at_end=len(waiting), native_onboard_at_end=len(onboard),
            ledger_waiting_at_end=observer["ledger"]["waiting_passengers"],
            ledger_onboard_at_end=observer["ledger"]["in_vehicle_passengers"],
            native_endpoint_path="native_endpoint.json")
        checks = dict(original_observer=observer["status"] == "complete", root_binding=binding["all_pass"],
            accounting_from_zero=self.start == 0., end_matches_ledger=self.end == observer["ledger"]["time_seconds"],
            groups_disjoint=sum(len(members) for members in ids.values()) == len(set().union(*ids.values())),
            groups_cover_departed=set().union(*ids.values()) == set(departed),
            groups_reconcile=all(sum_costs(group[period] for group in groups.values()) == whole[period]
                                 for period in ("full_day", "before_root", "after_root")),
            selected_root_costs_nonnegative=all(value >= 0 for row in riders+[target]
                                               for value in row["after_root"].values()))
        summary = dict(status="complete" if all(checks.values()) else "HOLD", all_pass=all(checks.values()),
            checks=checks, start=self.start, root_clock=binding["clock"], end=self.end,
            whole_network=whole, groups=groups, root_receiver_riders=riders, diagnostic_target=target,
            observer=observer, log_manifest=self.log_manifest,
            semantics="W and R are the unchanged native passenger ledger integrals; G=2W+R, in passenger-seconds. Full-day accounts include the common prefix. Root receiver riders, diagnostic target and all other departed passengers are disjoint. Native endpoint membership and ledger endpoint counts are reported separately. Phase costs are descriptive subdivisions already included in R and are never added again. Incomplete passengers retain all cost accrued through the actual endpoint.")
        self._save("accounting_summary.json", summary)
        return summary
