"""Execution-trigger traces linked to exact one-second passenger memberships."""

from review_common import HERE, EXPERIMENTS, P, module, write, distribution
import argparse
from collections import defaultdict
from pathlib import Path
import json
from types import MethodType

old = module("round3_attribution", EXPERIMENTS / "sumo_eval/attribution_v1/run.py")
base = old.common


class PersonLedger:
    def __init__(self, bridge):
        self.bridge = bridge
        self.time = float(bridge.current_time)
        self.active = {"wait": {}, "ride": {}}
        self.total = {"wait": defaultdict(float), "ride": defaultdict(float)}
        self.waiting_by_stop, self.onboard_by_bus = {}, {}

    def sample(self, now):
        import libsumo as traci
        self.waiting_by_stop = {stop: set(traci.busstop.getPersonIDs(stop)) for stop in self.bridge.stop_obj_dic}
        self.onboard_by_bus = {bus: set(traci.vehicle.getPersonIDList(bus)) for bus in self.bridge.active_bus_ids}
        memberships = {"wait": set().union(*self.waiting_by_stop.values()),
                       "ride": set().union(*self.onboard_by_bus.values())}
        self.advance(now, memberships)

    def advance(self, now, memberships):
        for state, present in memberships.items():
            active = self.active[state]
            for person in active.keys() - present:
                self.total[state][person] += now-active.pop(person)
            for person in present - active.keys():
                active[person] = now
        self.time = float(now)

    def finalize(self):
        return {state: {person: self.total[state].get(person, 0.) +
                               (self.time-active[person] if person in active else 0.)
                        for person in self.total[state].keys() | active.keys()}
                for state, active in self.active.items()}


class Trace:
    def __init__(self, out):
        self.out, self.env, self.ledger = out, None, None
        self.records, self.pending = [], {}
        self.current_execution = None

    def install(self, env):
        self.env = env
        bridge = env._bridge
        self.ledger = PersonLedger(bridge)
        self.ledger.sample(float(bridge.current_time))
        sync = bridge._synchronize_passenger_time_counts
        transition = bridge._synchronize_bus_passenger_transition

        def count_hook(owner, now):
            sync(now)
            self.ledger.sample(float(now))

        def transition_hook(owner, bus_id, previous_state, current_state, bus_obj, now):
            transition(bus_id, previous_state, current_state, bus_obj, now)
            if previous_state == "Stop" and current_state != "Stop" and bus_id in self.pending:
                import libsumo
                row = self.pending.pop(bus_id)
                onboard = set(libsumo.vehicle.getPersonIDList(bus_id))
                row["departure_time"] = float(now)
                row["boarded_after_decision"] = sorted(onboard-set(row["onboard_at_decision"]))
                row["onboard_at_departure"] = sorted(onboard)

        bridge._synchronize_passenger_time_counts = MethodType(count_hook, bridge)
        bridge._synchronize_bus_passenger_transition = MethodType(transition_hook, bridge)

    def record(self, event, local, action, hold):
        import libsumo
        proposed, parent, potential = self.current_execution
        changed = hold > parent+1e-4
        eligible = potential > parent+1e-4
        if not eligible:
            return
        row = dict(bus=str(event.bus_id), stop=str(event.stop_id), line=str(event.line_id),
                   time=float(event.sim_time), proposed_hold=proposed, parent_hold=parent,
                   executed_hold=hold, changed=changed,
                   onboard_at_decision=sorted(libsumo.vehicle.getPersonIDList(event.bus_id)),
                   waiting_at_decision=sorted(libsumo.busstop.getPersonIDs(event.stop_id)),
                   boarded_after_decision=[], onboard_at_departure=[])
        self.records.append(row)
        self.pending[event.bus_id] = row

    def finish(self):
        bridge = self.env._bridge
        snapshot = bridge.passenger_time_ledger.snapshot()
        # The environment closes the last count interval at the horizon without
        # another membership query; use the same right boundary for individuals.
        self.ledger.time = float(snapshot.time_seconds)
        amounts = self.ledger.finalize()
        errors = dict(wait=sum(amounts["wait"].values())-snapshot.accrued_wait_seconds,
                      ride=sum(amounts["ride"].values())-snapshot.accrued_in_vehicle_seconds)
        if max(abs(value) for value in errors.values()) > 1e-5:
            raise RuntimeError(f"Individual membership accounting disagrees with system ledger: {errors}")
        cohorts = {name: set() for name in ("waiting_at_decision", "onboard_at_decision", "boarded_after_decision")}
        for event in self.records:
            for name in cohorts:
                cohorts[name].update(event[name])
        rows = []
        for person, departure in sorted(bridge.departed_passenger_times.items()):
            completion = bridge.completed_passenger_times.get(person)
            rows.append(dict(person=person, departure=departure, completion=completion,
                             wait=amounts["wait"].get(person, 0.), ride=amounts["ride"].get(person, 0.)))
        # Individual records stay remote; collection includes JSON summaries only.
        for name, values in (("passengers", rows), ("triggers", self.records)):
            with (self.out / f"{name}.jsonl").open("w") as stream:
                for value in values:
                    stream.write(json.dumps(value)+"\n")
        result = dict(potential_trigger_events=len(self.records),
                      changed_events=sum(r["changed"] for r in self.records), ledger_balance=errors,
                      unresolved_departures=len(self.pending), cohorts={})
        for name, people in cohorts.items():
            selected = [r for r in rows if r["person"] in people]
            result["cohorts"][name] = dict(passengers=len(selected),
                 completed=sum(r["completion"] is not None for r in selected),
                 wait_seconds=distribution([r["wait"] for r in selected]),
                 ride_seconds=distribution([r["ride"] for r in selected]))
        write(self.out / "results/service_chain.json", result)


def cases():
    return [dict(method=method, execution_layer=layer, demand_scale=demand, **block)
            for method in P["guard"]["methods"] for layer in P["guard"]["layers"]
            for demand in P["guard"]["demands"] for block in P["guard"]["blocks"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results").mkdir(exist_ok=True)
    case = dict(cases()[args.case], case_id=f"case{args.case:03d}")
    p = json.loads((old.HERE / "protocol.json").read_text())
    p["evaluation_horizon_seconds"] = 1200 if args.smoke else P["guard"]["horizon_seconds"]
    trace = Trace(args.out)
    original_env, original_execute, original_record = base.AllLinesSumoEnv, old.execute, base._event_record

    class TracedEnv(original_env):
        def reset(self, *a, **kw):
            result = super().reset(*a, **kw)
            trace.install(self)
            return result

        def close(self):
            if trace.ledger is not None:
                trace.finish()
            super().close()

    def execute(raw, proposed, layer, config, scale):
        action, parent = original_execute(raw, proposed, layer, config, scale)
        potential, _ = original_execute(raw, proposed, "early60", config, scale)
        hold = lambda a: float(base.policy_action_to_hold_seconds_v7(a[0], config))
        trace.current_execution = (hold(proposed), hold(parent), hold(potential))
        return action, parent

    def record(event, local, action, hold):
        trace.record(event, local, action, hold)
        return original_record(event, local, action, hold)

    base.AllLinesSumoEnv, old.execute, base._event_record = TracedEnv, execute, record
    cfg, objects = base.build_scenario(args.out, case["demand_scale"], case["demand_seed"], 120)
    row = old.evaluate((case, p, str(args.out), cfg, objects))
    write(args.out / "results/ready.json", dict(case=case, horizon=row["horizon_seconds"]))


if __name__ == "__main__":
    main()
    print("DONE review supplement", flush=True)
