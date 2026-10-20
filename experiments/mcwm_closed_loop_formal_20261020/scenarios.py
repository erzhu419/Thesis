"""Native SUMO speed intervention and fixed-time passenger-cost observations.

The intervention is outside the policy observation and changes one real road
edge.  A native StepListener applies it between simulation ticks, independently
of when a holding decision happens.  Existing bridge/experiment files are read
only.  Cost snapshots contain cumulative accrued time, including unfinished
passengers; they do not discard incomplete trips.
"""

from __future__ import annotations

import math


EDGE_ID = "E_J282_J328.50"
LANE_IDS = tuple(f"{EDGE_ID}_{index}" for index in range(4))
AFFECTED_LINES = ("7X", "102X", "705X")
UPSTREAM_STOP = "7X13_102X26_705X27"
DOWNSTREAM_STOP = "7X14_102X27_705X28"


class ScenarioRecorder:
    """One full-day scenario; :func:`install_scenario` adds native inheritance."""

    def __init__(self, api, ledger, name, *, shock_at=9000., factor=.65,
                 metric_every=300.):
        if name not in ("stable", "local_speed_drop"):
            raise ValueError(f"Unknown target scenario: {name}")
        if not 0 < factor < 1 or shock_at <= 0 or metric_every <= 0:
            raise ValueError("The speed shock and metric interval must be positive")
        self.api, self.ledger, self.name = api, ledger, name
        self.shock_at, self.factor = float(shock_at), float(factor)
        self.metric_every = float(metric_every)
        self.installed_at = float(api.simulation.getTime())
        if name == "local_speed_drop" and self.installed_at >= self.shock_at:
            raise ValueError("Install the scenario after reset and before its speed shock")
        self.applied_at = None
        self.lane_caps = []
        self.metrics = []
        self.listener_id = None
        self._closed = False
        self._next_metric = (math.floor(self.installed_at / metric_every)+1)*metric_every
        self._record(self.installed_at)

    def _record(self, now, scheduled_at=None):
        snapshot = self.ledger.snapshot()
        # The native listener runs after simulationStep but before the bridge
        # advances its ledger.  Integrating its current passenger counts over
        # that last tick observes the same endpoint without mutating the ledger.
        elapsed = float(now)-float(snapshot.time_seconds)
        if elapsed < -1e-8:
            raise RuntimeError("SUMO passenger-time ledger is ahead of its native clock")
        wait = snapshot.accrued_wait_seconds + snapshot.waiting_passengers*max(elapsed, 0.)
        ride = snapshot.accrued_in_vehicle_seconds + snapshot.in_vehicle_passengers*max(elapsed, 0.)
        record = dict(time_seconds=float(now),
            scheduled_time_seconds=None if scheduled_at is None else float(scheduled_at),
            waiting_passengers=int(snapshot.waiting_passengers),
            in_vehicle_passengers=int(snapshot.in_vehicle_passengers),
            accrued_wait_seconds=float(wait), accrued_in_vehicle_seconds=float(ride),
            accrued_generalized_passenger_seconds=float(2*wait+ride))
        if not self.metrics or now > self.metrics[-1]["time_seconds"]:
            self.metrics.append(record)
        else:
            self.metrics[-1] = record

    def step(self, _requested_step=0):
        # SUMO passes the simulationStep argument (normally zero), not the
        # actual simulator time; always ask the native simulation clock.
        now = float(self.api.simulation.getTime())
        if (self.name == "local_speed_drop" and self.applied_at is None
                and now >= self.shock_at):
            for lane_id in LANE_IDS:
                original = float(self.api.lane.getMaxSpeed(lane_id))
                requested = original*self.factor
                self.api.lane.setMaxSpeed(lane_id, requested)
                observed = float(self.api.lane.getMaxSpeed(lane_id))
                if not math.isclose(observed, requested, rel_tol=1e-10, abs_tol=1e-10):
                    raise RuntimeError(f"SUMO did not apply the speed cap on {lane_id}")
                self.lane_caps.append(dict(lane_id=lane_id, original_mps=original,
                    requested_mps=requested, observed_mps=observed))
            self.applied_at = now
        if now >= self._next_metric:
            self._record(now, self._next_metric)
            self._next_metric = (math.floor(now/self.metric_every)+1)*self.metric_every
        return True

    def capture_final(self):
        self._record(float(self.api.simulation.getTime()))
        return self.metrics[-1]

    def metadata(self):
        return dict(name=self.name, installed_at_seconds=self.installed_at,
            scheduled_shock_at_seconds=self.shock_at,
            actual_shock_at_seconds=self.applied_at,
            speed_factor=self.factor, edge_id=EDGE_ID, lane_ids=list(LANE_IDS),
            edge_length_meters=723.30, directly_affected_lines=list(AFFECTED_LINES),
            upstream_stop=UPSTREAM_STOP, downstream_stop=DOWNSTREAM_STOP,
            lane_caps=list(self.lane_caps), metric_every_seconds=self.metric_every,
            mechanism="native lane speed cap; no intervention indicator in policy input",
            time_semantics="after the native tick at actual_shock_at_seconds; subsequent motion uses the new cap",
            cost_semantics="2*accrued_wait_seconds + accrued_in_vehicle_seconds; includes unfinished passengers")

    def close(self):
        if not self._closed and self.listener_id is not None:
            self.api.removeStepListener(self.listener_id)
        self._closed = True


def install_scenario(env, name, *, shock_at=9000., factor=.65, metric_every=300.):
    """Install after target ``reset``; close before closing its SUMO backend."""
    from bus_h2o.sumo_env import rl_bridge

    api = rl_bridge.traci

    class NativeScenario(ScenarioRecorder, api.StepListener):
        pass

    hook = NativeScenario(api, env._bridge.passenger_time_ledger, name,
        shock_at=shock_at, factor=factor, metric_every=metric_every)
    hook.listener_id = api.addStepListener(hook)
    if hook.listener_id is None:
        raise RuntimeError("SUMO rejected the native scenario StepListener")
    return hook


def is_direct_interval(line_id, station_id, next_station_id):
    """The next-stop physical label that crosses the intervened edge."""
    return (str(line_id) in AFFECTED_LINES and str(station_id) == UPSTREAM_STOP
            and str(next_station_id) == DOWNSTREAM_STOP)


def prediction_period(started_at, *, shock_at=9000., early_seconds=3000., late_at=15000.):
    """Assign by prediction issue time, never by the later label-release time."""
    if started_at < shock_at:
        return "before"
    if started_at < shock_at+early_seconds:
        return "early_after"
    if started_at >= late_at:
        return "late_after"
    return "middle_after"
