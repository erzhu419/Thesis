"""Separate exogenous arrivals and measure fixed-horizon passenger cohorts."""

from types import MethodType

import numpy as np


class ArrivalFlow:
    def __init__(self, seed, line_index, scene, start, end, extra_multiplier):
        streams = np.random.SeedSequence([seed, line_index, 81008]).spawn(3)
        self.baseline_rng, self.extra_rng, self.order_rng = (
            np.random.default_rng(stream) for stream in streams
        )
        self.scene, self.start, self.end = scene, start, end
        self.extra_multiplier = extra_multiplier

    def draw(self, rates, interval, time):
        baseline = self.baseline_rng.poisson(np.asarray(rates) * interval)
        extra = np.zeros_like(baseline)
        if self.scene == "burst50" and self.start <= time < self.end:
            extra = self.extra_rng.poisson(
                np.asarray(rates) * interval * self.extra_multiplier
            )
        return baseline, extra


def passenger_class(original):
    class CohortPassenger(original):
        __slots__ = ("cohort",)

        def __init__(self, time, station, destination, cohort):
            super().__init__(time, station, destination)
            self.cohort = cohort

    return CohortPassenger


def install_arrivals(line, passenger_type, flow):
    def generate(self, current_time, update_interval):
        hour = max(6, min(6 + int(current_time) // 3600, 19))
        if self._pax_cache_hour != hour:
            self._rebuild_pax_cache(hour)
        if not len(self._pax_flat_rates):
            return
        baseline, extra = flow.draw(self._pax_flat_rates, update_interval, current_time)
        for original_count, extra_count, (station, destination) in zip(
            baseline, extra, self._pax_flat_map
        ):
            passengers = [
                passenger_type(current_time, station, destination, "baseline")
                for _ in range(int(original_count))
            ]
            passengers.extend(
                passenger_type(current_time, station, destination, "extra")
                for _ in range(int(extra_count))
            )
            # Equal-time arrivals must not give the baseline cohort priority.
            if original_count and extra_count:
                flow.order_rng.shuffle(passengers)
            station.waiting_passengers.extend(passengers)
            station.total_passenger.extend(passengers)

    line._batch_passenger_arrival = MethodType(generate, line)


def passenger_times(passenger, horizon):
    boarding = horizon if passenger.boarding_time is None else min(
        horizon, passenger.boarding_time
    )
    waiting = max(0.0, boarding - passenger.appear_time)
    riding = 0.0
    if passenger.boarding_time is not None:
        arrival = horizon if passenger.arrive_time is None else min(
            horizon, passenger.arrive_time
        )
        riding = max(0.0, arrival - boarding)
    return waiting, riding


def summarize_passengers(passengers, horizon, start, end):
    names = ("all", "baseline", "extra", "baseline_before", "baseline_during", "baseline_after")
    groups = {
        name: dict(passengers=0, waiting_seconds=0.0, onboard_seconds=0.0,
                   boarded=0, completed=0)
        for name in names
    }
    for passenger in passengers:
        waiting, riding = passenger_times(passenger, horizon)
        keys = ["all", passenger.cohort]
        if passenger.cohort == "baseline":
            period = "before" if passenger.appear_time < start else (
                "during" if passenger.appear_time < end else "after"
            )
            keys.append("baseline_" + period)
        for key in keys:
            row = groups[key]
            row["passengers"] += 1
            row["waiting_seconds"] += waiting
            row["onboard_seconds"] += riding
            row["boarded"] += int(passenger.boarding_time is not None
                                  and passenger.boarding_time <= horizon)
            row["completed"] += int(passenger.arrive_time is not None
                                    and passenger.arrive_time <= horizon)
    for row in groups.values():
        count = row["passengers"]
        row["mean_wait_seconds"] = row["waiting_seconds"] / count if count else None
        row["mean_onboard_seconds"] = row["onboard_seconds"] / count if count else None
        row["mean_generalized_seconds"] = (
            2 * row["waiting_seconds"] + row["onboard_seconds"]
        ) / count if count else None
        row["completion_rate"] = row["completed"] / count if count else None
    return groups
