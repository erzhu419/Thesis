"""Exercise historical bus interfaces without importing their training entrypoints."""

import ast
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import h5py
import numpy as np

from rewards import ridge_reward


HERE = Path(__file__).resolve().parent
SUMO = HERE.parents[2] / "sumo-rl"
H2O = SUMO / "H2Oplus"
CURRENT = HERE.parent / "mcwm_bus/runtime/H2Oplus/bus_h2o/sim_core/bus.py"


def blob(repo, commit, path):
    return subprocess.check_output(
        ["git", "show", f"{commit}:{path}"], cwd=repo, text=True
    )


def definitions(source, names, namespace):
    tree = ast.parse(source)
    selected = [node for node in tree.body if getattr(node, "name", None) in names]
    assert len(selected) == len(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), "historical", "exec"), namespace)
    return namespace


def method(source, name):
    tree = ast.parse(source)
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(matches) == 1
    scope = {"np": np, "DecisionEvent": SimpleNamespace}
    exec(compile(ast.Module(body=matches, type_ignores=[]), name, "exec"), scope)
    return scope[name]


class InterleavedEnv:
    max_agent_num = 2
    sequence = ((0, 0, 0.), (1, 0, 0.), (0, 1, -101.), (1, 1, -202.))

    def reset(self):
        self.index, self.commands, self.done = 0, [], False
        self.update()

    def update(self):
        bus, station, reward = self.sequence[self.index]
        obs = [0., float(bus), float(station)] + [0.] * 12
        self.state, self.reward = {bus: [obs]}, {bus: reward}

    def step_fast(self, actions):
        return self.state, self.reward, self.done

    def capture_full_system_snapshot(self):
        return {}

    def step_to_event(self, actions):
        self.commands.append({str(k): v for k, v in actions.items() if v is not None})
        self.index += 1
        self.done = self.index == len(self.sequence)
        if not self.done:
            self.update()
        return self.state, self.reward, self.done


class Buffer:
    _valid_snap_indices = []

    def __init__(self):
        self.rows = []

    def append(self, **row):
        self.rows.append(row)


def sampler_test():
    source = blob(H2O, "3f20b4a", "SimpleSAC/bus_sampler.py")
    scope = definitions(source, {"_map_raw_to_env", "_extract_active_buses", "BusStepSampler"},
                        {"np": np, "extract_structured_context": lambda snapshot: np.zeros(30)})
    env, buffer = InterleavedEnv(), Buffer()

    def policy(observation, deterministic=False):
        bus = int(observation[0, 1])
        return np.array([[-.5, .8] if bus == 0 else [.5, -.8]], np.float32)

    scope["BusStepSampler"](env, buffer, max_traj_events=4, p_reset=0.).sample(policy, 4)
    rows = [r for r in buffer.rows if r["done"] == 0.]
    assert len(rows) == 2
    for row, reward in zip(rows, (-101., -202.)):
        assert row["obs"][1] == row["next_obs"][1]
        assert row["obs"][2] == 0. and row["next_obs"][2] == 1.
        assert row["reward"] == reward
        np.testing.assert_array_equal(row["next_obs"][-2:], row["action"])
    np.testing.assert_allclose(env.commands[0]["0"], [15., 1.2])
    np.testing.assert_allclose(env.commands[1]["1"], [45., .8])
    return {"nonterminal_rows": len(rows), "same_bus_settlement": True,
            "next_observation_own_previous_action": True, "physical_commands": env.commands}


def speed_test():
    old = method(blob(H2O, "3f20b4a", "bus_h2o/sim_core/bus.py"), "_advance_on_route")
    old.__globals__["LINE_SIGNAL_CONFIG"] = {}
    current = method(CURRENT.read_text(), "_advance_on_route")
    results = []
    for speed in (.8, 1.2):
        route = SimpleNamespace(speed_limit=10., runtime_calibration={},
                                runtime_fixed_overhead_seconds=0,
                                sample_runtime_seconds=lambda *a, **kw: 100.)
        common = dict(current_route=route, _speed_ratio=speed, acceleration=2., deceleration=2.,
                      current_speed=10. * speed, direction_int=1., next_station_dis=500.,
                      last_station_dis=0., absolute_distance=0., _signal_wait_remaining=0.,
                      _signal_stopped=False, _segment_effective_speed=10.,
                      _empirical_runtime_calibrated=True, _calibrated_travel_steps_remaining=None,
                      launch_time=0.)
        legacy, calibrated = SimpleNamespace(**common), SimpleNamespace(**common)
        old(legacy)
        current(calibrated)
        first_step = calibrated.absolute_distance
        for _ in range(99):
            current(calibrated)
        results.append(dict(speed_ratio=speed, historical_first_step_m=legacy.last_station_dis,
                            historical_absolute_distance_increment_m=legacy.absolute_distance,
                            calibrated_first_step_m=first_step,
                            calibrated_distance_after_100_ticks_m=calibrated.absolute_distance))
    assert results[0]["historical_first_step_m"] != results[1]["historical_first_step_m"]
    for row in results:
        assert row["historical_absolute_distance_increment_m"] == 2 * row["historical_first_step_m"]
    assert results[0]["calibrated_first_step_m"] == results[1]["calibrated_first_step_m"]
    for row in results:
        np.testing.assert_allclose(row["calibrated_distance_after_100_ticks_m"], 500.)
    return results


def reward_test():
    old = method(blob(SUMO, "3416538", "SUMO_ruiguang/online_control/rl_env.py"),
                 "_compute_reward_linear")
    rows = []
    for f, b in ((360., 360.), (460., 260.), (545., 360.), (600., 600.)):
        event = SimpleNamespace(forward_headway=f, backward_headway=b,
                                target_forward_headway=360., target_backward_headway=360.,
                                forward_bus_present=True, backward_bus_present=True)
        historical, current = float(old(None, event, 360.)), float(ridge_reward(f, b))
        rows.append(dict(forward_seconds=f, backward_seconds=b,
                         historical_reward=historical, current_reward=current))
    np.testing.assert_allclose(rows[1]["historical_reward"], rows[1]["current_reward"])
    assert rows[2]["historical_reward"] != rows[2]["current_reward"]
    return rows


def dataset_sample():
    with h5py.File(H2O / "bus_h2o/datasets_v2/merged_all_v2.h5") as data:
        index = np.linspace(0, len(data["actions"])-1, 4096, dtype=int)
        s, ns, a = (data[name][index] for name in ("observations", "next_observations", "actions"))
        active = data["terminals"][index].reshape(-1) < .5
        counts = dict(sampled_rows=len(index), nonterminal_rows=int(active.sum()),
                      different_line_or_bus=int(np.count_nonzero(active & np.any(s[:, :2] != ns[:, :2], axis=1))),
                      different_direction=int(np.count_nonzero(active & (s[:, 4] != ns[:, 4]))),
                      previous_action_mismatches=int(np.count_nonzero(active & np.any(abs(ns[:, -2:]-a) > 1e-4, axis=1))),
                      variable_speed_rows=int(np.count_nonzero(abs(a[:, 1]) > 1e-5)))
        assert counts["different_line_or_bus"] == counts["different_direction"] == counts["previous_action_mismatches"] == 0
        assert counts["variable_speed_rows"] > 0
        return dict(observation_shape=list(data["observations"].shape), action_shape=list(data["actions"].shape),
                    **counts, provenance="Current local v2 file, not identified as the February training dataset")


if __name__ == "__main__":
    result = dict(parent_commit="3416538", h2oplus_commit="3f20b4a", sampler=sampler_test(),
                  speed=speed_test(), rewards=reward_test(), local_dataset=dataset_sample())
    (HERE / "historical_comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
