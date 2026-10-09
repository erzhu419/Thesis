"""Exercise the deployed SUMO inputs through the first real decision event."""

import json

import run
from evaluate_control import AllLinesSumoEnv, ROOT, SUMO_ROOT, build_scenario

out = run.HERE / "runs/runtime_smoke"
cfg, objects = build_scenario(out, 1.0, 13401, 120)
env = AllLinesSumoEnv(str(SUMO_ROOT / "online_control"),
                     str(ROOT / "bus_h2o/network_data/a_sorted_busline_edge.xml"),
                     episode_seconds=24000, sumo_seed=13301, sumo_cfg=cfg,
                     passenger_obj_xml=objects, observation_schema="v5", sumo_backend="libsumo")
try:
    env.reset()
    env.step_to_event({})
    import libsumo
    assert env._current_events and not env.done
    print(json.dumps({"runtime_smoke": "passed", "version": libsumo.getVersion(),
                      "first_events": len(env._current_events),
                      "time": env._bridge.current_time}), flush=True)
finally:
    env.close()
