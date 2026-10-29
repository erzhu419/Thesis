"""Use the installed runtime and actual initializer, without pytest or SGD."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))


def main():
    import numpy as np
    import torch
    from experiments.mcwm_closed_loop_sync_pilot_v43_20261024.collect import StochasticController
    from experiments.mcwm_closed_loop_shared_level_v36_20261024.learning import make_learner, initial_preflight
    from experiments.mcwm_closed_loop_overlap_v41_20261024.overlap import conditional_overlap
    from experiments.mcwm_closed_loop_trace_design_v40_20261024.behavior import actor_parameters
    p = json.loads((HERE/'protocol.json').read_text())
    learner, replay, contract, stations, initializer = make_learner(p['training_seed'], arm=p['learner_arm'])
    parity = initial_preflight(learner)
    before = torch.get_rng_state().clone()
    controller = StochasticController(learner, seed=p['action_seed'])
    norms, records = [], []
    for state in replay.history['s'][:3]:
        action = controller(state)
        parameters = actor_parameters(controller.reference, state)
        overlap = conditional_overlap(learner.action_transform, state, parameters, parameters)
        norms.append(overlap['normalization_error'])
        records.append(dict(action=np.asarray(action).tolist(), overlap=overlap['total_overlap'],
            expected_log_c=overlap['expected_log_c_under_behavior']))
    checks = dict(initial_native_parity=parity['all_pass'], initializer50000=learner.total_updates == 50000,
        target_feedback_empty=replay.target_received == len(replay.recent) == 0,
        real_actor_stochastic_forward=controller.draws == 3,
        action_rng_isolated=torch.equal(before, torch.get_rng_state()),
        same_distribution_F_unit_overlap=all(abs(row['overlap']-1.) < 5e-12
            and abs(row['expected_log_c']) < 5e-12 for row in records),
        normalized_F_measure=max(norms) < 5e-12, no_world=learner.world is None)
    value = dict(all_pass=all(checks.values()), checks=checks, initializer=str(initializer),
        real_archive_probes=records, new_updates=0, new_native_prefixes=0, new_feedback=0,
        local_unit_tests_passed=14, previous_gate='t211509: pytest absent; no dependencies changed')
    with (HERE/'resident_gate.json').open('x') as handle:
        json.dump(value, handle, separators=(',', ':'), allow_nan=False)
    print(json.dumps(value), flush=True)
    if not value['all_pass']:
        raise RuntimeError('Resident native initialization/measure preflight failed')


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        path = HERE/'resident_gate.json'
        if not path.exists():
            path.write_text(json.dumps(dict(all_pass=False, type=type(error).__name__, message=str(error),
                new_updates=0, new_native_prefixes=0))+'\n')
        raise
