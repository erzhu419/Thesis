"""Read-only target integration check using resident arrays; no optimizer or env."""
from copy import deepcopy
import json
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.mcwm_closed_loop_fresh_attribution_v25_20261023.learning import (
    _batch, _join, load_learner as original_loader)
from experiments.mcwm_closed_loop_fresh_value_v64_20261025.learning import load_learner
import numpy as np
import torch


def snapshot(learner):
    return (deepcopy(learner.rng.bit_generator.state), torch.get_rng_state().clone(),
            np.random.get_state(), random.getstate())


def restore(learner, state):
    learner.rng.bit_generator.state = deepcopy(state[0])
    torch.set_rng_state(state[1])
    np.random.set_state(state[2])
    random.setstate(state[3])


def rng_equal(left, right):
    return (left[0] == right[0] and torch.equal(left[1], right[1])
            and left[2][0] == right[2][0] and np.array_equal(left[2][1], right[2][1])
            and left[2][2:] == right[2][2:] and left[3] == right[3])


def main():
    old = original_loader(6201, arm='no_fresh')[0]
    unchanged = load_learner(6201, arm='no_fresh')[0]
    fresh = load_learner(6201, arm='fresh')[0]
    replay = old.replay_buffer
    batch = _join(_join(_batch(replay.history, np.arange(64), old.device),
                       _batch(replay.history, np.arange(64, 128), old.device)),
                  _batch(replay.source, np.arange(128), old.device))
    start = snapshot(old)
    results, states = [], []
    for learner in (old, unchanged, fresh):
        restore(learner, start)
        results.append(learner._target(batch, backup='h2oplus'))
        states.append(snapshot(learner))
    with torch.no_grad():
        expected = (fresh.config.reward_scale*batch['rewards'][64:128]
                    + batch['discounts'][64:128]*(1-batch['terminals'][64:128])
                    * fresh.value(batch['next_observations'])[64:128])
    checks = dict(no_fresh_target_exact=torch.equal(results[0], results[1]),
        historical_rows_exact=torch.equal(results[0][:64], results[2][:64]),
        source_rows_exact=torch.equal(results[0][128:], results[2][128:]),
        fresh_value_target_exact=torch.equal(expected, results[2][64:128]),
        no_fresh_rng_exact=rng_equal(states[0], states[1]),
        fresh_rng_exact=rng_equal(states[0], states[2]),
        initialization_unchanged=all(x.total_updates == 50000 for x in (old, unchanged, fresh)),
        no_feedback_admitted=all(x.replay_buffer.target_received == 0 for x in (old, unchanged, fresh)))
    report = dict(all_pass=all(checks.values()), checks=checks,
                  training_updates=0, native_branches=0, feedback_slots=0)
    with (HERE/'component_precheck.json').open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps(report))
    if not report['all_pass']:
        raise RuntimeError('Native target component integration failed')


if __name__ == '__main__':
    main()
