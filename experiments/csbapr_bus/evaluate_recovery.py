"""Frozen bus-policy response to a finite demand pulse, evaluated where weights reside."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from io import StringIO
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import time

import numpy as np
import torch

ROOT = Path(os.environ.get('THESIS_CSBAPR_ROOT', '/home/erzhu419/csbapr_exp'))
sys.path.insert(0, str(ROOT / 'CS-BAPR'))
sys.path.insert(0, str(ROOT / 'offline-sumo/env'))
from scripts.test_multiline_convergence import run_episode_multiline
from sim_core.sim import MultiLineEnv
from csbapr.agent import CSBAPRAgent
from csbapr.config import CSBAPRConfig

METHODS = {'bapr-pristine': 'logs_hopper_v2', 'csbapr-no-nau': 'logs_hopper_v2',
           'csbapr': 'logs_v14'}
PROTOCOL = dict(training_seeds=[0, 1], evaluation_seed_base=14301,
                methods=list(METHODS), horizon_seconds=18000, window_seconds=300,
                pulse_start=3600, pulse_end=5400, pulse_multiplier=5.,
                recovery_relative_tolerance=.10, recovery_absolute_tolerance=20.,
                recovery_consecutive_windows=3)


class ObservedEnv(MultiLineEnv):
    def __init__(self, path):
        super().__init__(path)
        self.windows = {}

    def window(self, index):
        return self.windows.setdefault(index, dict(samples=0, waiting=0., onboard=0.,
                                                   hold_sum=0., imbalance_sum=0., decisions=0))

    def step(self, actions, **kwargs):
        before = self.current_time
        if before >= PROTOCOL['horizon_seconds']:
            self.done = True
            return self._aggregate_state(), self._aggregate_reward(), True
        waiting = sum(len(station.waiting_passengers)
                      for line in self.line_map.values() for station in line.stations)
        onboard = sum(len(bus.passengers)
                      for line in self.line_map.values() for bus in line.bus_all)
        row = self.window(int(before)//PROTOCOL['window_seconds'])
        row['waiting'] += waiting
        row['onboard'] += onboard
        row['samples'] += 1
        state, reward, done = super().step(actions, **kwargs)
        return state, reward, done or self.current_time >= PROTOCOL['horizon_seconds']


class ObservedPolicy:
    def __init__(self, agent, env):
        self.agent, self.env = agent, env

    def select_action(self, state, deterministic=False):
        action = self.agent.select_action(state, deterministic=deterministic)
        row = self.env.window(int(round(float(state[10])*72000))//PROTOCOL['window_seconds'])
        row['decisions'] += 1
        row['hold_sum'] += float(np.clip((action[0]+1)*30, 0, 60))
        row['imbalance_sum'] += abs(float(state[5]-state[6]))*600
        return action


def evaluate(job):
    method, seed, scene, out = job[:4]
    evaluation_seed = job[4] if len(job) > 4 else PROTOCOL['evaluation_seed_base'] + seed
    schedule = job[5] if len(job) > 5 else None
    out = Path(out)
    suffix = f'_eval{evaluation_seed}' if len(job) > 4 else ''
    path = out / f'{method}_seed{seed}_{scene}{suffix}.json'
    if path.exists():
        return json.loads(path.read_text())
    torch.set_num_threads(1)
    checkpoint = ROOT / METHODS[method] / 'ckpt' / f'{method}_seed{seed}_best.pt'
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    config = CSBAPRConfig(**payload['config'])
    config.device = 'cpu'
    agent = CSBAPRAgent(15, 1, config)
    agent.actor.load_state_dict(payload['actor'])
    agent.actor.eval()
    random.seed(evaluation_seed)
    np.random.seed(evaluation_seed)
    torch.manual_seed(evaluation_seed)
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='thesis-csbapr-') as temporary:
        temporary = Path(temporary)
        source = ROOT / 'offline-sumo/env/calibrated_env'
        shutil.copy2(source / 'config.json', temporary / 'config.json')
        os.symlink(source / 'data', temporary / 'data', target_is_directory=True)
        with redirect_stdout(StringIO()):
            env = ObservedEnv(str(temporary))
            if schedule is not None:
                env.set_ood_schedule(schedule)
            elif scene == 'pulse':
                env.set_ood_schedule([(0, 1.), (PROTOCOL['pulse_start'], PROTOCOL['pulse_multiplier']),
                                      (PROTOCOL['pulse_end'], 1.)])
            rewards, decisions, _ = run_episode_multiline(
                env, ObservedPolicy(agent, env), deterministic=True, train=False, config=None)
        rows = []
        for index, window in sorted(env.windows.items()):
            if not window['samples']:
                continue
            rows.append(dict(time_start=index*300, time_end=(index+1)*300,
                             mean_waiting=window['waiting']/window['samples'],
                             mean_onboard=window['onboard']/window['samples'],
                             wait_passenger_seconds=window['waiting'],
                             ride_passenger_seconds=window['onboard'],
                             mean_hold=window['hold_sum']/max(window['decisions'], 1),
                             mean_headway_imbalance=window['imbalance_sum']/max(window['decisions'], 1),
                             decisions=window['decisions'], samples=window['samples']))
        if env.current_time != PROTOCOL['horizon_seconds']:
            raise RuntimeError(f'Episode ended before observation horizon: {env.current_time}')
        people = [p for line in env.line_map.values() for station in line.stations
                  for p in station.total_passenger]
        result = dict(method=method, training_seed=seed, evaluation_seed=evaluation_seed,
                      scene=scene, horizon_seconds=env.current_time,
                      generated_passengers=len(people),
                      arrived_passengers=sum(p.arrived for p in people),
                      waiting_tail=sum(len(s.waiting_passengers) for line in env.line_map.values() for s in line.stations),
                      reward=sum(rewards.values()), decisions=decisions,
                      wait_passenger_seconds=sum(row['wait_passenger_seconds'] for row in rows),
                      ride_passenger_seconds=sum(row['ride_passenger_seconds'] for row in rows),
                      windows=rows, elapsed_seconds=time.monotonic()-start)
    path.write_text(json.dumps(result, indent=2)+'\n')
    return result


def main(out, workers):
    out.mkdir(parents=True, exist_ok=True)
    (out / 'protocol.json').write_text(json.dumps(PROTOCOL, indent=2)+'\n')
    jobs = [(method, seed, scene, str(out)) for method in METHODS
            for seed in PROTOCOL['training_seeds'] for scene in ('normal', 'pulse')]
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as pool:
        futures = [pool.submit(evaluate, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            print(json.dumps({key: result[key] for key in ('method', 'training_seed', 'scene',
                                                          'elapsed_seconds', 'wait_passenger_seconds')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    main(args.out.resolve(), args.workers)
