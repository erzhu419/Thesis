"""The visible packet51 actor and frozen, real-clock remaining-episode costs."""
from pathlib import Path
import json
import time

import torch

from experiments.mcwm_closed_loop_tail_policy_exchange_v71_20261026.support import (
    PrefixComplete, acquire_tail_slot)

ARMS = {'low': 'fresh', 'high': 'no_fresh'}
PACKETS, RECORDS, UPDATES = 51, 255, 408
COUNTS = dict(critic=UPDATES, value=UPDATES, actor=UPDATES, alpha=UPDATES,
              policy=UPDATES, world=0, trust=0)


def snapshot(point):
    value = {key: float(getattr(point, key)) for key in (
        'time_seconds', 'accrued_wait_seconds', 'accrued_in_vehicle_seconds',
        'waiting_passengers', 'in_vehicle_passengers')}
    value['accrued_generalized_passenger_seconds'] = (
        2*value['accrued_wait_seconds']+value['accrued_in_vehicle_seconds'])
    return value


def validate_snapshot(value, *, seed, arm):
    return bool(value.get('seed') == seed and value.get('source_arm') == arm
        and value.get('packets') == PACKETS and value.get('counts') == COUNTS
        and value.get('covariates') == RECORDS
        and value.get('complete_outcomes') == (RECORDS if arm == 'fresh' else 0)
        and value.get('publication_deadline') == 16200
        and value.get('data_deadline') == 16200
        and 0 <= value.get('max_available_at', -1) <= 16200
        and value.get('published_clock', -1) >= 16200
        and value.get('processed_at') == value.get('published_clock')
        and value.get('future_training_used') is False)


def publish_actor(path, learner, *, seed, arm, counts, published_clock, max_available_at):
    if counts != COUNTS:
        raise RuntimeError('Actor must be published after exactly408 packet51 updates')
    metadata = dict(seed=seed, source_arm=arm, packets=PACKETS, counts=counts,
        covariates=RECORDS, complete_outcomes=RECORDS if arm == 'fresh' else 0,
        published_clock=float(published_clock), publication_deadline=16200,
        data_deadline=16200, max_available_at=float(max_available_at), processed_at=float(published_clock),
        future_training_used=False)
    if not validate_snapshot(metadata, seed=seed, arm=arm):
        raise RuntimeError('Actor publication has the wrong visible feedback deadline')
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    pending = path.with_suffix('.pending')
    torch.save(dict(metadata=metadata, actor={key: value.detach().cpu().clone()
        for key, value in learner.actor.state_dict().items()}), pending)
    pending.replace(path)
    return metadata


def opposite_actor(path, *, seed, arm, timeout=3600):
    path = Path(path)
    started = last = time.monotonic()
    while not path.exists():
        failure = path.parent/'failure.json'
        if failure.exists():
            raise RuntimeError('Peer prefix failed; no substitute actor: '+str(failure))
        now = time.monotonic()
        if now-started > timeout:
            raise TimeoutError('Peer packet51 actor not ready; restored prefix retained')
        if now-last >= 45:
            print('SCHEDULEURM_PROGRESS '+json.dumps(dict(
                phase='wait_same_seed_packet51_actor', seed=seed, arm=arm)), flush=True)
            last = now
        time.sleep(.2)
    value = torch.load(path, map_location='cpu', weights_only=False)
    if not validate_snapshot(value.get('metadata', {}), seed=seed, arm=arm):
        raise RuntimeError('Peer actor contains a different seed, budget or future training')
    return value['actor'], value['metadata']


def trace_child(old, path):
    from experiments.mcwm_closed_loop_response_residual_v11_20261021.prefix_trace import NativePrefixTrace
    observed, matched = old.observed_issues, old.matched_issues
    old.close()
    result = NativePrefixTrace(path)
    result.observed_issues, result.matched_issues = observed, matched
    return result


def strip_profiler(value):
    if isinstance(value, dict):
        return {key: strip_profiler(item) for key, item in value.items()
                if key != 'regularizer_wall_seconds'}
    if isinstance(value, list):
        return [strip_profiler(item) for item in value]
    return value


def cost_windows(metrics, denominator, cut_snapshot):
    if denominator != 15079:
        raise RuntimeError('Departed passenger cohort changed')
    points = {str(int(row['time_seconds'])): row for row in metrics
              if row['time_seconds'] in (18300, 20100, 24000)}
    if set(points) != {'18300', '20100', '24000'}:
        raise RuntimeError('Missing actual native cost endpoint')
    starts = dict(remaining=cut_snapshot, tail=points['18300'], post20100=points['20100'])
    final, result = points['24000'], {}
    for name, initial in starts.items():
        wait = final['accrued_wait_seconds']-initial['accrued_wait_seconds']
        ride = final['accrued_in_vehicle_seconds']-initial['accrued_in_vehicle_seconds']
        result[name] = dict(wait=wait, ride=ride, cost=2*wait+ride,
            denominator=denominator, cost_per_passenger=(2*wait+ride)/denominator)
    return points, result
