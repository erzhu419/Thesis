"""Resident sequential evidence for every native prefix issue and command."""

import json
from pathlib import Path

import numpy as np


class NativePrefixTrace:
    def __init__(self, path, reference=None):
        self.path = Path(path)
        self.writer = self.path.open('w', encoding='utf-8')
        self.reference = None if reference is None else Path(reference).open(encoding='utf-8')
        self.observed_issues = self.matched_issues = 0
        self.comparison_exact = True

    def observe(self, event, snapshot, nominal_action=None):
        row = {key: str(event[key]) for key in (
            'agent', 'line_id', 'vehicle_id', 'trip_id', 'station_id', 'domain')}
        row.update(sequence=int(event['sequence']), episode_id=int(event['episode_id']),
            started_at=float(event['started_at']),
            issue_ledger={key: float(getattr(snapshot, key)) for key in (
                'time_seconds', 'waiting_passengers', 'in_vehicle_passengers',
                'accrued_wait_seconds', 'accrued_in_vehicle_seconds')})
        row.update({key: np.asarray(event[key]).tolist() for key in (
            's', 'raw_s', 'event', 'a', 'proposed_a')})
        self.writer.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
        self.observed_issues += 1
        if self.reference is None:
            return
        line = self.reference.readline()
        if not line:
            self.comparison_exact = False
            raise RuntimeError(f'Native prefix reference ended before issue {row["sequence"]}')
        expected = json.loads(line)
        compared = dict(row)
        if nominal_action is not None:
            compared['a'] = np.asarray(nominal_action).tolist()
        differing = [key for key in expected.keys() | compared.keys()
                     if expected.get(key) != compared.get(key)]
        if differing:
            self.comparison_exact = False
            raise RuntimeError(f'Native prefix differs at issue {row["sequence"]}: '
                               + ', '.join(sorted(differing)))
        self.matched_issues += 1

    def close(self):
        self.writer.close()
        if self.reference is not None:
            self.reference.close()
