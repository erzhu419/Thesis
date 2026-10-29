"""Reduce resident passenger ledgers; no learner or environment is constructed."""
import json
from pathlib import Path
import sys

ANCHORS = (0, 9000, 12000, 18300, 24000)
N = 15079


def analyze(base):
    result = dict(new_native_seconds=0, new_optimizer_updates=0, scenes={})
    for scene in ('stable', 'local_speed_drop'):
        tables, finals = {}, {}
        for arm in ('fresh', 'no_fresh'):
            folder = base/'runs'/scene/arm
            summary = json.loads((folder/'summary.json').read_text())
            rows = json.loads((folder/'metrics.json').read_text())
            table = {float(row['time_seconds']): row for row in rows}
            table[0] = dict(accrued_wait_seconds=0, accrued_in_vehicle_seconds=0)
            table[24000] = summary['ledger']
            # Without exact anchors the interval cannot be attributed as reported.
            if any(t not in table for t in ANCHORS):
                raise ValueError('Missing exact ledger anchor: '+scene+'/'+arm)
            tables[arm] = table
            finals[arm] = summary['passengers']['state_counts_at_horizon']

        def cumulative(t):
            a, b = tables['fresh'][t], tables['no_fresh'][t]
            wait = 2*(a['accrued_wait_seconds']-b['accrued_wait_seconds'])/N
            ride = (a['accrued_in_vehicle_seconds']-b['accrued_in_vehicle_seconds'])/N
            return dict(weighted_wait=wait, ride=ride, cost=wait+ride)

        values = {str(t): cumulative(t) for t in ANCHORS}
        segments = []
        for start, end in zip(ANCHORS, ANCHORS[1:]):
            a, b = values[str(start)], values[str(end)]
            segments.append(dict(start=start, end=end,
                fresh_minus_matched={k: b[k]-a[k] for k in a}))
        result['scenes'][scene] = dict(cumulative=values, segments=segments,
            final_passenger_states=finals)
    result['interpretation'] = ('Fixed ledger intervals, all divided by 15079; '
        'time of accrued cost is not the time or cause of the responsible action.')
    return result


if __name__ == '__main__':
    print(json.dumps(analyze(Path(sys.argv[1])), separators=(',', ':'), allow_nan=False))
