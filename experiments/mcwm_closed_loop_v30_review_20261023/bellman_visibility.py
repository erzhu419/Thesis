"""Project original next-event label differences; no replay or native stepping."""
from pathlib import Path
import json

ROOT = Path('/home/zhengliang01/scheduleurm_work/Thesis/experiments/mcwm_closed_loop_physical_execution_v30_20261023')


def project():
    cases = {}
    for scenario in ('stable','local_speed_drop'):
        root = ROOT/'runs'/'seed6202'/scenario/'branches'
        rows = {arm:json.loads((root/arm/'summary.json').read_text())['global_transition']
            for arm in ('low','mid','high')}
        contrasts = {}
        for low,high in (('low','mid'),('low','high'),('mid','high')):
            result = {key:rows[low][key] == rows[high][key]
                for key in ('r','dt','terminal','next_vehicle_id','completed_at','ns','raw_ns')}
            for key in ('ns','raw_ns'):
                difference = [[index,a,b] for index,(a,b) in enumerate(zip(rows[low][key],rows[high][key])) if a != b]
                result[key+'_difference_count'] = len(difference)
                result[key+'_first_differences'] = difference[:6]
            contrasts[high+'_minus_'+low] = result
        cases[scenario] = dict(labels={arm:{key:row[key] for key in
            ('a','r','dt','terminal','next_vehicle_id','completed_at')} for arm,row in rows.items()},
            contrasts=contrasts)
    return dict(cases=cases,source='Existing original v30 global_transition JSON only; exact field comparisons, no training or native branch.')


if __name__ == '__main__':
    encoded = json.dumps(project(),separators=(',',':'),allow_nan=False)
    assert len(encoded.encode())<=5000
    print(encoded)
