"""Project existing v29 JSON and small logs; never modify native runs."""
import json
from pathlib import Path

ROOT = Path('/home/zhengliang01/scheduleurm_work/Thesis/experiments/mcwm_closed_loop_tail_truth_v29_20261023')


def read(path):
    return json.loads(path.read_text())


def project():
    cases = {}
    for scenario in ('stable','local_speed_drop'):
        root = ROOT/'runs'/'seed6202'/scenario
        main = read(root/'summary.json')
        branches = {arm:read(root/'branches'/arm/'summary.json') for arm in ('low','high')}
        logs = {}
        for arm in branches:
            path = root/'branches'/(arm+'.log')
            matches = []
            count = 0
            with path.open() as handle:
                for line in handle:
                    if any(word in line for word in ('WARNING','Error:','Traceback')):
                        count += 1
                        matches.append(line.strip()[:400])
                        matches = matches[-3:]
            logs[arm] = dict(bytes=path.stat().st_size,warning_error_lines=count,last_matches=matches,
                files={p.name:p.stat().st_size for p in (root/'branches'/arm).iterdir() if p.is_file()})
        transitions = [branches[arm]['global_transition'] for arm in ('low','high')]
        cases[scenario] = dict(
            parent_status=main['status'], parent_wall_seconds=main['wall_seconds'],
            parent_execution_wall=main['execution_wall'],budget=main['budget'],
            initial_updates=main['initializer']['initial_updates'],
            main_training_stopped_at=main['main_training_stopped_at'],
            post_budget_counts=main['post_budget_counts'],
            post_budget_shadow_counts=main['post_budget_shadow_counts'],
            world_updates=main['world_updates'],trust_updates=main['trust_updates'],
            suffix_counters={arm:value['counters'] for arm,value in branches.items()},
            suffix_native_steps={arm:value['native']['suffix_steps'] for arm,value in branches.items()},
            next_transition_equal={key:transitions[0].get(key)==transitions[1].get(key)
                for key in ('ns','raw_ns','r','dt','completed_at','next_vehicle_id','terminal')
                if key in transitions[0]},
            logs=logs)
    return dict(cases=cases,remote_summary_bytes=(ROOT/'truth_summary.json').stat().st_size)


if __name__ == '__main__':
    value = project()
    encoded = json.dumps(value,separators=(',',':'),allow_nan=False)
    assert len(encoded.encode()) < 10000
    print(encoded)
