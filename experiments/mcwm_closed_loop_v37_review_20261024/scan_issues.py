"""Read-only issue/F support after18000; never opens outcomes or checkpoints."""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from experiments.mcwm_closed_loop_action_local_v3_20261020.action_contract import ExecutionAction
from experiments.mcwm_closed_loop_paired_coverage_v37_20261024.collect import action_choices

LINES = ('7X', '102X', '705X')
SCENARIOS = ('stable', 'local_speed_drop')
START, END, BIN = 18000, 24000, 300


def scan_trace(path, transform):
    bins = {line: {clock: [clock, 0, 0, None, None]
                  for clock in range(START, END, BIN)} for line in LINES}
    total, selected = 0, 0
    with Path(path).open() as source:
        for text in source:
            row = json.loads(text)
            total += 1
            line, clock = row['line_id'], float(row['started_at'])
            if line not in bins or not START <= clock < END:
                continue
            selected += 1
            cell = bins[line][START + int((clock - START) // BIN) * BIN]
            cell[1] += 1
            cell[2] += action_choices(transform, row['s']) is not None
            cell[3] = clock if cell[3] is None else min(cell[3], clock)
            cell[4] = clock if cell[4] is None else max(cell[4], clock)
    lines = {}
    for line, values in bins.items():
        rows = list(values.values())
        active = [row for row in rows if row[1]]
        lines[line] = dict(issues=sum(row[1] for row in rows),
            F_jointly_reachable=sum(row[2] for row in rows),
            first=active[0][3] if active else None,
            last=active[-1][4] if active else None, bins=rows)
    return dict(trace_rows=total, focus_issues_after18000=selected, lines=lines)


def run(root):
    root = Path(root).resolve()
    repo = HERE.parents[1]
    metadata = json.loads((repo / 'experiments/mcwm_closed_loop_formal_20261020'
                          '/inputs/history_metadata.json').read_text())
    transform = ExecutionAction(metadata['contract'])
    return dict(root=str(root), interval=[START, END], bin_seconds=BIN,
        columns=['bin_start', 'issues', 'F_jointly_reachable', 'first', 'last'],
        cases={scenario: scan_trace(root / 'runs/collection' / scenario
                                   / 'prefix_trace.jsonl', transform)
               for scenario in SCENARIOS},
        scope=dict(reads='prefix issue line, native issue clock and canonical state; contract metadata',
            F_holds_seconds=[0.0, 1.875], pair_truth_read=False,
            model_checkpoint_read=False, new_training=False, selects_windows=False,
            physical_feasibility='Not certified: trace has no actual isAtBusStop or actionStepLength.',
            interpretation='F reachability counts candidate issues only; native stop and action-step checks remain necessary.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    payload = json.dumps(run(args.root), ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    if len(payload.encode()) > 8000:
        raise RuntimeError('Issue support output exceeds8000-byte transfer budget')
    print(payload)
