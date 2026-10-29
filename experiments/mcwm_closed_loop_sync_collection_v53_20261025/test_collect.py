"""Budget replacement and genuine event endpoints for the new factual stream."""
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('THESIS_H2O_ROOT', str(Path(__file__).resolve().parents[1]/'sumo_eval/runtime/H2Oplus'))
import numpy as np

from experiments.mcwm_closed_loop_sync_collection_v53_20261025.collect import (
    WarmupBudget, ContinuousStream, flow_coverage,
)
from experiments.mcwm_closed_loop_critic_memory_v32_20261023.test_memory import MemoryNativeEnv
from experiments.mcwm_closed_loop_critic_memory_v32_20261023.memory import ExecutionMemory
from experiments.mcwm_closed_loop_paired_coverage_v37_20261024.test_collect import native_row
from experiments.mcwm_closed_loop_v1.stream import TransitMDPConfigV7


def test_cancel_fifth_ordinary_entitlement_before_third_selection():
    budget = WarmupBudget(dict(feedback_first_deadline=1200, feedback_last_deadline=1800,
        feedback_interval_seconds=300, feedback_records_per_packet=5))
    sizes = []
    for index, deadline in enumerate((1200, 1500, 1800)):
        # Third window deliberately contains four and cannot support a request
        # for five followed by dropping one after consumption.
        count = (5, 5, 4)[index]
        budget.ingest([SimpleNamespace(sequence=index*10+j, available_at=deadline-1,
            dt=1., terminal=False) for j in range(count)])
        selected, discarded = budget.take(deadline)
        sizes.append(len(selected))
        assert discarded == []
    assert sizes == [5, 5, 4]
    assert sum(row['records'] for row in budget.receipts) == 14


def test_271_real_closures_keep_zero_dt_and_continue_to_actual_endpoint(tmp_path):
    batches = [(10, [('A', 0), ('B', 0)])]
    batches += [(clock, [(f'B{clock}', 0)]) for clock in range(11, 290)]
    env = MemoryNativeEnv(batches)
    issues, closed = [], []

    def issue(event):
        event['behavior_s'] = dict(actor_snapshot='new_post24_from14')
        issues.append(deepcopy(event))

    with (tmp_path/'rows.jsonl').open('x') as handle:
        stream = ContinuousStream(env, [3], TransitMDPConfigV7(horizon_seconds=24000),
            execute_transform=False, delay_seconds=30., on_issue=issue,
            on_response_row=closed.append, response_file=handle)
        stream.reset()
        stream.memory = ExecutionMemory(env._bridge)
        stream.response_start = 0
        while len(issues) < 273:
            stream.step(lambda state: np.asarray([-1.], np.float32))
        assert len(closed) == len(stream.response_rows) == 271
        assert closed[0].dt == 0.
        assert all(not row.terminal for row in closed)
        assert closed[-1].sequence == 270
        assert closed[-1].completed_at == issues[271]['started_at']
        np.testing.assert_array_equal(closed[-1].ns, issues[271]['s'])
        assert closed[-1].memory_ns == issues[271]['memory_s']
        assert stream.issued_count >= 273 and not stream.done
        stream.memory.close()
    assert len((tmp_path/'rows.jsonl').read_text().splitlines()) == 271


def native_flow(last_clock):
    clocks = [9000., 9001., 9002.]
    clocks += [9002.+(last_clock-9002.)*i/269 for i in range(1, 270)]
    rows = [native_row(100+i, clocks[i], clocks[i+1], -1.) for i in range(271)]
    for i, row in enumerate(rows):
        row.line_id = ('7X', '102X', '705X')[min(i, 2)]
        row.feedback_id = f'continuous/{i}'
        row.memory_s = dict(observed_at=row.started_at, vector=[0.]*84)
        row.memory_ns = dict(observed_at=row.completed_at, vector=[0.]*84)
    endpoint = dict(sequence=371, started_at=last_clock, line_id='7X',
        s=rows[-1].ns.copy(), memory_s=deepcopy(rows[-1].memory_ns))
    return rows, endpoint


def test_three_line_300_coverage_requires_real_next_issue_and_keeps_short_line_missing():
    rows, endpoint = native_flow(9302.)
    proof = flow_coverage(rows, endpoint)
    assert proof['all_pass']
    assert proof['root_coverage']['705X']['final_endpoint_span'] == 300.
    missing = flow_coverage(rows, None)
    assert not missing['all_pass'] and missing['checks']['real272nd_issue'] is False
    short_rows, short_endpoint = native_flow(9301.)
    short = flow_coverage(short_rows, short_endpoint)
    assert short['missing300'] == ['705X']
    assert short['root_coverage']['705X']['first_real_300_endpoint_clock'] is None
    short_endpoint['s'][0] = 1.
    assert not flow_coverage(short_rows, short_endpoint)['checks']['real272nd_issue']
