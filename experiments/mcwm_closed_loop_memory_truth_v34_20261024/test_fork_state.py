"""Two concurrent native-state children preserve the paused parent and parser."""
import json
import os
import random
import time

import numpy as np
import torch

from experiments.mcwm_closed_loop_physical_execution_v30_20261023.test_fork_state import trace_files, detach, finish
from experiments.mcwm_closed_loop_memory_truth_v34_20261024.fork_state import fork_children


def test_only_two_children_inherit_rng_and_preserve_parent_trace_and_state(tmp_path):
    torch.set_num_threads(1)
    random.seed(419); np.random.seed(419); torch.manual_seed(419)
    py = random.Random(); py.setstate(random.getstate())
    array = np.random.RandomState(); array.set_state(np.random.get_state())
    tensor = torch.Generator().set_state(torch.get_rng_state())
    expected = [py.random(), float(array.random_sample()), float(torch.rand((), generator=tensor))]
    trace, counters, samples = trace_files(tmp_path), dict(policy=456, feedback=285), []

    def prepare(arm):
        detach(trace, tmp_path/arm)
        samples.extend([random.random(), float(np.random.random_sample()), float(torch.rand(()))])
        assert trace.reference.readline() == 'two\n'
        trace.writer.write(arm+'\n'); trace.writer.flush()

    result = fork_children(tmp_path, trace, lambda: 18002., lambda: counters.copy(), prepare)
    if isinstance(result, str):
        counters['policy'] = 999
        finish(tmp_path/result, samples)
    try:
        assert result['all_pass'] and result['overlap_seconds'] > 0
        assert set(result['child_pids']) == {'low', 'high'}
        assert len(set(result['child_pids'].values())) == 2
        assert counters == dict(policy=456, feedback=285)
        assert trace.reference.readline() == 'two\n'
        assert (tmp_path/'parent.txt').read_text() == 'parent\n'
        for arm in ('low', 'high'):
            assert json.loads((tmp_path/arm/'summary.json').read_text())['samples'] == expected
    finally:
        trace.writer.close(); trace.reference.close()


def test_two_child_route_refills_have_independent_parent_offsets(tmp_path):
    content = b''.join(f'{number:07d}\n'.encode() for number in range(10000))
    route = tmp_path/'buffered.rou.xml'
    route.write_bytes(content)
    trace = trace_files(tmp_path)
    with route.open('rb', buffering=4096) as parser:
        assert parser.read(13) == content[:13]
        kernel_before = os.lseek(parser.fileno(), 0, os.SEEK_CUR)
        child_samples = []

        def prepare(arm):
            detach(trace, tmp_path/arm)
            child_samples.append(parser.read(16384).decode())

        result = fork_children(tmp_path, trace, lambda: 18002., lambda: dict(policy=456), prepare)
        if isinstance(result, str):
            finish(tmp_path/result, child_samples)
        try:
            assert result['all_pass'] and result['route_streams']['count'] == 1
            assert result['route_streams']['parent_offsets_unchanged']
            assert os.lseek(parser.fileno(), 0, os.SEEK_CUR) == kernel_before
            assert parser.read(16384) == content[13:13+16384]
            for arm in ('low', 'high'):
                assert json.loads((tmp_path/arm/'summary.json').read_text())['samples'] == [content[13:13+16384].decode()]
        finally:
            trace.writer.close(); trace.reference.close()
