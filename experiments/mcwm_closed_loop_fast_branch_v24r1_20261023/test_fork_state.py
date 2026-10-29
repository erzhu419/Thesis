"""Process/RNG tests only; these do not certify native SUMO replay."""
import json
import os
from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.mcwm_closed_loop_fast_branch_v24r1_20261023.fork_state import fork_children


def trace_files(tmp_path):
    source = tmp_path/'source.txt'
    source.write_text('one\ntwo\nthree\n')
    trace = SimpleNamespace(writer=(tmp_path/'parent.txt').open('w'), reference=source.open())
    trace.writer.write('parent\n')
    assert trace.reference.readline() == 'one\n'
    return trace


def detach(trace, directory):
    cookie = trace.reference.tell()
    source = Path(trace.reference.name)
    trace.writer.close()
    trace.reference.close()
    trace.writer = (directory/'trace.txt').open('w')
    trace.reference = source.open()
    trace.reference.seek(cookie)


def finish(directory, samples=None):
    start = time.time()
    time.sleep(.12)
    value = dict(status='complete', integrity=dict(all_pass=True), samples=samples,
                 execution_wall=dict(pid=os.getpid(), native_started_wall=start,
                                     native_finished_wall=time.time()))
    (directory/'summary.json').write_text(json.dumps(value))
    os._exit(0)


def test_both_children_restore_rng_without_advancing_parent_or_shared_trace(tmp_path):
    torch.set_num_threads(1)
    random.seed(419)
    np.random.seed(419)
    torch.manual_seed(419)
    py = random.Random()
    py.setstate(random.getstate())
    array = np.random.RandomState()
    array.set_state(np.random.get_state())
    tensor = torch.Generator().set_state(torch.get_rng_state())
    expected = [py.random(), float(array.random_sample()), float(torch.rand((), generator=tensor))]
    trace = trace_files(tmp_path)
    counters = dict(policy=5, world=7)
    samples = []

    def prepare(arm):
        detach(trace, tmp_path/arm)
        samples.extend([random.random(), float(np.random.random_sample()), float(torch.rand(()))])
        assert trace.reference.readline() == 'two\n'
        trace.writer.write(arm+'\n')
        trace.writer.flush()

    result = fork_children(tmp_path, trace, lambda: 1559., lambda: counters.copy(), prepare)
    if isinstance(result, str):
        counters['policy'] = 99
        finish(tmp_path/result, samples)
    try:
        assert result['all_pass'] and result['overlap_seconds'] > 0
        assert counters == dict(policy=5, world=7)
        assert trace.reference.readline() == 'two\n'
        assert (tmp_path/'parent.txt').read_text() == 'parent\n'
        for arm in ('low', 'high'):
            value = json.loads((tmp_path/arm/'summary.json').read_text())
            assert value['samples'] == expected
    finally:
        trace.writer.close()
        trace.reference.close()


def test_child_route_parser_refill_preserves_parent_buffer_and_kernel_offset(tmp_path):
    content = b''.join(f'{number:07d}\n'.encode() for number in range(10000))
    source = tmp_path/'buffered.rou.xml'
    source.write_bytes(content)
    trace = trace_files(tmp_path)
    with source.open('rb', buffering=4096) as parser:
        assert parser.read(13) == content[:13]
        kernel_before = os.lseek(parser.fileno(), 0, os.SEEK_CUR)
        assert kernel_before > parser.tell()
        child_samples = []

        def prepare(arm):
            detach(trace, tmp_path/arm)
            # Consume the copied buffer and refill from an independent open file.
            child_samples.append(parser.read(16384).decode())

        result = fork_children(tmp_path, trace, lambda: 1559., lambda: dict(policy=5), prepare)
        if isinstance(result, str):
            finish(tmp_path/result, child_samples)
        try:
            assert result['all_pass']
            assert result['route_streams']['count'] == 1
            assert result['route_streams']['parent_offsets_unchanged']
            assert os.lseek(parser.fileno(), 0, os.SEEK_CUR) == kernel_before
            assert parser.read(16384) == content[13:13+16384]
            for arm in ('low', 'high'):
                value = json.loads((tmp_path/arm/'summary.json').read_text())
                assert value['samples'] == [content[13:13+16384].decode()]
        finally:
            trace.writer.close()
            trace.reference.close()


@pytest.mark.parametrize('failure', ['exit', 'missing_summary', 'timeout'])
def test_failure_keeps_receipt_and_reaps_children(tmp_path, failure):
    trace = trace_files(tmp_path)

    def prepare(arm):
        detach(trace, tmp_path/arm)
        (tmp_path/arm/'log.txt').write_text('child started\n')
        if failure == 'exit' and arm == 'low':
            os._exit(3)
        if failure == 'missing_summary' and arm == 'low':
            os._exit(0)
        while True:
            time.sleep(1)

    error = TimeoutError if failure == 'timeout' else FileNotFoundError if failure == 'missing_summary' else RuntimeError
    try:
        with pytest.raises(error):
            fork_children(tmp_path, trace, lambda: 1559., lambda: dict(policy=5), prepare,
                          timeout_seconds=.15)
        receipt = json.loads((tmp_path/'live_fork.json').read_text())
        assert not receipt['all_pass']
        assert receipt['parent_rng_unchanged'] and receipt['parent_trace_cookie_unchanged']
        assert len(receipt['child_exitcodes']) == 2
        for pid in receipt['child_pids'].values():
            with pytest.raises(ChildProcessError):
                os.waitpid(pid, os.WNOHANG)
    finally:
        trace.writer.close()
        trace.reference.close()
