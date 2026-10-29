"""Run synthetic state round trips on the original resident runtime, without environment."""
import json
from pathlib import Path
import platform
import time
import unittest

import torch

HERE = Path(__file__).resolve().parent
started = time.monotonic()
steps = 0
original_step = torch.optim.Adam.step


def count_step(self, *args, **kwargs):
    global steps
    steps += 1
    return original_step(self, *args, **kwargs)


torch.optim.Adam.step = count_step
try:
    suite = unittest.defaultTestLoader.discover(str(HERE), pattern='test_*state.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
finally:
    torch.optim.Adam.step = original_step
record = dict(all_pass=result.wasSuccessful(), tests=result.testsRun,
    failures=len(result.failures), errors=len(result.errors),
    synthetic_optimizer_steps=steps, source_environment_steps=0, target_native_seconds=0,
    real_data_training_updates=0, seconds=time.monotonic()-started,
    runtime=dict(host=platform.node(), python=platform.python_version(), torch=torch.__version__))
with (HERE/'runtime_tests.json').open('x') as handle:
    json.dump(record, handle, indent=2)
    handle.write('\n')
print(json.dumps(record), flush=True)
if not record['all_pass']:
    raise SystemExit(1)
