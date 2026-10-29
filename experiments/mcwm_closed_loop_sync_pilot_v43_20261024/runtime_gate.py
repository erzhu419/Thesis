"""Check resident runtime imports and budget/measure wiring without collection."""
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent

if __name__ == '__main__':
    tests = ['test_collect.py', 'test_credit.py', 'test_learning.py']
    result = subprocess.run([sys.executable, '-m', 'pytest', '-q', *[str(HERE/name) for name in tests]],
        cwd=HERE.parents[1], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    value = dict(all_pass=result.returncode == 0, exit_code=result.returncode,
        result_tail=result.stdout[-2200:], new_native_prefixes=0, new_updates=0)
    with (HERE/'runtime_gate.json').open('x') as handle:
        json.dump(value, handle, separators=(',', ':'))
    print(json.dumps(value), flush=True)
    sys.exit(result.returncode)
