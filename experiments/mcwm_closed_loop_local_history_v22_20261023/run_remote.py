"""Stage only v22 code and collect one bounded training-only diagnostic JSON."""
import json
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from experiments.mcwm_closed_loop_execution_phase_v19_20261022.infra import (
    PYTHON, remote, scheduler,
)


def main():
    protocol = json.loads((HERE/'protocol.json').read_text())
    result_path = HERE/'training_diagnostic.json'
    if result_path.exists():
        raise FileExistsError(result_path)
    files = [HERE/'diagnostic.py', HERE/'protocol.json']
    scheduler.run_on('node001', 'mkdir -p '+shlex.quote(remote(HERE)), timeout=45, check=True)
    subprocess.run(['rsync', '-a', '--timeout=30', '-e', scheduler._ssh_rsync_shell_for_node('node001'),
        *map(str, files), scheduler._ssh_target_for_node('node001')+':'+remote(HERE)+'/'],
        check=True, timeout=90)
    source = Path(remote(ROOT/'experiments'))/protocol['source']
    rc, output, error = scheduler.run_on('node001', shlex.join([PYTHON, '-u', remote(HERE/'diagnostic.py'),
        '--source', str(source), '--out', remote(result_path)]), timeout=45, check=False)
    if rc:
        sys.stderr.write(error[-2500:])
        raise RuntimeError('Resident diagnostic failed, rc='+str(rc))
    print(output.strip(), flush=True)
    code = 'from pathlib import Path; p=Path('+repr(remote(result_path))+'); assert p.stat().st_size<=24000; print(p.read_text(),end="")'
    _, output, _ = scheduler.run_on('node001', shlex.join([PYTHON, '-c', code]), timeout=45, check=True)
    result = json.loads(output)
    with result_path.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    receipt = dict(status='complete', priority='high', mode='resident_read_only_algebra',
        new_scheduler_tasks=0, uploaded_files=[p.name for p in files],
        uploaded_bytes=sum(p.stat().st_size for p in files), collected_bytes=len(output.encode()),
        new_training_labels=0, new_native_seconds=0)
    with (HERE/'execution.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
