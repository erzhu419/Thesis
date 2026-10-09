"""Stream the original host libraries into this campaign's private runtime."""
import argparse
import base64
import os
import shlex
import signal
import subprocess
import sys

sys.path.insert(0, "/home/erzhu419/mine_code/scheduleurm/skill")
import scheduler

SOURCE_ROOT = "/home/erzhu419/mine_code/sumo-rl/H2Oplus/reproducibility/deployments/h2oplus_v20_formal_confirmation_evaluation_v4/workspace/H2Oplus"
SOURCE_PYTHON = "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/python"
PRODUCER = r'''
import contextlib, json, pathlib, subprocess, sys, tarfile
output = sys.stdout.buffer
with contextlib.redirect_stdout(sys.stderr):
    import torch, numpy, h5py, libsumo
    from SimpleSAC import evaluate_transit_v20_formal_confirmation
paths = set()
for line in pathlib.Path('/proc/self/maps').read_text().splitlines():
    parts = line.split(None, 5)
    if len(parts) == 6 and parts[5].startswith(('/lib/', '/usr/lib/')):
        p = pathlib.Path(parts[5])
        if p.is_file() and '.so' in p.name and not p.name.startswith('libcuda'):
            paths.add(p)
for binary in ('/bin/bash', '/bin/dash', '/usr/bin/env',
               sys.prefix + '/lib/python3.8/site-packages/sumo/bin/sumo'):
    result = subprocess.run(['ldd', binary], stdout=subprocess.PIPE, universal_newlines=True, check=True)
    for line in result.stdout.splitlines():
        for word in line.split():
            if word.startswith(('/lib/', '/usr/lib/')) and pathlib.Path(word).is_file():
                paths.add(pathlib.Path(word).resolve())
files = {p.name: p for p in paths}
for directory in {p.parent for p in paths}:
    for p in directory.iterdir():
        if p.is_symlink() and p.resolve() in paths:
            files[p.name] = p
print(json.dumps({'library_names': len(files),
                  'bytes': sum(p.stat().st_size for p in files.values())}), file=sys.stderr)
with tarfile.open(fileobj=output, mode='w|gz', dereference=True) as archive:
    for name, p in sorted(files.items()):
        archive.add(str(p), arcname=name, recursive=False)
    for name, p in (('runtime_bin/bash', '/bin/bash'),
                    ('runtime_bin/sh', '/bin/dash'), ('runtime_bin/env', '/usr/bin/env')):
        archive.add(p, arcname=name, recursive=False)
'''


def transfer(node, destination):
    code = "import base64;exec(base64.b64decode(%r))" % base64.b64encode(PRODUCER.encode()).decode()
    source_command = (
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
        "LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcurl.so.4 "
        "PYTHONPATH=" + shlex.quote(SOURCE_ROOT) + " " + SOURCE_PYTHON +
        " -c " + shlex.quote(code))
    destination_command = ("mkdir -p " + shlex.quote(destination) +
                           " && tar -xzf - -C " + shlex.quote(destination))
    transfer_commands("jtl110gpu", node, source_command, destination_command)


def transfer_commands(source_node, node, source_command, destination_command):
    source_args = scheduler._ssh_base_args(source_node) + [source_command]
    destination_args = scheduler._ssh_base_args(node) + [destination_command]
    guard = scheduler._remote_exec_deps().transport_guard
    with guard(source_args), guard(destination_args):
        source = subprocess.Popen(source_args, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, start_new_session=True)
        destination_process = None
        try:
            destination_process = subprocess.Popen(
                destination_args, stdin=source.stdout, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, start_new_session=True)
            source.stdout.close()
            source.stdout = None
            _, destination_error = destination_process.communicate(timeout=1800)
            _, source_error = source.communicate(timeout=15)
            if source.returncode or destination_process.returncode:
                raise RuntimeError((source.returncode, destination_process.returncode,
                                    source_error.decode()[-1500:], destination_error.decode()[-1500:]))
            print(source_error.decode().strip())
            print("Streamed campaign files from " + source_node + " to " + node)
        finally:
            for process in (source, destination_process):
                if process is not None and process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def prepare_rootfs(node, destination):
    code = r'''
from pathlib import Path
import shutil
libs = Path(DESTINATION)
root = libs.parent / 'guestroot'
for directory in ('lib/x86_64-linux-gnu', 'lib64', 'usr/lib/x86_64-linux-gnu',
                  'usr/bin', 'bin', 'home', 'tmp', 'proc', 'dev', 'sys', 'etc'):
    path = root / directory
    if directory in ('lib/x86_64-linux-gnu', 'usr/lib/x86_64-linux-gnu'):
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_symlink(): path.symlink_to(libs)
    else:
        path.mkdir(parents=True, exist_ok=True)
loader = root / 'lib64/ld-linux-x86-64.so.2'
if not loader.is_symlink(): loader.symlink_to('/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2')
for source, target in (('bash', 'bin/bash'), ('sh', 'bin/sh'), ('env', 'usr/bin/env')):
    shutil.copy2(str(libs / 'runtime_bin' / source), str(root / target))
print(str(root))
'''.replace('DESTINATION', repr(destination))
    python = scheduler.NODES[node].get("resume_scan_python", "python3")
    command = shlex.quote(python) + " -c " + shlex.quote(code)
    _, output, _ = scheduler.run_on(node, command, timeout=15, check=True)
    print(output.strip())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--prepare-rootfs", action="store_true")
    args = parser.parse_args()
    if args.prepare_rootfs:
        prepare_rootfs(args.node, args.destination)
    else:
        transfer(args.node, args.destination)
