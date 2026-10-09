"""Stream the frozen evaluation inputs to an isolated second-node runtime."""
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
RUNTIME = str(HERE / "runtime")
SOURCE = "/home/erzhu419/mine_code/sumo-rl/H2Oplus/reproducibility/deployments/h2oplus_v20_formal_confirmation_evaluation_v4"
DATASET = "/home/erzhu419/mine_code/sumo-rl/H2Oplus/reproducibility/deployments/v7_formal_v2_training_v7r1/workspace/bus_h2o/datasets_v7/formal_v2_full/transit_smdp_v7_formal_v2.h5"
SUMO = "/home/erzhu419/mine_code/sumo-rl/SUMO_ruiguang"
PYTHON = "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/python"


def ssh(host, code, **kwargs):
    command = "PYTHONDONTWRITEBYTECODE=1 python3 -c " + shlex.quote(code)
    return subprocess.run(["ssh", "-o", "BatchMode=yes", host, command],
                          check=True, **kwargs)


producer = r'''
from pathlib import Path
import json, sys, tarfile
source=Path(SOURCE)
h=source/'workspace/H2Oplus'
files={}
def add(p, name=None):
    if not p.is_file(): raise FileNotFoundError(p)
    files[str(name or p.relative_to(source))]=p
for folder in ('SimpleSAC','bus_h2o','collect_policy'):
    for p in (h/folder).rglob('*.py'):
        add(p)
for p in (h/'reproducibility').glob('*.py'): add(p)
for p in (h/'reproducibility').glob('*.json'): add(p)
for p in (h/'bus_h2o/network_data').rglob('*'):
    if p.is_file() and p.suffix in ('.xml','.json'): add(p)
for p in (h/'bus_h2o/formal_confirmation_scenarios_v20/v1').rglob('*'):
    if p.is_file(): add(p)
for p in (source/'workspace/SUMO_ruiguang/online_control').rglob('*'):
    if p.is_file() and '__pycache__' not in p.parts: add(p)
refs=source/'remote_results_v4/single_jtl110gpu'
selected=[]
for block in range(8101,8111):
    for seed in range(6201,6206):
        for demand in ('0p75','1','1p25'):
            p=refs/('v20-formal-confirmation-%02d'%(block-8100))/('sac-online-seed%d-events000025000'%seed)/('demand_%s_jitter120s_seed%d'%(demand,block+100))/('sumo_seed%d.json'%block)
            add(p)
            selected.append(p)
for seed in range(6201,6206):
    add(h/('reproducibility/h2oplus_v8_fewshot_results/training/sac-online_seed%d/fewshot_policies/online_events_000025000.pt'%seed))
for p in json.loads(selected[0].read_text())['source_sha256']:
    add(Path(p))
print(json.dumps({'files':len(files),'uncompressed_bytes':sum(p.stat().st_size for p in files.values()),'references':len(selected)}),file=sys.stderr)
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
    for name,p in sorted(files.items()):
        archive.add(str(p),arcname=name,recursive=False)
'''.replace("SOURCE", repr(SOURCE), 1)


def main():
    ssh("jtl110gpu2", "from pathlib import Path; Path(%r).mkdir(parents=True,exist_ok=True)" % RUNTIME)
    source_command = "PYTHONDONTWRITEBYTECODE=1 python3 -c " + shlex.quote(producer)
    dest_command = "tar -xzf - -C " + shlex.quote(RUNTIME)
    src = subprocess.Popen(["ssh", "-o", "BatchMode=yes", "jtl110gpu", source_command], stdout=subprocess.PIPE)
    dst = subprocess.Popen(["ssh", "-o", "BatchMode=yes", "jtl110gpu2", dest_command], stdin=src.stdout)
    src.stdout.close()
    destination_status = dst.wait()
    source_status = src.wait()
    if destination_status or source_status:
        raise RuntimeError((source_status, destination_status))
    finish = r'''
from pathlib import Path
import json, subprocess, os
root=Path(RUNTIME)
h=root/'workspace/H2Oplus'
s=root/'workspace/SUMO_ruiguang'
def link(path,target):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.is_symlink():
        if path.resolve()!=Path(target).resolve(): raise ValueError(str(path))
    elif path.exists(): raise FileExistsError(path)
    else: path.symlink_to(target)
for name in ('b_network','c_social_vehicle_rou','d_bus_rou','e_passenger_rou'):
    target=Path(SUMO)/name
    if target.exists(): link(s/name,target)
link(h/'SUMO_ruiguang','../SUMO_ruiguang')
dataset=h/'bus_h2o/datasets_v7/formal_v2_full/transit_smdp_v7_formal_v2.h5'
link(dataset,DATASET)
(root/'tmp').mkdir(exist_ok=True)
env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(h), LD_PRELOAD='/usr/lib/x86_64-linux-gnu/libcurl.so.4', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
code="import sys,torch,numpy,h5py,libsumo; from SimpleSAC import evaluate_transit_v20_formal_confirmation as v; print(sys.version,torch.__version__,numpy.__version__,h5py.__version__,libsumo.getVersion()); print(v.__file__)"
check=subprocess.run([PYTHON,'-c',code],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
report=dict(node='jtl110gpu2',root=str(h),dataset=str(dataset),dataset_resolved=str(dataset.resolve()),sumo_dir=str(s/'online_control'),edge=str(s/'online_control/intersection_delay/a_sorted_busline_edge.xml'),refs=str(root/'remote_results_v4/single_jtl110gpu'),runtime_lock=str(h/'reproducibility/h2oplus_v8_stage_a_sumo_runtime_lock_py38.json'),python=PYTHON,import_status=check.returncode,import_output=check.stdout,reference_count=len(list((root/'remote_results_v4/single_jtl110gpu').rglob('sumo_seed*.json'))),actors=len(list((h/'reproducibility/h2oplus_v8_fewshot_results/training').glob('sac-online_seed*/fewshot_policies/online_events_000025000.pt'))),scenario_manifests=len(list((h/'bus_h2o/formal_confirmation_scenarios_v20/v1').glob('*/manifest_portable_v8.json'))))
(root/'staging.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
if check.returncode: raise SystemExit(check.returncode)
'''
    values = dict(RUNTIME=RUNTIME, DATASET=DATASET, SUMO=SUMO, PYTHON=PYTHON)
    prelude = "\n".join(key + " = " + repr(value) for key, value in values.items())
    ssh("jtl110gpu2", prelude + "\n" + finish)


if __name__ == "__main__":
    main()
