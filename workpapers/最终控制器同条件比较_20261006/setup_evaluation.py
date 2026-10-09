"""Stage the common evaluator on the existing Linux server runtimes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/home/erzhu419/mine_code/scheduleurm/skill")
import scheduler

HPC = "/home/zhengliang01/scheduleurm_work"
REMOTE = HPC + "/Thesis/workpapers/" + HERE.name
PYENV = "/home/erzhu419/anaconda3/envs/LSTM-RL"
HPC_BASE = HPC + "/thesis-h2o-source-guard-20261006"
LAB_BASE = str(HERE.parent / "H2O纯源同执行诊断_20261006/runtime")
ORIGINAL = "/home/erzhu419/mine_code/sumo-rl/H2Oplus/reproducibility/deployments/h2oplus_v20_formal_confirmation_evaluation_v4"
ALL_NODES = [f"node{i:03d}" for i in range(1, 8)] + ["jtl110gpu", "jtl110gpu2", "jtl311linux"]


def layout(node):
    hpc = node.startswith("node")
    directory = REMOTE if hpc else str(HERE)
    base = HPC_BASE if hpc else ORIGINAL if node == "jtl110gpu" else LAB_BASE if node == "jtl110gpu2" else str(HERE / "runtime")
    source = (HPC + "/Thesis/experiments/sumo_eval/assets/SUMO_ruiguang" if hpc else
              base + "/workspace/SUMO_ruiguang" if node == "jtl311linux" else
              "/home/erzhu419/mine_code/sumo-rl/SUMO_ruiguang")
    return directory, base, source


def shell(node):
    directory, base, _ = layout(node)
    cmd = []
    if node != "jtl110gpu":
        cmd = [base + "/proot", "-R", base + "/guestroot", "-b", "/home"]
        if node.startswith("node"):
            cmd += ["-b", base + "/env:" + PYENV]
        cmd += ["-w", directory]
    cmd += ["/usr/bin/env", "PATH=" + PYENV + "/bin:/usr/bin:/bin",
            "LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcurl.so.4",
            "PYTHONPATH=" + base + "/workspace/H2Oplus", "LIBSUMO_AS_TRACI=quiet",
            "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1",
            "NUMEXPR_NUM_THREADS=1", "PYTHONDONTWRITEBYTECODE=1", PYENV + "/bin/python"]
    return "#!/bin/bash\nset -e\nexec " + shlex.join(cmd) + ' "$@"\n'


def setup(node):
    directory, base, source = layout(node)
    generated = HERE / "deployments" / node
    generated.mkdir(parents=True, exist_ok=True)
    wrapper = generated / "eval_python.sh"
    wrapper.write_text(shell(node))
    scheduler.run_on(node, "mkdir -p " + shlex.quote(directory), timeout=30, check=True)
    files = [HERE / name for name in ("evaluate.py", "prepare_scenarios.py", "node_case.py", "protocol.json")]
    subprocess.run(["rsync", "-a", "-e", scheduler._ssh_rsync_shell_for_node(node),
                    *map(str, files), str(wrapper), scheduler._ssh_target_for_node(node) + ":" + directory + "/"], check=True)
    # The existing official evaluator normally re-reads 589 MB for each case.
    # Verify it once here, then use its supported size/mtime cache for this batch.
    expected = json.loads((HERE / "replay_reference.json").read_text())["dataset"]["sha256"]
    code = '''
import hashlib,json,os
from pathlib import Path
here,base,expected=Path(HERE),Path(BASE),EXPECTED
h=base/'workspace/H2Oplus'
dataset=h/'bus_h2o/datasets_v7/formal_v2_full/transit_smdp_v7_formal_v2.h5'
digest=hashlib.sha256()
with dataset.open('rb') as stream:
    for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
if digest.hexdigest()!=expected:raise ValueError('dataset differs from formal training data')
stat=dataset.stat()
row={'h2o-root':str(h),'dataset':str(dataset),'sumo-dir':str(base/'workspace/SUMO_ruiguang/online_control'),
 'edge-xml':str(base/'workspace/SUMO_ruiguang/online_control/intersection_delay/a_sorted_busline_edge.xml'),
 'sumo-runtime-lock':str(h/'reproducibility/h2oplus_v8_stage_a_sumo_runtime_lock_py38.json'),
 'verified-dataset-sha256':expected,'verified-dataset-size-bytes':stat.st_size,'verified-dataset-mtime-ns':stat.st_mtime_ns}
(here/'eval_runtime.json').write_text(json.dumps(row,indent=2)+chr(10))
print(json.dumps(row))
'''.replace("HERE", repr(directory)).replace("BASE", repr(base)).replace("EXPECTED", repr(expected))
    python = scheduler.NODES[node].get("resume_scan_python", "python3")
    _, output, _ = scheduler.run_on(node, shlex.join([python, "-c", code]), timeout=120, check=True)
    (generated / "eval_runtime.json").write_text(output)
    command = ('if [ ! -f ' + shlex.quote(directory + "/scenarios/scenario_index.json") + ' ]; then ' +
        shlex.join(["bash", directory + "/eval_python.sh", directory + "/prepare_scenarios.py",
                    "--h2o-root", base + "/workspace/H2Oplus", "--sumo-root", source,
                    "--output-root", directory + "/scenarios"]) + '; fi')
    scheduler.run_on(node, command, timeout=300, check=True)
    read = '''
import json
from pathlib import Path
p=Path(ROOT)/'scenarios'
d=json.loads((p/'scenario_index.json').read_text())
for c in d['cases']:
 m=json.loads((p/c['scenario_manifest']).read_text())
 c['passenger_input_digests']={k:v for k,v in m['scenario_sha256'].items() if not k.endswith('.sumocfg')}
print(json.dumps(d))
'''.replace("ROOT", repr(directory))
    _, output, _ = scheduler.run_on(node, shlex.join([python, "-c", read]), timeout=30, check=True)
    (generated / "scenario_index.json").write_text(output)
    print("Evaluation deployment ready: " + node, flush=True)
    return node


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", action="append", required=True)
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=4) as pool:
        done = list(pool.map(setup, args.node))
    indexes = [json.loads((HERE / "deployments" / n / "scenario_index.json").read_text()) for n in done]
    if any(value != indexes[0] for value in indexes[1:]):
        raise RuntimeError("Generated passenger inputs differ across nodes")
    (HERE / "evaluation_deployment.json").write_text(json.dumps({"setup_nodes":done,
        "eligible_nodes":ALL_NODES,"new_cases":len(indexes[0]["cases"]),"passenger_inputs_match":True},indent=2)+"\n")
