"""Stream frozen actor/prefix files between servers without local checkpoint files."""

import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, "/home/erzhu419/mine_code/scheduleurm/skill")
import scheduler

HERE = Path(__file__).resolve().parent
DESTINATION = "/home/zhengliang01/scheduleurm_work/Thesis/workpapers/最终控制器同条件比较_20261006"
DEPLOYMENTS = "/home/erzhu419/mine_code/sumo-rl/H2Oplus/reproducibility/deployments"


def stream(source_node, destination_node, source_command, destination_command):
    """SSH endpoints stream through an in-memory pipe; HPC extraction uses shared /home."""
    source = subprocess.Popen(scheduler._ssh_base_args(source_node) + [source_command],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    destination = subprocess.Popen(scheduler._ssh_base_args(destination_node) + [destination_command],
                                   stdin=source.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    source.stdout.close()
    try:
        output, destination_error = destination.communicate(timeout=600)
        source_error = source.stderr.read()
        source.wait(timeout=15)
        if source.returncode or destination.returncode:
            raise RuntimeError(dict(source_returncode=source.returncode,
                destination_returncode=destination.returncode,
                source_error=source_error.decode()[-3000:], destination_error=destination_error.decode()[-3000:]))
        return output.decode()
    finally:
        for process in (source, destination):
            if process.poll() is None:
                process.kill()
                process.wait()


def main():
    files = {}
    for seed in range(6201, 6206):
        files["inputs/h2oplus_seed%d.pt" % seed] = (
            DEPLOYMENTS + "/h2oplus_v14_terminal_taper_pilot_v1/workspace/reproducibility/"
            "h2oplus_v14_initializers_v1/h2oplus_seed%d.pt" % seed)
        files["policies/h2oplus_final_seed%d.pt" % seed] = (
            DEPLOYMENTS + "/h2oplus_v20_formal_confirmation_evaluation_v4/workspace/H2Oplus/"
            "reproducibility/h2oplus_v14_terminal_taper_results/branches_v1/"
            "h2oplus_v14_terminal_quarter_seed%d/v14_policies/online_events_000025000.pt" % seed)
    producer = ("import sys,tarfile; files=" + repr(files) + "\n"
                "with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:\n"
                " for name,path in files.items(): archive.add(path,arcname=name,recursive=False)\n")
    source_command = "python3 -c " + shlex.quote(producer)
    destination_command = "mkdir -p " + shlex.quote(DESTINATION) + " && tar -xzf - -C " + shlex.quote(DESTINATION)
    stream("jtl110gpu", "node001", source_command, destination_command)
    old = "/home/zhengliang01/scheduleurm_work/thesis-h2o-source-guard-20261006/workspace/H2Oplus"
    links = {DESTINATION + "/policies/sac_guarded_seed%d.pt" % seed:
        old + "/reproducibility/h2oplus_v8_fewshot_results/training/sac-online_seed%d/fewshot_policies/online_events_000025000.pt" % seed
        for seed in range(6201, 6206)}
    link_code = ("from pathlib import Path; links=" + repr(links) + "\n"
                 "for destination,source in links.items():\n"
                 " p=Path(destination)\n"
                 " if not p.exists():p.symlink_to(source)\n")
    scheduler.run_on("node001", "/home/zhengliang01/scheduleurm_work/conda_envs/csbapr-gpu-py310/bin/python -c "
                     + shlex.quote(link_code), timeout=20, check=True)
    record = dict(source_node="jtl110gpu", destination_root=DESTINATION,
                  local_checkpoint_files=False, copied_files=files, reused_sac_links=links)
    (HERE / "asset_deployment.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(copied_files=len(files), destination=DESTINATION), ensure_ascii=False))


def stage_jtl311():
    source = "/home/erzhu419/mine_code/Thesis/workpapers/H2O纯源同执行诊断_20261006/runtime"
    target = str(HERE / "runtime")
    producer = r'''
import pathlib,tarfile,sys
root=pathlib.Path(SOURCE)
files={}
for name in ['proot','system_libs','workspace/SUMO_ruiguang']:
 files[name]=root/name
h=root/'workspace/H2Oplus'
for folder in ['SimpleSAC','bus_h2o','collect_policy']:
 for p in (h/folder).rglob('*.py'): files[str(p.relative_to(root))]=p
for p in (h/'reproducibility').glob('*'):
 if p.suffix in ['.py','.json']: files[str(p.relative_to(root))]=p
for p in (h/'bus_h2o/network_data').rglob('*'):
 if p.is_file() and p.suffix in ['.xml','.json']: files[str(p.relative_to(root))]=p
names=[
 'workspace/H2Oplus/bus_h2o/formal_confirmation_scenarios_v20/v1/demand_0p75_jitter120s_seed8201',
 'workspace/H2Oplus/reproducibility/h2oplus_v8_fewshot_results/training/sac-online_seed6201/fewshot_policies/online_events_000025000.pt',
 'remote_results_v4/single_jtl110gpu/v20-formal-confirmation-01/sac-online-seed6201-events000025000/demand_0p75_jitter120s_seed8201/sumo_seed8101.json']
for name in names: files[name]=root/name
for name in ['control_sim_traci_period.sumocfg','initialize_obj/save_obj_passenger.add.xml']:
 files['workspace/SUMO_ruiguang/online_control/'+name]=pathlib.Path('/home/erzhu419/mine_code/sumo-rl/SUMO_ruiguang/online_control')/name
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz',dereference=True) as archive:
 for name,path in files.items():archive.add(str(path),arcname=name,recursive=path.is_dir())
'''.replace("SOURCE", repr(source), 1)
    stream("jtl110gpu2", "jtl311linux", "python3 -c " + shlex.quote(producer),
           "mkdir -p " + shlex.quote(target) + " && tar -xzf - -C " + shlex.quote(target))
    sys.path.insert(0, str(HERE.parent / "H2O纯源同执行诊断_20261006"))
    from stage_runtime_libs import prepare_rootfs
    prepare_rootfs("jtl311linux", target + "/system_libs")
    code = r'''
from pathlib import Path
r=Path(TARGET)
h=r/'workspace/H2Oplus'
p=h/'bus_h2o/datasets_v7/formal_v2_full/transit_smdp_v7_formal_v2.h5'
p.parent.mkdir(parents=True,exist_ok=True)
if not p.exists():p.symlink_to('/home/erzhu419/mine_code/Thesis/experiments/mcwm_bus/runtime/dataset.h5')
(r/'tmp').mkdir(exist_ok=True)
'''.replace("TARGET", repr(target), 1)
    scheduler.run_on("jtl311linux", "python3 -c " + shlex.quote(code), timeout=20, check=True)
    policies = ["%s_seed%d.pt" % (method, seed)
                for method in ("h2oplus_final", "sac_guarded") for seed in range(6201, 6206)]
    policy_target = str(HERE / "policies")
    stream("node001", "jtl311linux",
        "tar -chzf - -C " + shlex.quote(DESTINATION + "/policies") + " " + " ".join(policies),
        "mkdir -p " + shlex.quote(policy_target) + " && tar -xzf - -C " + shlex.quote(policy_target))
    (HERE / "jtl311_runtime_deployment.json").write_text(json.dumps(dict(
        node="jtl311linux", runtime=target, source_node="jtl110gpu2",
        source_runtime=source, local_large_files=False,
        python="/home/erzhu419/anaconda3/envs/LSTM-RL/bin/python",
        dataset_reused="/home/erzhu419/mine_code/Thesis/experiments/mcwm_bus/runtime/dataset.h5"),
        ensure_ascii=False, indent=2) + "\n")
    print("Staged original isolated runtime on jtl311linux")


if __name__ == "__main__":
    stage_jtl311() if "--jtl311" in sys.argv else main()
