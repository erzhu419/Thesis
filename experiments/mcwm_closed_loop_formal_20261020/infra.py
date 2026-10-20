"""Scheduler relay, explicit code staging, and compact formal-run collection."""

from pathlib import Path
import json
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
LOCAL = Path("/home/erzhu419/mine_code")
HPC = Path("/home/zhengliang01/scheduleurm_work")
SCHEDULER = LOCAL / "scheduleurm/skill/scheduler.py"
PYTHON = str(HPC / "conda_envs/csbapr-gpu-py310/bin/python")
CPUS = [f"node{i:03d}" for i in range(1, 7)]
sys.path.insert(0, str(SCHEDULER.parent))
import scheduler


def remote(path):
    return str(HPC / Path(path).resolve().relative_to(LOCAL))


def environment():
    experiments = HPC / "Thesis/experiments"
    sumo = str(experiments / "sumo_eval/assets/SUMO_ruiguang")
    return dict(
        THESIS_H2O_ROOT=str(experiments / "sumo_eval/runtime/H2Oplus"),
        THESIS_MCWM_ROOT=str(experiments / "mcwm_bus/runtime/MC-WM"),
        THESIS_SOURCE_ENV=str(experiments / "mcwm_bus/runtime/source_env"),
        THESIS_DATASET=str(experiments / "mcwm_bus/runtime/dataset.h5"),
        THESIS_SUMO_ROOT=sumo, SUMO_HOME=sumo,
        PYTHONPATH=str(HPC / "runtimes/h2oplus-eval-v7-sumo125-overlay-v1/site-packages"),
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
        SDL_VIDEODRIVER="dummy", MPLBACKEND="Agg", LIBSUMO_AS_TRACI="quiet",
    )


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")


def stage(files, *, source_prior=None, node="node001"):
    """Send named code/config files into this new experiment's shared directory."""
    paths = [Path(path).resolve() for path in files]
    destinations = [(path, Path(remote(path))) for path in paths]
    if source_prior is not None:
        destinations.append((Path(source_prior).resolve(), Path(remote(HERE)) / "inputs/source_9101.npz"))
    for source, destination in destinations:
        scheduler.run_on(node, "mkdir -p "+shlex.quote(str(destination.parent)), timeout=45, check=True)
        subprocess.run([
            "rsync", "-a", "--timeout=30", "-e", scheduler._ssh_rsync_shell_for_node(node),
            str(source), scheduler._ssh_target_for_node(node)+":"+str(destination),
        ], check=True, timeout=90)
    return dict(node=node, files=len(destinations), destination=remote(HERE),
                uploaded_bytes=sum(path.stat().st_size for path, _ in destinations))


def stage_initializers(*, node="node001", source_node="jtl110gpu2"):
    """Copy resident IQL artifacts into the fresh experiment, without local files."""
    destination = remote(HERE / "inputs")
    scheduler.run_on(node, "mkdir -p "+shlex.quote(destination), timeout=45, check=True)
    existing = HPC / "Thesis/experiments/mcwm_alignment_v2/inputs"
    names = [f"h2oplus_seed{seed}.pt" for seed in (6201, 6202, 6203)]
    scheduler.run_on(node, "cp -n "+" ".join(shlex.quote(str(existing/name)) for name in names)
                     +" "+shlex.quote(destination)+"/", timeout=45, check=True)
    source = (LOCAL / "sumo-rl/H2Oplus/reproducibility/deployments/"
              "h2oplus_v14_terminal_taper_pilot_v1/workspace/reproducibility/"
              "h2oplus_v14_initializers_v1")
    names = [f"h2oplus_seed{seed}.pt" for seed in (6204, 6205)]
    # Run rsync on the scheduler's reachable relay: checkpoint bytes move
    # directly from that server to shared HPC storage.
    target = scheduler._ssh_target_for_node(node)+":"+destination+"/"
    command = ["rsync", "-a", "--timeout=60", "-e",
               "ssh -o BatchMode=yes -o ConnectTimeout=15",
               *[str(source/name) for name in names], target]
    scheduler.run_on(source_node, shlex.join(command), timeout=180, check=True)
    return dict(destination=destination, seeds=list(range(6201, 6206)), local_files_written=0)


def task(protocol, label, argv, *, waits=(), ram_mb=8000):
    prefix = protocol["name"]
    command = [PYTHON, "-u", remote(HERE / "run.py"), *map(str, argv)]
    row = dict(
        project="Thesis", description=prefix+" "+label,
        signature=prefix+"/"+label, cwd=str(HERE), cmd=shlex.join(command),
        allowed_nodes=CPUS, cpu=1, ram_mb=ram_mb, vram=0,
        env=environment(), resource_family=prefix+"/cpu",
        ram_resource_family=prefix+"/cpu", vram_resource_family=prefix+"/cpu",
        allow_cpu_training=True,
        cpu_training_justification="Single-thread paired SUMO target adaptation, shared resident data and initializers.",
        allow_no_ckpt=True, allow_no_resume=True, skip_launch_staging=True,
    )
    if waits:
        row["wait_for_files"] = [str(Path(path)) for path in waits]
    return row


def submit(rows, name="campaign"):
    output = HERE / (name+"_submission.json")
    if output.exists():
        raise FileExistsError(output)
    jobs = HERE / (name+"_jobs.json")
    write(jobs, rows)
    result = json.loads(subprocess.check_output([
        sys.executable, str(SCHEDULER), "submit-jsonl", "--file", str(jobs),
        "--trusted", "--json",
    ], text=True))
    write(output, result)
    return result


def status(ids):
    result = json.loads(subprocess.check_output([
        sys.executable, str(SCHEDULER), "status", "--readonly", "--json", "--brief",
        "--lock-timeout", "30", "--ids", ",".join(ids),
    ], text=True))
    return [{key: row.get(key) for key in (
        "id", "status", "node", "signature", "log_path", "remote_pids",
    )} for row in result["tasks"]]


def collect_json(relative_path, *, node="node001"):
    """Download one named compact JSON report, never trajectory or checkpoint data."""
    path = HERE / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "rsync", "-a", "--timeout=30", "-e", scheduler._ssh_rsync_shell_for_node(node),
        scheduler._ssh_target_for_node(node)+":"+remote(path), str(path),
    ], check=True, timeout=60)
    return json.loads(path.read_text())
