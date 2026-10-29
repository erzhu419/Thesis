"""Explicit v14 staging and small resident summaries; every task uses high priority."""

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
RESIDENT_INPUTS = HPC / "Thesis/experiments/mcwm_closed_loop_formal_20261020/inputs"
RESIDENT_SOURCE = HPC / "Thesis/experiments/mcwm_closed_loop_diagnostic_v2_20261020/inputs"
RESIDENT_V7 = HPC / "Thesis/experiments/mcwm_closed_loop_action_fork_v7_20261020"
RESIDENT_V8 = HPC / "Thesis/experiments/mcwm_closed_loop_action_response_v8_20261021"
RESIDENT_V9 = HPC / "Thesis/experiments/mcwm_closed_loop_response_ablation_v9_20261021"
RESIDENT_V10 = HPC / "Thesis/experiments/mcwm_closed_loop_controlled_calibration_v10_20261021"
RESIDENT_V13 = HPC / "Thesis/experiments/mcwm_closed_loop_separated_contrast_v13_20261021"
RESIDENT_V12 = HPC / "Thesis/experiments/mcwm_closed_loop_state_interaction_v12_20261021"
RESIDENT_V11 = HPC / "Thesis/experiments/mcwm_closed_loop_response_residual_v11_20261021"
CPUS = [f"node{i:03d}" for i in range(1, 7)]
PRIORITY = "high"
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
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def stage(files, *, node="node001"):
    """Upload named v14 code only; resident data and earlier versions stay read only."""
    groups = {}
    for path in files:
        path = Path(path).resolve()
        path.relative_to(HERE)
        groups.setdefault(path.parent, []).append(path)
    if not groups:
        return dict(files=0, uploaded_bytes=0)
    scheduler.run_on(node, "mkdir -p " + shlex.join([remote(p) for p in groups]), timeout=45, check=True)
    for directory, paths in groups.items():
        subprocess.run([
            "rsync", "-a", "--timeout=30", "-e", scheduler._ssh_rsync_shell_for_node(node),
            *map(str, paths), scheduler._ssh_target_for_node(node) + ":" + remote(directory) + "/",
        ], check=True, timeout=90)
    paths = [path for values in groups.values() for path in values]
    return dict(node=node, files=len(paths), file_names=[str(path.relative_to(HERE)) for path in paths],
                uploaded_bytes=sum(path.stat().st_size for path in paths), destination=remote(HERE))


def task(protocol, label, argv=(), *, runner="run.py", ram_mb=8000):
    prefix = protocol["name"]
    return dict(
        project="Thesis", description=prefix + " " + label, signature=prefix + "/" + label,
        cwd=str(HERE), cmd=shlex.join([PYTHON, "-u", remote(HERE / runner), *map(str, argv)]),
        priority=PRIORITY, allowed_nodes=CPUS, cpu=1, ram_mb=ram_mb, vram=0, env=environment(),
        resource_family=prefix + "/cpu", ram_resource_family=prefix + "/cpu",
        vram_resource_family=prefix + "/cpu", allow_cpu_training=True,
        cpu_training_justification="Single-thread native SUMO assignment innovation calibration and paired truth using resident initializers.",
        allow_no_ckpt=True, allow_no_resume=True, skip_launch_staging=True,
    )


def submit(rows, name):
    output = HERE / (name + "_submission.json")
    if output.exists():
        raise FileExistsError(output)
    jobs = HERE / (name + "_jobs.json")
    write(jobs, rows)
    result = json.loads(subprocess.check_output([
        sys.executable, str(SCHEDULER), "submit-jsonl", "--file", str(jobs), "--trusted", "--json",
    ], text=True))
    write(output, result)
    return result


def status(ids):
    result = json.loads(subprocess.check_output([
        sys.executable, str(SCHEDULER), "status", "--readonly", "--json",
        "--lock-timeout", "30", "--ids", ",".join(ids),
    ], text=True))
    return [{key: row.get(key) for key in (
        "id", "status", "priority", "node", "signature", "log_path", "remote_pids",
    )} for row in result["tasks"]]


def collect_json(relative_path, *, node="node001", max_bytes=200000):
    """Reject oversized resident output before copying even the first byte locally."""
    path = HERE / relative_path
    _, size, _ = scheduler.run_on(node, "stat -c %s " + shlex.quote(remote(path)), timeout=45, check=True)
    if int(size.strip()) > max_bytes:
        raise RuntimeError("Resident summary exceeds the small-result budget; reduce it on server first")
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "rsync", "-a", "--timeout=30", "-e", scheduler._ssh_rsync_shell_for_node(node),
        scheduler._ssh_target_for_node(node) + ":" + remote(path), str(path),
    ], check=True, timeout=60)
    return json.loads(path.read_text())
