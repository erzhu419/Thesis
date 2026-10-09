"""Release the 32 fixed cases only after an exact replay, then collect the analysis."""
import json
from pathlib import Path
import shlex
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SCHEDULER = "/home/erzhu419/mine_code/scheduleurm/skill/scheduler.py"
sys.path.insert(0, str(Path(SCHEDULER).parent))
import scheduler

HPC = "/home/zhengliang01/scheduleurm_work/Thesis"
REMOTE = HPC + "/workpapers/" + HERE.name
PYTHON = "/home/zhengliang01/scheduleurm_work/conda_envs/csbapr-gpu-py310/bin/python"


def wait(pattern):
    subprocess.run([sys.executable, SCHEDULER, "wait-for", "--signature", pattern,
                    "--poll", "30", "--timeout", "21600", "--verbose", "--refresh"], check=True)


def pull(paths):
    subprocess.run(["rsync", "-a", "-e", scheduler._ssh_rsync_shell_for_node("node001"),
                    *[scheduler._ssh_target_for_node("node001") + ":" + path for path in paths],
                    str(HERE) + "/"], check=True)


wait("thesis-f2-execution-scale-20261006/scale0.25-case00")
pull([REMOTE + "/runs/scale_0.25/case_00/replay_check.json"])
if not json.loads((HERE / "replay_check.json").read_text())["passed"]:
    raise RuntimeError("The original F-2 replay has not passed")
submission = HERE / "scale1_submission.json"
if not submission.exists():
    output = subprocess.check_output([sys.executable, SCHEDULER, "submit-jsonl", "--file",
                                      str(HERE / "scale1_jobs.json"), "--trusted", "--json"], text=True)
    submission.write_text(output)
print("Exact replay passed; all 32 fixed cases submitted.", flush=True)
wait("thesis-f2-execution-scale-20261006/scale1-case*")
command = shlex.join([PYTHON, REMOTE + "/analyze.py", "--original-root",
                      HPC + "/experiments/sumo_eval/runs/sumo125_v3", "--out-root", REMOTE + "/runs"])
code, output, error = scheduler.run_on("node001", command, timeout=60, check=True)
print(output, flush=True)
if json.loads(output)["status"] != "complete":
    (HERE / "进度.json").write_text(output)
    raise RuntimeError("F-2 results remain incomplete; no comparison concluded")
pull([REMOTE + "/runs/scale_1/analysis.json", REMOTE + "/runs/scale_1/analysis_figure_cells.csv"])
report = json.loads((HERE / "analysis.json").read_text())
lines = ["# F-2执行幅度诊断结果", "", "32个冻结策略评价已齐全；原系数回放逐项一致。", "",
         "| 比较（前者减后者） | 原额外系数0.25：平均时间差/秒 | 系数1：平均时间差/秒 | 原/新前者较好格数 |",
         "|---|---:|---:|---:|"]
for candidate, control, label in (("h2o", "online", "早期H2O+减纯源"),
                                 ("mcwm", "online", "早期世界模型减纯源"),
                                 ("mcwm", "h2o", "早期世界模型减早期H2O+")):
    rows = [r for r in report["method_contrasts"] if r["candidate"] == candidate and r["control"] == control]
    before = statistics.mean(r["old_delta_seconds"] for r in rows)
    after = statistics.mean(r["new_delta_seconds"] for r in rows)
    wins_before = sum(r["old_delta_seconds"] < 0 for r in rows)
    wins_after = sum(r["new_delta_seconds"] < 0 for r in rows)
    lines.append(f"| {label} | {before:+.2f} | {after:+.2f} | {wins_before}/8 → {wins_after}/8 |")
lines += ["", "负差值表示前者的广义乘客时间更低。平均差按原图8种条件等权计算；重复显示的参照不算新的独立回合。", "",
          "这项事后诊断只改变额外动作系数。训练预算、密度比修正、模型置信及策略本身均保持原状，"
          "不能把新结果称为最终H2O+或最终MC-WM的性能，也不据此选取赢家版本替换旧图。",
          "各条件的时间、完成率、停站量变化见analysis.json；40格映射见analysis_figure_cells.csv。"]
(HERE / "结果.md").write_text("\n".join(lines) + "\n")
print("F-2 diagnostic complete: " + str(HERE / "结果.md"), flush=True)
