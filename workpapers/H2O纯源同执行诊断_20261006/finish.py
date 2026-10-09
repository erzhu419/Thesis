"""Wait once for this fixed campaign, then pull compact results and summarize."""
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SCHEDULER = "/home/erzhu419/mine_code/scheduleurm/skill/scheduler.py"
signature = "thesis-h2o-source-guard-20261006/jtl110gpu/sac_guarded*"
wait = subprocess.run([sys.executable, SCHEDULER, "wait-for", "--signature", signature,
                       "--poll", "60", "--timeout", "86400", "--verbose", "--refresh"])
nodes = ["jtl110gpu"]
placement = HERE / "distributed_placement.json"
if placement.exists():
    nodes += json.loads(placement.read_text())["result_nodes"]
collect_args = [sys.executable, str(HERE / "collect.py")]
for node in dict.fromkeys(nodes):
    collect_args += ["--node", node]
subprocess.run(collect_args, check=True)
subprocess.run([sys.executable, str(HERE / "analyze.py")], check=True)
report = json.loads((HERE / "analysis.json").read_text())
if report["status"] != "complete":
    (HERE / "结果.md").write_text(
        f"本批尚未完成：已回传{report['complete_new_results']}/150个回合。"
        "缺失案例见analysis.json，目前不形成两种策略优劣的结论。\n")
    raise SystemExit(1)

means = report["aggregate_method_means"]
comparison = report["comparisons"]["candidate_minus_guarded_sac"]
difference = comparison["mean_difference"]
interval = comparison["descriptive_block_t95_interval"]["cost_seconds"]
directions = comparison["direction_counts"]["cost_seconds"]
rows = ["# 同执行条件诊断结果", "", "150个固定配对回合已齐全。", "",
        "| 控制器 | 广义乘客时间（秒） | 乘客完成率 |", "|---|---:|---:|"]
for key, label in (("candidate", "原完整H2O+候选"), ("raw_sac", "原纯源SAC"),
                   ("guarded_sac", "采用相同执行规则的纯源SAC")):
    value = means[key]
    rows.append(f"| {label} | {value['cost_seconds']:.2f} | {value['completion_percent']:.4f}% |")
rows += ["", f"完整候选减同规则SAC：时间差{difference['cost_seconds']:+.2f}秒，"
         f"完成率差{difference['completion_percent']:+.4f}个百分点。",
         f"10个环境块的描述性95%区间为[{interval[0]:+.2f}, {interval[1]:+.2f}]秒；"
         f"候选时间更低的环境块为{directions['blocks']['improved']}/10，"
         f"训练种子均值更低的为{directions['training_seeds']['improved']}/5。", "",
         "负时间差有利于完整候选。逐案例、需求分组和时间分解见analysis.json。", "",
         "本批复用原测试日，属于事后机制诊断。两方案源交互数相同，训练更新总量不同；"
         "本结果不能单独归因于离线数据，也不能外推为MC-WM与其他算法的统一排名。"]
(HERE / "结果.md").write_text("\n".join(rows) + "\n")
print("Completed all 150 paired evaluations: " + str(HERE / "结果.md"), flush=True)
