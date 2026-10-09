"""Build the current Word's data map and copy its existing small external sources."""

import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "workpapers/论文数据对照_20261009"
EXTERNAL = ROOT / "data/external"


def read(path):
    return json.loads(path.read_text())


def resolve(source):
    path = Path(source)
    if path.is_absolute():
        return path
    if source.startswith("../") or source.split("/")[0] in {
            "experiments", "scripts", "manuscript", "workpapers", "data"}:
        return (ROOT / path).resolve()
    return ROOT.parent / path


def main():
    index = read(OUT / "正文图表索引.json")
    registry = read(OUT / "版本登记.json")
    families = registry["families"]
    mapping = {}
    for family in families:
        for number in family["figures"] + family["tables"]:
            assert number not in mapping, number
            mapping[number] = family["id"]
    concepts = registry["concept_figures"] + registry["concept_tables"]
    captions = index["figures"] + index["tables"]
    assert set(mapping) | set(concepts) == {item["number"] for item in captions}

    sources = {source for family in families for source in family["sources"]}
    for figure in index["figures"]:
        for record in figure["figure_source_records"]:
            sources.update(record["sources"])
    # Operational summaries accompany the already indexed extension reward arrays.
    sources.update(str(path) for path in
                   (ROOT.parent / "RE-SAC/results/bus_seed_extension/logs").glob("*/operational_metrics.json"))
    snapshots, unresolved = [], []
    for source in sorted(sources):
        path = resolve(source)
        if path.is_relative_to(ROOT):
            if not path.exists():
                unresolved.append(dict(source=source, reason="本地未找到"))
            continue
        if path.is_dir() and source.endswith("/audits"):
            candidates = sorted(path.glob("seed_*/audit.json"))
        else:
            candidates = [path]
        for candidate in candidates:
            if not candidate.is_file():
                unresolved.append(dict(source=source, reason="描述性来源、目录或本地未找到"))
                continue
            if candidate.name == "paper.tex":
                continue
            if candidate.suffix not in {".json", ".csv", ".npy", ".npz", ".xlsx", ".xml", ".md", ".py", ".tex"}:
                unresolved.append(dict(source=str(candidate), reason="不属于本次数据/代码快照格式"))
                continue
            if candidate.stat().st_size > 10 * 1024 * 1024:
                unresolved.append(dict(source=str(candidate), reason="超过10MiB，仅登记原位置"))
                continue
            destination = EXTERNAL / candidate.relative_to(ROOT.parent)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, destination)
            snapshots.append(dict(original=str(candidate), local=str(destination.relative_to(ROOT)),
                                  bytes=candidate.stat().st_size))
    snapshots = list({row["local"]: row for row in snapshots}.values())
    (OUT / "外部数据快照清单.json").write_text(
        json.dumps(dict(read_date="2026-10-09", files=snapshots, unresolved=unresolved),
                   ensure_ascii=False, indent=2) + "\n")
    copied = {row["original"]: row["local"] for row in snapshots}

    def link(source):
        path = resolve(source)
        if path.is_relative_to(ROOT) and path.exists():
            relative = str(path.relative_to(ROOT))
        elif str(path) in copied:
            relative = copied[str(path)]
        else:
            return "`" + source + "`"
        return f"[{relative}](../../{relative})"

    lines = [
        "# 论文数据与实验版本对照", "",
        "基准：`output/排版完成_参考文献复核_20261007_导师批改.docx`。整理日期：2026-10-09。",
        "本次只读取该文件，没有修改其文字、数值、批注、样式或图表。下列位置按该Word正文重新识别，不沿用旧Markdown稿编号。",
        "",
        "## 阅读方式", "",
        "先在各章对照表找到图表，再用版本代号查后面的数据登记。`正文图表索引.json`保存全部63张图、61张表、164个标题及表格原值；正文块位置用于脚本定位，不是页码。",
        "六张框架图提取了Word中实际嵌入的可编辑Visio源。图3-2按图片替代描述中的源路径定位；其余56张图找到了字节一致的本地图像，其中包括图F-2。",
        "",
        "## 全文结论的来源", "",
        "| 论文部分 | 实际支撑数据 |",
        "| --- | --- |",
        "| 摘要及第1章研究递进 | 汇总第3—6章各版本；没有独立的摘要实验数据。当前摘要的CS乘客目标退化对应C-STABLE-V5，不对应已选定的软SINDy新版。 |",
        "| 第2章输入、源/目标一致性与协议 | BASE、FID；表2-5—2-8是协议/适用条件说明，不是新增结果。 |",
        "| 第3章随机环境及小结 | 原RE-SAC归档、温度诊断、类别/奖励/集成消融，以及R-MATRIX-OLD的单驻站子集。 |",
        "| 第4章非平稳环境及小结 | 原CS报告、早期BAPR公交比较、恢复/重复冲击、BC交叉、C-STABLE-V5；正文另有C-NATIVE-V8比较。 |",
        "| 第5章跨保真学习及小结 | offline-sumo原研究、H-FORMAL-V20、短预算作用分离、RATIO-V2及开发访问台账。 |",
        "| 第6章世界模型及小结 | 原通用RL确认与类别反事实、公交适配、机制/成本、归一化、20次初始化确认及v19时距补充。 |",
        "| 第7章总结与创新 | 复述上述各章证据；不应将不同参照、奖励、完成保障或初始化范围合并为一个统一排名。 |",
        "| 附录F | 早期接入、最终225回合比较、早期偏差探索三批；不是一套实验的连续扩展。 |", ""
    ]
    for chapter in dict.fromkeys(item["chapter"] for item in captions):
        lines += [f"## {chapter}", "", "| 正文位置 | 图表及名称 | 数据版本 |", "| --- | --- | --- |"]
        for item in sorted((row for row in captions if row["chapter"] == chapter), key=lambda row: row["block"]):
            version = mapping.get(item["number"], "示意/协议，无控制结果数据")
            lines.append(f"| {item['section']} | {item['number']} {item['title']} | {version} |")
        lines.append("")
    lines += ["## 各版本的数据登记", ""]
    for family in families:
        lines += [f"### {family['id']}：{family['name']}", "", family["note"], "",
                  "现用位置：" + "、".join(family["figures"] + family["tables"] or ["第4.4.5节正文直接比较"]) + "。", "",
                  "数据与记录："]
        lines += ["- " + link(source) for source in family["sources"]]
        lines.append("")
    lines += ["## 图源与逐图统计说明", "",
              "下表列Word实际使用的本地图源。实验图统计方式沿用成图来源登记，保留各版本自身的统计单位。完整原始来源列表在`正文图表索引.json`的`figure_source_records`中。", "",
              "| 图号 | 实际图源 | 统计/用途 |", "| --- | --- | --- |"]
    for item in index["figures"]:
        if item["editable_diagrams"]:
            image_source = link(item["editable_diagrams"][0]["local"])
        elif item["exact_image_matches"]:
            image_source = link(item["exact_image_matches"][0])
        else:
            image_source = link(item["image_metadata_paths"][0])
        stats = "；".join(record["statistics"] for record in item["figure_source_records"])
        if item["number"] == "图F-2":
            stats = "5个训练重复内各平均15种测试情形；再汇总5个种子及配对差，95% t区间；225回合不作225次独立训练。"
        if not stats:
            stats = "可编辑原理/信息流程示意，无抽样结果。"
        lines.append(f"| {item['number']} | {image_source} | {stats.replace('|', '/')} |")
    pending = registry["pending_adoption"]
    lines += ["", "## 已选定但尚未写入此Word", "", pending["name"] + "，别名`" + pending["alias"] + "`。",
              pending["status"], "", "- " + link(pending["sources"][0]),
              "- " + link(pending["sources"][2]), "",
              "该版本乘客奖励下六类场景的平均成本均低于原配方RE-SAC，降幅0.083%—0.549%；时距奖励下五类场景改善，持续10倍需求下轻微退化。不能把它直接覆盖到表4-10、4-11的旧BAPR参照或把时距奖励改善写成所有间隔指标改善。", "",
              "## 理论附录与非数据材料", "",
              "附录A—E为数学证明，不用公交评价数据。理论来源和实际编译记录分别保留：", "",
              "- " + link("workpapers/最终验收与人工扩展证明_20261003/人工扩展证明对应.md"),
              "- " + link("workpapers/最终验收与人工扩展证明_20261003/AppendixExtensions.lean"),
              "- " + link("workpapers/最终验收与人工扩展证明_20261003/ExistingAppendixAudit.lean"),
              "- " + link("workpapers/三项闭合_20261003/cvar_status.json"), "",
              "这些历史证明记录不是本次重新编译验收。部分导入依赖位于`/home/erzhu419/mine_code/proof`工程，仓库中保留的脚本与记录不能冒称自包含Lean环境。TransitDuet补充数据及其他未进入此Word的实验仍保存在仓库，但不登记为现用主结果。", "",
              "## 版本边界与资料缺口", "",
              "1. 第3章双向走廊和第4章部分走廊配对，与第5—6章多线路SUMO，规模、乘客集合及成本数值不能跨表直接排序。",
              "2. 第4章当前两目标补充、软SINDy定稿选择和近期直接SINDy/温度诊断是不同算法端点；当前Word尚未合入新定稿结果。",
              "3. 第5章正式保障候选、短预算H2O+、同预算比率修正组、第6章v13无模型基线与附录F.2完整H2O+不是同一个控制器。",
              "4. Daganzo式参照也包含前向/双向反馈及是否叠加完成保障的区别，应沿用对应协议，不能只凭名称拼接。",
              "5. 表3-2仅有原论文报告值，逐回合数据尚未恢复。基础交通资料的采集年份与完整处理历史仍未确证。部分旧批次未记录SUMO版本。",
              "6. 邻近项目快照是2026-10-09本次读取的文件；成图时冻结的`manuscript/figures/source_data`与实验汇总同时保留，前者不能静默代替后者。",
              "7. 未从服务器拉取新CSV、轨迹或检查点。大量训练输入和模型仍在服务器；各批次的protocol、jobs、node_runtime、submission及日志保留具体路径和节点。本仓库不是完整训练数据/所有权重的镜像。", "",
              "## 本次归档范围", "",
              f"本地邻近项目现有来源快照：{len(snapshots)}个文件，合计{sum(row['bytes'] for row in snapshots)/1024**2:.2f}MiB，见[外部数据快照清单.json](外部数据快照清单.json)。",
              "仓库保留实验代码、协议、已在本地的逐例结果与权重、分析汇总、绘图资产、图源、证明及修改过程记录；排除论文PDF/Word/PPT、旧章节正文、全文备份包、完整正文提取和整页预览、虚拟环境与缓存。没有删除本地被排除的文件。", ""]
    (OUT / "论文数据与版本对照.md").write_text("\n".join(lines))
    print(json.dumps(dict(figures=len(index["figures"]), tables=len(index["tables"]),
                         families=len(families), snapshots=len(snapshots),
                         snapshot_mib=sum(row["bytes"] for row in snapshots)/1024**2), ensure_ascii=False))


if __name__ == "__main__":
    main()
