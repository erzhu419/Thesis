from pathlib import Path
from shutil import copy2
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_1_2_1 import W
from edit_section_4_4 import replace_paragraph


ROOT = Path(__file__).resolve().parents[2]
DOCUMENT = ROOT / "output/排版完成_逐节校订_第2轮_20261010.docx"
BACKUP = Path(__file__).with_name("校订前_第2轮_2.2.docx")
REVISIONS = {
    410: (
        "本文公交H2O+与MC-WM采用上述有限时域无折扣目标，非终止转移的延续因子为1，终止转移为0。其他任务沿用各自的奖励、折扣和终止约定；第3章的贝尔曼收缩分析则要求折扣因子严格小于1。",
    ),
    411: (
        "同一时刻可能有多辆车需要决策。公交H2O+与MC-WM按服务和车辆标识排列这些车辆，逐车收集驻站动作，收集完后再推进仿真时钟。前面的转移只更新本批已选动作等决策信息，时间间隔和乘客成本均为零；最后一个转移记录推进时钟后经过的时间及系统成本。",
    ),
    412: (
        "策略每次为一辆车输出动作，后续车辆的决策状态包含本批先前已选动作，但不包含这些动作尚未产生的运行结果。这一顺序表示决策依赖，不增加物理等待时间。",
    ),
    415: (
        "输入维度随控制任务而定。多线路公交离线学习使用15个原始观测量及上一驻站动作。公交H2O+与MC-WM最多同时编码16辆待决策车辆，每辆车占据一个编码位置；原始状态为245维。将各车重复包含的全局量统一表示，并用容量归一化载客量后，状态为229维。有效车辆标记和本批已选驻站时长表示决策进度。同一任务的比较方法共用输入，不要求不同任务采用相同维数。",
    ),
    418: (
        "第5章在必要上下客完成后决策。令", "为本次服务完成时刻，",
        "为前车在同站最近一次有效服务完成时刻，前向间隔由两者之差计算。后车尚未完成本站服务，只能利用当前运行信息估计其后向间隔。记同一线路坐标下的本车和后车位置为",
        "、", "，用于估计的有效速度为", "。后车已上线时，采用",
    ),
    420: (
        "其中", "，",
        "为后向计划间隔。估计间隔的上限取2400秒与三倍计划间隔中的较大值，速度下限取1米/秒。SUMO采用最近仿真步的车道平均速度，无有效读数时取10米/秒；简易仿真使用已有标定记录，在对应需求条件下按决策时间选取邻近的站点速度样本，不使用测试日未来速度。该后向量是决策时的近似预测，未逐项计入后车后续路口等待、上下客及驻站，不等同于最终实现的服务间隔。",
    ),
    424: (
        "简易仿真对已分配到线路的分时OD求和，再换算为每秒需求率。该输入来自场景设定，既不是下一名乘客的实际到达时刻，也不是由最近上客人数反推的估计。需求强度或线路选择发生变化时，预设需求率与实际客流之间、车辆运行预测与实际后果之间均可能出现偏差。",
    ),
    443: (
        "上述成本也可按人累计。令", "为时域内已出发乘客集合，", "、",
        "分别为乘客", "在时刻",
        "的候车与在车指示变量，处于对应状态取1，否则取0：",
    ),
    449: (
        "对于同一时刻的", "个动作，前",
        "个转移的成本为零，最后一个转移记录本批动作确定后至下一决策批次的乘客成本",
        "，总和为",
        "。因此，给定同一批动作，逐车记录转移保留无折扣总成本，并使后续车辆利用已选动作；各步的价值学习目标仍需按对应转移定义计算。",
    ),
    455: (
        "多线路公交离线学习按已完成行程段统计乘客时间，公交H2O+与MC-WM则按规定时域内全部已出发乘客统计。第5章分别报告两类结果，各任务内的比较采用共同统计对象和分母。",
    ),
    456: (
        "训练转移的时间区间也需区分。第3、4章公交训练按同一车辆的相邻观测计算时间增量，不同车辆的区间可能重叠，不能把这些训练奖励直接相加作为系统成本。公交H2O+与MC-WM按全系统事件顺序结算，时间区间不重复。乘客时间补充比较另行累计全系统人数时间积分，在同一任务内使用共同分母和评价时域。",
    ),
    466: (
        "折扣任务采用", "；按实际时间折扣时，须指定时间单位，以",
        "计算延续因子。公交H2O+与MC-WM无折扣累计乘客成本，事件间隔计入奖励；第3章收缩分析采用严格折扣。奖励、折扣与终止规则按各任务设定。",
    ),
    473: (
        "各车在各自决策时刻调用共享演员，记录进入共同回放。动作控制当前车辆，训练利用车队经验；类别特征区分服务身份，连续量描述运行条件。后续各章在此基础上分别改进动作评价、变化适应及目标数据的利用。",
    ),
    477: (
        "若运行规律由不能直接观测的模式",
        "决定，可将公交状态与该模式共同建模，并根据历史观测和动作计算模式后验",
        "。历史只包含决策前已获得的观测与动作。给定模式演化和观测模型后，这构成POMDP；历史编码器可近似提取模式信息，元学习则进一步在多个训练任务上学习适应方法。概率上下文与元强化学习提供了相应实例。",
        "第4章分别说明受控任务中的模式识别、公交训练中的近期统计量调节，以及需要跨任务训练的元学习，三者并非同一种过程。",
    ),
    492: (
        "上述关系支撑本文的递进研究。未预先获得目标域转移训练数据时，第3章用RE-SAC研究平稳随机运行下的动作评价，第4章用BAPR及CS-BAPR研究运行规律变化后的识别与适应。获得固定目标离线数据后，第5章在H2O+框架下用目标记录辅助初始化和混合学习，新增交互仍来自简易仿真。第6章在相同数据条件下学习世界模型，在保留源预测并修正其误差与直接目标建模之间选择，以缓解标定不足造成的预测偏差。",
    ),
}


def main():
    with ZipFile(DOCUMENT) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        assert "".join(body[398].itertext()) == "2.2 强化学习基础"
        assert "".join(body[493].itertext()) == "2.3 迁移学习基础"
        updated = original
        for index, prose in REVISIONS.items():
            assert 398 < index < 493
            updated = replace_paragraph(updated, body[index], prose)

        # Only prose changes; equations, citations, fields and layout remain intact.
        mask = lambda xml: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", xml, flags=re.S)
        assert mask(original) == mask(updated), "Non-text XML changed."
        final_body = ET.fromstring(updated).find(W + "body")
        assert len(body) == len(final_body)
        changed = [i for i, (a, b) in enumerate(zip(body, final_body)) if ET.tostring(a) != ET.tostring(b)]
        assert set(changed).issubset(REVISIONS), changed
        if not BACKUP.exists():
            copy2(DOCUMENT, BACKUP)
        temporary = DOCUMENT.with_suffix(".tmp")
        with ZipFile(temporary, "w") as target:
            for item in source.infolist():
                target.writestr(item, updated.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename))
    temporary.replace(DOCUMENT)
    with ZipFile(BACKUP) as before, ZipFile(DOCUMENT) as after:
        assert before.namelist() == after.namelist()
        assert [n for n in before.namelist() if before.read(n) != after.read(n)] == ["word/document.xml"]
        baseline = before.read("word/document.xml").decode("utf-8")
        assert mask(baseline) == mask(updated), "Protected structure changed."
        initial_body = ET.fromstring(baseline).find(W + "body")
        final_changed = [i for i, (a, b) in enumerate(zip(initial_body, final_body)) if ET.tostring(a) != ET.tostring(b)]
        assert final_changed == list(REVISIONS), final_changed
    print({"changed_blocks": final_changed, "paragraph_count": len(REVISIONS),
           "backup": str(BACKUP), "output": str(DOCUMENT)})


if __name__ == "__main__":
    main()
