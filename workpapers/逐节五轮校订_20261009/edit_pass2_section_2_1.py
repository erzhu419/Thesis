from pathlib import Path
from shutil import copy2
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_1_2_1 import W
from edit_section_4_4 import replace_paragraph


ROOT = Path(__file__).resolve().parents[2]
DOCUMENT = ROOT / "output/排版完成_逐节校订_第2轮_20261010.docx"
BACKUP = Path(__file__).with_name("校订前_第2轮_2.1.docx")
REVISIONS = {
    367: (
        "公交系统包括道路网络、线路、站点、计划班次、车辆和乘客行程。线路上、下行各有站点序列与班次。本文将线路与方向的组合称为线路—方向组合，其集合为",
        "，任一组合", "的站点序列为",
        "。该序列按行驶次序列出物理站点，最后一个站点的序号也是该方向的站点总数。同一物理站点可被多条线路使用；物理站点标识表示空间位置，线路内站点序号表示车辆在当前班次中的行驶进度，两者需要区分。",
    ),
    369: (
        "车辆执行班次时，依次经历发车、路段行驶、到站、上下客、附加驻站和离站。必要上下客时间取决于实际服务人数和单人服务时间，控制器不能缩短或取消必要服务，只能在其后选择附加驻站时长。两类时间分别计算，以区分正常服务所需的停留与控制额外增加的时间代价。",
    ),
    375: ("式", "分别描述前车至本车、本车至后车在同一站点的到站间隔。"),
    379: (
        "为说明时距偏差如何向下一站传递，考虑同一方向的相邻班次。以", "和",
        "分别表示班次", "在第", "站的到站时刻与实际离站时刻，以",
        "表示必要上下客时间，", "表示附加驻站时长，",
        "表示上述两部分结束后至车辆实际驶离的等待时间。各段时间满足",
    ),
    381: (
        "其中",
        "表示从实际驶离当前站点至到达下一站的道路运行时间。比较简易仿真与SUMO时，各段时间应按相同边界划分。若统计的道路运行时间已包含上下客、附加驻站和驶离等待，再在简易仿真中分别模拟这些过程，就会重复计算延误。",
    ),
    384: (
        "式",
        "说明，下一站的车头时距由当前站的时距，以及相邻班次在必要服务、附加驻站、驶离等待和道路运行上的时间差共同决定。",
    ),
    385: (
        "控制器直接选择的是附加驻站时长", "，而不能直接指定下一站的车头时距",
        "。必要上下客时间随实际服务人数变化，驶离等待与道路运行时间受交通条件影响；此前的驻站还可能改变后续到站时刻和上客人数，间接影响服务时间。其他条件不变时，本车增加驻站会拉大下一站的前向时距。因此，当前前向间隔已过大时，继续驻站未必有利，仍需结合后车位置、下游需求及乘客时间代价判断。",
    ),
    386: (
        "在车辆容量充足、无上次遗留候车乘客，且乘客以近似不变的强度",
        "到达的局部近似下，本次新增上车人数的期望满足",
        "，其中，期望符号内的变量为本班次在该站新增的上车人数。若每名上车乘客的平均服务时间为",
        "，上客时间随前向时距增长的近似变化率为",
        "。较大的到站间隔可能带来更多上客和更长服务，使车辆进一步延误，形成正反馈。车辆达到容量上限后，新增需求不再等量增加本车上客人数，而可能使未能上车的乘客继续候车。需求冲击实验因此需要区分容量充足与容量受限两种情况。",
    ),
    387: (
        "上述时间关系描述车辆怎样运行，并未指定驻站控制的目标。按时刻表控制以计划离站时刻为参照，按车头时距控制以相邻车辆的服务间隔为参照，乘客导向控制则直接考虑候车与车内延误。三者的优化对象和所需信息不同。以下用典型形式说明这种差别，具体控制律还取决于各项研究对站点、需求和执行条件的假设。",
    ),
    390: (
        "其中",
        "分别为前、后向间隔的反馈增益；驻站命令的上标区分时刻表反馈与车头时距反馈，帽号表示决策时刻的后向时距估计。时刻表反馈通过延后离站纠正提前运行，但无法消除已有晚点。车头时距反馈中，前向间隔偏小或后向间隔偏大时，相应项倾向于增加驻站。两项相加后，将结果限制在允许范围内。车辆容量和换乘需求仍需结合服务目标考虑。",
    ),
    392: (
        "驻站决策可采用三种方式：直接反馈依据少量实时运行量计算动作；预测优化用运行模型推演后续状态并求解动作；强化学习利用交互经验学习价值函数或策略，将主要计算放在训练阶段。预测优化的效果取决于预测能否反映实际运行，学习型控制的效果则取决于训练经验能否支持部署时的动作选择。本文采用强化学习进行驻站决策，在上述运行关系、间隔反馈和乘客成本的基础上，依次研究平稳随机运行、非平稳变化以及仿真与目标环境差异。",
    ),
    394: (
        "驻站控制通过延后离站改变前后车辆的时间关系，也可能改变下游乘客在车辆之间的分配。它可能缓解间隔失衡，同时增加车内乘客时间。因此，时距改善与乘客时间收益需要分别评价，不能由其中一项直接推断另一项。",
    ),
}


def main():
    with ZipFile(DOCUMENT) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        assert "".join(body[365].itertext()) == "2.1 公交运行与控制"
        assert "".join(body[398].itertext()) == "2.2 强化学习基础"
        updated = original
        for index, prose in REVISIONS.items():
            assert 365 < index < 398
            updated = replace_paragraph(updated, body[index], prose)

        # Only prose changes; preserve equations, citations, fields and layout.
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
