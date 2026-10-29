from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_pass2_section_5_2 import BACKUP, DOCUMENT, REVISIONS
from edit_section_3_2 import M, W, prose_groups
from edit_section_4_4 import replace_paragraph


UPDATES = {
    1228: (
        "本节沿用第2章的事件决策模型，分别说明三种训练条件及两项公交任务的状态、目标和驻站动作。",
    ),
    1234: (
        "各学习过程统记为",
        "，下标分别表示纯离线学习、目标在线微调和跨保真学习。纯离线学习依赖已有采集记录；目标在线微调直接尝试新动作；跨保真学习在源仿真中尝试新动作，需处理源目标运行差异。回报应在各自数据条件下比较。",
    ),
    1235: (
        "目标离线数据记录已执行动作的运行结果，源仿真响应当前策略的新动作。两者结合可扩展动作尝试，但新增源经验并非新增目标经验。",
    ),
    1240: (
        "第一项任务在18,000秒多线路SUMO场景中，以3,374,285条离线驻站转移比较固定数据学习与30,000次额外SUMO决策。第二项H2O+公交任务在24,000秒场景中，用另一套3,284,718条离线转移结合新增源仿真经验训练。",
    ),
    1257: (
        "随机种子为20261002。图上方为采样过程；下方为经验与再采样累积分布函数（Cumulative Distribution Function，CDF）相对母分布的误差，横轴为对数坐标。各组来自同一母分布。",
    ),
    1259: (
        "第一项任务含12个线路—方向组合、389个计划班次、353个物理站点及459个线路—站点组合记录。各班次同回合运行，由同一策略控制符合条件的到站事件。",
    ),
    1260: (
        "策略输入为15个原始观测量及上一附加驻站动作，包括线路、车辆、站点位置、时段和方向等类别信息，以及车头时距、候车人数、计划间隔、停站信息、运行时间、同线间隔和路段速度。站点位置分47类，表示线路内次序。",
    ),
    1261: (
        "策略输出映射为0至60秒附加驻站，速度倍率固定为1.0。离线数据由多种策略采集，学习动作与实际执行的附加驻站量一致。",
    ),
    1262: (
        "第二项跨保真任务采用同时到站车辆的批次状态和混合驻站分布。两环境共用第2章的基础运营资料及控制定义，以不同机制推进事件。固定数据来自SUMO，新增训练经验来自简易仿真。",
    ),
    1282: (
        "其中",
        "在归一化动作",
        "处表示不额外驻站，",
        "为连续时长分支密度。动作经",
        "映射为秒。确定性评价时，若",
        "，以高斯均值经上述变换得到时长，否则选择0秒附加驻站。",
    ),
    1283: (
        "固定目标离线数据中，80.2246%的动作是不额外驻站。单一连续分布在该点的概率为零，故采用混合分布，分别学习是否额外驻站及增加多久。",
    ),
    1288: (
        "0秒样本仅学习门控概率，大于0秒的样本还学习时长分布。雅可比仅用于连续分支，0秒分支计算离散概率；离线初始化和源仿真续训均保留精确0秒动作。",
    ),
}


with ZipFile(DOCUMENT) as source:
    current = source.read("word/document.xml").decode("utf-8")
    body = ET.fromstring(current).find(W + "body")
    updated = current
    for index, prose in UPDATES.items():
        updated = replace_paragraph(updated, body[index], prose)

    with ZipFile(BACKUP) as before:
        original = before.read("word/document.xml").decode("utf-8")
        baseline = ET.fromstring(original).find(W + "body")
        final = ET.fromstring(updated).find(W + "body")
        mask = lambda xml: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", xml, flags=re.S)
        assert mask(original) == mask(updated)
        changed = [i for i, (a, b) in enumerate(zip(baseline, final)) if ET.tostring(a) != ET.tostring(b)]
        assert changed == sorted(set(REVISIONS) | set(UPDATES)), changed
        for index in changed:
            old_nodes, _, protected = prose_groups(baseline[index])
            new_nodes, _, _ = prose_groups(final[index])
            assert all(old_nodes[i].text == new_nodes[i].text for i in protected)
            for tag in (M + "oMath", W + "pPr", W + "hyperlink", W + "instrText", W + "fldChar"):
                assert [ET.tostring(n) for n in baseline[index].iter(tag)] == [ET.tostring(n) for n in final[index].iter(tag)]
            assert re.findall(r"\d+(?:\.\d+)?", "".join(baseline[index].itertext())) == re.findall(r"\d+(?:\.\d+)?", "".join(final[index].itertext()))

    temporary = DOCUMENT.with_suffix(".tmp")
    with ZipFile(temporary, "w") as target:
        for item in source.infolist():
            target.writestr(item, updated.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename))
temporary.replace(DOCUMENT)
with ZipFile(BACKUP) as before, ZipFile(DOCUMENT) as after:
    assert before.namelist() == after.namelist()
    assert [n for n in before.namelist() if before.read(n) != after.read(n)] == ["word/document.xml"]
print({"changed_blocks": changed, "paragraphs": len(changed), "output": str(DOCUMENT)})
