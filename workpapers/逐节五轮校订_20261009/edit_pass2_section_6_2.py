from pathlib import Path
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_3_2 import W
from edit_section_4_4 import replace_paragraph


ROOT = Path(__file__).resolve().parents[2]
DOCUMENT = ROOT / "output/排版完成_逐节校订_第2轮_20261010.docx"
BACKUP = Path(__file__).with_name("校订前_第2轮_6.2.docx")
REVISIONS = {
    1557: (
        "世界模型预测给定状态和动作之后的运行结果，动作仍由策略选择。受控连续任务采用固定决策步长，模型预测状态增量和奖励；公交任务的相邻事件间隔不固定，还需预测下一批决策车辆及其运行状态、事件持续时间和乘客成本。本节先区分两类任务的数据条件，再说明预测模型的构建。",
    ),
    1560: (
        "对于固定决策步长的受控连续任务，将源仿真与目标环境表示为两个马尔可夫决策过程：",
    ),
    1562: (
        "两者共享状态空间、动作空间、折扣因子和初始状态分布，转移核与期望单步奖励则可以不同。策略以最大化目标环境中的期望累计奖励为目标：",
    ),
    1565: (
        "上述固定步长、固定折扣的无限时域表述用于受控连续任务。公交任务沿用第2章的事件决策，在规定运营时域内无折扣累计奖励。以乘客时间为目标时，事件持续时间已通过人数与时间的积分计入成本，不再另设时间折扣。公交的数据条件与第5章H2O+一致：SUMO目标离线记录固定，新增训练交互仅来自简易仿真。",
    ),
    1567: (
        "两类任务均先用源数据拟合源模型",
        "并固定其参数，再用相同的初始目标数据拟合两类候选模型：残差模型",
        "保留源预测并学习修正量；直接目标模型",
        "不依赖源预测，直接学习目标环境中的预测关系。",
    ),
    1568: (
        "受控连续任务的源模型和直接目标模型各含5个概率网络。每个网络设3个隐藏层，每层200个神经元，激活函数为Sigmoid线性单元（Sigmoid Linear Unit，SiLU）；输出状态增量与奖励的高斯均值和对角方差。输入、输出缩放参数在各自初次拟合时确定。残差模型固定源预测，状态修正结构见第6.3.1节。",
    ),
    1569: (
        "两类候选模型在同一组未参与参数拟合的目标验证记录上比较，选择预测误差较小者。受控连续任务仅在初始目标数据上选择模型类别，新增目标记录只更新所选模型。公交的一次拟合实验也固定初始选择；分阶段实验则随源交互增加，重新拟合并选择模型，见第6.3.3节。",
    ),
    1570: (
        "下文的“H2O+混合续训”同时利用目标离线记录与源仿真经验更新策略；“世界模型辅助”在此基础上加入MC-WM生成的一步转移，并按置信度加权参与价值更新；“纯离线续训”仅利用固定SUMO记录；“纯源训练”仅利用简易仿真交互。这些公交比较均不追加目标环境训练交互。",
    ),
    1571: (
        "本章公交策略只决定附加驻站时长，MC-WM与H2O+共用状态和动作定义，执行时速度倍率固定为1。偏差实验改变道路平均速度、客流OD、路口延误及必要停站服务时间，影响环境转移规律，不改变动作定义。同组方法采用相同驻站上限与完成保障规则。",
    ),
    1572: (
        "公交世界模型联合预测类别、计数与连续量，称为混合事件模型：线路、站点为类别，人数为计数，车头时距、事件持续时间和乘客成本为连续量。同一时刻各车辆依次决策时，物理时间不推进，已选动作的记录按已知规则更新；完成该批决策后，模型预测物理时间推进后的下一批事件，包括决策车辆数量、线路与站点、车辆运行量、服务状态标记、事件持续时间和乘客成本。生成模型转移时，后继事件的线路与站点由模型预测，不使用其真实类别作为输入。再依据线路可达站点及有效事件标记，将预测结果转换为策略观测。",
    ),
    1573: (
        "公交源模型使用已计入训练预算的源交互拟合。固定目标离线记录分为三组：拟合记录用于估计参数，提前停止验证记录用于判断何时停止拟合，模型选择验证记录用于比较两类候选模型的综合预测误差。所选模型由3个成员组成，对成员输出的预测参数取平均后生成一步转移。运营末段仍使用目标离线记录与源仿真轨迹训练；模型转移仅以运营时域前75%的已观测状态为起点，以减小服务终点预测误差对更新的影响。",
    ),
    1574: (
        "综合预测误差由六类损失等权平均得到，分别对应系统连续量、车辆连续量、下一批事件数量、线路类别、站点类别及前后车存在标记。持续时间和非负乘客成本先作",
        "变换，其中",
        "为待变换的非负标量，再与系统人数变化一起按源拟合记录的均值和标准差标准化，标准差下限为0.05。车辆连续量也按源拟合记录中的有效车辆数据标准化。前两类损失取各维平方误差的平均，车辆连续量仅在有效车辆位置计算；事件数量、线路和站点采用交叉熵，前后车存在标记采用二元交叉熵。综合误差用于模型选择及样本加权，并非单一物理量的误差，不能与受控连续任务的原始状态均方误差直接比较。",
    ),
}


def main():
    assert BACKUP.exists(), "Refine from the preserved pre-edit backup."
    with ZipFile(BACKUP) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        assert "".join(body[1556].itertext()) == "6.2 模型构建"
        assert "".join(body[1575].itertext()) == "6.3 求解方法"
        updated = original
        for index, prose in REVISIONS.items():
            updated = replace_paragraph(updated, body[index], prose)

        # Only prose may change; preserve all objects, fields and layout XML.
        mask = lambda xml: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", xml, flags=re.S)
        assert mask(original) == mask(updated), "Non-text XML changed."
        final = ET.fromstring(updated).find(W + "body")
        assert len(body) == len(final) == 2924
        changed = [i for i, (a, b) in enumerate(zip(body, final)) if ET.tostring(a) != ET.tostring(b)]
        assert changed == list(REVISIONS), changed
        for index in changed:
            numbers = lambda p: re.findall(r"\d+(?:\.\d+)?", "".join(p.itertext()))
            assert numbers(body[index]) == numbers(final[index]), index
        temporary = DOCUMENT.with_suffix(".tmp")
        with ZipFile(temporary, "w") as target:
            for item in source.infolist():
                target.writestr(item, updated.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename))
    temporary.replace(DOCUMENT)
    with ZipFile(BACKUP) as before, ZipFile(DOCUMENT) as after:
        assert before.namelist() == after.namelist()
        assert [n for n in before.namelist() if before.read(n) != after.read(n)] == ["word/document.xml"]
    print({"changed_blocks": changed, "backup": str(BACKUP), "output": str(DOCUMENT)})


if __name__ == "__main__":
    main()
