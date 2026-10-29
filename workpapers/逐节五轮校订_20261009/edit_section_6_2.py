import re
import xml.etree.ElementTree as ET
from pathlib import Path
from shutil import copy2
from zipfile import ZipFile

from edit_section_1_2_1 import DOCUMENT, M, W
from edit_section_4_4 import replace_paragraph


BACKUP = Path(__file__).with_name("校订前_6.2.docx")
REVISIONS = {
    1557: (
        "世界模型根据状态和动作预测目标环境的后继状态与奖励。受控连续任务采用固定决策步长，模型预测状态增量和奖励；公交任务的相邻事件间隔不固定，模型还需预测下一批决策车辆的数量、线路与站点类别，以及乘客人数、车辆运行状态、事件持续时间和乘客成本。以下先区分两类任务可以获得的训练数据，再说明预测模型的构建。",
    ),
    1560: (
        "受控连续任务采用固定决策步长，将源仿真与目标环境表示为两个具有相同状态和动作定义的马尔可夫决策过程",
    ),
    1562: (
        "两者共享状态空间、动作空间、折扣因子和初始状态分布，但转移核与期望单步奖励可以不同。策略以最大化目标环境中的期望累计奖励为目标：",
    ),
    1564: (
        "其中，",
        "、",
        "分别为状态和动作维度，",
        "、",
        "；",
        "为目标环境产生的奖励。记",
        "，转移的终止标记为",
        "，取1表示终止。源数据集为",
        "，初始目标数据集为",
        "。受控连续任务允许继续与目标环境交互，将新增转移记录加入目标经验池。",
    ),
    1565: (
        "上述固定步长、固定折扣的无限时域表述用于受控连续任务。公交任务沿用第2章的事件决策，在规定运营时域内无折扣累计奖励。以乘客时间为目标时，事件持续时间已通过人数与时间的积分计入成本，不再据此另设时间折扣。公交的SUMO离线数据固定，新增训练交互来自简易仿真，与第5章H2O+的数据条件一致。因此，受控连续任务可以补充目标转移记录，公交任务则利用固定目标记录辅助新的源仿真交互。",
    ),
    1567: (
        "两类任务均先用源数据拟合源模型",
        "，随后固定其参数。在相同初始目标数据上建立两类候选模型：残差模型",
        "保留源预测，拟合目标观测与源预测之差；直接目标模型",
        "直接利用目标转移记录学习预测关系。受控连续任务预测状态增量与奖励，公交任务预测的事件变量在后文说明。",
    ),
    1568: (
        "受控连续任务的源模型和直接目标模型均由5个概率网络组成。每个网络包含3个隐藏层，每层200个Sigmoid线性单元（Sigmoid Linear Unit，SiLU），输出状态增量与奖励的高斯均值和对角方差。输入与输出的缩放参数在各自初始拟合时确定。残差模型保留固定的源预测，其状态修正结构见第6.3.1节。",
    ),
    1569: (
        "两类候选模型使用同一组未参与拟合的目标记录比较预测误差，据此选择残差模型或直接目标模型。受控连续任务在初始目标数据上完成选择，本次训练中不再更换模型类别；后续目标记录只用于更新所选模型。公交的一次拟合比较也固定初始选择，分阶段拟合比较则随源交互增加重新拟合候选模型并再次选择，具体安排见第6.3.3节。",
    ),
    1570: (
        "下文的“H2O+混合续训”同时利用目标离线记录与源仿真经验更新策略；“世界模型辅助”在此基础上加入MC-WM生成的转移及可信度权重；“纯离线续训”仅利用固定SUMO记录更新策略；“纯源训练”仅利用简易仿真交互。这些公交比较均不追加目标环境训练交互。",
    ),
    1571: (
        "本章公交策略只决定附加驻站时长，MC-WM与H2O+共用状态和动作定义，执行时速度倍率固定为1。偏差实验改变道路平均速度、客流OD、路口延误及基本服务时间，这些因素影响环境转移规律，不改变策略的动作定义。同组方法采用相同驻站上限与完成保障规则。",
    ),
    1572: (
        "公交世界模型联合预测类别、计数与连续量，本文称为混合事件模型。线路和站点属于类别，人数属于计数，车头时距、持续时间和乘客成本属于连续量。同一时刻的多车决策不推进物理时间，已选动作的记录按已知规则更新；完成这一批决策后，由模型预测下一批事件。预测内容包括决策车辆数量、线路与站点、车辆运行量、服务状态标记、事件持续时间和乘客成本。下一事件的线路与站点由模型生成，不将实际后继事件的类别作为输入。模型输出依据线路可达站点及有效事件标记，转换为策略所需的观测。",
    ),
    1573: (
        "公交模型同样比较源预测加残差修正与直接目标建模。先利用已计入训练预算的源交互拟合源模型，再将固定目标离线记录分为三组：拟合记录用于估计参数，提前停止验证记录用于判断何时停止拟合，模型选择验证记录用于比较两类候选模型的综合预测误差。所选模型由3个成员组成，对其预测参数取平均后生成一步转移。运营末段仍使用目标离线记录与源仿真采样轨迹训练；模型转移仅从观测运营时域前75%内的状态生成，以减少尚未解决的服务终点预测误差对控制更新的影响。",
    ),
    1574: (
        "综合预测误差对六类损失等权取平均：系统连续量、车辆连续量、下一批事件数量、线路类别、站点类别及前后车存在标记。持续时间和非负乘客成本先作",
        "变换，其中",
        "为待变换的非负标量，再与系统人数变化一起按源拟合记录的均值和标准差标准化，标准差下限为0.05。车辆连续量按源拟合记录中的有效车辆位置标准化。前两项分别对各维平方误差取平均，事件数量、线路和站点采用交叉熵，前后车存在标记采用二元交叉熵；车辆连续量的损失只在有效车辆位置计算。这一综合误差用于模型选择与样本加权，不对应单一物理量，不能与连续控制任务的原始状态均方误差直接比较。",
    ),
}


def main():
    with ZipFile(DOCUMENT) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        assert "".join(body[1556].itertext()) == "6.2 模型构建"
        assert "".join(body[1575].itertext()) == "6.3 求解方法"
        updated = original
        for index, prose in REVISIONS.items():
            updated = replace_paragraph(updated, body[index], prose)

        mask = lambda xml: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", xml, flags=re.S)
        assert mask(original) == mask(updated), "Non-text XML changed."
        final_body = ET.fromstring(updated).find(W + "body")
        assert len(body) == len(final_body)
        changed = [i for i, (a, b) in enumerate(zip(body, final_body)) if ET.tostring(a) != ET.tostring(b)]
        assert set(changed).issubset(REVISIONS), changed
        assert [ET.tostring(m) for m in body.iter(M + "oMath")] == [ET.tostring(m) for m in final_body.iter(M + "oMath")]
        if not BACKUP.exists():
            copy2(DOCUMENT, BACKUP)
        temporary = DOCUMENT.with_suffix(".section-edit.docx")
        with ZipFile(temporary, "w") as target:
            for item in source.infolist():
                data = updated.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename)
                target.writestr(item, data)
    temporary.replace(DOCUMENT)
    with ZipFile(BACKUP) as before, ZipFile(DOCUMENT) as after:
        assert before.namelist() == after.namelist()
        assert [n for n in before.namelist() if before.read(n) != after.read(n)] == ["word/document.xml"]
        initial = before.read("word/document.xml").decode("utf-8")
        assert mask(initial) == mask(updated)
        initial_body = ET.fromstring(initial).find(W + "body")
        final_changed = [i for i, (a, b) in enumerate(zip(initial_body, final_body)) if ET.tostring(a) != ET.tostring(b)]
        assert final_changed == list(REVISIONS), final_changed
    print({"changed_body_blocks": final_changed, "backup": str(BACKUP), "non_text_xml_preserved": True})


if __name__ == "__main__":
    main()
