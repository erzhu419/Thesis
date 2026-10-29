from pathlib import Path
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_3_2 import W
from edit_section_4_4 import replace_paragraph


ROOT = Path(__file__).resolve().parents[2]
DOCUMENT = ROOT / "output/排版完成_逐节校订_第2轮_20261010.docx"
BACKUP = Path(__file__).with_name("校订前_第2轮_6.3.docx")
REVISIONS = {
    1576: (
        "MC-WM算法",
        "先拟合残差模型与直接目标模型，在模型选择验证集上择优，再生成一步转移，按置信度加权用于价值更新。源模型参数保持固定，执行动作仍由演员输出。第6.3.1至6.3.2节介绍允许新增目标交互的受控连续任务，第6.3.3节介绍仅使用固定目标离线数据的公交训练，第6.3.4节分析模型误差与置信权重的尺度。",
    ),
    1578: (
        "受控连续任务将10,000条初始目标转移按固定随机方式划分为三组，比例为80%、10%、10%。拟合集用于估计参数；特征筛选与提前停止验证集用于筛选残差特征，并决定直接模型何时停止拟合；模型选择验证集用于比较拟合后的候选模型。三组记录不重叠，但按转移而非完整轨迹划分，相邻转移可能分属不同组。控制泛化另用新测试回合评价。",
    ),
    1584: (
        "该误差也称经验预测风险，衡量预测偏差，不是交通事故概率。模型选择验证用于比较候选预测模型，与道路速度等仿真参数的标定不同。选取规则为",
    ),
    1587: (
        "特征筛选和提前停止均会影响候选模型，因而另用未参与这些决策的同一组记录比较拟合结果，减少模型选择对筛选与停止过程的依赖。",
    ),
    1589: (
        "模型选择取决于本次目标数据与拟合预算，不预设残差模型始终更简单。源预测误差可由少量关系描述时，残差修正能够利用已有结构；源模型反映的规律与目标不符时，直接模型无需先抵消不合适的源预测。",
    ),
    1591: (
        "以下先介绍残差模型的构造，再说明类别选定后如何利用新增目标经验更新所选模型。",
    ),
    1594: (
        "可解释动力学模型采用不同结构：方程学习网络组合基本运算，人工智能（Artificial Intelligence，AI）符号回归方法AI Feynman利用对称性、可分性等性质搜索表达式，结合SINDy的强化学习方法（SINDy Reinforcement Learning，SINDy-RL）将稀疏动力学、奖励和策略表示与强化学习结合。",
        "本章以稀疏关系描述源预测与目标观测的差异。线性分支仅更新选定的输出与特征之间的连接，神经网络补充修正项使用各输出所选特征的并集；后续拟合不再改变这一特征集合。",
    ),
    1602: (
        "其中，",
        "为特征数，",
        "为经均方根缩放的特征，",
        "为稀疏系数矩阵。掩码",
        "标记至少被一个状态坐标选中的特征。补充修正网络含一个32单元SiLU隐藏层，输出层初始化为零，起始时不改变线性预测；随后各状态坐标均可使用并集中的全部特征。因此，完整残差还可依赖该坐标的线性分支未选中的特征。",
    ),
    1606: (
        "线性系数和输入掩码分别限定两个分支可用的特征。稀疏方程只能解释线性修正，不能代表包含神经网络输出的完整残差。不同训练所选特征是否一致，以特征集合的重合程度评价。",
    ),
    1615: (
        "目标记录上的预测误差用于构造局部置信，再与后续状态的置信评价共同确定模型样本的训练权重；局部置信不是模型预测正确的概率。对记录",
        "定义",
    ),
    1626: (
        "每250次目标交互，从目标经验池抽取400个状态，以当前策略动作生成一步模型转移。置信权重仅作用于损失，不修改预测奖励。只生成一步转移，在扩充局部训练样本的同时避免连续多步预测误差的累积。",
    ),
    1629: (
        "模型样本的可信程度与动作是否产生违例分别处理：前者由",
        "调节，后者用二元标记",
        "表示。环境转移的违例标记来自环境诊断，模型转移则按同一环境的状态判据生成标记。",
    ),
    1639: (
        "概括模型选择与更新：连续任务仅更新初始所选模型，公交分阶段重新选择；一步模型转移按置信度加权参与价值学习。",
    ),
    1643: (
        "公交训练将第6.2.2节的混合事件模型接入第5章的混合学习：模型预测类别、计数与连续量，策略学习共同使用目标离线数据与源仿真经验。目标数据按完整运行日或需求场景组划分为拟合集、提前停止验证集和模型选择验证集；源模型由初始源交互经验拟合，随后固定参数。一次拟合在策略续训前完成目标模型拟合与类别选择，续训中不再拟合模型；分阶段拟合则随源交互增加，按近期策略经验为固定目标记录加权，重新拟合候选并选择类别。所选模型含3个成员，以六类综合预测误差构造局部置信。两种方式均不追加目标训练数据，以下说明其置信度递推更新、混合价值更新及归一化改进。",
    ),
    1651: (
        "式",
        "中，混合经验项与模型项的系数分别为0.75、0.25。一次拟合实验仅在附加更新中采用该损失，其余更新沿用原混合价值损失；分阶段拟合则在每次常规评论家更新中采用上述组合，不增加优化步数。",
    ),
    1654: (
        "模型选择决定使用哪类预测，置信加权决定模型样本对价值学习的贡献。策略网络、停站幅度变换与完成保障沿用各组公交参照设置，在相同SUMO任务与服务时域下评价控制效果。",
    ),
    1655: (
        "原置信度递推更新将当前对数置信与折扣后的未来评价相加。即使预测误差不变，只要局部置信小于1，对数置信便为负，递推累积会压低样本权重。归一化调整当前项的系数，使未终止转移中的当前与未来两项构成加权平均，减小递推累积对权重尺度的影响。尺度关系见第6.3.4节，训练与服务效果见第6.4.3至6.4.5节。",
    ),
    1658: (
        "局部置信、误差尺度更新、下一动作、模型拟合与类别选择均保持不变，仅以",
        "替换模型损失中的",
        "。乘客时间主要实验采用0.75倍混合经验损失与0.25倍模型损失；第6.4.5节的时距复核将模型项系数改为0.125。损失系数调节模型项的总体作用，各样本置信权重调节单条模型转移的作用，预测奖励和执行时的驻站范围均不改变。",
    ),
    1661: (
        "修正只用于非终止、持续时间大于零且处于运营时域前75%的转移。当前事件线路须与参考记录相同，其标准化状态与动作到2048条目标拟合记录的最近邻距离，不超过提前停止验证集相应距离的95%分位数。其余转移沿用原目标；同一时刻各车辆依次决策时，动作记录仍按原规则更新。该操作组合的是标量价值目标，不平均离散事件身份，也不改变演员使用的数据来源。",
    ),
    1663: (
        "固定策略与模型后，先分析由累计环境奖励定义的价值，不计SAC熵项；再分析对数置信的递推尺度。前者说明模型误差如何影响价值评价，后者说明置信递推如何改变模型样本的训练权重，分别对应后续模型类别与样本权重的比较。",
    ),
    1688: (
        "模型选择采用标准化后的状态增量与奖励预测误差，理论价值界则采用转移核总变差误差，前者不能直接代入后者。贝尔曼更新还通过后继价值反映转移误差的影响，而模型选择误差未考虑各坐标对长期价值的敏感程度。因此，同等预测改善可能产生不同控制效果。置信加权将预测样本用于价值学习，最终作用仍由更新后的策略表现评价。",
    ),
    1695: (
        ".9命题E.1证明式(6-31)。无终止且",
        "恒为",
        "时，归一化权重为",
        "，原权重为",
        "。正幂变换保留权重排序，但改变数值尺度。因此另设固定0.5权重作比较：若两种权重均有效，收益可能主要来自恢复模型损失的作用，而非状态相关置信的差异。",
    ),
}


def main():
    assert BACKUP.exists(), "Refine from the preserved pre-edit backup."
    with ZipFile(BACKUP) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        assert "".join(body[1575].itertext()) == "6.3 求解方法"
        assert "".join(body[1696].itertext()) == "6.4 仿真实验与结果分析"
        updated = original
        for index, prose in REVISIONS.items():
            updated = replace_paragraph(updated, body[index], prose)

        # Detect any change beyond the selected prose, including native objects.
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
