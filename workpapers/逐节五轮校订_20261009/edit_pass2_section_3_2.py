from pathlib import Path
from shutil import copy2
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_1_2_1 import W
from edit_section_3_2 import prose_groups
from edit_section_4_4 import replace_paragraph


ROOT = Path(__file__).resolve().parents[2]
DOCUMENT = ROOT / "output/排版完成_逐节校订_第2轮_20261010.docx"
BACKUP = Path(__file__).with_name("校订前_第2轮_3.2.docx")
REVISIONS = {
    649: (
        "沿用第2章的事件决策过程，本章固定线路、车辆供给与发车计划，学习附加驻站策略。状态包含车辆身份、运行位置和时段，转移按同一车辆的后续服务事件构造，训练经验全部来自仿真交互。不同车辆的经验共同参与学习，所得策略由全车队使用。",
    ),
    650: (
        "平稳假设约束的是给定状态和动作后的转移与奖励规律，而非每次运行的时距、客流和回报。类别信息用于区分运行条件，随机运行则提供相应的动作后果。以下说明状态表示、决策与反馈的对应关系，以及时距奖励、保守评价和乘客时间成本之间的关系。",
    ),
    654: (
        "RE-SAC走廊实验保留上述四类类别信息，连续输入采用前后间隔、下一站服务时间估计和同方向路段速度向量。后向间隔取后车最近保存的前向间隔观测，并非当前时刻的精确值；下一站服务时间根据当前候车量、路段距离与速度参数估计，不使用未来信息。具体设置见第3.4.1节。早期方案与走廊实验的连续输入维度可以不同。",
    ),
    655: (
        "共享策略使不同车辆的经验共同参与学习，类别输入则使策略能够区分车辆及运行位置。因此，本章改变的是状态表示与学习组织，并未忽略车辆在运行中的相互影响。",
    ),
    656: (
        "第1.2.2节介绍的MARL也可以共享参数，不能仅凭参数共享区分单智能体与多智能体。本章由一个学习器依次处理全车队的车辆决策，统一更新策略与动作价值网络，不逐车训练独立决策器，也不另设联合动作价值分解或车辆间通信。第4章沿用这一组织方式，研究运行规律变化时的适应。",
    ),
    657: (
        "类别信息表示车辆及运行位置，连续量反映时距、人数、速度和计划偏差等运行状态。具体输入取决于各实验的信息条件。",
    ),
    661: (
        "实体嵌入为每个类别分配可训练向量，使类别关系由学习任务确定。",
        "作者前期将其用于公交策略与价值函数，本章沿用这一表示。",
        "不同车辆通过共享网络学习共同关系，与多任务学习中的共享表征相联系。",
    ),
    666: (
        "各类信息分别使用嵌入矩阵；同一策略或价值网络内的嵌入与后续层由各车辆共用，并随相应学习目标更新。类别编号不被当作物理距离，也无需为每辆车或每个站点另建完整策略。",
    ),
    667: (
        "类别嵌入使网络能够区分车辆与站点，但不会增加经验不足类别的样本量。各车参与训练的次数仍取决于运行过程和回放采样，下一小节将说明其关系。",
    ),
    669: (
        "设某类别共有 ",
        " 个取值，第一隐藏层宽度为 ",
        "。独热输入的对应权重有 ",
        " 个参数；采用 ",
        " 维嵌入后，线性映射分解为两个矩阵，参数量为 ",
        "，秩不超过 ",
        "。较小的 ",
        " 限定了类别的线性表示；但 ",
        " 较小时，嵌入未必减少参数。因此，选择嵌入是为了学习类别关系，而非保证参数压缩。",
    ),
    670: (
        "车辆编号和方向类别没有连续距离的含义；站点虽有顺序，客流、换乘与路况也不必随编号线性变化。因此，将类别信息与连续运行量分别表示，再由共享网络学习二者的关系。",
    ),
    673: (
        "共享回放缓存收集各车的转移经验。训练按车辆实际到站顺序组织，不要求各车同时决策，也不要求一次更新包含等量的每车样本。",
    ),
    676: (
        "RE-SAC走廊实现按车辆暂存决策，待同车后续服务事件的状态和奖励可用后，构造下式转移。奖励下标对应第k次动作，不是反馈到达的时刻：",
    ),
    678: (
        "不同车辆的完整转移进入同一经验池，当前动作与后续反馈仍按同一车辆对应。后述正则项组合比较继续记录车辆周转后的运行，不将单个班次结束视为整日任务结束；其他实验的价值终止条件按各自任务定义设置。",
    ),
    681: (
        "因此，贡献转移较多的车辆在均匀回放中被采到的次数也更多。共享网络使经验较少的车辆能够利用其他车辆的经验，但并未使每辆车获得相同训练量。共享学习与按车辆均衡采样是不同的组织方式。",
    ),
    682: (
        "优先经验回放按样本对学习的重要程度调整采样概率。",
        "它不同于按车辆或站点均衡采样。本章采用均匀共享回放，不额外平衡各车样本数量；类别嵌入用于区分输入信息，并不改变采样概率。",
    ),
    683: (
        "时段输入可以区分常规峰谷，却不能揭示尚未观测到的突发客流；输入已有车辆编号也不意味着策略能够适应新车辆、新线路。本章使用可获得的运行观测，第4章进一步利用反馈识别运行规律变化，第5章再引入固定目标离线数据。",
    ),
    699: (
        "为单独分析局部时距均衡，暂以两侧间隔相对计划的平方偏差之和为目标。这一简化目标不同于实际训练奖励：",
    ),
    703: (
        "式",
        "给出上述局部平方偏差目标的最优动作，并非学习策略的控制律，也尚未计入驻站增加的车内时间。",
    ),
    712: (
        "图中上排为解析曲面，红线标出两侧间隔相等的截面；下排固定两侧间隔相等或两侧平均间隔为360秒。为同图展示，计划间隔奖励用 ",
        " 表示，指数项保留原值。该正仿射变换不改变状态排序；曲面说明函数形状，并非训练结果，不能据其高度判断算法优劣。",
    ),
    715: (
        "以下先说明保守价值评价的依据，再分析时距均衡何时能够降低乘客时间成本。",
    ),
    716: (
        "第2.2.4节已区分风险敏感与鲁棒评价：前者改变对不同结果的偏好，后者在允许的分布集合内考察最不利后果。CVaR的概率重加权形式联系了这两类评价。",
    ),
    719: (
        "式",
        "将最不利评价写为名义评价减去价值正则项；当",
        "时，该项非负。不同状态动作对的后继分布可以独立选择时，这一关系可用于鲁棒贝尔曼更新。Derman等讨论了相应不确定性与正则化的对应。",
        "",
        ".7的命题A.2、A.3和A.4分别给出该恒等式、分布距离下界及CVaR风险包络的推导。",
    ),
    722: (
        "上述关系为保守价值评价提供依据，但正则项的具体形式取决于不确定集合。RE-SAC根据目标网络权重范数设置价值偏移，并利用集成评价器的估计差异调节价值与策略更新。前者不是转移方差的直接估计，后者也不等同于运行结果的随机波动。具体更新见第3.3节。",
    ),
    723: (
        "RE-SAC并未直接估计回报分布或优化其中最差5%或10%的结果，不能将其网络正则项直接视为CVaR。其作用需结合第3.3节的更新定义与运行实验判断。",
    ),
    724: (
        "为分析时距与乘客时间的关系，考虑容量充足、按离站时刻完成服务的局部近似。乘客以恒定强度 ",
        "（人/秒）到达，由各区间末的车辆接走。前车至当前车、当前车至后车的期望候车总时间分别为 ",
        " 和 ",
        "。固定前后车服务时刻，附加驻站引起的候车总时间变化为",
    ),
    730: (
        "式",
        "将时距均衡所需的时长减去车内时间代价对应的时长，再限制在可行动作范围内。其他条件相同时，车内人数增加使局部最优驻站缩短或保持不变。这里计算的是乘客时间总成本变化，并非人均指标。",
    ),
    733: (
        "当 ",
        " 且驻站上限满足 ",
        " 时，存在降低局部成本的驻站动作，但时长仍须满足上述成本条件。当 ",
        " 且该动作可执行时，候车收益与车内代价相抵。车内有乘客时，若可执行时长超过这一临界值但仍小于 ",
        "，时距虽更均衡，乘客总成本却增加。车内人数增加会降低临界值；车内有人时，增大到达强度与候车权重会提高临界值。当 ",
        " 且车内有人时，额外驻站只增加局部成本。",
    ),
    735: (
        "因此，时距改善须结合乘客构成与后续服务评价。第3.4节考察规则与学习策略的运营效果，第3.4.5至3.4.6节比较两类训练奖励；第4章研究需求变化与供给限制，第5章用规定时域内的人均广义时间成本和完成率评价跨保真控制。第6章进一步考察源仿真标定不足时的预测修正。",
    ),
}


def main():
    assert not BACKUP.exists(), "Keep the existing pre-edit backup."
    with ZipFile(DOCUMENT) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        assert "".join(body[647].itertext()) == "3.2 模型构建"
        assert "".join(body[736].itertext()) == "3.3 求解方法"
        updated = original
        for index, prose in REVISIONS.items():
            assert 647 < index < 736 and body[index].tag == W + "p"
            updated = replace_paragraph(updated, body[index], prose)

        mask = lambda xml: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", xml, flags=re.S)
        assert mask(original) == mask(updated), "Non-text XML changed."
        final_body = ET.fromstring(updated).find(W + "body")
        assert len(body) == len(final_body) == 2924
        changed = [i for i, (a, b) in enumerate(zip(body, final_body)) if ET.tostring(a) != ET.tostring(b)]
        assert changed == list(REVISIONS), changed
        protected_count = 0
        for index in changed:
            if index not in (654, 716):
                before_numbers = re.findall(r"\d+(?:\.\d+)?", "".join(body[index].itertext()))
                after_numbers = re.findall(r"\d+(?:\.\d+)?", "".join(final_body[index].itertext()))
                assert after_numbers == before_numbers + (["6"] if index == 735 else []), index
            old_nodes, _, protected = prose_groups(body[index])
            new_nodes = list(final_body[index].iter(W + "t"))
            assert all(ET.tostring(old_nodes[j]) == ET.tostring(new_nodes[j]) for j in protected), index
            protected_count += len(protected)

        copy2(DOCUMENT, BACKUP)
        temporary = DOCUMENT.with_suffix(".tmp")
        with ZipFile(temporary, "w") as target:
            for item in source.infolist():
                target.writestr(item, updated.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename))
    temporary.replace(DOCUMENT)
    with ZipFile(BACKUP) as before, ZipFile(DOCUMENT) as after:
        assert before.namelist() == after.namelist()
        assert [n for n in before.namelist() if before.read(n) != after.read(n)] == ["word/document.xml"]
    print({"changed_blocks": changed, "protected_text_nodes": protected_count,
           "backup": str(BACKUP), "output": str(DOCUMENT)})


if __name__ == "__main__":
    main()
