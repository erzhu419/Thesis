from pathlib import Path
from shutil import copy2
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_3_2 import DOCUMENT, M, W, W14, prose_groups, revise


BACKUP = Path(__file__).with_name("校订前_4.3.docx")
REVISIONS = {
    1021: (
        "本节承接第3章的保守价值评价，依次说明非平稳环境下的训练反馈、稳定化设计与策略输出，并分析变化后的动作误差。执行器补偿用于检验模式识别的作用；公交仍使用既定驻站动作。",
    ),
    1022: (
        "执行器模式可能改变动作的符号或执行含义。BAPR后续研究据模式变换指令，目标是恢复原执行效果。",
        "中，",
        "给出有界命令能够精确补偿的条件，",
        "证明极性变换保持动作边界。附录B.3、B.4分别分析共同噪声下的逐路径等价和激励不足时的不可辨识，B.5至B.8连接误判、动作误差与回报损失。因此，控制代价还取决于动作幅度和局部动作价值响应，不能只看识别准确率。第4.4.4节检验HalfCheetah执行器补偿，公交结果见第4.4.2至4.4.5节。",
    ),
    1023: (
        "MDP同态网络利用状态动作空间的群对称性构造等变策略和价值表示。",
        "本文",
        "则研究给定执行器变换下的补偿条件：模式识别选择变换，动作映射形成指令，二者共同影响执行效果。",
    ),
    1024: (
        "CS-BAPR承接BAPR的适应学习，进一步研究变化后的状态覆盖与驻站响应。近期反馈调节训练保守程度，算术输出改变状态到动作的函数表示；执行器补偿改变指令到实际执行的映射，三者作用不同。",
    ),
    1025: (
        "公交训练中，共享经验用于更新价值网络，近期样本用于形成反馈并调节策略更新。公交评价时网络参数固定，由当前观测直接输出驻站动作；受控执行器实验另行启用在线模式估计和动作补偿。附录D.6分析固定符号惩罚下的模式贝尔曼算子，D.7进一步说明输入依赖反馈对固定算子条件的影响。",
    ),
    1026: (
        "CS-BAPR",
        "的下述公交配置采用共享策略和集成价值网络，结合近期反馈、训练稳定化与不同输出结构，符号发现和IRM另作扩展比较。奖励尺度、价值估计和反馈调节相互影响，以下六项设计用于稳定这一学习过程。",
    ),
    1029: (
        "目标聚合决定后继价值如何进入贝尔曼目标；策略更新中的集成均值和标准差则用于评价候选动作。二者分别影响价值回归与动作选择。",
    ),
    1030: (
        "式(4-5)列出最小值聚合与熵项。公交主配置还对每个目标评论家保留系数0.01的正参数范数偏移，该范数为全部权重和非输出层偏置的L1范数之和，连同聚合值乘以折扣因子及非终止标记。评论家回归损失另加入系数0.1的当前批动作价值集成标准差均值。因此，实际价值更新同时包含上述聚合、目标偏移与集成分歧正则。",
    ),
    1031: (
        "第二，将奖励除以运行奖励标准差，方差统计采用衰减系数0.99的指数移动平均，分母下限为",
        "。这一缩放只调整奖励幅度，不减去运行均值，因而保留奖励零点。",
    ),
    1032: (
        "第三，将熵温度下限设为0.01，限制训练后期熵项的过度减弱。",
    ),
    1035: (
        "其中，",
        "是经验动作，",
        "是当前策略重参数化采样的动作；每个状态采样一次，再对小批量平均。策略总损失加入",
        "，由该系数统一加权。原配置的算术输出（表中记为NAU/NMU）和科尔莫戈罗夫-阿诺尔德网络（Kolmogorov-Arnold Network，KAN）关闭BC，普通多层感知机（Multilayer Perceptron，MLP）保留正权重。第4.4.1至4.4.3节沿用这一设置；第4.4.5节的100回合补充比较则对两种稳定化输出均关闭BC，并采用相同权重衰减。",
    ),
    1036: (
        "第五，设置反馈预热：前100次训练更新不启用历史窗口调节，并将原惩罚放大系数由5降至2，以减弱初始价值估计尚不稳定时的反馈。",
    ),
    1037: (
        "第六，使用最近256条当前策略采集的样本，计算奖励偏差和在线评论家的集成价值标准差，并只将标准差上升作为分歧增强信号。记相邻诊断窗口的标准差统计值为",
        "和",
        "，采用下式计算其单向相对增量。统计序号随诊断更新递增，不表示物理时间或运行回合：",
    ),
    1039: (
        "式(4-7)不将价值分歧下降计为新的不稳定信号。以上六项共同构成本章的稳定化训练配置。",
    ),
    1040: (
        "近期偏差量取三个统计量的最大值：近期奖励均值相对运行均值的标准化偏差、式(4-7)的集成价值标准差增量，以及在线与目标评论家权重范数差的一半；再以系数0.1作指数平滑。原算法将其称为惊异度，它是训练偏差统计量，而非观测的负对数似然。随后为不同历史长度维护权重，用于调节演员更新；长度按诊断更新次数计量，不是已确认的交通模式持续时间。",
    ),
    1041: (
        "长度索引",
        "取0至19，初始权重均匀。对平滑后的偏差量",
        "，构造证据权重",
        "，与原长度权重相乘并归一化；再按预设0.02的重置比例向长度0分配权重，其余权重向下一长度移位，截断后重新归一化。这是基于规定证据函数的训练反馈，与式(4-2)由环境模式似然计算的后验概率不同。",
    ),
    1042: (
        "记归一化长度权重为",
        "，其加权量为",
        "。前100次训练更新将该量置零，之后以",
        "确定演员目标中的价值标准差系数。该系数为负，加权量增大时，对价值分歧的惩罚更强；它与集成均值和熵项共同影响策略更新，只在训练期间调节。",
    ),
    1044: (
        "驻站策略由状态特征网络和动作输出组成。",
        "为特征网络，",
        "与",
        "分别为NAU线性输出和NMU启发的二次修正，",
        "为混合权重。下文以",
        "表示输出映射、",
        "表示Fréchet导数，",
        "表示复合映射的雅可比；前后车头时距仍记为",
        "与",
        "。",
    ),
    1045: (
        "不同函数结构具有不同的外推响应。神经算术逻辑单元（Neural Arithmetic Logic Unit，NALU）利用算术运算和门控学习数值关系；NAU与NMU进一步约束加法和乘法结构，以研究超出训练数值范围后的预测。",
        "方程学习网络通过组合基本运算形成较简洁的关系，并将其用于外推和控制。",
        "",
    ),
    1046: (
        "KAN将可学习一元函数放在网络连接上，与采用固定节点激活的普通MLP形成不同参数化。",
        "本章将其作为另一种输出表示，与算术输出及普通MLP比较。",
    ),
    1047: (
        "本章比较ReLU、tanh、算术和KAN四类输出。算术配置以特征网络",
        "提取状态表示，再组合NAU线性输出与NMU启发的二次修正：",
    ),
    1049: (
        "归一化输出映射至0至60秒附加驻站。NAU提供线性组合；二次修正先将特征线性投影至动作维度，再逐分量平方并乘以可训练系数。该支路借鉴NMU的数值运算思想，但不是标准NMU对选定输入坐标求积的结构。",
        "表中沿用NAU/NMU名称对应原实验配置，外层tanh限定输出幅度。",
    ),
    1050: (
        "动作有界与响应平缓是不同性质：指令即使始终处于允许范围，也可能随状态变化而剧烈改变。策略雅可比表示动作对状态各分量的局部变化率，因此应分析特征变换和输出头组成的完整映射。",
    ),
    1051: (
        "设",
        "为输出头、",
        "为特征变换，二者在所讨论区域内可微。下式前两行分别限定两类导数随输入变化的幅度，第三行给出导数范数上界，末行限定特征映射本身的变化幅度；导数范数均采用相容的算子范数：",
    ),
    1056: (
        "完整策略的响应需同时考虑这两部分。",
        "中的",
        "给出标量复合导数界，附录D.5将其扩展至向量映射；",
        "与",
        "则分析导数跳跃为何不满足这一全局条件。",
    ),
    1057: (
        "为说明两类常数的来源，记",
        "，加减同一项得到",
    ),
    1059: (
        "第一项反映特征变化引起的输出头导数变化，第二项反映特征网络自身的导数变化。按式(4-9)分别估计，有",
    ),
    1061: (
        "两项相加即得式(4-10)。例如，ReLU特征跨越激活边界时可能发生导数跳跃，平滑输出头不一定消除这一变化。因此，外推分析须给出状态区域内完整映射的导数条件，不能仅由最后一层的结构判断。",
    ),
    1063: (
        "下面将完整策略的动作响应与参考控制关系比较，分析输入离开训练覆盖后，动作误差上界怎样随状态偏移变化。",
    ),
    1064: (
        "为待评价输入，",
        "为基准输入，",
        "为参考控制映射；",
        "与",
        "分别为基准点的函数值误差和导数误差上界，",
        "为策略与参考映射的导数Lipschitz常数，",
        "为输入偏移。下文分析完整确定性动作映射的误差，而非随机策略条件动作分布的距离。",
    ),
    1065: (
        "依据",
        "，可将动作误差上界分解为基准点误差、导数差异和输入偏移的作用。",
    ),
    1066: (
        "设",
        "为完整确定性动作映射，",
        "为参考映射，",
        "为训练邻域内的基准点。若该点的函数值误差不超过",
        "、雅可比误差不超过",
        "，且二者在连接基准点与待评价点的线段上可微，雅可比分别具有Lipschitz常数",
        "与",
        "，则",
    ),
    1068: (
        "式",
        "为",
        "在本节的应用，三个上界项分别与偏移无关、随距离线性变化及随距离平方变化。",
    ),
    1069: (
        "第一项保留基准点的拟合误差，第二项反映策略与参考关系的一阶响应差异，第三项反映两类雅可比沿路径的变化。满足上述条件时，控制完整映射的导数变化幅度可以限制误差上界的增长。",
    ),
    1070: (
        "为推导距离项，令",
        "，沿线段",
        "积分，得到",
    ),
    1072: (
        "前两项的范数分别不超过",
        "与",
        "；后两项的范数之和不超过",
        "，积分为二分之一，得到二次距离项。",
    ),
    1073: (
        "需求增大同时改变候车队列、服务时间、车内人数与后续间隔，需求倍率不是状态偏移距离。式(4-13)分析给定条件下的动作误差，需求实验则评价状态、动作与乘客后果共同变化后的闭环表现。",
    ),
    1074: (
        "为获得局部动作关系的经验参照，研究用5个同分布回合的47,247个决策样本拟合线性参考映射，样本等比例分为拟合和留出两部分。留出决定系数为0.70，所检查样本的动作均未触及0秒或60秒边界。这说明线性关系能够解释采样区域内的一部分动作变化，导数误差和跨环境共享关系仍需另行判断。附录D.8将策略与参考关系的两层误差连接，D.9区分共享关系的估计误差与环境特有部分，D.10据基准误差和导数条件给出传播界，从而明确符号参考用于外推分析所需的条件。",
    ),
    1075: (
        "概括本节的方法：近期运行反馈调节训练保守程度，奖励缩放与目标聚合稳定价值更新，算术输出改变驻站策略的函数表示。后文分别以需求变化、执行器变化和组件对照检验这些设计。",
    ),
}


def fix_citation_brackets(raw):
    pattern = r'(<w:hyperlink\b[^>]*w:anchor="ref-zhang2026csbapr"[^>]*>.*?<w:t>)\[158\](</w:t>.*?</w:hyperlink>)'
    result, count = re.subn(pattern, r"\g<1>158\2", raw, flags=re.S)
    assert count == 1 or "[158]" not in raw
    return result


def main():
    with ZipFile(DOCUMENT) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        assert "".join(body[1019].itertext()) == "4.3 求解方法"
        assert "".join(body[1076].itertext()) == "4.4 仿真实验与结果分析"
        updated = original
        for index, prose in REVISIONS.items():
            paragraph_id = body[index].get(W14 + "paraId")
            pattern = r'<w:p\b[^>]*w14:paraId="' + paragraph_id + r'"[^>]*>.*?</w:p>'
            matches = list(re.finditer(pattern, updated, flags=re.S))
            assert len(matches) == 1
            match = matches[0]
            revised = revise(match.group(0), body[index], prose)
            if index == 1049:
                revised = fix_citation_brackets(revised)
            updated = updated[:match.start()] + revised + updated[match.end():]

        mask = lambda raw: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", raw, flags=re.S)
        assert mask(original) == mask(updated), "Non-text XML changed."
        new_body = ET.fromstring(updated).find(W + "body")
        assert len(body) == len(new_body)
        changed = [i for i, (a, b) in enumerate(zip(body, new_body)) if ET.tostring(a) != ET.tostring(b)]
        assert set(changed).issubset(REVISIONS), changed
        protected_count = 0
        for index in changed:
            old_nodes, _, protected = prose_groups(body[index])
            new_nodes, _, _ = prose_groups(new_body[index])
            for position in protected:
                expected = old_nodes[position].text
                if index == 1049 and expected == "[158]":
                    expected = "158"
                assert new_nodes[position].text == expected, (index, position)
            protected_count += len(protected)
            for tag in (M + "oMath", W + "pPr", W + "hyperlink"):
                old = [ET.tostring(n, encoding="unicode") for n in body[index].iter(tag)]
                new = [ET.tostring(n, encoding="unicode") for n in new_body[index].iter(tag)]
                if index == 1049 and tag == W + "hyperlink":
                    old = [n.replace(">[158]<", ">158<") for n in old]
                assert old == new, (index, tag)
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
        changed_parts = [n for n in before.namelist() if before.read(n) != after.read(n)]
        assert changed_parts == ["word/document.xml"], changed_parts
        prior_body = ET.fromstring(before.read("word/document.xml")).find(W + "body")
        final_changed = [i for i, (a, b) in enumerate(zip(prior_body, new_body)) if ET.tostring(a) != ET.tostring(b)]
        assert final_changed == list(REVISIONS), final_changed
    print({"changed_blocks": changed, "protected_text_nodes": protected_count,
           "changed_parts": changed_parts, "backup": str(BACKUP)})
    for index in changed:
        print(index, "".join(n.text or "" for n in new_body[index].iter() if n.tag in (W + "t", M + "t")))


if __name__ == "__main__":
    main()
