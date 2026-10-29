from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_pass2_section_4_3 import BACKUP, DOCUMENT, REVISIONS
from edit_section_3_2 import W, prose_groups
from edit_section_4_4 import replace_paragraph


SHORTENED = {
    1030: (
        "式(4-5)列出最小值聚合与熵项。公交主配置还保留系数0.01的正向参数范数修正。参数范数为每个目标评论家的全部权重及非输出层偏置的L1范数之和，乘上述系数后加到相应目标价值，连同聚合值乘以折扣因子及非终止标记。评论家回归损失另加入系数0.1的当前小批量动作价值集成标准差均值。",
    ),
    1032: (
        "第三，将熵温度下限设为0.01，避免熵项系数在训练后期继续趋近于零。",
    ),
    1040: (
        "近期偏差量取以下三项的最大值：近期奖励均值相对运行均值的标准化偏差、式(4-7)的集成价值标准差增量，以及在线与目标评论家权重范数差的一半；再以系数0.1作指数平滑。原算法称其为惊异度，实为训练偏差统计，而非观测的负对数似然。不同历史长度的权重据此更新，再用于调节演员；长度按诊断次数计量，不代表交通模式的持续时间。",
    ),
    1042: (
        "记归一化长度权重为",
        "，其加权量为",
        "。前100次训练更新将该量置零，之后以",
        "确定演员目标中的价值标准差系数。该系数为负，加权量越大，价值分歧惩罚越强；它与集成均值和熵项共同参与演员更新。",
    ),
    1049: (
        "归一化输出映射至0至60秒附加驻站。NAU提供线性组合；二次支路把特征线性投影至动作维度，逐分量平方后乘以可训练系数，并非标准NMU对选定输入坐标求积。",
        "表中沿用NAU/NMU名称，外层tanh限定幅度。后续软符号配置关闭NAU，以可学习基准动作加神经修正形成完整动作；减去同一网络在参考状态的输出，使该状态的修正为零。SINDy关系对动作及其导数施加软约束，不替代策略决策。",
    ),
    1066: (
        "设完整确定性动作映射",
        "与参考映射",
        "在训练邻域内的基准点",
        "的函数值误差不超过",
        "、雅可比误差不超过",
        "。二者沿基准点到待评价点的线段可微，雅可比分别具有Lipschitz常数",
        "与",
        "时，有",
    ),
}


with ZipFile(DOCUMENT) as source:
    original = source.read("word/document.xml").decode("utf-8")
    body = ET.fromstring(original).find(W + "body")
    updated = original
    for index, prose in SHORTENED.items():
        updated = replace_paragraph(updated, body[index], prose)
    mask = lambda xml: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", xml, flags=re.S)
    assert mask(original) == mask(updated)
    with ZipFile(BACKUP) as before:
        prior = before.read("word/document.xml").decode("utf-8")
        initial = ET.fromstring(prior).find(W + "body")
        final = ET.fromstring(updated).find(W + "body")
        assert mask(prior) == mask(updated)
        assert len(initial) == len(final) == 2924
        changed = [i for i, (a, b) in enumerate(zip(initial, final)) if ET.tostring(a) != ET.tostring(b)]
        assert changed == list(REVISIONS)
        for index in changed:
            old_nodes, _, protected = prose_groups(initial[index])
            new_nodes, _, _ = prose_groups(final[index])
            assert all(old_nodes[i].text == new_nodes[i].text for i in protected)
            assert re.findall(r"\d+(?:\.\d+)?", "".join(initial[index].itertext())) == re.findall(r"\d+(?:\.\d+)?", "".join(final[index].itertext())), index
    temporary = DOCUMENT.with_suffix(".tmp")
    with ZipFile(temporary, "w") as target:
        for item in source.infolist():
            target.writestr(item, updated.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename))
temporary.replace(DOCUMENT)
with ZipFile(BACKUP) as before, ZipFile(DOCUMENT) as after:
    assert before.namelist() == after.namelist()
    assert [n for n in before.namelist() if before.read(n) != after.read(n)] == ["word/document.xml"]
print({"shortened_blocks": list(SHORTENED), "final_changed_blocks": changed})
