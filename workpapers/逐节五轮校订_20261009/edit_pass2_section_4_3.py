from pathlib import Path
from shutil import copy2
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_3_2 import M, W, prose_groups
from edit_section_4_4 import replace_paragraph


ROOT = Path(__file__).resolve().parents[2]
DOCUMENT = ROOT / "output/排版完成_逐节校订_第2轮_20261010.docx"
BACKUP = Path(__file__).with_name("校订前_第2轮_4.3.docx")
REVISIONS = {
    1021: (
        "承接第3章的保守价值评价，本节说明非平稳环境下如何利用近期反馈调节训练，再介绍驻站策略的输出结构，分析状态偏移对动作误差的影响。执行器补偿另用于检验模式识别的作用，不改变公交驻站动作的定义。",
    ),
    1022: (
        "执行器模式可能改变指令的符号或实际作用。BAPR后续研究先估计模式，再按执行关系变换动作指令，以恢复基准策略的执行效果。",
        "中，",
        "给出有界命令能够精确补偿的条件，",
        "证明极性变换保持动作边界。附录B.3、B.4分别分析共同噪声下的逐路径等价和激励不足时的不可辨识，B.5至B.8连接误判、动作误差与回报损失。同一次模式误判造成的损失，还取决于指令幅度及动作价值对执行误差的敏感程度，不能只由识别准确率判断。第4.4.4节检验HalfCheetah执行器补偿，公交结果见第4.4.2至4.4.5节。",
    ),
    1024: (
        "CS-BAPR进一步研究运行条件改变后的驻站响应。近期反馈调节训练保守程度，算术输出与符号参考约束影响状态到动作的关系；执行器补偿则改变指令到实际执行的映射。以下分别说明这些作用。",
    ),
    1025: (
        "公交共享经验用于更新价值网络，近期样本用于调节策略更新；评价时策略参数固定。受控执行器实验则在执行期间更新模式判断并补偿动作。附录D.6分析固定符号惩罚下的模式贝尔曼算子，D.7说明反馈随输入变化时，哪些固定算子条件不再直接适用。",
    ),
    1026: (
        "CS-BAPR",
        "的早期公交配置采用共享策略和集成价值网络，比较近期反馈、训练稳定化与不同输出结构的组合作用，符号发现和IRM另作扩展。奖励缩放改变奖励与熵项的相对尺度，价值估计又影响反馈及动作评价，以下六项设计共同调节这一学习过程。",
    ),
    1029: (
        "目标聚合决定后继价值怎样进入评论家的回归目标；演员更新中的集成均值和标准差则评价候选动作。前者改变学习所依据的价值目标，后者影响策略偏向哪些动作。",
    ),
    1030: (
        "式(4-5)列出最小值聚合与熵项。公交主配置还保留系数0.01的正向参数范数修正。参数范数为每个目标评论家的全部权重及非输出层偏置的L1范数之和，乘上述系数后加到相应目标价值，连同聚合值乘以折扣因子及非终止标记。评论家回归损失另加入系数0.1的当前小批量动作价值集成标准差均值。因此，最小值聚合、参数范数修正和集成分歧正则共同参与价值更新。",
    ),
    1031: (
        "第二，将奖励除以运行奖励标准差，方差统计采用衰减系数0.99的指数移动平均，分母下限为",
        "。该处理不减去运行均值，保留奖励零点；它调整奖励幅度，也改变奖励项相对熵项的大小。",
    ),
    1032: (
        "第三，将熵温度下限设为0.01，避免熵项系数在训练后期继续趋近于零。奖励缩放与温度下限共同影响价值目标和熵项的相对作用。",
    ),
    1036: (
        "第五，前100次训练更新不启用历史窗口调节，以减少初始价值估计误差对反馈的干扰；同时将历史反馈对价值分歧惩罚的放大系数由5降至2，减弱其调节幅度。",
    ),
    1037: (
        "第六，从当前策略最近采集的256条样本计算奖励偏差及在线评论家的集成价值标准差，只把标准差上升作为分歧增强信号。记相邻两次诊断的标准差统计值为",
        "和",
        "，其单向相对增量为下式。统计序号按诊断更新次数递增，不代表物理时间或运行回合：",
    ),
    1040: (
        "随后取三个统计量的最大值形成近期偏差量：近期奖励均值相对运行均值的标准化偏差、式(4-7)的集成价值标准差增量，以及在线与目标评论家权重范数差的一半；再以系数0.1作指数平滑。原算法称其为惊异度，实际含义是训练统计量的偏差，而非观测的负对数似然。该量用于更新不同历史长度的权重，进而调节演员更新；长度按诊断更新次数计量，不代表已识别的交通模式持续时间。",
    ),
    1041: (
        "长度索引",
        "取0至19，初始权重均匀。对平滑后的偏差量",
        "，构造证据权重",
        "，与原长度权重相乘并归一化；再按预设0.02的重置比例向长度0分配权重，其余权重向下一长度移位，截断后重新归一化。这一递推使用预设证据函数形成训练反馈，不同于式(4-2)根据环境模式似然计算后验概率。",
    ),
    1042: (
        "记归一化长度权重为",
        "，其加权量为",
        "。前100次训练更新将该量置零，之后以",
        "确定演员目标中的价值标准差系数。该系数为负，加权量增大时，价值分歧惩罚更强。它与集成价值均值、熵项共同决定策略更新，并不直接指定某个交通模式下的驻站动作。",
    ),
    1044: (
        "驻站策略由状态特征网络和动作输出组成。",
        "为特征网络，",
        "与",
        "分别为NAU线性输出和NMU启发的二次修正，",
        "为混合权重。下文以",
        "表示输出映射、",
        "表示弗雷歇（Fréchet）导数，",
        "表示复合映射的雅可比；前后车头时距仍记为",
        "与",
        "。",
    ),
    1049: (
        "归一化输出映射至0至60秒附加驻站。NAU提供线性组合；二次支路先将特征线性投影至动作维度，再逐分量平方并乘以可训练系数。该结构借鉴NMU的数值运算思想，并非标准NMU对选定输入坐标求积。",
        "早期表格沿用NAU/NMU名称，外层tanh限定输出幅度。后续软符号配置关闭NAU，以可学习基准动作加神经修正形成完整动作；减去同一网络在参考状态的输出，使修正在该状态为零。SINDy参考关系只对动作及其导数施加软约束，不直接替代策略决策。两种配置的作用由后文分别检验。",
    ),
    1050: (
        "动作有界不等于响应平缓：指令可以始终处于允许范围，却随状态的小幅变化而剧烈改变。策略雅可比描述动作对状态各分量的局部变化率，因此需要分析特征变换与输出头组成的完整映射。",
    ),
    1063: (
        "下面比较完整策略与参考控制关系，分析输入离开训练覆盖区域后，动作误差上界如何随状态偏移增长。",
    ),
    1064: (
        "为待评价输入，",
        "为基准输入，",
        "为参考控制映射；",
        "与",
        "分别为基准点的函数值误差和导数误差上界，",
        "为策略与参考映射的导数Lipschitz常数，",
        "为输入偏移。这里比较的是确定性动作映射的输出，不是随机策略的动作分布。",
    ),
    1065: (
        "依据",
        "，动作误差可由基准点误差与沿输入偏移路径积累的导数差异共同估计。",
    ),
    1066: (
        "设完整确定性动作映射",
        "与参考映射",
        "在基准点",
        "的函数值误差不超过",
        "、雅可比误差不超过",
        "。二者沿基准点到待评价点的线段可微，雅可比分别具有Lipschitz常数",
        "与",
        "时，有",
    ),
    1069: (
        "第一项是基准点的拟合误差，第二项来自该点策略与参考关系的导数差异，第三项来自两类雅可比沿路径的变化。上述条件成立时，限制完整映射的导数变化幅度，可以减缓这一误差上界随距离的增长。",
    ),
    1074: (
        "为获得局部动作关系的经验参照，本文用5个同分布回合的47,247个决策样本拟合线性参考映射，样本等比例分为拟合与留出两部分。留出决定系数为0.70，所检查样本的动作均未触及0秒或60秒边界。该结果说明线性关系解释了采样区域内的部分动作变化，并未估计导数误差或证实跨环境共享关系。附录D.8连接策略拟合与参考估计的两层误差，D.9区分共享关系的估计误差与环境特有部分，D.10据基准误差和导数条件给出传播界，为符号参考的外推分析明确适用条件。",
    ),
    1075: (
        "概括早期配置：近期反馈调节训练保守程度，奖励缩放与目标聚合参与价值更新，算术输出改变驻站策略的函数表示。后续配置再以参考状态约束和SINDy关系调节动作响应。下面通过需求变化、执行器变化及组件比较，分别检验训练反馈、动作补偿与输出结构的作用。",
    ),
}


def main():
    assert not BACKUP.exists(), "Keep the existing pre-edit backup."
    with ZipFile(DOCUMENT) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        assert "".join(body[1019].itertext()) == "4.3 求解方法"
        assert "".join(body[1076].itertext()) == "4.4 仿真实验与结果分析"
        updated = original
        for index, prose in REVISIONS.items():
            updated = replace_paragraph(updated, body[index], prose)

        # Preserve everything except the selected prose text nodes.
        mask = lambda xml: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", xml, flags=re.S)
        assert mask(original) == mask(updated), "Non-text XML changed."
        final = ET.fromstring(updated).find(W + "body")
        assert len(body) == len(final) == 2924
        changed = [i for i, (a, b) in enumerate(zip(body, final)) if ET.tostring(a) != ET.tostring(b)]
        assert changed == list(REVISIONS), changed
        protected_count = 0
        for index in changed:
            assert 1019 < index < 1076 and index not in (1020, 1043, 1062)
            old_nodes, _, protected = prose_groups(body[index])
            new_nodes, _, _ = prose_groups(final[index])
            assert all(old_nodes[i].text == new_nodes[i].text for i in protected)
            protected_count += len(protected)
            for tag in (M + "oMath", W + "pPr", W + "hyperlink", W + "instrText", W + "fldChar"):
                assert [ET.tostring(n) for n in body[index].iter(tag)] == [ET.tostring(n) for n in final[index].iter(tag)]
            assert re.findall(r"\d+(?:\.\d+)?", "".join(body[index].itertext())) == re.findall(r"\d+(?:\.\d+)?", "".join(final[index].itertext())), index
        copy2(DOCUMENT, BACKUP)
        temporary = DOCUMENT.with_suffix(".tmp")
        with ZipFile(temporary, "w") as target:
            for item in source.infolist():
                target.writestr(item, updated.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename))
    temporary.replace(DOCUMENT)
    with ZipFile(BACKUP) as before, ZipFile(DOCUMENT) as after:
        assert before.namelist() == after.namelist()
        assert [n for n in before.namelist() if before.read(n) != after.read(n)] == ["word/document.xml"]
    print({"changed_blocks": changed, "paragraphs": len(changed), "protected_text_nodes": protected_count,
           "backup": str(BACKUP), "output": str(DOCUMENT)})


if __name__ == "__main__":
    main()
