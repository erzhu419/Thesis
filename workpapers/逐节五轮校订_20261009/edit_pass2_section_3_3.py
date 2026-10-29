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
BACKUP = Path(__file__).with_name("校订前_第2轮_3.3.docx")
REVISIONS = {
    737: (
        "本节说明共享驻站策略的价值评价与更新。RE-SAC",
        "沿用SAC的演员评论家结构",
        "，以多个评论家估计驻站动作的长期后果，通过价值目标修正和集成分歧调节学习。以下先介绍实际更新，再分析固定惩罚下的价值迭代性质。",
    ),
    738: (
        "价值学习的不同改进针对不同误差来源。双重深度Q网络（Double Deep Q-Network，Double DQN）分开动作选择与评价，以减弱最大化操作造成的价值高估；竞争式深度Q网络（Dueling Deep Q-Network，Dueling DQN）分解状态价值与动作优势，改善价值表示；Rainbow组合多项改进并分析各自贡献。",
        "本章以连续控制SAC为基础，引入集成评价与价值目标偏移。实验除比较控制结果，还分别考察奖励、共享经验和熵温度的影响。",
    ),
    751: (
        "实际演员用参数化分布近似上述密度。温度较高时，动作价值差对选择概率的影响较弱；温度较低时，概率更集中于高价值动作。训练时采样动作会改变驻站时长和乘客时间，最终则采用确定性动作，两种执行方式分别评价。",
    ),
    753: (
        "第2.2.4节已区分偶然不确定性与认知不确定性，两者在平稳和非平稳环境中均可存在。第3、4章递进的是从给定运行规律下的随机学习到运行规律变化后的适应，而非分别处理两类不确定性。",
    ),
    754: (
        "RE-SAC采用两类保守调整。目标网络权重范数用于修正价值目标；目标评论家的估计差异也进入后继价值目标，在线评论家的估计差异则用于约束价值拟合并调节演员的动作评价。以下分别定义这些量及其使用位置。",
    ),
    765: (
        "为第",
        "个目标评论家第",
        "层的权重矩阵，",
        "表示其参数；",
        "表示矩阵各元素绝对值之和。",
        "与",
        "分别调节权重范数偏移和目标集成分歧惩罚的强度。层索引遍历目标网络的全部权重层。",
    ),
    770: (
        "式",
        "中，同一成员的权重范数项在一次更新内不随后继状态和动作变化，集成方差项则随二者变化。它们均通过后继价值目标影响评论家拟合；式(3-29)的分歧项直接调节演员评价。因此，价值整体下降不等于动作排序必然改变。",
    ),
    771: (
        "终止转移不包含后继价值。目标网络和 ",
        " 在本次评论家回归中不参与梯度计算，范数项通过目标数值影响在线评论家的拟合，而非另对在线权重施加梯度惩罚。",
    ),
    772: (
        "目标网络更新后，范数偏移量也随之变化。它取决于网络参数，并非环境转移方差的直接估计。以无偏置的两层整流线性单元（Rectified Linear Unit，ReLU）函数为例，对任意",
        "，有",
    ),
    774: (
        "两组参数表示同一函数，但权重范数之和由",
        "变为",
        "，一般不同。等价缩放不改变网络预测或环境随机性，却会改变范数偏移。因此，范数项的控制作用需在指定网络结构与更新设置下由实验评价。",
    ),
    777: (
        "各评论家拟合自己的目标，同时受在线成员分歧项约束。独立目标保留成员差异，分歧惩罚限制差异程度，二者共同影响集成评价。因此，不能仅凭评论家数量判断控制效果，集成规模和惩罚强度仍需结合实验选择。",
    ),
    781: (
        "最小化上述损失，相当于最大化 ",
        " 与熵项之和。由于 ",
        " 非正，平均价值相同时，分歧越大的动作评价越低。策略仍综合平均价值与分歧判断动作，并非只选择分歧最小的动作。",
    ),
    785: (
        "以下在正则化MDP框架下分析固定惩罚的价值更新。",
        "分析采用给定转移核，并将RE-SAC的保守修正固定为即时惩罚项，以考察该项对价值迭代的影响。",
    ),
    788: (
        "其中 ",
        "，也可采用固定正温度下的对数指数聚合。网络权重范数与集成分歧在此构成固定惩罚 ",
        "。要与式",
        "的采样目标对应，需先对后继惩罚取条件期望，再计入折扣。",
    ),
    789: (
        "采样目标在后继状态动作上计算惩罚。固定环境、策略",
        "、温度与目标网络，对非终止转移定义惩罚的条件期望及固定策略下的软价值：",
    ),
    791: (
        "其中",
        "、",
        "。式",
        "中的期望惩罚需乘以折扣因子，再从即时奖励与折扣后继软价值之和中扣除，才与采样目标的条件期望一致。比较两个输入",
        "时，固定惩罚与熵项抵消，后继价值的概率平均不放大输入差异。终止掩码使后继价值系数处于",
        "，不增大收缩上界。",
    ),
    792: (
        "动作最大值、固定温度的软聚合与固定策略期望虽不同，都不放大价值函数之间的最大差异。",
        "据此分析固定即时项下的算子性质。",
    ),
    796: (
        "证明。记 ",
        "，则逐点有 ",
        "。动作最大值具有单调性；所有动作价值加同一常数时，最大值也增加该常数，故 ",
        "。软聚合具有相同性质：所有动作价值加常数 ",
        "后，指数和可提出因子 ",
        "，取对数后得到相同平移量，因而也满足这一差值界。",
    ),
    800: (
        "有限维价值空间在无穷范数下完备，故该收缩算子有唯一固定点 ",
        "。从任意初始价值出发，精确迭代满足 ",
        "。固定惩罚不改变收缩因子，但固定点表示的是修正后的评价目标，而非原问题的最优价值。",
    ),
    804: (
        "当 ",
        " 时，逼近误差的累计上界趋于 ",
        "，理想算子的收缩性并不消除持续拟合误差。固定惩罚下的单调性与收缩性见",
        "，熵正则化形式见",
        "和",
        "，精确迭代的固定点结论见",
        "。",
        "给出输入依赖惩罚的反例。实际训练持续改变网络、策略和采样分布，故上述固定算子结论不能直接证明联合训练收敛；训练稳定性与控制收益由实验检验。",
    ),
    807: (
        "惩罚系数为正时，若长驻站的分歧更大，其平均价值优势需抵消额外惩罚，才能在有效评价中占优；若分歧更小，该项反而有利于长驻站。两动作分歧相同或系数为零时，排序不变。因此，应考察分歧在候选动作之间的差别，而非只看集成标准差的整体大小。",
    ),
    808: (
        "保守评价并不预设长驻站一定较差。即使长驻站经验较少，是否延长驻站仍取决于当前状态及预期服务后果。本章早期环境不在附加驻站期间交换乘客；第5、6章允许继续上客，长驻站还可能增加搭乘机会。各组需按自身服务机制评价动作，不能将统一缩短驻站视为最优控制。",
    ),
    809: (
        "训练时从共享经验中采样转移，利用后继策略动作和各成员的目标网络构造价值目标，更新在线评论家，再更新演员、熵温度与目标网络。以下实验将分别考察这些更新与驻站服务的关系。",
    ),
    810: (
        "展示上述更新流程：类别信息与连续运行量共同进入共享演员和集成评论家，各评论家通过自己的目标网络学习动作价值，演员据集成评价选择驻站时长。",
    ),
}


def main():
    assert not BACKUP.exists(), "Keep the existing pre-edit backup."
    with ZipFile(DOCUMENT) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        assert "".join(body[736].itertext()) == "3.3 求解方法"
        assert "".join(body[811].itertext()) == "3.4 仿真实验与结果分析"
        updated = original
        for index, prose in REVISIONS.items():
            assert 736 < index < 811 and body[index].tag == W + "p"
            updated = replace_paragraph(updated, body[index], prose)

        mask = lambda xml: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", xml, flags=re.S)
        assert mask(original) == mask(updated), "Non-text XML changed."
        final_body = ET.fromstring(updated).find(W + "body")
        assert len(body) == len(final_body) == 2924
        changed = [i for i, (a, b) in enumerate(zip(body, final_body)) if ET.tostring(a) != ET.tostring(b)]
        assert changed == list(REVISIONS), changed
        protected_count = 0
        for index in changed:
            if index not in (753, 808, 809):
                before_numbers = re.findall(r"\d+(?:\.\d+)?", "".join(body[index].itertext()))
                after_numbers = re.findall(r"\d+(?:\.\d+)?", "".join(final_body[index].itertext()))
                assert after_numbers == before_numbers, index
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
