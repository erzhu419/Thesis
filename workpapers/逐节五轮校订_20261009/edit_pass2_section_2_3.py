from pathlib import Path
from shutil import copy2
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_1_2_1 import W
from edit_section_4_4 import replace_paragraph


ROOT = Path(__file__).resolve().parents[2]
DOCUMENT = ROOT / "output/排版完成_逐节校订_第2轮_20261010.docx"
BACKUP = Path(__file__).with_name("校订前_第2轮_2.3.docx")
REVISIONS = {
    494: (
        "获得固定目标离线数据后，可比较源目标经验并辅助策略训练。本节介绍样本加权与运行预测两种方法，分别用于第5章的混合训练和第6章标定不足时的世界模型修正。",
    ),
    496: (
        "第3、4章未预先获得目标域转移训练数据，分别研究平稳随机运行和非平稳适应。第5章引入固定目标记录。重要性采样（importance sampling）用密度比调整样本权重，以采样分布的数据估计目标分布下的期望。设目标密度为",
        "，采样密度为", "，且在", "为正的位置上，", "也为正，则",
    ),
    498: (
        "式中，x为采样对象，f(x)为待估计期望的可积函数，w(x)为重要性权重。目标密度相对采样密度越大，权重越大。加权不补足缺失的目标转移；大权重可能使少量样本主导估计，实际方法常对权重截断或归一化，并需考虑由此引入的估计偏差。因此，需结合数据覆盖和更新位置解释加权作用。",
    ),
    501: (
        "左侧是转移样本的联合密度，右侧的",
        "为状态动作分布，",
        "为给定状态动作下后继状态与事件间隔的条件转移核。采集策略影响哪些状态动作更常出现，运行机制影响同一动作的后果，并进一步改变后续状态分布。因此，单凭分类器能区分源目标样本，不能判断差异来自采集行为还是运行机制。",
    ),
    502: (
        "满足上述覆盖条件时，给定状态动作下估计目标条件期望的源样本权重为",
        "。联合转移分类器输入状态、动作、后继状态与事件间隔，状态动作分类器只输入状态与动作。二者均用等量源样本与目标样本训练，分别以",
        "和", "表示样本来自目标环境的概率。估计准确时，有",
    ),
    504: (
        "其中，logit将分类概率v转换为对数几率。两个分类器采用相同源目标样本比例时，先验项相减后抵消，比例不一致才需修正。比率精度还受概率估计误差影响。DARC用该比率修正训练奖励，H2O与H2O+进一步在源目标混合训练中处理动力学差异。",
        "联合分类器不输入事件间隔时，仅估计后继状态的条件转移比。第5章说明比率修正是否启用、权重用于哪项损失，再以分类诊断与控制结果评价其作用。",
    ),
    509: (
        "其中，",
        "为模型参数。联合预测分布描述后继状态与事件间隔，奖励预测给出该转移对应的奖励。概率模型拟合条件分布，确定性模型可预测状态增量或条件均值；奖励可单独预测，也可在获得乘客人数轨迹后由时间积分计算。事件间隔影响成本累计和时域终止，仅预测后继状态不足以描述完整的半马尔可夫过程。第6章按公交事件的数据类型构建预测，并用于价值更新。",
    ),
    510: (
        "模型拟合还需考虑事件之间的时间关联。相邻公交事件共享车辆、乘客及外部运行条件，随机打散后划分拟合与验证记录，会使同一运行日的信息进入两组。本文公交模型按完整运行日或需求场景组划分记录，用验证记录比较预测误差，再在新SUMO运行日评价训练后的策略。模型类别、数据量和计算成本的比较，进一步说明预测修正如何影响控制。",
    ),
    513: (
        "式中，j、l分别为求和与乘积索引，预测长度以事件数计，空乘积取1。带帽的状态、奖励与延续因子来自模型预测，终止后的延续因子取零。增加预测步数可考虑更远的后果，也会累积预测误差。第6章只生成一步模型转移，构造SAC价值回归目标，后继价值按式",
        "计入策略熵，模型样本权重调节其对评论家更新的影响。模型集成分歧或验证预测误差用于判断预测可靠程度。",
    ),
    514: (
        "样本加权改变已有经验对学习的影响，世界模型则预测动作后果并提供价值更新样本。第5章先用固定目标离线数据辅助源仿真训练；第6章在相同数据条件下，针对标定不足比较残差修正与直接目标建模，再按预测可信程度使用模型样本。分类与预测诊断解释学习过程，两章最终均以目标环境中的公交服务结果评价方法。",
    ),
}


def main():
    with ZipFile(DOCUMENT) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        assert "".join(body[493].itertext()) == "2.3 迁移学习基础"
        assert "".join(body[515].itertext()) == "2.4 仿真平台与实验设置"
        updated = original
        for index, prose in REVISIONS.items():
            assert 493 < index < 515
            updated = replace_paragraph(updated, body[index], prose)

        # Compare protected structure separately from the edited prose.
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
