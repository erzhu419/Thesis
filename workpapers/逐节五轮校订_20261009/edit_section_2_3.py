from pathlib import Path
from shutil import copy2
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_1_2_1 import DOCUMENT, W, M, W14, revise_paragraph
from edit_section_1_2_4 import replace_text_nodes


BACKUP = Path(__file__).with_name("校订前_2.3.docx")

REVISIONS = {
    494: (
        "获得固定的目标离线数据后，可利用这些记录比较源仿真经验与目标运行条件。本节介绍两种基础方法：调整源样本参与学习的权重，以及学习或修正动作执行后的运行预测。第5章据此结合目标离线数据与源仿真交互训练公交策略，第6章再研究标定不足时的世界模型修正。",
    ),
    496: (
        "第3、4章在训练前没有目标域转移数据，分别研究给定随机规律下的鲁棒学习和运行规律变化后的适应。第5章获得目标离线数据后，可进一步比较源目标经验。重要性采样（importance sampling）利用密度比对样本加权，以采样分布的数据估计目标分布下的期望。设目标密度为",
        "，采样密度为",
        "，且在",
        "为正的位置上，",
        "也为正，则",
    ),
    498: (
        "式中，x表示所采样的对象，f(x)为待计算期望的可积函数，w(x)为重要性权重。目标分布中相对更常出现的样本获得较大权重，采样分布中过多出现的样本获得较小权重。加权改变已有样本对估计的影响，不会补出未采集到的目标转移。权重过大时，少量样本可能主导估计；实际方法常采用截断或归一化控制这种影响，但也可能引入估计偏差。因此，离线与跨动力学学习需要结合数据覆盖范围，解释样本权重及其在更新中的作用。",
    ),
    499: (
        "公交转移样本的联合分布可分解为状态动作分布和给定状态动作后的条件转移核。源环境与目标环境分别表示为",
    ),
    501: (
        "等式左侧为完整转移样本的联合密度。右侧的",
        "表示数据中的状态动作分布，",
        "描述给定状态动作后的后继状态与事件间隔。采集策略不同，会改变数据中各状态动作出现的频率；运行机制不同，也可能使同一动作产生不同后果，并进一步改变后续状态的分布。因此，仅凭分类器能够区分源目标样本，还不能判断差异来自采集行为、运行机制，还是二者共同作用。",
    ),
    502: (
        "给定状态动作时，源样本用于估计目标条件期望的权重为",
        "。可分别训练联合转移分类器和状态动作分类器，前者输入当前状态、动作、后继状态与事件间隔，后者只输入当前状态与动作。两个分类器均以等量源样本和目标样本训练，分别用",
        "和",
        "表示样本来自目标环境的预测概率。若分类器准确给出这两个概率，则有",
    ),
    504: (
        "其中，logit将介于0与1之间的分类概率v转换为对数几率。若源目标类别比例不等，还需扣除相应先验比；有限数据训练所得的比率则受概率估计误差影响。DARC据此修正动力学差异造成的奖励偏差，H2O与H2O+进一步在源目标混合训练中利用动力学差异信息。",
        "若联合分类器不输入事件间隔，估计的只是后继状态的转移比，不能直接解释为状态与事件时间的联合转移比。第5章分别说明各训练方案是否启用比率修正及权重作用于何种损失，并以分类诊断和公交控制结果评价其作用。",
    ),
    506: (
        "世界模型从数据中学习环境的演化规律，用于预测动作执行后的状态、奖励或后续观测。已有方法既可在低维状态上学习概率动力学，也可在潜在空间中预测观测及回报。",
        "本文公交任务具有明确的类别身份和连续运行量，因此直接研究当前状态、动作与后继运行结果之间的预测关系。",
    ),
    507: (
        "公交按事件决策，相邻事件经过的时间不固定。相应模型需要描述后继状态、事件间隔与奖励，其预测对象可写为",
    ),
    509: (
        "其中，",
        "为模型参数。前一个带帽对象表示后继状态与事件间隔的联合预测分布，后一个表示给定转移条件下的奖励预测。概率模型拟合条件分布，确定性模型可预测状态增量或条件均值；奖励可单独预测，也可在获得乘客人数轨迹后由时间积分计算。事件间隔影响成本累计和时域终止，仅预测后继状态不足以描述完整的半马尔可夫过程。第6章据公交事件的数据类型构建预测，并将实际预测量用于价值更新。",
    ),
    510: (
        "模型拟合还需考虑数据的时间关联。相邻公交事件共享车辆、乘客及外部运行条件，随机打散事件后再划分拟合与验证数据，会使同一运行日的信息进入两组。本文公交模型的相应比较按完整运行日划分数据，在未参与拟合的运行日上评价预测误差。同时比较模型类别、拟合数据量和计算成本，再通过目标环境中的服务结果判断预测改进是否有助于控制。",
    ),
    511: (
        "模型既可用于规划，也可生成预测样本辅助价值学习。模型辅助价值估计将短程预测中的奖励与末端状态价值相加，构造当前状态的价值估计。",
        "仅累计环境奖励时，由状态",
        "向前预测",
        "步，得到",
    ),
    514: (
        "重要性采样调整已有源经验对学习的影响，世界模型则学习或修正动作后果，并生成用于价值更新的预测样本。第5章先研究固定目标离线数据辅助源仿真训练，第6章进一步比较残差修正与直接目标建模，并按预测可信程度使用模型样本。两章均以目标环境中的公交服务效果检验方法，分类精度与预测误差用于解释相应学习过程。",
    ),
}

FIELD_PARAGRAPH = {
    0: "式中，预测长度以事件数计，空乘积取",
    1: "1",
    2: "，预测终止后的延续因子取零。带帽的状态、奖励和延续因子由模型生成。增加预测步数可纳入更多后续影响，也会使预测误差随之累积。第",
    3: "6",
    4: "章采用一步模型转移构造",
    5: "SAC",
    6: "价值回归目标，后继价值按式",
    8: "计入策略熵，模型样本权重调节其对评论家更新的影响。式中的",
    9: "j",
    10: "和",
    11: "l",
    12: "分别为求和与乘积索引。模型集成分歧或留出预测误差可用于评价预测可靠程度，模型辅助效果则由目标环境中的控制结果检验。",
}


def main():
    assert not BACKUP.exists(), "Preserve the existing backup; inspect before rerunning."
    with ZipFile(DOCUMENT) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        assert "".join(body[493].itertext()) == "2.3 迁移学习基础"
        assert "".join(body[515].itertext()) == "2.4 仿真平台与实验设置"
        indices = sorted([*REVISIONS, 513])
        updated = original
        for index in indices:
            paragraph = body[index]
            paragraph_id = paragraph.get(W14 + "paraId")
            pattern = r'<w:p\b[^>]*w14:paraId="' + paragraph_id + r'"[^>]*>.*?</w:p>'
            matches = list(re.finditer(pattern, updated, flags=re.S))
            assert len(matches) == 1, (index, paragraph_id)
            match = matches[0]
            raw = match.group(0)
            if index == 513:
                assert list(paragraph.iter(W + "t"))[7].text == "(2-18)"
                revised = replace_text_nodes(raw, FIELD_PARAGRAPH)
            else:
                revised = revise_paragraph(raw, paragraph, REVISIONS[index])
            updated = updated[:match.start()] + revised + updated[match.end():]

        # Compare all non-text XML to catch any unintended formatting or object change.
        mask = lambda raw: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", raw, flags=re.S)
        assert mask(original) == mask(updated)
        new_body = ET.fromstring(updated).find(W + "body")
        assert len(body) == len(new_body)
        changed = [i for i, (a, b) in enumerate(zip(body, new_body)) if ET.tostring(a) != ET.tostring(b)]
        assert changed == indices, changed
        for index in changed:
            for tag in (M + "oMath", W + "hyperlink", W + "instrText", W + "fldChar", W + "pPr", W + "rPr"):
                assert [ET.tostring(n) for n in body[index].iter(tag)] == [ET.tostring(n) for n in new_body[index].iter(tag)]
            old_citations = [ET.tostring(n) for n in body[index].iter(W + "r") if n.find(W + "rPr/" + W + "vertAlign") is not None]
            new_citations = [ET.tostring(n) for n in new_body[index].iter(W + "r") if n.find(W + "rPr/" + W + "vertAlign") is not None]
            assert old_citations == new_citations
        assert list(new_body[513].iter(W + "t"))[7].text == "(2-18)"

        copy2(DOCUMENT, BACKUP)
        temporary = DOCUMENT.with_suffix(".section-edit.docx")
        with ZipFile(temporary, "w") as target:
            for item in source.infolist():
                data = updated.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename)
                target.writestr(item, data)
    temporary.replace(DOCUMENT)
    with ZipFile(BACKUP) as before, ZipFile(DOCUMENT) as after:
        assert before.namelist() == after.namelist()
        assert [name for name in before.namelist() if before.read(name) != after.read(name)] == ["word/document.xml"]
    print("Changed body paragraphs:", changed)
    print("Only section 2.3 prose changed; abstracts, other sections, appendices, equations, citations, fields and formatting preserved.")
    for index in changed:
        print(index, "".join(n.text or "" for n in new_body[index].iter() if n.tag in (W + "t", M + "t")))
    print("OUTPUT", DOCUMENT)


if __name__ == "__main__":
    main()
