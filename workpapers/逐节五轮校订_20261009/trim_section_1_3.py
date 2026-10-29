from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_1_2_1 import DOCUMENT, W, revise_paragraph
from edit_section_1_3 import paragraph_pattern


REVISIONS = {
    341: (
        "（3）第5章研究固定目标域离线数据辅助的跨保真公交驻站训练。基于已有H2O/H2O+框架，完成公交事件状态、驻站动作与训练流程的适配。",
        "先比较离线学习与新增目标交互的在线微调，再结合固定SUMO离线数据和快速仿真交互训练策略。编码同时到站车辆的当前状态与已执行动作，用混合动作表示不额外驻站与额外驻站时长，并结合驻站幅度调整和行程完成保障形成控制器。比较混合训练与纯离线训练，并对照同一策略的不同执行规则，以Daganzo式规则作为参照，分别分析训练与执行对乘客服务的作用。",
    ),
    342: (
        "（4）第6章针对源仿真标定不足，提出MC-WM",
        "。利用目标域离线数据学习源预测的残差修正，并与直接学习目标预测的模型比较；用未参与拟合的数据选择模型，按预测可信度控制模型样本参与价值更新的权重。公交模型预测下一事件的车辆与站点类别、连续运行状态、事件间隔和乘客成本，并对置信度递推更新作归一化，防止模型样本权重在递推中迅速减小。通过不同标定偏差下与H2O+、无世界模型混合训练基线的比较，检验预测修正对车头时距和乘客服务的作用。",
    ),
}


with ZipFile(DOCUMENT) as source:
    original = source.read("word/document.xml").decode("utf-8")
    body = ET.fromstring(original).find(W + "body")
    updated = original
    for index, prose in REVISIONS.items():
        match = re.search(paragraph_pattern(body[index]), updated, flags=re.S)
        revised = revise_paragraph(match.group(), body[index], prose)
        mask = lambda raw: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", raw, flags=re.S)
        assert mask(match.group()) == mask(revised)
        updated = updated[:match.start()] + revised + updated[match.end():]
    new_body = ET.fromstring(updated).find(W + "body")
    assert len(body) == len(new_body)
    changed = [i for i, (a, b) in enumerate(zip(body, new_body)) if ET.tostring(a) != ET.tostring(b)]
    assert changed == list(REVISIONS)
    temporary = DOCUMENT.with_suffix(".section-edit.docx")
    with ZipFile(temporary, "w") as target:
        for item in source.infolist():
            target.writestr(item, updated.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename))
temporary.replace(DOCUMENT)
print("Trimmed only the two research items in 1.3:", changed)
