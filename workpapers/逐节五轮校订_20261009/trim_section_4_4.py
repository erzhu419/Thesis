from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_4_4 import DOCUMENT, W, replace_paragraph


REVISIONS = {
    1183: "供给与需求日程相同时，完整配置改善了时距回报，候车与完成率变化较小。长期变化与有限冲击均支持这一控制作用，但乘客收益仍需单独评价。",
    1191: "列出输出结构与BC的交叉比较，检验稳定化后两个组件的增量。",
    1200: "惩罚为第3.2.3节负奖励的绝对值，对决策取平均，包含加权计划偏差、间隔不均衡和大偏差项，未用训练标准差缩放。惩罚降幅取BAPR减CS-BAPR后除以BAPR；计划偏差变化取CS-BAPR相对BAPR的变化率。各格为两次训练均值。",
    1213: "公交固定策略评价不单独检验在线参数适应。单次脉冲采用两个训练种子和成对输入，六次运行均未恢复；重复冲击采用三个训练种子与三个评价日交叉，早期BAPR与RE-SAC仅复评一个训练种子。后续软符号组采用两个训练种子，超参数与RE-SAC不同。各批均值对应各自参照；累计候车量受需求日程与总量影响，并非恢复速度，训练低回报比例也不代表现实风险。",
    1214: "训练稳定化改善学习基础，输出结构与软符号约束影响特定条件的响应，但没有一种结构始终最好。乘客目标下的软符号策略同时改善平均乘客时间和时距指标，体现控制目标与策略表示的共同作用。第5章引入固定目标离线数据，研究源仿真训练的迁移收益。",
}

with ZipFile(DOCUMENT) as source:
    original = source.read("word/document.xml").decode("utf-8")
    body = ET.fromstring(original).find(W + "body")
    updated = original
    for index, text in REVISIONS.items():
        updated = replace_paragraph(updated, body[index], (text,))
    mask = lambda xml: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", xml, flags=re.S)
    assert mask(original) == mask(updated)
    final_body = ET.fromstring(updated).find(W + "body")
    assert len(body) == len(final_body)
    changed = [i for i, (a, b) in enumerate(zip(body, final_body)) if ET.tostring(a) != ET.tostring(b)]
    assert set(changed) == set(REVISIONS)
    temporary = DOCUMENT.with_suffix(".section-edit.docx")
    with ZipFile(temporary, "w") as target:
        for item in source.infolist():
            target.writestr(item, updated.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename))
temporary.replace(DOCUMENT)
print({"trimmed_blocks": changed, "formatting_unchanged": True})
