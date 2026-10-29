from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_pass2_section_4_4 import BACKUP, DOCUMENT, REVISIONS
from edit_section_3_2 import W, prose_groups
from edit_section_4_4 import replace_paragraph


SHORTENED = {
    1077: (
        "本节检验运行条件变化时的驻站控制。先以BAPR为参照，比较训练稳定化与输出结构，再分析乘客服务、冲击后积压和模式识别。随后分离输出与BC的作用，分别以时距和乘客时间为目标，与RE-SAC比较。前几组解释稳定化和算术输出的作用，后续软符号配置则检验动作约束能否改善乘客服务。",
    ),
    1128: (
        "并列低回报比例与最差回报。算术输出低于−1200千的比例较小，最差回报却较低，典型表现与最差结果需分别观察。比例按10次训练计算。",
    ),
    1146: (
        "不额外驻站在五种条件下的乘客成本均最低，正常运行奖励却为−424.80千，低于两种学习策略。本组时距目标下的策略尚无乘客净时间收益。第3.2.4节的局部分析也说明，候车节省需超过车内增量，驻站才可能降低成本。",
        "给出全程乘客时间，检验这种净收益。",
    ),
    1181: (
        "相对稳定化ReLU，算术输出在正常、单次2倍、三次2倍、单次5倍、三次5倍下，分别提高7.69、7.57、9.51、13.06和4.62千。这组中等压力下有额外收益，第4.4.3节高强度变化中却略差于ReLU，说明输出收益随需求条件变化。",
        "并列全部回报增益与候车统计。",
    ),
    1183: (
        "相同供给和需求日程下，持续变化与有限冲击均显示：时距回报改善，但候车时间和完成率变化较小，尚无明显乘客时间收益。",
    ),
    1186: (
        "每策略在正常、单次5倍及三次5倍脉冲下，各评价三个共同运行日，共180回合。两类脉冲均累计1800秒。先平均评价日，再汇总五次训练。",
    ),
    1206: (
        "软符号配置采用可学习基准动作与参考状态下为零的神经修正，关闭NAU，加入SINDy软约束；使用3个评论家、256批量。两类奖励训练100回合，各用两个训练种子，每场景三个共同评价日；RE-SAC使用10个评论家、2048批量和10⁻⁵学习率。乘客目标下，正常、单次5倍、重复5倍、持续5倍、持续10倍及短时50倍的成本分别降低62.80、67.36、44.63、27.78、14.18、51.58秒，即0.549%、0.440%、0.333%、0.172%、0.083%、0.245%；计划偏差减少7.35至12.09秒，时距不均衡度减少11.88至20.08秒。",
    ),
    1208: (
        "以上结果逐步回答本章问题：500回合比较支持完整稳定化相对BAPR的回报收益；100回合交叉比较尚未确定输出与BC各自的稳定增量；后续软符号配置在六类需求中降低了相对RE-SAC的平均乘客成本。公交近期反馈用于训练，评价时策略固定。",
    ),
    1211: (
        "软符号配置以时距为目标时，五类场景的平均奖励惩罚降低10.41%至17.43%，持续10倍需求下却增加0.27%；计划偏差增加0.33至1.41秒，时距不均衡度也增加。奖励改善未对应全部间隔指标改善。以乘客时间为目标时，每场景六个配对评价有三个成本下降、三个上升，总体均值仍下降，说明平均服务改善与各次运行差异并存。",
    ),
    1214: (
        "训练稳定化提高时距回报，算术输出的增益随需求变化。软符号配置以乘客时间为目标时，平均乘客成本和两类时距指标均改善；以时距为目标时仍有取舍。本章未预先使用目标域转移训练数据，第5章再引入固定目标离线记录，研究跨保真训练。",
    ),
}


with ZipFile(DOCUMENT) as source:
    original = source.read("word/document.xml").decode("utf-8")
    body = ET.fromstring(original).find(W + "body")
    updated = original
    for index, prose in SHORTENED.items():
        updated = replace_paragraph(updated, body[index], prose)
    mask = lambda xml: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", xml, flags=re.S)
    with ZipFile(BACKUP) as before:
        prior = before.read("word/document.xml").decode("utf-8")
        initial = ET.fromstring(prior).find(W + "body")
        final = ET.fromstring(updated).find(W + "body")
        assert mask(prior) == mask(updated)
        assert len(initial) == len(final) == 2924
        changed = [i for i, (a, b) in enumerate(zip(initial, final)) if ET.tostring(a) != ET.tostring(b)]
        assert changed == sorted(set(REVISIONS) | set(SHORTENED)), changed
        for index in changed:
            assert 1076 < index < 1215
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
print({"shortened_blocks": list(SHORTENED), "final_changed_blocks": changed, "paragraphs": len(changed)})
