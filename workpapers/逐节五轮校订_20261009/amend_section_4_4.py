from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_4_4 import BACKUP, DOCUMENT, W, prose_groups, replace_paragraph


CHANGES = {
    1077: {
        "后续软符号策略比较连接摘要采用的乘客服务结果。": "后续软符号策略比较检验以乘客时间为目标时的服务改善。",
    },
    1205: {
        "参照已从前序BAPR变为RE-SAC，不能与表4-10、4-11混作同一组结果。": "两组实验采用不同参照，结果分别反映相对前序BAPR与RE-SAC的变化。",
    },
    1206: {
        "采用3评论家、256批量": "采用3个评论家、批量大小256",
        "采用10评论家、2048批量及10⁻⁵学习率": "采用10个评论家、批量大小2048及10⁻⁵学习率",
        "不均衡度减少11.88至20.08秒。": "时距不均衡度减少11.88至20.08秒。",
        "这是摘要采用的软符号策略结果，与早期稳定化MLP的收益不同。": "软符号策略取得了平均乘客时间收益，与早期稳定化MLP的结果不同。",
    },
    1211: {
        "前后不均衡也增加。奖励改善不能写成全部时距指标改善。": "时距不均衡度也增加。可见，奖励改善并不意味着全部时距指标都改善。",
    },
    1212: {
        "将完整动作分为可学习基准与神经修正": "将完整动作分为可学习的基准动作与神经网络修正项",
    },
    1213: {
        "各批均值按各自参照解释，不跨日程将累计候车量当作恢复速度，也不将十次训练的低回报比例当作现实事件风险。": "各批均值反映相对其参照方法的变化。不同需求日程下的累计候车量还受需求总量影响，不能直接衡量恢复速度；十次训练的低回报比例反映训练差异，不代表现实事件风险。",
    },
}

with ZipFile(DOCUMENT) as source:
    original = source.read("word/document.xml").decode("utf-8")
    body = ET.fromstring(original).find(W + "body")
    updated = original
    for index, replacements in CHANGES.items():
        nodes, groups, _ = prose_groups(body[index])
        prose = ["".join(nodes[j].text or "" for j in group) for group in groups]
        assert len(prose) == 1
        for old, new in replacements.items():
            assert prose[0].count(old) == 1, (index, old)
            prose[0] = prose[0].replace(old, new)
        updated = replace_paragraph(updated, body[index], tuple(prose))
    mask = lambda xml: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", xml, flags=re.S)
    assert mask(original) == mask(updated)
    after = ET.fromstring(updated).find(W + "body")
    assert len(body) == len(after)
    changed = [i for i, (a, b) in enumerate(zip(body, after)) if ET.tostring(a) != ET.tostring(b)]
    assert set(changed) == set(CHANGES)
    temporary = DOCUMENT.with_suffix(".section-edit.docx")
    with ZipFile(temporary, "w") as target:
        for item in source.infolist():
            target.writestr(item, updated.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename))
temporary.replace(DOCUMENT)
with ZipFile(BACKUP) as before, ZipFile(DOCUMENT) as after:
    assert before.namelist() == after.namelist()
    assert [n for n in before.namelist() if before.read(n) != after.read(n)] == ["word/document.xml"]
print({"amended_blocks": changed, "original_backup_preserved": str(BACKUP)})
