from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_pass2_section_3_4 import BACKUP, DOCUMENT, REVISIONS
from edit_section_1_2_1 import W
from edit_section_3_2 import prose_groups
from edit_section_4_4 import replace_paragraph


FINISH = {
    923: (
        "相对参数量相同的屏蔽组，正常运行成本减少38.17秒（2.32%），减少量区间[4.87，71.47]秒；客流激增和快速切换少49.97、39.08秒，区间跨零。",
        "保留各次训练结果，正常运行的收益较明确，另两种条件为均值改善。",
    ),
    927: (
        "本组固定目标价值中的集成惩罚与权重范数项，仅改变评论家损失分歧项和演员保守评价项是否启用，BC与温度更新不变。“两项均关闭”是指定位置的消融，既非去除全部集成惩罚，也不是改为SAC。",
    ),
    930: (
        "记",
        "为演员项标记",
        "与评论家项标记",
        "取0或1时的指标，1表示启用、0表示关闭。为区分两项的作用，在每个训练种子内，对另一项的两种设置等权平均，定义",
    ),
    954: (
        "中，2、5、10、20个评论家的平均训练耗时为81.97、83.14、82.37、94.17分钟，峰值已分配显存为27.70、38.75、55.76、91.18二进制兆字节（Mebibyte，MiB）。同种子四组共用节点，耗时受并发负载影响，所报显存并非进程总占用。部署仅调用演员，中央处理器（Central Processing Unit，CPU）单次调用平均0.407至0.443毫秒，未随集成规模持续增加。",
    ),
}

with ZipFile(DOCUMENT) as source, ZipFile(BACKUP) as backup:
    original = source.read("word/document.xml").decode("utf-8")
    baseline = backup.read("word/document.xml").decode("utf-8")
    body = ET.fromstring(original).find(W + "body")
    before = ET.fromstring(baseline).find(W + "body")
    updated = original
    for index, prose in FINISH.items():
        updated = replace_paragraph(updated, body[index], prose)
    final = ET.fromstring(updated).find(W + "body")
    mask = lambda xml: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", xml, flags=re.S)
    assert mask(baseline) == mask(updated)
    changed = [i for i, (a, b) in enumerate(zip(before, final)) if ET.tostring(a) != ET.tostring(b)]
    assert set(changed) == set(REVISIONS) | set(FINISH)
    for i in changed:
        assert 811 < i < 980
        assert re.findall(r"\d+(?:\.\d+)?", "".join(before[i].itertext())) == re.findall(r"\d+(?:\.\d+)?", "".join(final[i].itertext()))
        nodes, _, protected = prose_groups(before[i])
        after_nodes = list(final[i].iter(W + "t"))
        assert all(ET.tostring(nodes[j]) == ET.tostring(after_nodes[j]) for j in protected)
    temporary = DOCUMENT.with_suffix(".tmp")
    with ZipFile(temporary, "w") as target:
        for item in source.infolist():
            target.writestr(item, updated.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename))
temporary.replace(DOCUMENT)
with ZipFile(BACKUP) as before, ZipFile(DOCUMENT) as after:
    assert before.namelist() == after.namelist()
    assert [n for n in before.namelist() if before.read(n) != after.read(n)] == ["word/document.xml"]
print({"changed_blocks": changed, "paragraphs": len(changed)})
