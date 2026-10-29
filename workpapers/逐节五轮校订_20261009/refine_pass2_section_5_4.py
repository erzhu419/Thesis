from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_pass2_section_5_4 import BACKUP, DOCUMENT, REVISIONS, W, prose_groups, replace_paragraph


REFINEMENTS = {
    1388: (
        "据此，后续跨保真任务统计全部已出发乘客的时间，以人均广义时间成本衡量候车节省与车内代价，另报行程完成率，检验规定时域内的服务结果。",
    ),
    1409: (
        "正式确认的1350个评价回合均运行至24,000秒，各完成389个车辆班次。最大同时到站批次含9辆车，未超过16个车辆位置的容量；没有回合因批次容量限制或时间上限提前终止，全部计划评价均纳入统计。",
    ),
    1499: (
        "再比较各策略由无最低驻站改为60秒最低驻站的收益。将两种策略各自的保障前后成本差相减，混合策略相对不额外驻站建议为−5.87秒，区间[−15.93，4.20]秒；相对计划间隔规则为−0.67秒，区间[−10.60，9.27]秒。两区间均包含0，尚未确定混合策略能额外增强保障收益。",
    ),
    1537: (
        "数百万条转移来自30组需求与交通条件，五次训练反映固定数据下的学习差异，正式配对检验另以10个新场景组为单位。时间组成、线路和乘客群体分析解释已有运行结果，不增加独立训练重复；按实际服务经历选出的群体，也不单独确定一次保障动作的个体效应。",
    ),
}


def main():
    with ZipFile(DOCUMENT) as source, ZipFile(BACKUP) as backup:
        current = source.read("word/document.xml").decode("utf-8")
        original = backup.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(current).find(W + "body")
        initial = ET.fromstring(original).find(W + "body")
        updated = current
        for index, prose in REFINEMENTS.items():
            updated = replace_paragraph(updated, body[index], prose)
        mask = lambda xml: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", xml, flags=re.S)
        assert mask(original) == mask(updated)
        final = ET.fromstring(updated).find(W + "body")
        assert len(initial) == len(final) == 2924
        changed = [i for i, (a, b) in enumerate(zip(initial, final)) if ET.tostring(a) != ET.tostring(b)]
        assert changed == list(REVISIONS), changed
        for i in changed:
            old_nodes, _, protected = prose_groups(initial[i])
            new_nodes, _, _ = prose_groups(final[i])
            assert all(old_nodes[k].text == new_nodes[k].text for k in protected)
            assert re.findall(r"\d+(?:\.\d+)?", "".join(initial[i].itertext())) == re.findall(r"\d+(?:\.\d+)?", "".join(final[i].itertext())), i
        temporary = DOCUMENT.with_suffix(".tmp")
        with ZipFile(temporary, "w") as target:
            for item in source.infolist():
                target.writestr(item, updated.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename))
    temporary.replace(DOCUMENT)
    with ZipFile(DOCUMENT) as after, ZipFile(BACKUP) as before:
        assert before.namelist() == after.namelist()
        assert [n for n in before.namelist() if before.read(n) != after.read(n)] == ["word/document.xml"]
    print({"refined": list(REFINEMENTS), "total_changed_paragraphs": len(changed)})


if __name__ == "__main__":
    main()
