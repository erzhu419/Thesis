from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_pass2_section_5_3 import BACKUP, DOCUMENT, REVISIONS
from edit_section_3_2 import M, W, prose_groups
from edit_section_4_4 import replace_paragraph


UPDATES = {
    1298: (
        "本研究取",
        "、",
        "，另设较小的评论家权重正则项。",
        "沿用式",
        "的未加权平方误差，比较固定目标数据",
        "中的动作与重参数化采样动作，仅乘一次",
        "。负分歧系数降低高分歧动作的评价，BC限制策略偏离已有动作。本配置是前期RE-SAC",
        "在长沙多线路SUMO中的离线适配，不追加目标训练交互。保守评价思路与第3章相同，但状态、经验和BC设置不同，结果在本节任务内比较。",
    ),
    1303: (
        "在线及离线到在线方法均使用30,000次目标决策，每100次事件为一个训练周期，共300周期，在第50、100、150、200、250、300周期及结束时保存模型。周期用于组织更新，不是完整运营回合，预算按决策事件计量。",
    ),
    1330: (
        "各最终策略从同种子10,000源事件处的参数、优化器及回放状态续训至25,000个源事件。第15,000至25,000事件期间，演员学习率降至原值的0.25倍，其余日程不变。每个种子共完成70,000次训练更新，只评价25,000事件处的终点策略。",
    ),
    1335: (
        "每个状态对连续时长分支采样一次，同时直接计算0秒动作的价值。更新演员时固定评论家参数，但保留价值对动作的导数。式",
        "按两分支概率加权计算损失，门控输出因而可直接求导，无需对二值采样求导或采用直通梯度近似；连续时长通过重参数化求导。交互时按门控概率采样；确定性评价在额外驻站概率不小于0.5时，使用",
        "对应的时长，否则不额外驻站，最后应用第5.3.4节的执行变换。",
    ),
    1337: (
        "联合分类器纳入事件持续时间。式(5-18)中，",
        "与",
        "分别为联合和状态动作分类器的对数几率，",
        "与",
        "为参数，输入为",
        "与",
        "。目标样本为正类、源样本为负类。两类等量且类别概率准确时，对数几率之差等于目标相对源的条件转移对数比率，实际近似写为",
    ),
    1354: (
        "时间窗口限定干预阶段。较早驻站可能影响后续站点和班次，临近终点则可能推迟车内乘客完成行程。窗口和最低时长经开发评价确定，测试时固定。",
    ),
    1362: (
        "正式确认前，候选与直接参照、五个演员、场景组、主要指标、完成率允许下降的上限和检验方法均已固定，确认期间不再训练或更换控制器。配对差异反映最低驻站量与窗口调整的共同作用；混合训练另在第5.4.3节比较。",
    ),
}


with ZipFile(DOCUMENT) as source:
    current = source.read("word/document.xml").decode("utf-8")
    body = ET.fromstring(current).find(W + "body")
    updated = current
    for index, prose in UPDATES.items():
        updated = replace_paragraph(updated, body[index], prose)

    with ZipFile(BACKUP) as before:
        original = before.read("word/document.xml").decode("utf-8")
        baseline = ET.fromstring(original).find(W + "body")
        final = ET.fromstring(updated).find(W + "body")
        mask = lambda xml: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", xml, flags=re.S)
        assert mask(original) == mask(updated)
        changed = [i for i, (a, b) in enumerate(zip(baseline, final)) if ET.tostring(a) != ET.tostring(b)]
        assert changed == sorted(set(REVISIONS) | set(UPDATES)), changed
        for index in changed:
            old_nodes, _, protected = prose_groups(baseline[index])
            new_nodes, _, _ = prose_groups(final[index])
            assert all(old_nodes[i].text == new_nodes[i].text for i in protected)
            for tag in (M + "oMath", W + "pPr", W + "hyperlink", W + "instrText", W + "fldChar"):
                assert [ET.tostring(n) for n in baseline[index].iter(tag)] == [ET.tostring(n) for n in final[index].iter(tag)]
            assert re.findall(r"\d+(?:\.\d+)?", "".join(baseline[index].itertext())) == re.findall(r"\d+(?:\.\d+)?", "".join(final[index].itertext()))

    temporary = DOCUMENT.with_suffix(".tmp")
    with ZipFile(temporary, "w") as target:
        for item in source.infolist():
            target.writestr(item, updated.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename))
temporary.replace(DOCUMENT)
with ZipFile(BACKUP) as before, ZipFile(DOCUMENT) as after:
    assert before.namelist() == after.namelist()
    assert [n for n in before.namelist() if before.read(n) != after.read(n)] == ["word/document.xml"]
print({"changed_blocks": changed, "paragraphs": len(changed), "output": str(DOCUMENT)})
