from pathlib import Path
from shutil import copy2
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_1_2_1 import W
from edit_section_4_4 import replace_paragraph


ROOT = Path(__file__).resolve().parents[2]
DOCUMENT = ROOT / "output/排版完成_逐节校订_第2轮_20261010.docx"
BACKUP = Path(__file__).with_name("校订前_第2轮_3.5.docx")
REVISIONS = {
    981: (
        "本章研究未预先获得目标域转移训练数据时，平稳随机环境中的公交驻站控制。将车辆、站点、方向及时段的类别嵌入与连续运行量结合，建立车队共享策略；提出RE-SAC，以价值目标修正和集成分歧约束改进动作评价，并分析固定惩罚下贝尔曼算子的收缩性质。",
    ),
    982: (
        "在同一RE-SAC学习器下，五次重复训练表明，学习嵌入相对独热编码在正常运行、客流激增和快速切换中，分别降低5.51%、1.41%和4.91%的人均广义时间成本，时距不均衡度也降低；相对参数量相同的类别屏蔽组，正常运行成本下降2.32%。快速切换下，演员保守评价项平均减少2.298秒时距不均衡。多线路时距目标下，RE-SAC相对SAC使平均计划偏差由106.47降至101.47秒，时距不均衡度由126.79降至119.44秒；改用乘客时间目标未获平均改善。另在同一集成结构内，保留评论家分歧与演员保守评价两项，相对同时移除两项，在正常运行、客流激增和快速切换中使平均计划偏差分别下降5.82%、7.12%和5.05%，乘客时间目标下的人均成本分别下降14.20%、4.02%和11.29%。该消融比较的时距结果为均值改善，乘客成本的配对区间则均支持下降。",
    ),
    983: (
        "集成规模需结合训练预算选择，扩大规模未带来持续收益。固定温度缓解了长期退化，候选动作比较则表明，更小的价值拟合误差未必改善动作排序和控制表现。计划间隔奖励减小计划偏差，却增加所测乘客时间；指数衰减尺度与驻站代价也改变这一取舍。驻站调节不能替代供给与计划调整。运行规律发生突变或反复变化时，历史经验还可能不再适用。下一章据此研究BAPR与CS-BAPR的条件识别、训练调节及非平稳驻站控制。",
    ),
}


def main():
    assert not BACKUP.exists(), "Keep the existing pre-edit backup."
    with ZipFile(DOCUMENT) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        assert "".join(body[980].itertext()) == "3.5 本章小结"
        assert "".join(body[985].itertext()).startswith("第4章 ")
        updated = original
        for index, prose in REVISIONS.items():
            updated = replace_paragraph(updated, body[index], prose)
        mask = lambda xml: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", xml, flags=re.S)
        assert mask(original) == mask(updated), "Non-text XML changed."
        final = ET.fromstring(updated).find(W + "body")
        assert len(body) == len(final) == 2924
        changed = [i for i, (a, b) in enumerate(zip(body, final)) if ET.tostring(a) != ET.tostring(b)]
        assert changed == list(REVISIONS), changed
        for index in changed:
            assert re.findall(r"\d+(?:\.\d+)?", "".join(body[index].itertext())) == re.findall(r"\d+(?:\.\d+)?", "".join(final[index].itertext()))
        copy2(DOCUMENT, BACKUP)
        temporary = DOCUMENT.with_suffix(".tmp")
        with ZipFile(temporary, "w") as target:
            for item in source.infolist():
                target.writestr(item, updated.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename))
    temporary.replace(DOCUMENT)
    with ZipFile(BACKUP) as before, ZipFile(DOCUMENT) as after:
        assert before.namelist() == after.namelist()
        assert [n for n in before.namelist() if before.read(n) != after.read(n)] == ["word/document.xml"]
    print({"changed_blocks": changed, "backup": str(BACKUP), "output": str(DOCUMENT)})


if __name__ == "__main__":
    main()
