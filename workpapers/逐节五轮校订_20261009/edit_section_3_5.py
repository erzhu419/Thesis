from pathlib import Path
from shutil import copy2
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_1_2_1 import DOCUMENT, W, W14, revise_paragraph


BACKUP = Path(__file__).with_name("校订前_3.5.docx")
REVISIONS = {
    981: (
        "本章研究未预先获得目标域转移训练数据时，平稳随机环境中的公交驻站控制。将四类身份信息与连续状态结合，构建共享策略；提出RE-SAC，通过保守价值目标和评价器分歧改进学习，并证明固定惩罚条件下贝尔曼算子的收缩性质。",
    ),
    982: (
        "五种子表征比较中，学习嵌入相对独热编码在正常运行、客流激增和快速切换下，人均广义时间成本分别下降5.51%、1.41%和4.91%，车头时距不均衡度也下降；相对尺寸匹配的类别信息屏蔽组，正常运行的成本下降2.32%。快速切换时，演员悲观项平均减少2.298秒的时距不均衡度。多线路时距目标下，RE-SAC相对SAC的平均计划偏差由106.47降至101.47秒，时距不均衡度由126.79降至119.44秒。另与相同集成学习器移除评论家分歧项和演员悲观项的参照比较，单一驻站下三种场景的平均计划偏差分别下降5.82%、7.12%和5.05%；乘客时间目标下，人均广义时间成本分别下降14.20%、4.02%和11.29%。",
    ),
    983: (
        "集成规模的收益随训练设置变化。长期训练和候选动作比较表明，探索强度与动作排序影响收益，价值拟合误差不足以判断控制质量。计划间隔奖励减小计划偏差，却增加所测乘客时间；指数奖励尺度与动作惩罚也影响车头时距稳定性和驻站代价的取舍。驻站难以独立补偿持续需求变化或计划失配。下一章进一步研究运行规律随时间变化时的模式识别与非平稳适应。",
    ),
}


def main():
    with ZipFile(DOCUMENT) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        assert "".join(body[980].itertext()) == "3.5 本章小结"
        assert "".join(body[985].itertext()).startswith("第4章 ")
        updated = original
        for index, prose in REVISIONS.items():
            paragraph_id = body[index].get(W14 + "paraId")
            pattern = r'<w:p\b[^>]*w14:paraId="' + paragraph_id + r'"[^>]*>.*?</w:p>'
            matches = list(re.finditer(pattern, updated, flags=re.S))
            assert len(matches) == 1
            match = matches[0]
            revised = revise_paragraph(match.group(0), body[index], prose)
            updated = updated[:match.start()] + revised + updated[match.end():]
        mask = lambda raw: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", raw, flags=re.S)
        assert mask(original) == mask(updated), "Non-text XML changed."
        new_body = ET.fromstring(updated).find(W + "body")
        assert len(body) == len(new_body)
        changed = [i for i, (a, b) in enumerate(zip(body, new_body)) if ET.tostring(a) != ET.tostring(b)]
        assert changed == list(REVISIONS), changed
        if not BACKUP.exists():
            copy2(DOCUMENT, BACKUP)
        temporary = DOCUMENT.with_suffix(".section-edit.docx")
        with ZipFile(temporary, "w") as target:
            for item in source.infolist():
                data = updated.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename)
                target.writestr(item, data)
    temporary.replace(DOCUMENT)
    with ZipFile(BACKUP) as before, ZipFile(DOCUMENT) as after:
        assert before.namelist() == after.namelist()
        changed_parts = [name for name in before.namelist() if before.read(name) != after.read(name)]
        assert changed_parts == ["word/document.xml"], changed_parts
    print({"changed_blocks": changed, "changed_parts": changed_parts, "backup": str(BACKUP)})


if __name__ == "__main__":
    main()
