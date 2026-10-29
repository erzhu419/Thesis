from pathlib import Path
from shutil import copy2
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_3_2 import M, W, prose_groups
from edit_section_4_4 import replace_paragraph


ROOT = Path(__file__).resolve().parents[2]
DOCUMENT = ROOT / "output/排版完成_逐节校订_第2轮_20261010.docx"
BACKUP = Path(__file__).with_name("校订前_第2轮_5.1.docx")
REVISIONS = {
    1224: (
        "第3、4章未预先获得目标域转移训练数据，分别研究平稳随机运行与运行规律变化下的驻站学习。本章进一步考虑已有目标离线数据的情况。记录包含目标环境中的状态、驻站动作及其后续结果，但未覆盖的状态和动作仍缺少直接经验。如何利用已有目标记录，又减少新增训练对目标环境反复试错的依赖，是本章研究的问题。",
    ),
    1225: (
        "本章以多线路SUMO为高保真目标环境，以简易仿真为可反复交互的源环境，研究两项任务。第一项比较纯离线学习与追加SUMO交互的在线微调，考察固定目标记录的作用及新增目标经验的收益。第二项基于H2O+框架进行公交适配，使用另一套固定SUMO数据，将新增训练交互转至源仿真，结合目标记录与源经验学习驻站策略。后一任务不追加SUMO训练转移，开发验证和最终测试轨迹也不进入策略训练。",
    ),
    1226: (
        "源目标经验中的状态和动作出现频率可能不同；即使状态和驻站动作相同，道路速度、客流到达及停站服务的差异也可能使后继结果的分布不同。有限采样还会使相同运行规律产生不同估计。本章据此区分状态动作分布差异、条件转移差异与采样误差，分别检验训练方案和执行规则的服务作用。两项任务的数据、输入、奖励和运营时域不同，结果在各自任务内比较。源仿真标定不足时的预测修正由第6章研究。",
    ),
}


def main():
    assert not BACKUP.exists(), "Keep the existing pre-edit backup."
    with ZipFile(DOCUMENT) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        assert "".join(body[1223].itertext()) == "5.1 问题描述"
        assert "".join(body[1227].itertext()) == "5.2 模型构建"
        updated = original
        for index, prose in REVISIONS.items():
            updated = replace_paragraph(updated, body[index], prose)

        # Only ordinary paragraph text may change; keep objects and formatting.
        mask = lambda xml: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", xml, flags=re.S)
        assert mask(original) == mask(updated), "Non-text XML changed."
        final = ET.fromstring(updated).find(W + "body")
        assert len(body) == len(final) == 2924
        changed = [i for i, (a, b) in enumerate(zip(body, final)) if ET.tostring(a) != ET.tostring(b)]
        assert changed == list(REVISIONS), changed
        for index in changed:
            old_nodes, _, protected = prose_groups(body[index])
            new_nodes, _, _ = prose_groups(final[index])
            assert all(old_nodes[i].text == new_nodes[i].text for i in protected)
            for tag in (M + "oMath", W + "pPr", W + "hyperlink", W + "instrText", W + "fldChar"):
                assert [ET.tostring(n) for n in body[index].iter(tag)] == [ET.tostring(n) for n in final[index].iter(tag)]
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
