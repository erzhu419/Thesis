from pathlib import Path
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_3_2 import W
from edit_section_4_4 import replace_paragraph


ROOT = Path(__file__).resolve().parents[2]
DOCUMENT = ROOT / "output/排版完成_逐节校订_第2轮_20261010.docx"
BACKUP = Path(__file__).with_name("校订前_第2轮_6.1.docx")
REVISIONS = {
    1552: (
        "第5章结合固定目标离线数据与源仿真交互，训练驻站策略。道路速度、客流、路口延误或停站服务标定不足时，同一动作在源仿真与目标环境中可能产生不同的到达时刻、乘客服务结果和前后时距，导致策略高估某些驻站动作的收益。本章据此研究如何用目标记录修正运行预测，并用于驻站学习。",
    ),
    1553: (
        "公交训练仍只追加源仿真交互，SUMO离线数据保持固定。世界模型有两种构建方式：先从源经验学习运行关系，再用目标记录拟合源预测的残差；或直接用目标记录建模。目标数据的覆盖范围与拟合预算会影响两种方式的效果，因此需用未参与拟合的目标记录比较预测误差，选择用于训练的模型。",
    ),
    1554: (
        "所选模型根据当前状态和驻站动作，预测后继状态、事件间隔与奖励，生成一步转移辅助价值学习。预测误差也可能随样本进入动作评价，因此需依据预测可靠程度确定训练权重。权重过大可能放大错误预测的影响，过小则削弱模型作用。除比较预测误差外，还需检验样本加权能否使模型信息有效参与训练，以及最终驻站控制是否改善。",
    ),
    1555: (
        "实验分别考察预测模型、样本使用和控制结果。受控连续任务允许追加目标交互，用于分析残差修正、直接建模与模型选择；公交任务保持目标离线数据固定，检验不同源标定偏差下的事件预测、驻站服务效果与计算代价。",
    ),
}


def main():
    assert BACKUP.exists(), "Use the retained pre-edit backup for this refinement."
    with ZipFile(BACKUP) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        assert "".join(body[1551].itertext()) == "6.1 问题描述"
        assert "".join(body[1556].itertext()).startswith("6.2 ")
        updated = original
        for index, prose in REVISIONS.items():
            updated = replace_paragraph(updated, body[index], prose)

        # Preserve objects, fields and layout; only the four prose blocks change.
        mask = lambda xml: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", xml, flags=re.S)
        assert mask(original) == mask(updated), "Non-text XML changed."
        final = ET.fromstring(updated).find(W + "body")
        assert len(body) == len(final) == 2924
        changed = [i for i, (a, b) in enumerate(zip(body, final)) if ET.tostring(a) != ET.tostring(b)]
        assert changed == list(REVISIONS), changed
        for index in changed:
            numbers = lambda p: re.findall(r"\d+(?:\.\d+)?", "".join(p.itertext()))
            assert numbers(body[index]) == numbers(final[index]), index
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
