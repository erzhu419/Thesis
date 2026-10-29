from pathlib import Path
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_3_2 import W
from edit_section_4_4 import replace_paragraph


ROOT = Path(__file__).resolve().parents[2]
DOCUMENT = ROOT / "output/排版完成_逐节校订_第2轮_20261010.docx"
BACKUP = Path(__file__).with_name("校订前_第2轮_6.5.docx")
REVISIONS = {
    1917: (
        "本章承接第5章利用目标离线数据和源仿真交互训练策略的研究，进一步处理源仿真标定不足造成的预测偏差。MC-WM保留已拟合的源模型，用目标记录分别学习残差修正和直接目标模型，依据模型选择验证集上的预测误差选择。公交模型预测事件类别、人数、连续运行状态、持续时间和乘客成本，生成一步转移样本，以归一化置信权重参与价值学习。公交训练仍只使用固定SUMO记录和新增源经验，部署仅调用演员网络。",
    ),
    1918: (
        "单一驻站任务中，20次独立训练采用乘客时间目标、0.25模型损失项系数。正常标定、联合50%及75%源偏差下，人均广义时间成本相对无世界模型混合基线分别减少51.14、69.93和87.05秒，相对共同完成保障的Daganzo式规则分别减少57.67、70.67和76.26秒；所测偏差越大，相对混合基线的平均降幅越大。该基线并非正式H2O+；共同源采样与执行规则下的完整控制器比较见",
        "，MC-WM相对正式H2O+及纯源SAC分别平均减少23.02和40.25秒，相对H2O+的配对区间包含零。两组分别评价模型辅助配置与完整控制器。",
    ),
    1919: (
        "机制分析需要区分预测误差与预测样本对价值更新的贡献。早期公交训练的预测误差基本稳定，原置信递推却使样本权重和梯度贡献持续衰减；归一化后，模型样本重新有效参与价值学习。固定0.5样本权重也有收益，说明改善不全来自按状态区分样本可信程度。三个受控动力学任务各训练10次，MC-WM相对Target SAC的回报分别增加1925.7、1184.5和133.4，配对区间均为正，违例减少；相对匹配预算DARC的回报也提高。Ant执行器变化下，选用直接目标模型比强制采用残差模型增加1529.9回报，说明源预测不适用时，直接建模可以改善控制。",
    ),
    1920: (
        "进一步修正源转移的价值目标，或根据识别的运行条件调节模型系数，均未带来持续的平均增益。时距奖励补充实验采用0.125模型损失项系数和两个独立初始化；共同完成保障下，正常标定及联合75%偏差时，MC-WM相对H2O+的人均广义时间成本分别减少1.03和13.36秒，但未改善全部时距指标。乘客时间主结果主要来自候车节省，部分条件下车内时间和运行间隔指标有所恶化。综上，模型辅助训练缓解了所测标定偏差下的乘客服务损失，收益还取决于训练目标、预测样本的使用程度和执行规则，不能单独归于残差结构。",
    ),
}


def main():
    assert BACKUP.exists(), "Refine from the preserved pre-edit backup."
    with ZipFile(BACKUP) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        assert "".join(body[1916].itertext()) == "6.5 本章小结"
        assert "第7章" in "".join(body[1922].itertext())
        with ZipFile(DOCUMENT) as current:
            current_body = ET.fromstring(current.read("word/document.xml")).find(W + "body")
            assert len(current_body) == len(body)
            assert all(ET.tostring(a) == ET.tostring(b) for i, (a, b) in enumerate(zip(body, current_body)) if i not in REVISIONS)
        updated = original
        for index, prose in REVISIONS.items():
            updated = replace_paragraph(updated, body[index], prose)

        # Preserve objects, fields, formatting, and all text outside this section.
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
    print({"changed_paragraphs": len(changed), "backup": str(BACKUP), "output": str(DOCUMENT)})


if __name__ == "__main__":
    main()
