from pathlib import Path
from shutil import copy2
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_1_2_1 import DOCUMENT, M, W
from edit_section_3_2 import prose_groups
from edit_section_4_4 import replace_paragraph


BACKUP = Path(__file__).with_name("校订前_6.5.docx")
REVISIONS = {
    1917: (
        "本章在第5章目标离线数据辅助源仿真训练的基础上，研究标定不足时的世界模型修正。MC-WM固定已拟合的源模型，以目标记录学习残差修正与直接目标模型，并根据模型选择验证集上的预测误差择优。公交模型分别预测事件类别、人数、连续运行状态、持续时间与乘客成本，再用归一化置信更新确定一步预测样本的训练权重。公交训练只使用固定SUMO记录和新增源经验，部署仍仅调用演员网络。",
    ),
    1918: (
        "单一驻站的公交任务中，20次独立训练表明，以乘客时间为目标、模型损失项系数为0.25时，正常标定、联合50%及75%源偏差下，人均广义时间成本相对无世界模型混合基线分别减少51.14、69.93和87.05秒，相对共同完成保障的Daganzo式规则分别减少57.67、70.67和76.26秒。无世界模型混合基线不同于正式H2O+控制器；共同源采样与执行规则下的最终控制器比较另见",
        "，MC-WM相对正式H2O+及纯源SAC分别平均减少23.02和40.25秒，其中相对H2O+的配对区间包含零。两组结果分别说明模型辅助配置和最终完整控制器的平均服务效果。",
    ),
    1919: (
        "机制分析表明，预测误差与模型样本的使用强度需要分别评价。早期公交训练中，预测误差基本稳定，原置信度递推更新却使权重和梯度贡献持续衰减；归一化后恢复了模型参与价值学习的作用。固定0.5样本权重也有效，说明收益不能全部归于状态相关置信排序。三个受控动力学任务、各10次训练中，MC-WM相对Target SAC的回报分别增加1925.7、1184.5和133.4，配对区间均为正，违例减少；相对匹配预算DARC的回报也提高。Ant执行器变化下，所选直接模型相对强制残差增加1529.9回报，表明源预测不合适时，替换模型可以改善控制。",
    ),
    1920: (
        "源转移价值目标修正与上下文系数调节没有持续增加平均收益。时距奖励补充实验采用0.125模型损失项系数和两个独立初始化；共同完成保障下，正常标定及联合75%偏差时，MC-WM相对H2O+的人均广义时间成本分别减少1.03和13.36秒，但未改善全部时距指标。乘客时间主结果主要来自候车节省，也伴随部分车内时间和运行间隔代价。本章结果支持模型辅助在所测标定偏差下的平均服务收益，其效果取决于训练目标、样本使用强度和执行规则，不能将完整配置的收益单独归于残差模型。",
    ),
}


def main():
    assert not BACKUP.exists(), "Do not overwrite the pre-edit backup."
    with ZipFile(DOCUMENT) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        assert "".join(body[1916].itertext()) == "6.5 本章小结"
        assert "".join(body[1922].itertext()) == "第7章 总结与展望"
        updated = original
        for index, prose in REVISIONS.items():
            updated = replace_paragraph(updated, body[index], prose)

        mask = lambda xml: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", xml, flags=re.S)
        assert mask(original) == mask(updated), "Non-text XML changed."
        final_body = ET.fromstring(updated).find(W + "body")
        assert len(body) == len(final_body)
        changed = [i for i, (a, b) in enumerate(zip(body, final_body)) if ET.tostring(a) != ET.tostring(b)]
        assert changed == list(REVISIONS), changed
        protected_count = 0
        for i in changed:
            for tag in (M + "oMath", W + "hyperlink", W + "instrText", W + "fldChar", W + "pPr"):
                assert [ET.tostring(n) for n in body[i].iter(tag)] == [ET.tostring(n) for n in final_body[i].iter(tag)], (i, tag)
            nodes, _, protected = prose_groups(body[i])
            new_nodes = list(final_body[i].iter(W + "t"))
            assert all(ET.tostring(nodes[j]) == ET.tostring(new_nodes[j]) for j in protected), i
            protected_count += len(protected)
        copy2(DOCUMENT, BACKUP)
        temporary = DOCUMENT.with_suffix(".section-edit.docx")
        with ZipFile(temporary, "w") as target:
            for item in source.infolist():
                data = updated.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename)
                target.writestr(item, data)
    temporary.replace(DOCUMENT)
    with ZipFile(BACKUP) as before, ZipFile(DOCUMENT) as after:
        assert before.namelist() == after.namelist()
        assert [n for n in before.namelist() if before.read(n) != after.read(n)] == ["word/document.xml"]
    print({"changed_blocks": changed, "protected_text_nodes": protected_count,
           "backup": str(BACKUP), "output": str(DOCUMENT)})


if __name__ == "__main__":
    main()
