import re
import xml.etree.ElementTree as ET
from pathlib import Path
from shutil import copy2
from zipfile import ZipFile

from edit_section_1_2_1 import DOCUMENT, W
from edit_section_4_4 import replace_paragraph


BACKUP = Path(__file__).with_name("校订前_5.5.docx")
REVISIONS = {
    1542: (
        "本章在前两章未预先使用目标域转移训练数据的基础上，进一步研究获得固定目标离线数据后的公交驻站学习。区分源目标状态动作分布、条件转移和有限采样误差，先比较固定数据学习与追加SUMO训练交互，再研究目标离线数据辅助源仿真训练。前一任务表明，固定数据已能支持有效策略，追加目标交互可提高部分控制回报，但车头时距稳定性与乘客时间未必同时改善。",
    ),
    1543: (
        "后一任务以广义乘客时间为目标，在H2O+框架下构建公交混合训练流程，分别编码同时到站车辆，并用混合动作分布表示不额外驻站和连续驻站时长。使用同一批目标离线数据、保持策略与评论家更新次数相同的五次训练中，混合方案相对纯离线IQL在正常及1.25倍需求下，人均广义时间成本分别减少16.66和19.75秒。其相对纯源训练的稳定优势尚未建立；附录F.2在新运行日采用共同执行规则比较完整控制器，H2O+相对纯源SAC平均减少17.23秒，但配对区间仍包含零。",
    ),
    1544: (
        "训练之外，驻站幅度校准与行程完成保障共同形成完整控制器。10个新场景组的1350回合评价中，该控制器的人均广义时间成本为2392.73秒、完成率为0.96650；相对采用原执行方式的纯源SAC，时间下降7.75%。另固定同一学习策略，将保障条件下的最低附加驻站由20秒改为60秒并扩大启用时段，时间减少92.12秒、下降3.71%，完成率提高0.466个百分点，10组方向一致。360回合交叉评价还表明，保障对六种基础策略均有收益；采用共同保障时，混合策略与计划间隔规则的结果接近。这些比较分别说明混合训练和执行规则的作用，保障的候车收益也伴随车内时间及大间隔事件增加。",
    ),
    1545: (
        "上述结果支持利用固定目标离线数据和源仿真交互训练驻站策略，但源仿真的运行与动作响应仍会影响训练效果。道路速度、客流和停站服务标定不足时，同一动作在源目标环境中可能产生不同后果。下一章据此研究MC-WM，进一步用目标数据学习世界模型，在残差修正与直接目标建模之间选择，并在共同动作、奖励和执行条件下评价模型辅助训练的服务效果。",
    ),
}


def main():
    assert not BACKUP.exists(), "Keep the existing pre-edit backup."
    with ZipFile(DOCUMENT) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        assert "".join(body[1541].itertext()) == "5.5 本章小结"
        assert "".join(body[1547].itertext()).startswith("第6章 ")
        updated = original
        for index, prose in REVISIONS.items():
            updated = replace_paragraph(updated, body[index], prose)

        mask = lambda xml: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", xml, flags=re.S)
        assert mask(original) == mask(updated), "Non-text XML changed."
        final_body = ET.fromstring(updated).find(W + "body")
        assert len(body) == len(final_body)
        changed = [i for i, (a, b) in enumerate(zip(body, final_body)) if ET.tostring(a) != ET.tostring(b)]
        assert changed == list(REVISIONS), changed
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
    print({"changed_body_blocks": changed, "backup": str(BACKUP), "non_text_xml_preserved": True})


if __name__ == "__main__":
    main()
