from pathlib import Path
from shutil import copy2
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_3_2 import W
from edit_section_4_4 import replace_paragraph


ROOT = Path(__file__).resolve().parents[2]
DOCUMENT = ROOT / "output/排版完成_逐节校订_第2轮_20261010.docx"
BACKUP = Path(__file__).with_name("校订前_第2轮_5.5.docx")
REVISIONS = {
    1542: (
        "本章进一步引入固定目标离线数据，研究已有目标记录如何支持公交驻站学习。区分状态动作分布、条件转移与有限采样误差，先比较仅使用固定数据与追加SUMO训练交互，再研究目标数据辅助源仿真训练。前一任务表明，固定数据已能支持有效策略，追加目标交互可提高部分控制回报，但时距稳定性与乘客时间未必同时改善。",
    ),
    1543: (
        "跨保真任务在H2O+框架下构建公交混合训练流程，以降低广义乘客时间成本为目标，分别编码同时到站车辆，以混合动作分布表示不额外驻站与连续驻站时长。在使用同一批目标离线数据、策略与评论家更新次数相同的五次重复训练中，混合方案相对纯离线IQL，在正常及1.25倍需求下分别减少16.66和19.75秒人均广义时间成本，与纯源训练的差异尚不明确。附录F.2另在新运行日采用共同执行规则比较完整控制器，H2O+相对纯源SAC平均减少17.23秒，配对区间仍包含零。",
    ),
    1544: (
        "将学习策略与驻站幅度校准、行程完成保障结合，形成完整控制器。10个新场景组的1350回合评价中，该控制器的人均广义时间成本为2392.73秒、完成率为0.96650；相对该批采用原执行方式的纯源SAC，成本下降7.75%。固定同一学习策略，将保障条件下的最低附加驻站由20秒改为60秒并扩大启用时段，成本减少92.12秒、下降3.71%，完成率提高0.466个百分点，10组方向一致。360回合交叉评价中，保障对六种基础策略均有收益；采用共同保障时，混合策略与计划间隔规则接近。因此，保障收益并非混合训练独有，候车节省也伴随车内时间及大间隔事件增加。",
    ),
    1545: (
        "固定目标记录与新增源仿真交互相结合，可以支持驻站策略训练，效果仍依赖源仿真的动作响应。道路速度、客流和停站服务标定不足时，同一动作在源目标环境中可能产生不同后果。下一章据此研究MC-WM，用固定目标记录学习世界模型，在源预测的残差修正与直接目标建模之间选择，并在共同动作、奖励和执行条件下检验模型辅助的服务效果。",
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

        # Only prose may change; preserve all objects, fields and layout XML.
        mask = lambda xml: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", xml, flags=re.S)
        assert mask(original) == mask(updated), "Non-text XML changed."
        final = ET.fromstring(updated).find(W + "body")
        assert len(body) == len(final) == 2924
        changed = [i for i, (a, b) in enumerate(zip(body, final)) if ET.tostring(a) != ET.tostring(b)]
        assert changed == list(REVISIONS), changed
        for index in changed:
            numbers = lambda p: re.findall(r"\d+(?:\.\d+)?", "".join(p.itertext()))
            assert numbers(body[index]) == numbers(final[index]), index
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
