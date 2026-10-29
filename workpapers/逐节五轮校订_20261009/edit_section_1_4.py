from pathlib import Path
from shutil import copy2
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_1_2_1 import DOCUMENT, W, W14
from edit_section_1_2_4 import replace_text_nodes


BACKUP = Path(__file__).with_name("校订前_1.4.docx")

PARAGRAPH_REVISIONS = {
    344: {
        0: "本文按照公交决策建模、仿真策略学习和目标数据辅助改进三个环节展开研究。",
        1: "首先，依据既定线路、班次和车辆供给，将车辆运行、乘客服务与驻站决策联系起来，明确动作后的状态变化，并分别设置车头时距和广义乘客时间成本目标。建模时区分给定规律下的随机波动与规律本身的变化：前者需评价动作的不同随机后果，后者还需根据近期观测判断运行条件。偶然不确定性与认知不确定性描述随机结果与知识不足，并非平稳与非平稳的对应分类。",
    },
    345: {
        0: "其次，在训练前尚无目标域转移数据时，先在仿真中学习并检验驻站策略。",
        1: "第3章在给定随机规律下，通过类别特征共享车辆经验，以RE-SAC的保守价值评价减少不可靠估计对动作选择的影响。第4章进一步考虑运行规律变化，利用近期记录调节训练，并研究策略输出结构及符号参考约束。域随机化为生成多种仿真条件提供思路，鲁棒评价与尾部风险分析用于解释保守处理。",
        2: "", 3: "", 4: "", 5: "",
        13: "公交BAPR及CS-BAPR在训练时利用近期运行信息，评价时保持策略参数固定，以变化条件下的控制结果检验适应效果；模式推断与元学习的关系及具体实现范围在第4章说明。",
        14: "", 15: "", 16: "", 17: "", 18: "", 19: "", 20: "",
    },
    346: {
        0: "最后，在获得固定目标离线数据后，将目标经验用于源仿真训练及预测修正。",
        1: "第5章将H2O+用于公交，把目标记录与源仿真新增经验共同用于策略更新。当源仿真标定不足时，第6章以MC-WM在留出目标数据上比较残差修正与直接建模，再依据预测可靠程度调节模型样本对价值更新的影响。由此，目标数据的用途从策略学习扩展到预测模型学习，而非必须增加数据量。公交新增训练交互仍在源仿真中完成，目标环境提供离线数据并用于开发、测试评价。",
        2: "", 3: "", 4: "", 5: "",
    },
    347: {
        0: "理论分析与仿真实验分别说明方法的成立条件和控制效果。鲁棒评价的最坏情形下界以不确定集合及所评价的目标为前提：当集合包含待部署环境时，下界也约束该环境下的表现；集合之外的变化仍需通过数据与实验检验。表",
        1: "",
        3: "汇总各章的运行条件、可用数据与方法重点。",
    },
    348: {3: "各研究章的运行条件、数据来源与方法重点"},
    350: {
        2: "给出上述技术路线。各研究章依次介绍问题、模型、求解方法及实验。模型构建说明相对第",
        4: "章增加的环境、状态、奖励或数据条件；结果分析结合车头时距、乘客时间与行程完成情况，评价控制效果、影响因素及适用范围。公交迁移在简易仿真与SUMO之间验证，受控连续任务按各自的数据条件分析识别、补偿及模型学习机制。",
    },
}

TABLE_REVISIONS = {
    (0, 1): "运行条件与可用数据",
    (0, 2): "建模与学习的扩展",
    (0, 3): "主要方法",
    (1, 1): "给定随机规律；无预先目标域转移数据",
    (1, 2): "类别共享与随机价值评价",
    (1, 3): "RE-SAC保守价值学习",
    (2, 1): "运行规律变化；利用观测及历史记录",
    (2, 2): "隐含运行模式识别与策略输出约束",
    (3, 2): "公交事件编码与源目标转移差异",
    (4, 1): "公交：固定目标数据与源交互；受控任务另有目标交互",
    (4, 2): "残差修正、直接建模与模型样本加权",
    (4, 3): "MC-WM与置信度归一化",
}


def main():
    with ZipFile(DOCUMENT) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        assert "".join(body[343].itertext()) == "1.4 技术路线"
        assert "".join(body[351].itertext()) == "1.5 论文结构"
        edits = [(body[index], replacements) for index, replacements in PARAGRAPH_REVISIONS.items()]
        rows = body[349].findall(W + "tr")
        for (row, column), text in TABLE_REVISIONS.items():
            paragraphs = rows[row].findall(W + "tc")[column].findall(W + "p")
            assert len(paragraphs) == 1
            nodes = list(paragraphs[0].iter(W + "t"))
            edits.append((paragraphs[0], {i: text if i == 0 else "" for i in range(len(nodes))}))
        updated = original
        mask = lambda raw: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", raw, flags=re.S)
        for paragraph, replacements in edits:
            paragraph_id = paragraph.get(W14 + "paraId")
            pattern = r'<w:p\b[^>]*w14:paraId="' + paragraph_id + r'"[^>]*>.*?</w:p>'
            matches = list(re.finditer(pattern, updated, flags=re.S))
            assert len(matches) == 1, paragraph_id
            match = matches[0]
            revised = replace_text_nodes(match.group(0), replacements)
            # Only prose changes: native objects, fields and formatting stay intact.
            assert mask(match.group(0)) == mask(revised)
            updated = updated[:match.start()] + revised + updated[match.end():]
        new_body = ET.fromstring(updated).find(W + "body")
        assert len(body) == len(new_body)
        changed = [i for i, (a, b) in enumerate(zip(body, new_body)) if ET.tostring(a) != ET.tostring(b)]
        assert changed == list(range(344, 351)), changed
        assert not BACKUP.exists(), "The existing backup must not be overwritten."
        copy2(DOCUMENT, BACKUP)
        temporary = DOCUMENT.with_suffix(".section-edit.docx")
        with ZipFile(temporary, "w") as target:
            for item in source.infolist():
                data = updated.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename)
                target.writestr(item, data)
    temporary.replace(DOCUMENT)
    with ZipFile(BACKUP) as before, ZipFile(DOCUMENT) as after:
        assert before.namelist() == after.namelist()
        assert [name for name in before.namelist() if before.read(name) != after.read(name)] == ["word/document.xml"]
    print("Changed body blocks:", changed)
    print("Only section 1.4 prose changed; other content, native objects and formatting preserved.")
    for index in changed:
        print(index, "".join(n.text or "" for n in new_body[index].iter(W + "t")))
    print("OUTPUT", DOCUMENT)


if __name__ == "__main__":
    main()
