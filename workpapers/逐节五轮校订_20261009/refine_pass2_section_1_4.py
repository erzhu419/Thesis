from pathlib import Path
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_1_2_1 import W, W14
from edit_section_1_2_4 import replace_text_nodes
from edit_section_4_4 import replace_paragraph


ROOT = Path(__file__).resolve().parents[2]
DOCUMENT = ROOT / "output/排版完成_逐节校订_第2轮_20261010.docx"
BACKUP = Path(__file__).with_name("校订前_第2轮_1.4.docx")
TEXT_REVISIONS = {
    344: {
        0: "首先，建立各研究章共用的公交驻站决策模型。",
        1: "在既定线路、班次和车辆供给下，将运行、上下客和附加驻站组织为事件决策，明确动作与后续状态、奖励的对应关系。分别设置车头时距与乘客广义时间成本目标，以运营指标评价效果；价值评价区分运行随机性与有限经验造成的估计不足。",
    },
    345: {
        0: "尚无预先目标域转移训练数据时，先在仿真中学习策略。",
        1: "第3章以类别特征共享车队经验，用RE-SAC改进平稳随机环境下的动作评价。域随机化提供多条件训练思路，鲁棒与尾部风险分析解释保守评价。",
        13: "第4章转向非平稳运行，研究BAPR的条件识别及CS-BAPR的驻站输出结构和符号约束。公交利用近期反馈调节训练，评价时固定策略参数；执行器实验检验在线模式识别与动作补偿。",
    },
    346: {
        0: "最后，利用固定目标离线数据辅助源仿真训练。",
        1: "第5章将H2O+用于公交，混合目标记录与源仿真新增经验更新策略，分别检验训练和行程完成保障的作用。第6章针对源仿真标定不足，用MC-WM比较残差修正与直接目标建模，由未参与拟合的数据选择模型，按预测可信度加权模型样本更新价值。对置信度递推更新作归一化，缓解模型权重过度减小。公交训练仅追加源仿真交互。",
    },
}
PROSE_REVISIONS = {
    347: "理论分析说明方法的成立条件，仿真实验检验控制效果。表1-2汇总各章运行条件、可用数据与方法重点。",
    350: "给出技术路线。公交实验比较两类奖励的运营效果及标定、执行规则的影响；受控任务检验模式识别、动作补偿与模型学习机制。",
}
TABLE_REVISIONS = {
    (1, 1): "平稳随机运行；无预先目标训练数据",
    (1, 2): "共享学习与保守价值评价",
    (2, 2): "条件识别与策略输出约束",
    (2, 3): "CS-BAPR及BAPR",
    (3, 1): "目标数据固定；源可交互",
    (3, 2): "事件编码与混合经验学习",
    (4, 1): "公交目标数据固定、源交互；受控任务另有目标交互",
    (4, 2): "模型选择与样本加权",
}


def main():
    with ZipFile(DOCUMENT) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        updated = original
        for index, changes in TEXT_REVISIONS.items():
            paragraph_id = body[index].get(W14 + "paraId")
            matches = list(re.finditer(r'<w:p\b[^>]*w14:paraId="' + paragraph_id + r'"[^>]*>.*?</w:p>', updated, re.S))
            assert len(matches) == 1
            match = matches[0]
            revised = replace_text_nodes(match.group(0), changes)
            updated = updated[:match.start()] + revised + updated[match.end():]
        for index, text in PROSE_REVISIONS.items():
            updated = replace_paragraph(updated, body[index], (text,))
        rows = body[349].findall(W + "tr")
        for (row, column), text in TABLE_REVISIONS.items():
            paragraph = rows[row].findall(W + "tc")[column].find(W + "p")
            updated = replace_paragraph(updated, paragraph, (text,))
        mask = lambda xml: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", xml, flags=re.S)
        assert mask(original) == mask(updated)
        with ZipFile(BACKUP) as backup:
            initial = backup.read("word/document.xml").decode("utf-8")
            assert mask(initial) == mask(updated)
            before = ET.fromstring(initial).find(W + "body")
            after = ET.fromstring(updated).find(W + "body")
            changed = [i for i, (a, b) in enumerate(zip(before, after)) if ET.tostring(a) != ET.tostring(b)]
            assert changed == [344, 345, 346, 347, 349, 350], changed
        temporary = DOCUMENT.with_suffix(".tmp")
        with ZipFile(temporary, "w") as target:
            for item in source.infolist():
                target.writestr(item, updated.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename))
    temporary.replace(DOCUMENT)
    print({"changed_blocks_from_backup": changed, "bold_topic_sentences_preserved": True})


if __name__ == "__main__":
    main()
