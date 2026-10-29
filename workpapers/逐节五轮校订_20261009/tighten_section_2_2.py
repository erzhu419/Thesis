from pathlib import Path
from shutil import copy2
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_1_2_1 import DOCUMENT, W, M, W14, revise_paragraph


BACKUP = Path(__file__).with_name("2.2文字初改.docx")
REVISIONS = {
    441: (
        "成本单位为加权后的乘客秒，候车权重为2，车内时间权重为1。人数对时间的积分累计各乘客已发生的时间，按事件间的实际持续时间结算，而非按事件次数计量；奖励为成本的负值。",
    ),
    466: (
        "折扣任务通常采用",
        "；按实际时间折扣时，需指定时间单位，再根据",
        "计算延续因子。公交H2O+无折扣累计终点前的乘客成本，事件间隔直接进入奖励；第3章固定算子分析则采用严格折扣。奖励、折扣和终止规则共同决定任务目标。",
    ),
    473: (
        "各车在自己的决策时刻调用共享演员，其记录进入共同回放；动作控制当前车辆，训练利用车队经历。类别特征区分服务身份，连续量描述运行条件。第3章改进保守价值评价，第4章利用近期证据调节并约束策略输出，第5章结合目标数据与源交互，第6章加入模型辅助价值更新。",
    ),
    491: (
        "左式的优化变量为成本阈值，正部函数只计入超过阈值的成本；右式的优化变量为非负的概率重加权系数，其期望为1，上限为尾部概率的倒数。当",
        "时，CVaR评价最差5%的平均成本；分位点处有相同成本的结果时，由优化式确定计入的概率份额。当",
        "时，恢复期望成本。右式将CVaR表示为受限概率重加权后的最坏期望，给出其分布鲁棒解释。",
        "期望成本关注整体平均，CVaR增加不利结果的权重；它们是不同的优化目标，也不同于用于表示估计不确定性的统计置信区间。",
    ),
}


def main():
    with ZipFile(DOCUMENT) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        updated = original
        for index, prose in REVISIONS.items():
            paragraph_id = body[index].get(W14 + "paraId")
            pattern = r'<w:p\b[^>]*w14:paraId="' + paragraph_id + r'"[^>]*>.*?</w:p>'
            matches = list(re.finditer(pattern, updated, flags=re.S))
            assert len(matches) == 1
            match = matches[0]
            revised = revise_paragraph(match.group(0), body[index], prose)
            mask = lambda text: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", text, flags=re.S)
            assert mask(match.group(0)) == mask(revised)
            updated = updated[:match.start()] + revised + updated[match.end():]
        assert not BACKUP.exists()
        copy2(DOCUMENT, BACKUP)
        temporary = DOCUMENT.with_suffix(".section-edit.docx")
        with ZipFile(temporary, "w") as target:
            for item in source.infolist():
                data = updated.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename)
                target.writestr(item, data)
    temporary.replace(DOCUMENT)
    print("Refined section 2.2 paragraphs:", list(REVISIONS))


if __name__ == "__main__":
    main()
