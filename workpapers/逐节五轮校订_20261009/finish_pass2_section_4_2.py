from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_pass2_section_4_2 import BACKUP, DOCUMENT, REVISIONS
from edit_section_3_2 import W, prose_groups
from edit_section_4_4 import replace_paragraph


SHORTENED = {
    1000: (
        "HiP-MDP以潜在因素表示相关环境的动力学差异，UP-OSI利用近期状态动作历史推断系统参数。",
        "这类表示区分即时状态与决定后续转移的环境条件。例如，相同车头时距可能来自一次随机延误，也可能伴随持续需求上升，对后续运行的影响不同。模式变量描述尚未直接观测的运行条件，而不替代车辆即时状态。",
    ),
    1006: (
        "设环境在",
        "时刻变化，至",
        "时刻形成明确模式判断，检测延迟",
        "影响初期响应；若冲击持续时间与",
        "接近，判断形成时需求可能已回落。因此，峰值相同的冲击，其控制难度仍受持续时间影响。附录C.8的定理C.3和命题C.5在证据持续有利于新模式的条件下，推导新旧模式后验概率比达到阈值所需的更新步数；实际检测时间还取决于观测间隔。",
    ),
    1007: (
        "历史窗口越长，越能减弱个别事件的影响，但也会保留较多变化前的记录；较短窗口更侧重近期信息，却更易受随机波动干扰。策略更新还会改变所访问的状态及其奖励和价值估计，故训练中价值分歧上升也可能来自新状态的经验不足，而非运行规律改变。",
    ),
    1008: (
        "CS-BAPR先预热，再启用变化反馈，以减少初始网络误差的干扰。反馈来自当前策略近期样本的奖励异常与价值分歧，其中分歧项只响应上升。反馈调节训练保守程度，策略还需从运行经验中学习驻站响应。",
    ),
    1009: (
        "PEARL和VariBAD将历史形成的任务表示输入策略，模型元学习则用近期数据调整动力学模型。",
        "下面用部分可观测模型说明：当前状态为何不足以判断后续动作结果，新增运行记录又如何改变模式判断。",
    ),
    1017: (
        "公交实现以近期奖励和价值统计量形成训练反馈权重，计算见第4.3.1节；该权重调节保守程度，并非交通模式的后验概率。评价时，固定策略根据当前状态选择驻站动作。受控执行器实验另行检验BAPR的在线模式识别和动作补偿。公交训练不另设跨任务元学习过程。CS-BAPR研究变化反馈下的训练稳定化，以及算术输出和符号参考约束的作用。",
    ),
}


with ZipFile(DOCUMENT) as source:
    original = source.read("word/document.xml").decode("utf-8")
    body = ET.fromstring(original).find(W + "body")
    updated = original
    for index, prose in SHORTENED.items():
        updated = replace_paragraph(updated, body[index], prose)
    mask = lambda xml: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", xml, flags=re.S)
    assert mask(original) == mask(updated)
    with ZipFile(BACKUP) as before:
        prior = before.read("word/document.xml").decode("utf-8")
        initial = ET.fromstring(prior).find(W + "body")
        final = ET.fromstring(updated).find(W + "body")
        assert mask(prior) == mask(updated)
        assert len(initial) == len(final) == 2924
        changed = [i for i, (a, b) in enumerate(zip(initial, final)) if ET.tostring(a) != ET.tostring(b)]
        assert changed == list(REVISIONS)
        for index in changed:
            old_nodes, _, protected = prose_groups(initial[index])
            new_nodes, _, _ = prose_groups(final[index])
            assert all(old_nodes[i].text == new_nodes[i].text for i in protected)
            assert re.findall(r"\d+(?:\.\d+)?", "".join(initial[index].itertext())) == re.findall(r"\d+(?:\.\d+)?", "".join(final[index].itertext()))
    temporary = DOCUMENT.with_suffix(".tmp")
    with ZipFile(temporary, "w") as target:
        for item in source.infolist():
            target.writestr(item, updated.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename))
temporary.replace(DOCUMENT)
with ZipFile(BACKUP) as before, ZipFile(DOCUMENT) as after:
    assert before.namelist() == after.namelist()
    assert [n for n in before.namelist() if before.read(n) != after.read(n)] == ["word/document.xml"]
print({"shortened_blocks": list(SHORTENED), "final_changed_blocks": changed})
