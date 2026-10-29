from pathlib import Path
from shutil import copy2
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_3_2 import M, W, prose_groups
from edit_section_4_4 import replace_paragraph


ROOT = Path(__file__).resolve().parents[2]
DOCUMENT = ROOT / "output/排版完成_逐节校订_第2轮_20261010.docx"
BACKUP = Path(__file__).with_name("校订前_第2轮_4.2.docx")
REVISIONS = {
    996: (
        "以运行模式概括道路速度分布、客流强度及局部交通限制等环境条件。记当前模式为隐变量",
        "，其对应的状态转移与期望奖励为",
    ),
    998: (
        "同一模式内，车辆运行和乘客到达仍有随机波动；模式改变则意味着转移或奖励规律发生变化。控制器无法直接观测全部环境参数，须结合近期运行历史判断条件是否改变，不能将一次车头时距偏差直接视为模式变化。",
    ),
    999: (
        "运行条件可用离散类别或连续参数表示。交通管制的启停可近似为离散切换，平均速度逐渐下降则表现为连续漂移；分布形态变化时，仅估计均值不足以描述新条件。以下采用有限候选模式推导识别过程，公交实验考察固定需求放大、阶跃变化和重复脉冲，未系统检验连续漂移。",
    ),
    1000: (
        "HiP-MDP以潜在因素表示相关环境的动力学差异，UP-OSI利用近期状态动作历史推断系统参数。",
        "这类表示区分即时状态与决定后续转移的环境条件。例如，相同车头时距可能来自一次随机延误，也可能伴随持续需求上升，对后续运行的影响不同。模式变量用于描述后一类尚未直接观测的条件。",
    ),
    1001: (
        "BAPR利用历史证据更新对运行条件的判断。在显式概率模型中，记候选模式集合为",
        "，已有历史下的模式后验概率为",
        "。完成本次动作并获得新观测证据",
        "后，若这段时间内模式保持不变，后验更新为",
    ),
    1003: (
        "式",
        "更新的是对运行模式的判断，尚未直接规定驻站动作；具体动作仍由策略和执行映射确定。",
    ),
    1004: (
        "其中，",
        "表示已有历史和本次动作给定时，候选模式产生新观测的似然，分母须大于零。在线变化点检测则维护自上次变化起的运行长度分布。",
        "模式概率可用于组合不同模式的价值，或调节保守程度。",
        "证明固定权重下的混合转移仍是转移核，",
        "给出相应价值更新的收缩性；",
        "说明权重随价值函数变化时，反馈可能破坏收缩。附录C.5、C.6分别讨论非负惩罚系数的作用方向及模式条件价值的共享表示。",
    ),
    1006: (
        "模式判断需要积累已发生的观测。设环境在",
        "时刻变化，至",
        "时刻形成明确判断，检测延迟",
        "会影响持续需求上升时的初期响应；若冲击持续时间与",
        "接近，判断形成时需求可能已回落。因此，峰值相同但持续时间不同的冲击，控制难度也不同。附录C.8的定理C.3和命题C.5在证据持续有利于新模式的条件下，推导新旧模式后验概率比达到阈值所需的更新步数；换算为实际检测时间还须考虑观测间隔。",
    ),
    1007: (
        "历史窗口的长短影响证据积累。较长窗口有助于减弱个别事件的影响，却会保留较多变化前的记录；较短窗口更侧重近期信息，也更易受随机波动干扰。训练中，策略更新还会改变所访问的状态及其奖励和价值估计。因此，价值分歧上升不一定意味着运行规律改变，也可能反映策略进入了经验较少的状态。",
    ),
    1008: (
        "为减少上述干扰，CS-BAPR的公交训练先预热，之后才启用变化反馈，统计当前策略近期样本的奖励异常与价值分歧，并只响应价值分歧的上升。预热减少初始网络误差的影响，近期样本则使反馈更贴近当前策略访问的状态。反馈调节学习的保守程度，驻站响应仍取决于策略能够表达并学到的动作关系。",
    ),
    1009: (
        "历史信息也可直接参与决策。PEARL和VariBAD将历史形成的任务表示输入策略，模型元学习则用近期数据调整动力学模型。",
        "这些思路都要求说明：当前观测为何不足以判断后续动作结果，新增运行记录又如何改变对环境的认识。下面用部分可观测模型说明这一问题。",
    ),
    1010: (
        "设",
        "为决策前已观测的状态动作历史，",
        "为模式后验。该推导假定运行状态",
        "可见、模式",
        "不可见，策略只能依据已观测历史选择动作，并仅由已完成的状态转移更新模式判断。",
    ),
    1011: (
        "设模式在每次车辆状态转移后按条件概率",
        "演化；给定当前状态、动作和模式后，下一运行状态与下一模式条件独立。若",
        "及各模式的",
        "不随事件序号变化，扩展状态",
        "具有下列平稳转移核",
    ),
    1013: (
        "控制器观察",
        "而不能直接观察模式，故式",
        "对应POMDP表述。离散状态、有限模式下，先用新观测的车辆转移更新本次动作所处模式的概率，再按模式转变规律预测下一模式，其更新为",
    ),
    1015: (
        "式(4-4)的分母为正时，分子非负；对",
        "求和并利用",
        "，可得后验概率之和为1。在这一模型内，更新后的",
        "保留预测后续转移所需的信息。连续观测需使用相应密度；若奖励也提供模式证据，则改用状态与奖励的联合似然。第4.2.1节假定模式保持不变，不含上述预测步骤，",
        ".4给出相应归一化证明。",
    ),
    1016: (
        "进一步从多个任务学习历史到任务表示的映射，再将该表示用于决策，便与上下文元强化学习相联系。",
        "这一联系关注未知环境身份引起的部分可观测性。Ghosh等的认知POMDP视角也说明，即使单个环境的状态完全可见，环境身份未知时仍可能需要利用历史信息决策。",
    ),
    1017: (
        "本章公交实现根据近期奖励和价值统计量形成训练反馈权重，计算见第4.3.1节；该权重调节训练保守程度，不直接给出交通模式的后验概率。评价时，固定策略根据当前运行状态选择驻站动作。受控执行器实验另行检验BAPR的在线模式识别和动作补偿。公交训练不另设跨任务的元学习训练过程，CS-BAPR重点研究变化反馈下的训练稳定化，以及算术输出和符号参考约束的作用。",
    ),
    1018: (
        "上述后验推导以给定模式和转移假设为前提，任意未知漂移未必满足这些条件；规律已知的时间变化也未必需要模式识别。第3章以保守价值评价处理不利后果，本章进一步利用运行历史调节学习响应。第4.3节据此给出训练反馈及策略输出的具体方法。",
    ),
}


def main():
    assert not BACKUP.exists(), "Keep the existing pre-edit backup."
    with ZipFile(DOCUMENT) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        assert "".join(body[994].itertext()) == "4.2 模型构建"
        assert "".join(body[1019].itertext()) == "4.3 求解方法"
        updated = original
        for index, prose in REVISIONS.items():
            updated = replace_paragraph(updated, body[index], prose)

        # Only prose may change; keep equations, fields and layout intact.
        mask = lambda xml: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", xml, flags=re.S)
        assert mask(original) == mask(updated), "Non-text XML changed."
        final = ET.fromstring(updated).find(W + "body")
        assert len(body) == len(final) == 2924
        changed = [i for i, (a, b) in enumerate(zip(body, final)) if ET.tostring(a) != ET.tostring(b)]
        assert changed == list(REVISIONS), changed
        protected_count = 0
        for index in changed:
            old_nodes, _, protected = prose_groups(body[index])
            new_nodes, _, _ = prose_groups(final[index])
            assert all(old_nodes[i].text == new_nodes[i].text for i in protected)
            protected_count += len(protected)
            for tag in (M + "oMath", W + "pPr", W + "hyperlink"):
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
    print({"changed_blocks": changed, "paragraphs": len(changed), "protected_text_nodes": protected_count,
           "backup": str(BACKUP), "output": str(DOCUMENT)})


if __name__ == "__main__":
    main()
