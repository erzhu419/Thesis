from pathlib import Path
from shutil import copy2
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_3_2 import DOCUMENT, M, W, W14, prose_groups, revise


BACKUP = Path(__file__).with_name("校订前_4.2.docx")
REVISIONS = {
    996: (
        "以运行模式概括道路速度分布、客流强度及局部交通限制等环境条件。用隐变量",
        "表示当前模式，状态转移与期望奖励为",
    ),
    998: (
        "同一模式内仍有车辆运行和乘客到达的随机波动；模式变化则表示转移或奖励规律改变。控制器不能直接观测全部环境参数，需要结合运行历史判断当前条件，而不是将某一次状态偏差直接视为模式变化。",
    ),
    999: (
        "运行条件可用离散类别或连续参数表示。平均速度逐渐下降对应连续漂移，交通管制开始或结束可近似为离散切换；分布形态改变时，仅更新均值可能不足。本章公交实验考察固定需求放大、阶跃变化和重复脉冲，后文推导采用有限候选模式。连续漂移可由一般框架描述，本章未系统检验。",
    ),
    1000: (
        "HiP-MDP以潜在因素表示相关环境的动力学差异，UP-OSI利用近期状态动作历史推断系统参数。",
        "这些思路提示：相同的当前车头时距，可能对应常规波动或持续需求上升，后续服务结果却不同。模式变量用于区分这种尚未直接观测的运行条件与车辆即时状态。",
    ),
    1001: (
        "BAPR利用历史证据更新对运行条件的判断。在显式概率模型中，记候选模式集合为",
        "，模式后验概率为",
        "。获得新的观测证据",
        "后，若候选模式本身不发生迁移，后验更新为",
    ),
    1003: (
        "式",
        "更新的是模式判断，具体控制动作仍由策略和执行映射确定。",
    ),
    1004: (
        "其中，",
        "为候选模式下的证据似然，分母须大于零。在线变化点检测维护自上次变化起的运行长度分布。",
        "模式后验可加权价值或调节保守系数，而非每次观测后切换策略。",
        "证明固定权重的模式混合是转移核，",
        "给出价值更新的收缩性；",
        "说明权重随价值函数变化时，反馈可能破坏收缩。附录C.5、C.6分别分析非负惩罚的系数方向和模式条件价值的共享表示。",
    ),
    1006: (
        "历史识别只能利用已发生的观测。设环境在",
        "时刻变化，至",
        "时刻形成明确判断，检测延迟",
        "影响持续需求上升的初期响应；若冲击持续时间与",
        "接近，判断形成时需求可能已回落。相同峰值因此不代表相同控制难度。附录C.8的定理C.3和命题C.5在逐步证据持续有利于新模式的条件下，推导新旧模式后验概率比达到阈值所需的更新步数；实际检测时间还取决于观测间隔。",
    ),
    1007: (
        "历史保留长度也影响识别。较长窗口有助于减弱随机波动的影响，却可能混合变化前后的记录；较短窗口更侧重近期信息，但更易受个别车辆或站点事件影响。策略更新也会改变奖励和价值分歧。因此，训练中的分歧上升既可能来自环境变化，也可能来自策略访问了新状态，需要结合训练阶段判断。",
    ),
    1008: (
        "CS-BAPR的公交训练先进行预热，再统计当前策略近期样本的奖励异常与价值分歧。预热结束后才启用变化反馈，以减少初始网络误差的干扰；近期样本反映当前策略访问的状态，价值分歧反馈只响应其上升。这些安排控制反馈如何参与学习，新需求下的驻站响应仍取决于策略能够表达和学到的动作关系。",
    ),
    1009: (
        "历史信息也可直接参与决策：PEARL和VariBAD将任务表示作为策略输入，模型元学习用近期数据调整动力学模型。",
        "本章公交方法则调节训练保守程度与策略输出，不重新识别完整环境模型；通过需求幅度及变化日程对照检验控制效果。",
    ),
    1010: (
        "为说明未观测的运行模式如何形成部分可观测性，设",
        "为已观测的状态动作历史，",
        "为模式后验。以下假定运行状态",
        "可见、模式",
        "不可见，并仅以已发生的状态转移作为识别证据。",
    ),
    1011: (
        "设模式在每次车辆状态转移后按",
        "演化；给定当前状态、动作和模式后，下一运行状态与下一模式条件独立。若",
        "及各模式的",
        "不随事件序号变化，扩展状态",
        "具有平稳转移核",
    ),
    1013: (
        "控制器仅观察",
        "而不能直接观察模式，故式",
        "构成POMDP表述。对于离散状态与有限模式，先根据观测到的转移更新原模式的概率，再按模式迁移规律得到下一模式的后验。具体更新为",
    ),
    1015: (
        "式(4-4)的分母为正时，分子非负；对",
        "求和并利用",
        "，可得后验概率之和为1。在这一模型内，更新后的",
        "保留预测后续转移所需的信息。连续观测需使用相应密度；若奖励也提供模式证据，则改用状态与奖励的联合似然。第4.2.1节的固定模式更新不含模式迁移预测，",
        ".4给出其归一化证明。",
    ),
    1016: (
        "若进一步从多个任务学习历史到模式表示的映射，再据此决策，就与上下文元强化学习相联系。PEARL的概率上下文和VariBAD的近似任务后验提供了相应实现。",
        "在公交中，这类表示可用于判断近期转移是否提示拥堵或需求变化。Ghosh等的认知POMDP视角也说明，即使单个环境的状态完全可见，未知环境身份仍可造成部分可观测性。",
    ),
    1017: (
        "BAPR据此利用历史信息调节保守程度或补偿动作。公交实现由近期奖励和价值统计量形成训练反馈权重，其计算见第4.3.1节，不等同于上述交通模式后验；评价时执行固定策略。受控执行器实验另行检验在线模式识别和动作补偿。公交训练未设置独立的跨任务元训练外循环，CS-BAPR的重点是变化反馈下的训练稳定化与算术输出关系。",
    ),
    1018: (
        "上述后验推导依赖给定的模式表示和转移假设，任意未知漂移并不自动满足有限维马尔可夫表示；已知的时间变化也不必造成模式识别困难。第3章以保守价值评价处理不利后果，本章进一步利用历史信息判断运行条件、调节学习响应。第4.3节据此给出训练反馈及策略输出的具体方法。",
    ),
}


def main():
    with ZipFile(DOCUMENT) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        assert "".join(body[994].itertext()) == "4.2 模型构建"
        assert "".join(body[1019].itertext()) == "4.3 求解方法"
        updated = original
        for index, prose in REVISIONS.items():
            paragraph_id = body[index].get(W14 + "paraId")
            pattern = r'<w:p\b[^>]*w14:paraId="' + paragraph_id + r'"[^>]*>.*?</w:p>'
            matches = list(re.finditer(pattern, updated, flags=re.S))
            assert len(matches) == 1
            match = matches[0]
            revised = revise(match.group(0), body[index], prose)
            updated = updated[:match.start()] + revised + updated[match.end():]

        mask = lambda raw: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", raw, flags=re.S)
        assert mask(original) == mask(updated), "Non-text XML changed."
        new_body = ET.fromstring(updated).find(W + "body")
        assert len(body) == len(new_body)
        changed = [i for i, (a, b) in enumerate(zip(body, new_body)) if ET.tostring(a) != ET.tostring(b)]
        assert set(changed).issubset(REVISIONS), changed
        protected_count = 0
        for index in changed:
            old_nodes, _, protected = prose_groups(body[index])
            new_nodes, _, _ = prose_groups(new_body[index])
            assert all(old_nodes[i].text == new_nodes[i].text for i in protected)
            protected_count += len(protected)
            for tag in (M + "oMath", W + "pPr", W + "hyperlink"):
                assert [ET.tostring(n) for n in body[index].iter(tag)] == [ET.tostring(n) for n in new_body[index].iter(tag)]
        if not BACKUP.exists():
            copy2(DOCUMENT, BACKUP)
        temporary = DOCUMENT.with_suffix(".section-edit.docx")
        with ZipFile(temporary, "w") as target:
            for item in source.infolist():
                data = updated.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename)
                target.writestr(item, data)
    temporary.replace(DOCUMENT)
    with ZipFile(BACKUP) as before, ZipFile(DOCUMENT) as after:
        assert before.namelist() == after.namelist()
        changed_parts = [n for n in before.namelist() if before.read(n) != after.read(n)]
        assert changed_parts == ["word/document.xml"], changed_parts
        prior_body = ET.fromstring(before.read("word/document.xml")).find(W + "body")
        final_changed = [i for i, (a, b) in enumerate(zip(prior_body, new_body)) if ET.tostring(a) != ET.tostring(b)]
        assert final_changed == list(REVISIONS), final_changed
    print({"changed_blocks": changed, "protected_text_nodes": protected_count,
           "changed_parts": changed_parts, "backup": str(BACKUP)})
    for index in changed:
        print(index, "".join(n.text or "" for n in new_body[index].iter() if n.tag in (W + "t", M + "t")))


if __name__ == "__main__":
    main()
