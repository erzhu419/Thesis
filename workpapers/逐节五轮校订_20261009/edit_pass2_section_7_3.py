from pathlib import Path
from shutil import copy2
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_3_2 import W
from edit_section_4_4 import replace_paragraph


ROOT = Path(__file__).resolve().parents[2]
DOCUMENT = ROOT / "output/排版完成_逐节校订_第2轮_20261010.docx"
BACKUP = Path(__file__).with_name("校订前_第2轮_7.3.docx")
REVISIONS = {
    1942: (
        "本文通过简易仿真到SUMO的跨保真实验检验公交迁移方法，尚未在真实道路闭环部署。SUMO离线数据辅助简易仿真训练，与真实离线数据辅助SUMO训练，数据使用方式相似，但观测误差、驾驶执行和乘客行为不同。走向实际运营仍需真实记录检验预测、现场试验评价控制效果。",
    ),
    1943: (
        "算法机制方面，尚需明确各组件在什么条件下有效，以及组合后能否相互增强。各章按相应任务和控制动作分别评价，未检验全部组件共同使用的效果。类别表征、评论家与演员分歧项、集成规模已有五次独立训练比较，规模间的细微差异及长期训练退化仍待解释。CS-BAPR驻站输出结构与BC的独立作用尚未在交叉比较中确定，后续软符号配置的改善也不能全部归于SINDy。完整方法的平均改善不说明每个组件均不可替代，后续可在统一任务中进一步比较组件单独与组合的作用。",
    ),
    1944: (
        "目标数据辅助训练方面，H2O+混合方案相对所测纯离线方案已有改善，相对纯源训练及采用共同完成保障的间隔规则，稳定优势尚未建立。独立轨迹上的分类表现也不等于条件密度比估计准确，不能单独证明加权带来控制收益。MC-WM受控任务允许新增目标训练交互，公交则只用固定目标离线数据与新增源经验。最终控制器比较新增五次MC-WM续训，复用对应的H2O+与纯源策略；相对纯源的配对差置信区间低于零，相对H2O+的区间仍包含零。各组保留自身训练安排，不能单独判断世界模型的增量作用。后续应在相同数据、预算与执行规则下，分别检验比率修正和模型辅助的作用。",
    ),
    1945: (
        "模型作用还需在更多训练目标和运行条件下检验。公交乘客时间实验采用0.25模型损失项系数，包含20个独立初始化，配对区间反映固定测试情形下的训练差异；时距目标采用0.125系数，仅有两个初始化。按识别条件调节系数的两种扩展未取得一致额外收益，受控任务相对直接目标MBPO的回报优势也未确定。模型拟合、置信更新和样本生成增加训练成本，部署演员结构则不变。后续可在共同任务中比较服务改善与计算代价，并检验其他路网、OD空间分布和观测条件。各章数据、转移定义、需求强度、评价时域与统计分母不同，现有数值不直接用于跨章排名。",
    ),
    1946: (
        "理论分析需进一步接近实际训练过程。现有结论以固定算子、指定动作变换、函数光滑性或数据覆盖范围为条件；鲁棒评价与正则化精确对应，要求不确定集合与正则项匹配，CVaR结论也限于所设轨迹风险包络。网络权重范数与集成分歧用于修正学习，不能直接作为偶然与认知不确定性的方差估计。运行模式不可直接观测的POMDP分析依赖模式演化和观测假设，BAPR公交实现也未开展独立的跨任务元训练。后续需分析策略、价值网络与访问数据同时变化时，函数逼近、优化误差及环境非平稳性如何共同影响学习。",
    ),
    1947: (
        "真实数据验证可从车辆定位、进出站、客流和发车记录入手，按事件决策方式整理离线转移。除样本数量外，还需检查驻站动作覆盖、记录时间对齐及观测误差。先据此标定SUMO，再结合固定真实记录与仿真交互训练策略；以未用于标定和训练的运行日记录检验预测，分析控制建议的可执行性。",
    ),
    1948: (
        "现场验证可先记录策略建议而不干预运行，再在限定线路、时段和动作范围内开展受控试验。驻站决策需考虑驾驶员执行、站台容量、最大允许延误、末班服务及换乘需求。评价应同时考察候车与车内时间、行程完成和运行间隔，判断仿真中的服务改善及相应代价是否在实际运营中出现。",
    ),
    1949: (
        "非平稳适应可在现有单次与重复冲击实验基础上，分别改变需求变化的幅度、持续时间、发生位置和恢复过程，比较缓慢漂移、突发冲击与重复模式下的识别和响应。公交近期反馈目前用于训练调节，评价时执行固定策略；后续可研究运行中持续更新条件判断的效果，检验历史窗口长度与推断误差是否造成响应滞后。积累多个运行日和环境任务后，还可研究跨任务学习适应规则，分别评价训练稳定化与在线识别的贡献。",
    ),
    1950: (
        "世界模型研究需进一步回答：哪些预测改进能够改变驻站动作的优劣？事件类别、人数和持续时间的误差对候车及车内成本可能有不同影响，训练数据未覆盖的状态也需单独检验。可在现有一次拟合与分阶段拟合基础上，采用相同数据和计算预算，比较固定模型类别与重新选择类别的作用；再结合动作价值评价预测误差，同时考察模型样本权重和更新强度，使模型选择更直接地服务于驻站控制。",
    ),
    1951: (
        "上述研究方向延续本文从鲁棒评价、运行变化适应到目标数据辅助训练与预测修正的递进关系。后续重点是以真实记录完善状态估计与模型标定，明确各学习机制在何种条件下改善控制，再通过受控运营试验检验实际服务价值。",
    ),
}


def main():
    if not BACKUP.exists():
        copy2(DOCUMENT, BACKUP)
    with ZipFile(BACKUP) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        assert "".join(body[1941].itertext()) == "7.3 研究展望"
        assert "".join(body[1953].itertext()) == "参考文献"
        with ZipFile(DOCUMENT) as current:
            current_body = ET.fromstring(current.read("word/document.xml")).find(W + "body")
            assert len(current_body) == len(body)
            assert all(
                ET.tostring(a) == ET.tostring(b)
                for i, (a, b) in enumerate(zip(body, current_body))
                if i not in REVISIONS
            )
        updated = original
        for index, prose in REVISIONS.items():
            updated = replace_paragraph(updated, body[index], prose)

        # Keep equations, fields, objects and formatting outside prose intact.
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
