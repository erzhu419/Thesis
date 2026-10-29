from pathlib import Path
from shutil import copy2
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_1_2_1 import DOCUMENT, M, W
from edit_section_3_2 import prose_groups
from edit_section_4_4 import replace_paragraph


BACKUP = Path(__file__).with_name("校订前_7.3.docx")
REVISIONS = {
    1942: (
        "本文以简易仿真到SUMO的跨保真实验检验公交迁移方法，尚未在真实道路上闭环部署。SUMO离线数据辅助简易仿真训练，与真实离线数据辅助SUMO训练，在数据使用方式上相似，但观测误差、驾驶执行和乘客行为并不相同。因此，从跨保真仿真进一步走向实际运营，仍需真实数据和现场试验。",
    ),
    1943: (
        "现有研究也留下了若干算法问题。各章按所研究的任务和可采取的控制动作分别评价，尚未检验全部组件共同使用的效果。类别表征、评论家与演员分歧项、集成规模已有五次独立训练比较，规模间的细微差异及长期训练退化仍需研究。CS-BAPR输出头与BC的独立作用尚未在交叉比较中确定；后续软符号配置的改善，也不能全部归于SINDy。因此，完整方法的平均改善还不足以说明每个组件都不可替代。",
    ),
    1944: (
        "跨保真学习的收益取决于数据、训练和执行条件。H2O+混合训练相对所测纯离线方案的改善已有支持，相对纯源训练及采用共同完成保障的间隔规则，稳定优势尚未建立。独立轨迹诊断表明，比率分类器能够区分源目标数据，但这不等于条件密度比已估计准确，也不能单独证明加权的控制收益。MC-WM受控任务允许新增目标训练交互，公交任务则只用固定目标离线数据和新的源仿真经验。最终控制器比较新增五次MC-WM续训，复用对应的H2O+与纯源策略；相对纯源的配对区间低于零，相对H2O+仍包含零。各组保留自身训练安排，这一完整方案差异不能单独归因于世界模型。",
    ),
    1945: (
        "模型辅助的作用还需结合训练目标和使用方式判断。公交乘客时间结果采用0.25模型损失项系数，已有20个独立初始化；配对区间反映固定测试情形下的训练差异。时距目标补充实验采用0.125系数，只有两个独立初始化。上下文调节的两种扩展未取得一致的额外收益，增加复杂度并未持续改善控制。受控任务也尚未确定相对直接目标MBPO的回报优势。模型拟合、置信更新和样本生成增加训练成本，部署演员结构则不变。各章的数据、转移定义、需求强度、评价时域和统计分母不同，不能直接按数值排名；所测源参数偏差也未覆盖其他路网、OD空间分布和观测条件。",
    ),
    1946: (
        "理论分析同样有明确条件。现有结论针对固定算子、指定动作变换、函数光滑性或数据覆盖范围；鲁棒评价与正则化的精确对应，还要求不确定集合和正则项相匹配，CVaR分析采用指定的轨迹风险包络。网络权重范数和集成分歧是学习中的修正依据，不能直接视为偶然与认知不确定性的方差估计。隐含模式POMDP的分析依赖模式演化和观测假设，BAPR公交实现也没有独立的跨任务元训练外循环。后续需进一步分析策略、价值网络与访问数据同时变化时，函数逼近、优化误差和环境非平稳性如何共同影响学习。",
    ),
    1947: (
        "在真实数据方面，可利用车辆定位、进出站、客流及实际发车记录，按本文的事件决策方式整理离线转移。除样本数量外，还应检查不同驻站动作的覆盖、各类记录的时间对齐和观测误差。先以这些数据标定SUMO，再利用固定真实记录与仿真交互训练策略，并在未用于标定和训练的实际运行日检验预测与控制建议，逐步验证从仿真到现实的数据使用方式。",
    ),
    1948: (
        "现场先记录建议而不干预运行，再开展限定线路、时段和动作范围的受控试验。驻站需考虑驾驶员执行、站台容量、最大允许延误、末班服务与换乘需求，评价候车与车内时间、行程完成和运行间隔，检验仿真收益与代价。",
    ),
    1949: (
        "在非平稳适应方面，可在现有单次与重复冲击实验基础上，分别改变需求变化的幅度、持续时间、发生位置及恢复过程，比较缓慢漂移、突发冲击和重复模式下的识别与响应。公交现有比较中的近期反馈用于训练，评价时仍执行固定策略；下一步可研究运行中更新条件判断的效果，并检验历史窗口长度和推断误差是否造成响应滞后。在积累多个运行日与环境任务后，还可进一步研究跨任务学习适应规则，区分训练稳定化与在线识别各自的作用。",
    ),
    1950: (
        "在世界模型方面，需继续研究哪些预测改进真正改变驻站动作的优劣。事件类别、人数和持续时间的误差可能对候车与车内成本产生不同影响，训练数据未覆盖的状态也需单独检验。可在现有一次拟合与分阶段拟合基础上，采用相同数据和计算预算，比较固定模型类别与重新选择类别的作用；进一步结合动作价值评价预测误差，并同时考察模型样本权重与更新强度，使模型选择更直接地服务于控制。",
    ),
    1951: (
        "上述方向延续本文从鲁棒评价、运行变化适应到目标数据辅助训练与预测修正的递进关系。后续重点是利用真实记录完善状态估计和模型标定，明确各学习机制对控制的作用，再通过受控运营试验检验其实际服务价值。",
    ),
}


def main():
    assert not BACKUP.exists(), "Do not overwrite the pre-edit backup."
    with ZipFile(DOCUMENT) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        assert "".join(body[1941].itertext()) == "7.3 研究展望"
        assert "".join(body[1953].itertext()) == "参考文献"
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
                target.writestr(item, updated.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename))
    temporary.replace(DOCUMENT)
    with ZipFile(BACKUP) as before, ZipFile(DOCUMENT) as after:
        assert before.namelist() == after.namelist()
        assert [n for n in before.namelist() if before.read(n) != after.read(n)] == ["word/document.xml"]
    print({"changed_blocks": changed, "protected_text_nodes": protected_count,
           "backup": str(BACKUP), "output": str(DOCUMENT)})


if __name__ == "__main__":
    main()
