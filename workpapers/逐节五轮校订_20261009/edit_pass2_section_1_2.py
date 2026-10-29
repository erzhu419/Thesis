from pathlib import Path
from shutil import copy2
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_1_2_1 import M, W
from edit_section_3_2 import prose_groups
from edit_section_4_4 import replace_paragraph


ROOT = Path(__file__).resolve().parents[2]
DOCUMENT = ROOT / "output/排版完成_逐节校订_第2轮_20261010.docx"
BACKUP = Path(__file__).with_name("校订前_第2轮_1.2.docx")
REVISIONS = {
    278: (
        "两类方法并非在数学上截然分离。神经网络策略仍由有限参数表示，仿真优化也可以搜索参数化策略；强化学习通过动作价值估计和经验复用，将当前动作与后续运行收益联系起来。其控制效果取决于训练经验覆盖了哪些状态和动作，以及价值估计是否适用于使用环境。下面从公交强化学习研究出发，进一步讨论随机后果下的动作评价和运行变化后的策略适应。",
    ),
    295: (
        "分位数回归和隐式分位数网络分别学习分位点与分位函数，用于按回报分布评价风险。",
        "截断分位数评价器方法（Truncated Quantile Critics，TQC）结合分位数表示、截断和集成缓解价值高估；评价器多样化集成方法（Ensemble-Diversified Actor-Critic，EDAC）在离线学习中增强网络差异，以处理数据覆盖之外的动作。",
        "回报分布刻画随机结果，集成分歧反映估计差异。本文用集成保守评价选择驻站动作，不拟合完整回报分布。",
    ),
    300: (
        "环境识别方法利用已发生的状态转移推断隐含运行条件。隐参数马尔可夫决策过程（Hidden Parameter Markov Decision Process，HiP-MDP）以潜在参数描述相关任务的动力学差异；通用策略与在线系统辨识（Universal Policy with Online System Identification，UP-OSI）根据近期状态动作历史推断系统参数，再将其输入策略。",
        "概率嵌入式演员评论家方法（Probabilistic Embeddings for Actor-Critic Reinforcement Learning，PEARL）用概率上下文区分任务推断与控制；变分贝叶斯自适应深度强化学习（Variational Bayes-Adaptive Deep Reinforcement Learning，VariBAD）将任务不确定性用于动作选择；快速运动适应方法（Rapid Motor Adaptation，RMA）结合基本策略与适应模块。",
        "作者的BAPR前期研究（Bayesian Amnesic Piecewise-Robust）结合环境识别、模式后验更新与保守价值学习。",
        "本文沿此基础研究贝叶斯自适应策略正则化（Bayesian Adaptive Policy Regularization，BAPR），并进一步提出因果符号贝叶斯自适应策略正则化（Causal-Symbolic Bayesian Adaptive Policy Regularization，CS-BAPR）",
        "。公交实现以近期奖励和价值分歧调节训练保守程度，评价时固定策略；CS-BAPR进一步约束驻站输出结构与状态响应。受控执行器实验则检验在线模式识别与动作补偿。",
    ),
    302: (
        "因果建模研究关注环境关系对强化学习控制的作用",
        "；符号辨识学习可写成公式的关系。非线性动力系统稀疏辨识（Sparse Identification of Nonlinear Dynamics，SINDy）在候选函数中选择少量有效项，拟合动力学；不变风险最小化（Invariant Risk Minimization，IRM）研究在多个环境中保持一致的预测关系。",
        "本文据此用符号参考关系约束驻站动作及其对状态的响应。附录D分析参考拟合与策略导数误差，以及偏离参考状态后的动作误差。公交实验检验参考关系是否改善控制；稀疏拟合不等于因果识别。",
    ),
    304: (
        "演员评论家以连续动作表示驻站时长，并根据经验评价后续收益。本文先研究未预先获得目标域转移训练数据时的保守评价与非平稳适应，再研究固定目标数据如何辅助源仿真训练。下一小节讨论数据利用与仿真迁移。",
    ),
    314: (
        "作者与合作者的公交跨保真研究",
        "将H2O+用于公交驻站，并调整状态表示、动作分布和执行方式。车辆可能同时到站，必要上下客结束后可以选择不额外驻站，当前动作还会影响后续班次和规定时域内的服务。因此，本文按同时到站车辆组织状态，以混合动作分布分别表示不额外驻站的选择概率和连续驻站时长，并设置驻站幅度校准与行程完成保障。服务效果用人均广义时间成本和行程完成率评价，时间统计包括全部已出发乘客在规定时域内已发生的候车和车内时间。第5章分别比较固定数据学习、追加目标交互，以及结合目标记录与源仿真交互的训练。",
    ),
    315: (
        "价值学习与动力学修正作用于不同环节，可以结合。CQL以价值正则抑制缺少数据支持的动作高估，IQL在价值更新中避免查询数据未覆盖动作，Cal-QL使离线初始化的价值保持适当尺度；动力学重要性采样则调整源转移对价值学习的贡献。H2O+组合离线与在线价值更新，并可用转移比率修正源经验。第5章区分启用比率修正的比较方案与采用单位源权重的正式控制器，分别评价训练效果。",
    ),
    317: (
        "机器人虚实迁移综述从目标模型优化、源知识迁移和策略迭代梳理迁移路径",
        "；深度强化学习知识迁移综述",
        "与机器人学习综述",
        "提供了相应的方法和任务背景。针对动力学差异，Tan等结合执行器建模、延迟模拟与物理参数随机化开展机器人迁移；自动域随机化调整训练环境参数分布；仿真参数优化方法SimOpt则利用目标运行经验更新仿真参数分布。",
        "相关综述将域随机化、域适应与元学习归纳为不同迁移路径。",
        "它们体现两种思路：缺少目标运行记录时，扩大源训练覆盖的变化范围；获得目标记录后，用这些数据修正源目标差异。本文第5、6章研究后一条件下，固定目标数据如何辅助低成本源仿真训练。",
    ),
    327: (
        "上述研究表明，公交驻站学习既需处理当前运行规律下的随机后果，也需应对规律变化；能否预先获得目标运行记录，又决定了可采用的修正方法。本文按这两方面组织研究。",
        "对比获得固定目标数据后可采用的代表方法，说明第5、6章通用学习原理与本文公交适配的关系。",
    ),
    331: (
        "第一，未预先获得目标域转移训练数据、给定条件下的随机规律保持不变时，如何从仿真经验中学到有效的驻站策略？不同车辆、站点和时段的经验量不均衡，共享参数仍需区分控制对象；有限经验还可能使动作价值估计失真。第3章以类别特征嵌入保留对象差异，以RE-SAC改进保守价值评价，并通过表征与价值评价对照检验控制效果。",
    ),
    332: (
        "第二，客流或路况规律发生变化后，如何判断当前条件并作出合适响应？非平稳变化不同于原有规律下的随机扰动，仅凭一次观测难以区分二者，近期历史也可能受噪声和识别滞后影响。第4章从BAPR的条件识别出发，研究CS-BAPR的训练反馈、稳定化设计和驻站输出约束，以需求突变及反复变化检验时距控制与乘客服务。",
    ),
    333: (
        "第三，获得固定目标离线数据后，如何在不追加目标训练交互的条件下，利用源仿真继续改进策略？目标记录包含已有动作的运行后果，却不能覆盖所有新动作；源仿真可以提供新交互，后继结果又可能偏离目标环境。第5章将H2O+适配于公交，结合目标离线初始化与源仿真续训，并分别比较训练方案和行程完成保障的服务作用。",
    ),
    334: (
        "第四，源仿真标定不足时，如何用目标记录修正预测并辅助策略学习？经验重加权不直接修正源预测。第6章以MC-WM比较残差修正与直接目标建模，用预测误差构造置信权重，调节一步模型样本对价值学习的贡献，再通过标定偏差和模型使用方式对照检验乘客服务效果。",
    ),
}


def main():
    assert not BACKUP.exists(), "Keep the pre-edit backup."
    with ZipFile(DOCUMENT) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        assert "".join(body[262].itertext()) == "1.2 国内外研究综述"
        assert "".join(body[335].itertext()) == "1.3 研究目标与内容"
        assert "".join(body[2169].itertext()).startswith("附录A")
        updated = original
        for index, prose in REVISIONS.items():
            updated = replace_paragraph(updated, body[index], prose)

        # Preserve all formatting, equations and reference structures.
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
        with ZipFile(DOCUMENT.with_suffix(".tmp"), "w") as target:
            for item in source.infolist():
                target.writestr(item, updated.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename))
    DOCUMENT.with_suffix(".tmp").replace(DOCUMENT)
    with ZipFile(BACKUP) as before, ZipFile(DOCUMENT) as after:
        assert before.namelist() == after.namelist()
        assert [n for n in before.namelist() if before.read(n) != after.read(n)] == ["word/document.xml"]
    print({"changed_blocks": changed, "protected_text_nodes": protected_count,
           "backup": str(BACKUP), "output": str(DOCUMENT)})


if __name__ == "__main__":
    main()
