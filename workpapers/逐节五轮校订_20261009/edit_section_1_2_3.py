from pathlib import Path
from shutil import copy2
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_1_2_1 import DOCUMENT, W, M, W14, revise_paragraph


BACKUP = Path(__file__).with_name("校订前_1.2.3.docx")

# Prose segments retain the positions of the existing citation objects.
REVISIONS = {
    306: (
        "仿真到现实（sim-to-real transfer）研究在源环境训练的策略如何用于目标环境；离线、在线及离线到在线则区分训练时是否追加环境交互。两者分别描述迁移任务和数据获取方式，不能直接并列为算法类别。Levine等围绕固定数据中的分布偏移，讨论重要性采样、策略约束、保守价值学习和模型学习；Prudencio等区分算法结构与抑制分布偏移的机制；Zhao等则从域随机化、域适应及迁移梳理仿真到现实研究。",
        "据此，本文先区分三种训练条件：仅使用固定目标数据、允许追加目标交互，以及固定目标数据与新增源仿真交互相结合，再比较各类修正方法的作用。",
    ),
    307: (
        "首先考虑仅使用既有运行记录的情况。纯离线强化学习从固定数据学习策略，训练时不再与目标环境交互，相关综述按方法和应用梳理这一数据条件。",
        "具体方法包括通过优先采样提高已有经验的利用效率",
        "，以及结合扩散模型与期望最大化学习离线策略",
        "。学习策略可能选择数据采集策略很少采用的动作，使价值估计缺少对应经验。保守Q学习（Conservative Q-Learning，CQL）通过价值正则抑制对这类动作的乐观估计；隐式Q学习（Implicit Q-Learning，IQL）先评价数据中的已有动作，再通过优势加权的行为克隆（Behavior Cloning，BC）提取策略。BC依据已有状态与动作记录学习动作输出，优势加权则使评价较好的动作获得更大模仿权重；IQL由此避免在关键价值更新中查询数据未覆盖动作的价值。",
        "两者分别通过保守价值约束和更新方式，处理固定数据之外的动作估计问题。",
    ),
    308: (
        "固定数据学习还需控制估计误差在后续价值更新中的反复传递。离线强化学习综述将分布偏移及自举误差积累作为核心问题。",
        "批量约束Q学习（Batch-Constrained Q-Learning，BCQ）使候选动作接近已有数据中的动作；自举误差累积削减（Bootstrapping Error Accumulation Reduction，BEAR）通过限制动作选择减小这一误差；行为正则化演员评论家方法（Behavior Regularized Actor Critic，BRAC）统一讨论不同的行为正则形式。",
        "这些方法主要限制策略偏离已有动作的程度，CQL则将保守性直接用于价值学习。公交离线记录中，不额外驻站的动作较多，较长驻站动作较少。因此，除转移总条数外，还需考虑数据覆盖了哪些状态和动作。",
    ),
    309: (
        "策略也可以通过加权监督学习从已有动作中提取。优势加权回归（Advantage-Weighted Regression，AWR）结合价值回归与优势加权动作回归；评论家正则化回归（Critic Regularized Regression，CRR）依据评论家评价进行策略回归；结合BC的TD3方法（TD3+BC）在演员评论家更新中加入BC项，形成简洁的离线基线。",
        "这些方法以已有动作作为策略更新的重要依据，能够抑制缺少经验支持的动作变化，也可能限制策略超越采集行为的能力。本文离线RE-SAC同时采用集成保守评价和行为克隆约束，第5章在相同离线数据与更新预算下比较其完整学习效果。",
    ),
    310: (
        "离线数据的形成方式影响策略能够学习到的运行关系。深度数据驱动强化学习数据集（Datasets for Deep Data-Driven Reinforcement Learning，D4RL）纳入人类行为、规则控制和混合策略等数据来源；强化学习固定数据基准（Reinforcement Learning Unplugged，RL Unplugged）提供多类固定数据任务与评价协议。",
        "Agarwal等通过大规模经验回放分析数据规模、多样性与学习方法的共同作用。",
        "表征学习研究则讨论策略如何利用已有记录及应对分布偏移。",
        "公交中的随机驻站、时刻表规则和基于乘客时间的驻站规则，会产生不同的状态动作记录。混合多种采集行为与单纯增加同一种行为的运行回合数，提供的信息并不相同。",
    ),
    311: (
        "当允许追加目标环境交互时，离线到在线学习先用固定数据初始化，再利用新目标经验继续优化。利用先验数据的强化学习（Reinforcement Learning with Prior Data，RLPD）在在线异策略学习中混合已有离线数据；预热强化学习（Warm-Start Reinforcement Learning，WSRL）则先用在线预热经验调整初始价值估计，使后续微调不必持续保留原离线数据。",
        "比较这些方法时，需要同时说明新增目标交互量、学习更新量和最终运行效果，才能区分既有数据的作用与额外交互带来的改善。",
    ),
    312: (
        "从离线学习转向在线学习，除了获得新经验，还需避免策略和价值估计在切换初期退化。优势加权演员评论家方法（Advantage-Weighted Actor-Critic，AWAC）结合动态规划和最大似然策略更新，利用先验数据并继续微调；平衡回放方法使新增经验与接近当前策略的离线样本共同参与更新。",
        "校准Q学习（Calibrated Q-Learning，Cal-QL）关注离线价值初始化的尺度，策略扩展（Policy Expansion，PEX）则保留原离线策略，同时扩展可继续学习的新策略。",
        "这些方法分别调整经验使用、价值初始化和策略复用，说明在线微调的效果还取决于如何承接离线策略，而不只是新增样本数量。",
    ),
    313: (
        "若后续训练交互改由源仿真提供，还需判断源环境中的动作后果是否符合目标运行规律。基于分类器奖励的域适应方法（Domain Adaptation with Rewards from Classifiers，DARC）通过区分源目标转移的分类器估计奖励修正，在源环境训练中考虑动力学差异。",
        "H2O将固定目标离线数据与源仿真交互结合，在价值学习中处理动力学偏移；H2O+进一步支持不同离线与在线学习方法的组合。",
        "这一路线利用目标记录提供环境信息，由源仿真承担新增训练交互，为减少目标环境中的训练试错提供了方法基础。",
    ),
    314: (
        "作者与合作者的公交跨保真研究",
        "将H2O+用于公交驻站，并根据公交运行特点调整状态、动作和执行方式。车辆可能同时到站，完成必要上下客后也可以选择不额外驻站，当前动作还会影响后续班次及窗口结束时的服务。因此，本文按同时到站车辆组织状态，构建能够直接选择不额外驻站、也能输出正驻站时长的混合动作分布，调整学习策略的驻站幅度，并设置行程完成保障。服务效果采用人均广义时间成本与行程完成率评价，其中时间统计包括所有已出发乘客在规定窗口内已经发生的候车和车内时间。纯离线学习、追加目标交互的微调，以及仅在源仿真中继续交互的混合训练，分别比较。",
    ),
    315: (
        "价值约束与动力学修正作用于不同环节，可以在同一训练流程中结合。CQL、IQL与Cal-QL分别处理行为覆盖、未观察动作的价值估计及离线到在线的价值尺度；动力学重要性采样则通过权重调整源转移对目标环境价值学习的贡献。H2O+为组合离线与在线更新、处理两侧动力学差异提供了框架。第5章根据公交实验的实际配置，区分参与学习更新的修正与仅用于检查源目标差异的诊断量。",
    ),
    316: (
        "除混合经验与调整权重外，仿真迁移还可以修正运行预测本身。需要先区分观测差异与转移差异：视觉域随机化改变纹理、光照和物体外观，使基于合成图像训练的模型适应更广的观测变化，相关研究将其用于物体定位、检测和视觉导航。",
        "本文公交策略使用类别信息和连续运行量，重点关注道路速度、客流OD、路口延误及停站服务标定不足造成的转移偏差。这些因素改变同一驻站动作的后续结果，与仅改变观测外观的问题不同。",
    ),
    317: (
        "机器人虚实迁移综述从目标模型优化、源知识迁移和策略迭代梳理迁移路径",
        "；深度强化学习知识迁移综述",
        "与机器人学习综述",
        "提供了相应的方法和任务背景。针对动力学差异，Tan等结合执行器建模、延迟模拟与物理参数随机化开展机器人迁移；自动域随机化调整训练环境参数分布；仿真参数优化方法SimOpt则利用目标运行经验更新仿真参数分布。",
        "相关综述将域随机化、域适应与元学习归纳为不同迁移路径。",
        "它们体现两种思路：在尚无目标信息时，扩大源训练覆盖的变化范围；获得目标数据后，利用这些记录缩小或修正源目标差异。本文第5、6章研究后一条件下，固定目标数据如何支持低成本源仿真训练。",
    ),
    318: (
        "世界模型学习给定状态和动作之后的状态转移与奖励，用预测补充决策信息。概率集成轨迹采样方法（Probabilistic Ensembles with Trajectory Sampling，PETS）用概率动力学集成表达预测不确定性，并通过轨迹采样评价动作序列；基于模型的策略优化（Model-Based Policy Optimization，MBPO）从已有环境状态出发，只预测少量后续转移，以短模型滚动生成策略训练经验。",
        "Dreamer在压缩后的状态表示中生成预测轨迹，并据此学习行为，称为模型想象；它以这种方式连接状态表示与长期决策学习。",
    ),
    319: (
        "潜在空间强化学习综述分别讨论状态表示、动作表示和动力学模型",
        "；预测编码研究将环境模型与样本自适应的行动规划结合。",
        "Ha和Schmidhuber用学习得到的时空表示支持紧凑控制器；深度规划网络（Deep Planning Network，PlaNet）在潜在空间中在线规划；Dreamer系列进一步直接从模型想象中学习行为。",
        "时序差分模型预测控制（Temporal Difference Model Predictive Control，TD-MPC）及其第二版TD-MPC2，则将短时域轨迹规划与终端价值估计结合。",
        "因此，世界模型可以直接参与动作规划，也可以生成策略训练样本。本文采用后一用途，以预测的一步转移辅助演员评论家更新，部署时仍由演员输出驻站动作。",
    ),
    320: (
        "预测步数影响模型误差的累积。模型价值扩展（Model-Based Value Expansion，MVE）通过有限深度的模型展开改进价值估计；随机集成价值扩展（Stochastic Ensemble Value Expansion，STEVE）根据不确定性组合不同长度的展开结果。",
        "模型型强化学习综述将模型拟合与规划学习视为相联系的环节，讨论随机性、部分可观测性、计算预算及模型样本起点。",
        "对于源目标不一致的任务，既需选择合适的预测模型，也需限定预测用于训练的方式和步数。前者影响一步预测是否符合目标记录，后者影响误差在后续更新中的传播。",
    ),
    321: (
        "固定离线数据还要求防止策略利用模型在数据未覆盖区域的乐观预测。基于模型的离线策略优化（Model-Based Offline Policy Optimization，MOPO）以动力学不确定性惩罚预测奖励；基于模型的离线强化学习（Model-Based Offline Reinforcement Learning，MOReL）构造悲观模型决策过程；保守离线模型型策略优化（Conservative Offline Model-Based Policy Optimization，COMBO）在模型生成的分布外状态动作上施加保守价值正则。",
        "三者分别通过奖励、模型过程和价值函数限制模型误差的影响，说明生成更多预测样本仍不能代替数据覆盖。本文进一步比较保留源模型并修正其预测，与直接使用目标数据建模的效果。",
    ),
    322: (
        "模型用于训练的状态范围同样重要。Ross和Bagnell研究模型型强化学习中的系统辨识，强调模型应适应决策策略实际到达的状态分布。",
        "Asadi等从Lipschitz连续性分析模型误差传播，给出连接一步预测误差与多步后果的条件。",
        "因此，平均预测误差较低并不自动意味着控制回报较高，还需判断误差出现在什么状态、策略是否会到达这些状态，以及预测样本怎样参与训练。",
    ),
    323: (
        "已有源世界模型时，残差适应保留源预测，只学习目标结果与源预测之间的差。Lanier等在潜在状态动力学上构建残差，用目标数据调整已有世界模型。",
        "这种方法可以复用源环境中学到的关系，但若源预测已明显不适合目标运行，先抵消源误差再拟合目标也未必更有利。本文据此同时构建残差模型与直接目标模型，使用未参与模型拟合和结构筛选的留出数据比较预测误差、选择模型类别，再通过配对控制实验检验这一选择的实际作用。",
    ),
    324: (
        "真实机器人研究也体现了不同的数据利用方式。Feng等用机器人离线记录预训练世界模型，再通过真实环境交互微调，并在规划中平衡预期回报与模型认知不确定性，以抑制外推误差。",
        "Li等在2026年修订的工作中，为自回归世界模型加入认知不确定性估计，用不确定性惩罚支持离线策略训练，并在真实机器人上评价。",
        "前者允许追加目标交互，后者主要利用既有记录训练策略。本文公交研究采用固定目标离线数据辅助源仿真训练，通过SUMO跨保真实验检验控制效果，真实道路闭环应用仍需后续验证。",
    ),
    325: (
        "在公交H2O+研究基础上，本文进一步针对源仿真标定不足，提出基于模型修正的世界模型方法（Model-Corrected World Model，MC-WM）",
        "。公交实验利用固定目标离线数据学习预测修正，检验其对驻站控制与乘客服务的作用；受控连续实验使用接触多关节动力学平台（Multi-Joint dynamics with Contact，MuJoCo）",
        "，分别改变执行器、重力或摩擦，并允许追加目标交互，分析残差修正与直接建模的选择机制。两类实验分别检验公交控制效果与模型选择的作用，在第6章按各自数据条件和评价任务展开。",
    ),
}


def main():
    with ZipFile(DOCUMENT) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        assert "".join(body[305].itertext()) == "1.2.3 仿真迁移"
        assert "".join(body[326].itertext()) == "1.2.4 现有研究总结"
        updated = original
        for index, prose in REVISIONS.items():
            paragraph_id = body[index].get(W14 + "paraId")
            pattern = r'<w:p\b[^>]*w14:paraId="' + paragraph_id + r'"[^>]*>.*?</w:p>'
            matches = list(re.finditer(pattern, updated, flags=re.S))
            assert len(matches) == 1, (index, paragraph_id)
            match = matches[0]
            revised = revise_paragraph(match.group(0), body[index], prose)
            mask = lambda raw: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", raw, flags=re.S)
            assert mask(match.group(0)) == mask(revised)
            updated = updated[:match.start()] + revised + updated[match.end():]
        new_body = ET.fromstring(updated).find(W + "body")
        assert len(body) == len(new_body)
        changed = [i for i, (a, b) in enumerate(zip(body, new_body)) if ET.tostring(a) != ET.tostring(b)]
        assert changed == list(REVISIONS), changed
        for index in changed:
            for tag in (W + "hyperlink", M + "oMath", W + "pPr"):
                assert [ET.tostring(x) for x in body[index].iter(tag)] == [ET.tostring(x) for x in new_body[index].iter(tag)]
        assert not BACKUP.exists(), "Keep the earlier backup and inspect the current manuscript before rerunning."
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
    print("Only section 1.2.3 prose changed; abstracts, citations, native math and all formatting preserved.")
    for index in changed:
        print(index, "".join(n.text or "" for n in new_body[index].iter() if n.tag in (W + "t", M + "t")))
    print("OUTPUT", DOCUMENT)


if __name__ == "__main__":
    main()
