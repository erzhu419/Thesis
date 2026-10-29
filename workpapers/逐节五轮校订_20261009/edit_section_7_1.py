from pathlib import Path
from shutil import copy2
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_1_2_1 import DOCUMENT, M, W
from edit_section_3_2 import prose_groups
from edit_section_4_4 import replace_paragraph


BACKUP = Path(__file__).with_name("校订前_7.1.docx")
REVISIONS = {
    1924: (
        "本文围绕公交驻站控制，按可利用数据与运行环境变化逐步开展研究。未预先获得目标域转移训练数据时，先以RE-SAC处理给定运行规律下的随机波动，再以BAPR及CS-BAPR应对运行规律变化；获得固定目标离线数据后，利用H2O+框架开展公交跨保真训练，并进一步以MC-WM研究源仿真标定不足时的预测修正。公交实验采用走廊事件仿真与多线路SUMO，受控连续任务用于分析动作补偿和世界模型选择。主要结论如下。",
    ),
    1925: (
        "（1）类别表征使不同车辆共享驻站经验，同时保留各自的运行身份与服务需求。类别特征嵌入与连续运行量共同构成策略输入。在相同RE-SAC训练设置、五次独立训练中，学习嵌入相对独热编码在正常运行、客流激增和快速切换下，分别减少5.51%、1.41%和4.91%的人均广义时间成本，前后时距不均衡也下降；相对尺寸匹配的类别屏蔽组，正常运行下的成本减少2.32%。这些结果支持共享策略同时利用运行类别与连续状态。",
    ),
    1926: (
        "RE-SAC以集成动作价值及估计分歧调节价值学习和策略更新，减弱不可靠动作评价的影响。相同集成结构下，保留评论家分歧与演员保守评价，相对同时去除两项，在正常运行、客流激增和快速切换中分别降低14.20%、4.02%和11.29%的人均广义时间成本，配对区间均支持改善。另在分歧项交叉比较中，快速切换下的演员项平均减少2.298秒时距不均衡，评论家项与交互项的作用尚未分辨。统一100回合的规模比较中，五评论家组的平均计划偏差和乘客时间较低，扩大至10或20个未持续改善，长期训练也出现后期退化。多线路任务中，以车头时距为目标时，RE-SAC相对SAC的平均奖励提高12.46个单位，计划偏差和前后时距不均衡分别减少5.00秒和7.36秒；改用乘客时间目标后，未取得一致优势。因此，保守评价的作用需结合训练条件与服务指标判断。",
    ),
    1927: (
        "（2）运行规律变化时，BAPR利用历史证据更新条件判断，CS-BAPR进一步结合近期反馈、训练稳定化与驻站输出约束。公交实现以近期奖励和价值统计调节训练，评价时执行固定策略；受控执行器任务另检验在线模式识别与动作补偿。早期两次训练中，稳定化MLP相对BAPR在六类需求场景下降低58.82%至69.79%的平均时距惩罚，但改用乘客时间奖励后，人均广义时间成本增加0.068%至0.268%；算术输出也未在所有条件下增加收益。该组中，一个种子的训练退化得到缓解，对平均时距改善的贡献较大。后续两次训练采用可学习基准动作、参考状态下为零的神经修正及SINDy软约束，在正常需求、单次与重复5倍冲击、持续5倍与10倍需求及短时50倍需求下，与RE-SAC配对比较。以乘客时间为目标时，六类场景的成本均值均下降，降幅为0.083%至0.549%，计划偏差与前后时距不均衡分别减少7.35至12.09秒、11.88至20.08秒。以时距为目标时，五类场景的平均惩罚下降10.41%至17.43%，持续10倍需求下却增加0.27%，部分其他间隔指标也变差。后续配置改善了平均表现，但同结构去除符号约束的策略已有接近的乘客收益，不能将全部改善归于SINDy。",
    ),
    1928: (
        "（3）获得固定目标离线数据后，策略训练不必仅依靠源仿真经验。固定SUMO数据比较中，离线RE-SAC在不追加目标训练交互的方法中取得最高选定检查点回报；允许继续与SUMO交互后，WSRL进一步改善回报和车头时距偏差。各方法的乘客时间与间隔稳定性排序不同，较高时距回报不等同于较低乘客时间。",
    ),
    1929: (
        "为减少新增训练经验对部署环境交互的依赖，本文在H2O+框架下开展公交适配，以SUMO离线数据初始化，再结合简易仿真中新经验续训。策略分别编码同时到站车辆，用混合动作分布表示不额外驻站与连续驻站时长，并在执行端设置驻站幅度校准和行程完成保障。10个新场景组、三档需求及九种控制器的1350回合评价中，完整控制器的人均广义时间成本为2392.73秒、乘客行程完成率为0.96650；相对采用原执行方式的纯源SAC，成本下降7.75%。附录F.2另在新运行日采用共同执行规则，H2O+相对纯源SAC平均减少17.23秒（0.72%），配对区间包含零。两组差异分别对应原完整方案与共同执行条件下的控制器比较。",
    ),
    1930: (
        "训练与执行规则的作用分别由对照检验。使用同一批目标离线数据、保持策略与评论家更新次数相同的五次训练中，混合方案相对纯离线IQL在正常及1.25倍需求下分别减少16.66和19.75秒成本，95%配对区间均支持改善。两组学习目标不同，该结果反映完整训练方案的差异；相对纯源训练和360回合比较中的共同保障间隔规则，混合方案的稳定优势尚未建立。固定同一学习策略，将最低驻站由20秒改为60秒并扩大保障时段，成本减少92.12秒、下降3.71%，完成率提高0.466个百分点，10个场景组方向一致。候车节省抵消了车内增量，同时伴随大间隔事件增加。保障在多种基础策略中均有效，不是混合学习独有的收益。",
    ),
    1931: (
        "（4）MC-WM按目标验证误差选择残差修正或直接目标建模，以一步预测辅助价值学习。Ant、HalfCheetah、Walker各训练10次，相对Target SAC的回报分别增加1925.7、1184.5、133.4，95%配对区间均为正，违例减少；相对匹配预算DARC的回报也提高。Ant执行器变化下，直接模型比强制残差提高1529.9回报，支持在源预测不合适时替换模型。",
    ),
    1932: (
        "公交混合事件模型在固定SUMO数据条件下预测事件类别、人数、连续运行量、持续时间与乘客成本。五次训练的一次拟合比较中，按验证误差选择模型，相对无世界模型混合组在两档需求下平均减少0.04和3.74秒成本，区间均包含零；采用原置信更新的完整日程也未取得跨条件一致收益。训练诊断发现，预测误差基本稳定，累计对数置信却使模型权重与梯度贡献持续下降。归一化置信更新针对这一尺度问题，恢复模型样本参与价值学习的强度。固定0.5样本权重也有效，说明收益不能全部解释为状态相关置信排序。",
    ),
    1933: (
        "以乘客时间为目标、模型损失项系数为0.25时，20次独立训练中，正常标定、联合50%及75%源偏差下，MC-WM相对无世界模型混合基线分别减少51.14、69.93和87.05秒人均广义时间成本，三个配对区间均为负；相对共同完成保障的Daganzo式规则分别减少57.67、70.67和76.26秒。联合偏差改变源道路平均速度、OD强度、额外路口遭遇量和必要停站时间，目标SUMO路网与测试需求保持不变。收益主要来自候车节省，完成率提高，部分间隔指标仍不如规则。无世界模型混合基线并非正式H2O+控制器；最终控制器在共同源采样与执行规则下另行比较，MC-WM相对正式H2O+及纯源SAC分别平均减少23.02秒（0.97%）和40.25秒（1.68%），见",
        "。其中相对H2O+的区间包含零，该批平均改善不能单独确定世界模型的稳定增益。",
    ),
    1934: (
        "车头时距奖励补充实验采用0.125模型损失项系数、0.75混合经验损失项系数和两个独立初始化。采用共同完成保障，MC-WM相对H2O+在正常标定及联合75%源偏差下分别减少1.03和13.36秒人均广义时间成本；高偏差下成本下降约0.49%，时距奖励却降低0.26个单位。直接执行的高偏差条件下，成本反而增加2.34秒。源转移价值目标修正与上下文系数调节也未持续增加平均收益。因此，模型辅助的服务效果取决于训练目标、样本使用强度和执行规则，不能由模型预测误差单独判断。",
    ),
    1935: (
        "上述研究由共享公交经验与保守价值评价，递进到运行变化下的适应，再到固定目标数据辅助的跨保真训练和预测修正。各章分别检验类别表征、训练稳定化、混合学习与模型辅助的作用，未将不同任务的数值直接合并排名。公交服务收益主要体现为所测条件下的平均改善，时距与乘客时间仍有取舍，学习策略与执行规则也各有贡献。这些结果为减少训练对部署环境交互的依赖、缓解源仿真标定偏差提供了实验依据；真实道路上的控制收益仍需进一步验证。",
    ),
}


def main():
    assert not BACKUP.exists(), "Do not overwrite the pre-edit backup."
    with ZipFile(DOCUMENT) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        assert "".join(body[1923].itertext()) == "7.1 研究总结"
        assert "".join(body[1936].itertext()) == "7.2 主要创新"
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
