from collections import Counter
from pathlib import Path
from shutil import copy2
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_3_2 import W
from edit_section_4_4 import replace_paragraph


ROOT = Path(__file__).resolve().parents[2]
DOCUMENT = ROOT / "output/排版完成_逐节校订_第2轮_20261010.docx"
BACKUP = Path(__file__).with_name("校订前_第2轮_7.1.docx")
REVISIONS = {
    1924: (
        "本文按目标数据是否可用及运行规律是否变化，逐步研究公交驻站控制。未预先获得目标域转移训练数据时，先以RE-SAC处理给定运行规律下的随机波动，再以BAPR及CS-BAPR应对运行规律变化；获得固定目标离线数据后，在已有H2O+框架下开展公交跨保真训练，进一步以MC-WM处理源仿真标定不足造成的预测偏差。公交实验采用走廊事件仿真与多线路SUMO，受控连续任务用于分析动作补偿和世界模型选择。主要结论如下。",
    ),
    1925: (
        "（1）类别表征与保守价值评价共同支持平稳随机运行下的驻站学习。类别特征嵌入与连续运行量共同作为策略输入，使车辆共享经验，同时区分运行身份与服务需求。在相同RE-SAC设置、五次独立训练中，学习嵌入相对独热编码，在正常运行、客流激增和快速切换下分别降低5.51%、1.41%和4.91%的人均广义时间成本，前后时距不均衡也下降；相对尺寸匹配的类别屏蔽组，正常运行成本降低2.32%。该比较检验类别表征的作用，支持共享策略同时使用类别信息与连续状态。",
    ),
    1926: (
        "RE-SAC用集成动作价值及估计分歧调节价值学习和策略更新，减弱不可靠动作评价的影响。多线路任务以车头时距为目标时，相对SAC的平均奖励提高12.46个单位，计划偏差和前后时距不均衡分别减少5.00秒和7.36秒；改用乘客时间目标后，该组未取得一致优势。另在相同集成结构下，保留评论家分歧惩罚与演员保守评价，相对同时去除两项，在正常运行、客流激增和快速切换中分别降低14.20%、4.02%和11.29%的人均广义时间成本，配对区间均支持改善。分歧项交叉比较中，快速切换下的演员项平均减少2.298秒时距不均衡，评论家项及交互项的差值区间包含零。统一100回合的规模比较中，采用五个评论家的配置，平均计划偏差和乘客时间较低，扩大至10或20个未持续改善，长期训练还出现后期退化。保守评价的效果因此与训练设置和优化目标有关。",
    ),
    1927: (
        "（2）运行规律变化时，BAPR利用历史证据更新运行条件判断，CS-BAPR进一步结合近期反馈、训练稳定化与驻站输出约束。公交反馈用于训练调节，评价时执行固定策略；受控执行器任务另检验在线模式识别与动作补偿。后续两次训练采用可学习基准动作、参考状态下为零的神经修正及SINDy软约束，在正常需求、单次与重复5倍冲击、持续5倍与10倍需求及短时50倍需求下，与RE-SAC配对比较。以乘客时间为目标时，六类场景的人均广义时间成本均值均下降，降幅为0.083%至0.549%，计划偏差与前后时距不均衡分别减少7.35至12.09秒、11.88至20.08秒。以时距为目标时，五类场景的平均惩罚下降10.41%至17.43%，持续10倍需求下增加0.27%，六类场景的计划偏差与前后时距不均衡均增加。同结构去除符号约束的策略已有接近的乘客收益，改善不能全部归于SINDy。早期两次训练以稳定化MLP与BAPR比较，六类需求下的平均时距惩罚下降58.82%至69.79%，其中一次BAPR训练退化得到缓解，对均值改善贡献较大；改用乘客时间奖励后，成本增加0.068%至0.268%，算术输出也未在所有条件下增加收益。两批结果分别反映后续软符号配置相对RE-SAC、早期稳定化配置相对BAPR的效果。",
    ),
    1928: (
        "（3）获得固定目标离线数据后，可利用已有记录训练策略。在固定SUMO数据、不追加目标训练交互的比较中，离线RE-SAC的选定检查点回报最高；允许继续与SUMO交互后，WSRL进一步改善回报和车头时距偏差。该组乘客时间与间隔稳定性的排序不同，较高时距回报不等于较低乘客时间。",
    ),
    1929: (
        "在不追加目标环境训练交互的条件下，本文利用H2O+框架开展公交适配，以SUMO离线数据初始化策略，再结合简易仿真新经验混合续训。策略分别编码同时到站车辆，用混合动作分布表示不额外驻站与连续驻站时长，执行时结合驻站幅度校准和行程完成保障。10个新场景组、三档需求及九种控制器的1350回合评价中，完整控制器的人均广义时间成本为2392.73秒、乘客行程完成率为0.96650；相对采用原执行方式的纯源SAC，成本下降7.75%。附录F.2另在新运行日采用共同执行规则，H2O+相对纯源SAC平均减少17.23秒（0.72%），配对区间包含零。前一比较包含执行方式差异，后一比较采用共同规则。",
    ),
    1930: (
        "进一步分别检验混合训练与完成保障。使用同一批目标离线数据、匹配策略与评论家更新次数的五次训练中，混合方案相对纯离线IQL，在正常及1.25倍需求下分别减少16.66和19.75秒人均广义时间成本，95%配对区间均支持改善。两组学习目标不同，该结果对应完整训练方案的差异；相对纯源训练和360回合比较中采用共同保障的间隔规则，混合方案尚无稳定优势。固定同一学习策略，将保障条件下的最低附加驻站由20秒改为60秒并扩大启用时段，成本减少92.12秒、下降3.71%，完成率提高0.466个百分点，10个场景组方向一致。候车节省抵消了车内时间增量，同时伴随大间隔事件增加。保障收益也见于多种基础策略，并非混合学习独有。",
    ),
    1931: (
        "（4）源仿真标定不足时，MC-WM利用目标验证记录，在残差修正与直接目标建模之间选择，再用所选模型的一步预测辅助价值学习。Ant、HalfCheetah、Walker各训练10次，相对Target SAC的回报分别增加1925.7、1184.5、133.4，95%配对区间均为正，违例减少；相对匹配预算DARC的回报也提高。Ant执行器变化下，选择直接模型比强制采用残差模型增加1529.9回报，说明源预测不适用时，直接建模可以改善控制。",
    ),
    1932: (
        "公交世界模型在固定SUMO数据条件下，分别预测事件类别、人数、连续运行量、持续时间与乘客成本。五次独立训练中，模型仅在续训前拟合一次，按验证误差选择后，相对无世界模型混合组在两档需求下平均减少0.04和3.74秒成本，区间均包含零；分阶段拟合但沿用原置信更新，也未取得跨条件一致收益。诊断发现，预测误差基本稳定，但对数置信的负值在递推中累积，使模型权重与梯度贡献持续下降。归一化置信更新调整这一尺度，使模型样本重新有效参与价值学习。固定0.5样本权重也有收益，说明恢复模型样本的更新作用是重要原因，改善不能全部归于按状态区分样本可信程度。",
    ),
    1933: (
        "以乘客时间为目标、采用归一化置信更新及0.25模型损失项系数时，20次独立训练中，正常标定、联合50%及75%源偏差下，MC-WM相对无世界模型混合基线分别减少51.14、69.93和87.05秒人均广义时间成本，三个配对区间均为负；相对共同完成保障的Daganzo式规则分别减少57.67、70.67和76.26秒。联合偏差改变源道路平均速度、OD强度、额外路口遭遇量与必要停站时间，目标SUMO路网与测试需求不变。收益主要来自候车节省，完成率提高，部分间隔指标仍不如规则。该基线并非正式H2O+控制器；共同源采样与执行规则下另行比较最终控制器，MC-WM相对正式H2O+及纯源SAC分别平均减少23.02秒（0.97%）和40.25秒（1.68%），见",
        "。其中相对H2O+的成本差区间包含零。该批平均改善反映完整控制器效果，不能单独确定世界模型的稳定增益。",
    ),
    1934: (
        "车头时距奖励补充实验采用0.125模型损失项系数、0.75混合经验损失项系数和两个独立初始化。共同完成保障下，MC-WM相对H2O+在正常标定及联合75%源偏差下，分别减少1.03和13.36秒人均广义时间成本；高偏差下成本下降约0.49%，时距奖励却降低0.26个单位。直接执行的高偏差条件下，成本反而增加2.34秒。用目标模型修正源转移的价值目标，或按识别的源偏差条件调节模型系数，也未持续增加平均收益。模型辅助的服务效果因此与训练目标、模型样本的训练权重和执行规则有关，不能只由预测误差判断。",
    ),
    1935: (
        "上述研究由共享车辆经验与保守价值评价，递进到运行变化适应，再到固定目标数据辅助的跨保真训练和预测修正。各项结果按对应任务与比较条件解释，不直接按跨章数值排名。所测平均服务改善仍伴随时距与乘客时间的取舍，学习策略和执行规则各有贡献。研究支持减少训练对部署环境交互的依赖、缓解源标定不足；真实道路收益仍需验证。",
    ),
}


def main():
    if not BACKUP.exists():
        copy2(DOCUMENT, BACKUP)
    with ZipFile(BACKUP) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        assert "".join(body[1923].itertext()) == "7.1 研究总结"
        assert "".join(body[1936].itertext()) == "7.2 主要创新"
        with ZipFile(DOCUMENT) as current:
            current_body = ET.fromstring(current.read("word/document.xml")).find(W + "body")
            assert len(current_body) == len(body)
            assert all(ET.tostring(a) == ET.tostring(b) for i, (a, b) in enumerate(zip(body, current_body)) if i not in REVISIONS)
        updated = original
        for index, prose in REVISIONS.items():
            updated = replace_paragraph(updated, body[index], prose)

        # Keep every object, field, formatting property, and out-of-scope paragraph.
        mask = lambda xml: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", xml, flags=re.S)
        assert mask(original) == mask(updated), "Non-text XML changed."
        final = ET.fromstring(updated).find(W + "body")
        assert len(body) == len(final) == 2924
        changed = [i for i, (a, b) in enumerate(zip(body, final)) if ET.tostring(a) != ET.tostring(b)]
        assert changed == list(REVISIONS), changed
        for index in changed:
            numbers = lambda p: re.findall(r"\d+(?:\.\d+)?", "".join(p.itertext()))
            if index in (1926, 1927):
                assert Counter(numbers(body[index])) == Counter(numbers(final[index])), index
            else:
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
