from pathlib import Path
from shutil import copy2
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_3_2 import M, W, prose_groups
from edit_section_4_4 import replace_paragraph


ROOT = Path(__file__).resolve().parents[2]
DOCUMENT = ROOT / "output/排版完成_逐节校订_第2轮_20261010.docx"
BACKUP = Path(__file__).with_name("校订前_第2轮_5.3.docx")
REVISIONS = {
    1292: (
        "本节依次说明两项任务的求解。第5.3.1节比较固定目标数据学习与追加目标交互，第5.3.2节给出运营规则参照；第5.3.3节说明目标离线初始化与源仿真混合续训，第5.3.4节说明驻站幅度校准与行程完成保障。后续实验分别评价训练方法和执行规则的作用。",
    ),
    1294: (
        "离线学习需处理缺少数据支持时的动作价值估计。BCQ和BEAR限制动作偏离已有数据，BRAC采用行为正则，TD3+BC在演员目标中加入BC项。",
        "本章选用BC、CQL和离线RE-SAC，分别检验动作模仿、保守价值学习，以及集成评价与行为克隆约束的组合。",
    ),
    1298: (
        "本研究取",
        "、",
        "，另设较小的评论家权重正则项。",
        "沿用式",
        "的未加权平方误差，使用固定目标数据",
        "中的动作与重参数化采样的策略动作比较，损失仅乘一次",
        "。集成均值评价收益，负分歧系数降低高分歧动作的评价，BC限制策略偏离已有动作。这是前期RE-SAC",
        "在长沙多线路SUMO中的离线适配，不追加目标训练交互。其保守评价思路与第3章相同，但状态定义、经验来源和BC设置不同，结果只在本节共同任务内比较。",
    ),
    1299: (
        "集成保守评价不同于状态邻域内的平滑约束。EDAC增强价值网络多样性，以处理数据支持不足的动作；鲁棒离线强化学习（Robust Offline Reinforcement Learning，RORL）约束数据邻近状态下的策略与价值变化。",
        "本章检验固定目标记录能否支持有效驻站，第6章再研究源路况与服务参数标定不足时的训练与迁移。",
    ),
    1301: (
        "离线策略转入在线学习时，新旧经验分布与初始价值尺度会影响续训。AWAC通过优势加权利用已有数据，Cal-QL校准保守价值初始化，PEX保留离线策略并扩展新策略。",
        "本章选用WSRL与RLPD，分别检验在线预热后微调，以及持续混合离线数据与新经验两种方式。",
    ),
    1303: (
        "在线及离线到在线方法均使用30,000次目标决策，每100次事件组成一个训练周期，共300周期，在第50、100、150、200、250、300周期及训练结束时保存模型。周期用于组织更新，不是完整运营回合；交互预算按决策事件计量。",
    ),
    1304: (
        "集成规模比较另设10个评论家的WSRL-E10与RLPD-E10。RLPD-E10还比较更新数据比为4的配置，即每次新增交互对应4次学习更新。这些配置的交互量相同，但网络规模或优化计算量不同，故与主配置分别报告。",
    ),
    1312: (
        "上述规则均使用决策时可获得的运行量。式(5-12)依据Daganzo车头时距反馈思想设置前向比例规则，第6.4.5节则按前后间隔差构造双向规则；两者使用的信息与计算方式不同，各自在对应任务中比较。",
    ),
    1314: (
        "Niu等提出的H2O及H2O+结合目标离线经验与源仿真交互。",
        "作者与合作者据此研究公交跨保真驻站与行程完成保障。",
        "以下在第5.2节事件模型上，说明离线初始化、混合价值更新及源仿真续训。目标数据固定，源仿真随当前策略生成新经验；各组是否启用比率修正另行说明。",
    ),
    1315: (
        "概括控制器的形成过程：离线初始化与源仿真续训得到策略，开发评价确定驻站幅度变换与完成保障，随后固定策略及执行规则，在新SUMO场景中评价。",
    ),
    1317: (
        "离线阶段采用IQL期望分位（expectile）回归估计状态价值，再通过优势加权拟合已有动作；源仿真续训采用SAC式熵正则化演员更新，并按公交H2O+配置组合两类后继价值。",
    ),
    1323: (
        "状态价值回归损失为",
        "，其中",
        "、",
        "。正残差获得较大权重，使状态价值偏向数据中较高的动作价值；演员随后更着重模仿相对优势较高的已有动作。",
    ),
    1326: (
        "初始化使演员先模仿目标记录中价值较高的动作，再到源仿真中尝试新动作。优势权重设有上限，以免极大权重样本主导拟合。进入续训后，固定目标记录仍与新增源经验共同参与回放。",
    ),
    1330: (
        "最终策略从同种子10,000源事件处保存的参数、优化器及回放状态继续训练，累计至25,000个源事件。第15,000至25,000事件期间，演员学习率降为原值的0.25倍，其余更新日程不变。每个种子共完成70,000次训练更新，仅评价25,000事件处的终点策略。",
    ),
    1331: (
        "策略与价值学习只使用固定目标离线数据和新增源经验，不加入SUMO开发或测试轨迹。开发评价用于检查源仿真、选择控制器及执行规则，最终评价采用新场景组。因此，训练不追加目标转移，并不意味着开发与测试不访问SUMO；各类访问分别统计。",
    ),
    1332: (
        "固定离线数据覆盖30组需求与交通随机条件。第2.4节另以30个SUMO回合检查不额外驻站的运行，以210回合检查七类驻站干预；最后一轮控制器开发及正式确认分别使用405和1350回合。四类评价每回合24,000秒，对应模拟物理时长为200、1400、2700和9000小时，而非计算耗时。早期开发评价量见第5.4.4节，其他学习比较见",
        "及对应实验节。",
    ),
    1335: (
        "每个状态对连续时长分支采样一次，同时直接计算0秒动作的价值。更新演员时固定评论家参数，但保留价值对动作的导数。式",
        "对两分支概率求和，门控输出因而可直接求导，无需对二值采样求导或采用直通梯度近似；连续时长通过重参数化求导。交互时按门控概率采样；确定性评价在额外驻站概率不小于0.5时，使用",
        "对应的时长，否则不额外驻站，最后应用第5.3.4节的执行变换。",
    ),
    1336: (
        "源目标差异既可能来自状态动作的出现频率，也可能来自给定状态动作后的转移规律。DARC及H2O结合联合分类器与状态动作分类器的对数几率，估计条件转移比率。",
    ),
    1337: (
        "公交联合分类器还纳入事件持续时间。式(5-18)中，",
        "与",
        "分别为联合和状态动作分类器的对数几率，",
        "与",
        "为其参数，输入分别为",
        "与",
        "。目标样本为正类、源样本为负类。两类等量且类别概率准确时，对数几率之差等于目标相对源的条件转移对数比率，实际近似写为",
    ),
    1339: (
        "式",
        "的估计需用未参与分类器拟合的轨迹检查，分类与权重诊断仍需结合目标服务结果判断其控制作用。",
    ),
    1340: (
        "启用修正的同预算混合配置将估计对数比率取指数并截断至",
        "，所得源权重仅用于对应样本的评论家回归损失。正式控制器的留出分类表现较差，故采用单位源权重，保留目标离线初始化、混合回放和源续训。两种配置分别报告：正式控制器诊断见第5.4.2节，启用修正的预算比较与独立轨迹诊断见第5.4.3至5.4.4节。",
    ),
    1342: (
        "执行时先根据运行状态调整策略建议的附加驻站时长，再对满足服务条件的车辆设置最低驻站量。两步均不改变训练奖励或网络参数。",
    ),
    1347: (
        "在运营时域前75%，演员建议额外驻站时，式(5-20)按间隔偏差将时长限制在原输出的八分之一与原输出之间；其余情况取原输出的八分之一。该变换只缩小或保持学习时长，不把0秒改为正时长。满足下述服务条件时，再设置最低驻站量。",
    ),
    1348: (
        "行程完成保障依据车辆服务状态设置最低附加驻站量，条件为有前车、无在途后随车辆、载客率较低且仍有可服务客流。该规则与依据车头时距反馈的Daganzo式规则不同，实验分别评价两者，也评价共同使用的效果。",
    ),
    1351: (
        "式(5-21)的触发条件由车辆位置、载客率和客流决定，不依赖评论家估值；载客率阈值为25%，仅在车辆容量为正时计算。",
    ),
    1352: (
        "无在途后随车辆不等于本车是当天末班车。时间窗口按规定运营时域计算，规则效果用完成率检验，不保证所有乘客均能在时域内完成行程。",
    ),
    1353: (
        "前车提供间隔参照；当前没有在途后随车辆时，延长本车驻站可保留上车机会。低载条件限制已有乘客增加的车内时间，到达率为正则表示仍有可服务需求。因此，无新需求的空车和满载车辆均不触发该规则。",
    ),
    1354: (
        "时间窗口限制干预阶段：较早驻站可能影响后续站点和班次，临近评价终点又可能延后车内乘客的行程完成。窗口和最低时长由开发评价确定，最终测试时保持不变。",
    ),
    1360: (
        "60秒也是动作上限，因此候选触发时恰好附加驻站60秒。候选与直接参照共用演员、观测定义和前序变换，在相同需求与交通条件下配对；差别仅在最低驻站量和启用窗口。后续状态随各自动作演化，比较反映规则调整后的完整服务结果。",
    ),
    1361: (
        "驻站幅度和保障规则根据开发期的训练过程、动作时长、间隔响应及完成情况确定。若干开发轮次复用9个运行条件组合以选择方案，最终比较则采用新的场景组。",
    ),
    1362: (
        "正式确认前，候选和直接参照、五个演员、场景组、主要指标、完成率允许下降的上限及检验方法均已固定，确认期间不再训练或更换控制器。此配对比较检验最低驻站量与窗口共同调整的增量，混合训练的作用另由第5.4.3节比较。",
    ),
}


def main():
    assert not BACKUP.exists(), "Keep the existing pre-edit backup."
    with ZipFile(DOCUMENT) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        assert "".join(body[1291].itertext()) == "5.3 求解方法"
        assert "".join(body[1363].itertext()).startswith("5.4 ")
        updated = original
        for index, prose in REVISIONS.items():
            assert 1291 < index < 1363
            updated = replace_paragraph(updated, body[index], prose)

        # Text-only changes preserve native equations, fields and layout.
        mask = lambda xml: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", xml, flags=re.S)
        assert mask(original) == mask(updated), "Non-text XML changed."
        final = ET.fromstring(updated).find(W + "body")
        assert len(body) == len(final) == 2924
        changed = [i for i, (a, b) in enumerate(zip(body, final)) if ET.tostring(a) != ET.tostring(b)]
        assert changed == list(REVISIONS), changed
        for index in changed:
            old_nodes, _, protected = prose_groups(body[index])
            new_nodes, _, _ = prose_groups(final[index])
            assert all(old_nodes[i].text == new_nodes[i].text for i in protected)
            for tag in (M + "oMath", W + "pPr", W + "hyperlink", W + "instrText", W + "fldChar"):
                assert [ET.tostring(n) for n in body[index].iter(tag)] == [ET.tostring(n) for n in final[index].iter(tag)]
            old_numbers = re.findall(r"\d+(?:\.\d+)?", "".join(body[index].itertext()))
            new_numbers = re.findall(r"\d+(?:\.\d+)?", "".join(final[index].itertext()))
            assert old_numbers == new_numbers, (index, old_numbers, new_numbers)
        copy2(DOCUMENT, BACKUP)
        temporary = DOCUMENT.with_suffix(".tmp")
        with ZipFile(temporary, "w") as target:
            for item in source.infolist():
                target.writestr(item, updated.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename))
    temporary.replace(DOCUMENT)
    with ZipFile(BACKUP) as before, ZipFile(DOCUMENT) as after:
        assert before.namelist() == after.namelist()
        assert [n for n in before.namelist() if before.read(n) != after.read(n)] == ["word/document.xml"]
    print({"changed_blocks": changed, "paragraphs": len(changed), "backup": str(BACKUP), "output": str(DOCUMENT)})


if __name__ == "__main__":
    main()
