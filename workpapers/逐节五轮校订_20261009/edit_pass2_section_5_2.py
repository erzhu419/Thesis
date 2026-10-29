from pathlib import Path
from shutil import copy2
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_3_2 import M, W, prose_groups
from edit_section_4_4 import replace_paragraph


ROOT = Path(__file__).resolve().parents[2]
DOCUMENT = ROOT / "output/排版完成_逐节校订_第2轮_20261010.docx"
BACKUP = Path(__file__).with_name("校订前_第2轮_5.2.docx")
REVISIONS = {
    1228: (
        "本节沿用第2章的事件决策模型，说明固定目标离线数据、追加目标交互和追加源仿真交互三种训练条件，再分别给出两项公交任务的状态、目标及驻站动作。两项任务的数据和运营时域不同，以下分别说明。",
    ),
    1233: (
        "式",
        "所示策略的区别在于是否追加训练交互，以及新增经验来自目标环境还是源仿真。",
    ),
    1234: (
        "各学习过程统记为",
        "，下标分别表示纯离线学习、目标在线微调和跨保真学习。纯离线学习依赖采集策略已经尝试过的状态和动作；目标在线微调可获得新动作在目标环境中的结果；跨保真学习则在源仿真中尝试新动作，需处理源目标运行规律的差异。回报比较应结合各自获得经验的条件。",
    ),
    1235: (
        "目标离线数据提供已执行动作在目标环境中的结果，源仿真则响应当前策略选择的新动作。两者结合，可利用已有目标经验并扩展训练中的动作尝试，但新增源经验仍不等同于新增目标经验。",
    ),
    1238: (
        "式(5-2)中的差异有两层：状态和动作的出现频率不同，以及同一状态动作的后继结果分布不同。增加源交互能够扩展动作经验，却不能自动消除源目标转移差异；重复使用固定目标记录，也不能补出未执行动作的结果。因此，需分别考察目标数据覆盖、源仿真的动作响应和目标环境中的服务效果。",
    ),
    1239: (
        "第一项任务比较固定目标数据学习与追加目标交互；第二项改在源仿真中追加交互，检验目标数据辅助源仿真训练的效果。两项任务的具体设置如下。",
    ),
    1241: (
        "第一项任务以车头时距塑形奖励训练，同时报告乘客时间与运行指标；第二项以广义乘客时间成本训练，同时评价行程完成率。第一项的乘客时间统计对象为已完成乘车段，第二项则累计规定时域内所有已出发乘客已发生的时间，包括尚未完成行程者。因此，服务数值只在各自任务内比较。",
    ),
    1242: (
        "源目标差异还需与有限采样误差区分：运行规律不同可能使同一状态动作的后继结果分布不同，而同一运行规律的有限样本也可能给出不同估计。下面分别说明。",
    ),
    1244: (
        "仿真细节的多少不能直接衡量条件转移差异。若额外模拟的变量在给定共同状态后不再影响后继结果，或简化模型保留了这些变量造成的结果分布，两者仍可能具有相同的条件分布。本研究的简易仿真利用经验路段时间和服务关系推进事件，SUMO还模拟道路运动、跟驰及路口等待。省略的运行过程、有限数据标定和模型表达能力，都可能造成动作响应差异，其大小需由数据判断。",
    ),
    1245: (
        "状态简化也会影响这一比较。两种微观交通状态即使具有相同的候车人数、载客人数和车头时距，下游排队、信号相位或乘客组成仍可能不同。给定观测状态和动作时，后继分布还取决于这些未观测因素的分布，而不同环境或策略可能改变该分布。因此，经验响应差异既可能来自模型误差，也可能来自状态信息不完整，不能直接归因于转移机制不同；观测状态的马尔可夫性也不能由此保证。",
    ),
    1246: (
        "第2.4节的比较说明，总体服务接近不代表分项及动作响应相同。0.75、1.00和1.25倍需求下，源仿真相对SUMO的人均广义时间成本差为1.21%、0.25%和−1.85%，车内时间差为−2.07%、−3.59%和−5.36%。七种允许附加驻站的策略与三档需求构成21个组合，广义时间归一化响应差的各95%区间端点绝对值均不超过3.90%，完成率响应差的对应值为0.574个百分点。正常需求、载客较少且无后随车辆时，增加20秒驻站相对不额外驻站，使同车相邻决策间隔平均增加19.52秒（源仿真）和21.80秒（SUMO）；晚时段各需求条件下，扣除不额外驻站参照后的时间响应差最大绝对值为4.48秒。这些比较未参与标定，也未从相同微观状态逐条配对启动，反映的是服务与条件响应差异，而非转移核的总变差距离。",
    ),
    1253: (
        "利用有限运行记录建立仿真时，首次记录已有抽样误差，建模还可能引入状态简化与运行机制偏差。增加仿真样本主要减少后一次采样的波动，不能自动消除此前的估计与建模误差。因此，仍需结合目标数据判断这些差异，并在目标环境中检验控制效果。",
    ),
    1259: (
        "第一项多线路公交任务包含12个线路—方向组合、389个计划班次、353个物理站点和459个线路—站点组合记录。全部班次在同一回合运行，由同一策略控制符合条件的车辆到站事件。",
    ),
    1262: (
        "第二项跨保真任务将同时到站的待决策车辆组成一批，采用批次状态和混合驻站分布。简易仿真与SUMO共用第2章的基础运营资料及控制定义，以不同机制推进事件；固定目标数据来自SUMO，新增训练经验仅来自简易仿真。",
    ),
    1266: (
        "多辆公交可能在同一物理时刻需要驻站决策。策略分别编码各车的局部信息，再汇总同批车辆状态，使当前车辆的决策能够利用其他待决策车辆的信息。这一处理称为结构化状态编码。",
    ),
    1267: (
        "每批最多预留16个车辆位置，待决策车辆依次选择动作，空余位置不参与计算；车辆数超出上限时终止运行，不忽略多出的车辆。每个事件有16个原始观测量，将物理时刻、系统候车人数和车内人数移入全局信息后，剩余13个局部量。车辆容量用于人数和需求归一化，不再单列为输入，得到12个局部特征；加入有效标记和本批已选动作后，每个车辆位置为14维。五个全局量与16组车辆信息合计构成229维状态。",
    ),
    1268: (
        "演员、评论家和价值函数各自使用独立的编码器，同一编码器对各有效车辆使用相同参数。编码结果汇总为五类表示：当前车辆表征、全部有效车辆表征的均值与最大值、已选动作车辆表征的均值，以及尚未轮到决策车辆表征的均值。与五个全局量合并后形成256维表征，其中保留本批已选动作，供后续车辆决策使用。",
    ),
    1270: (
        "五个全局量为物理时刻、同时待决策车辆数、当前决策序号、系统候车人数和车内人数，均按规定尺度归一化。车辆表征反映局部运行情况，批次汇总反映共同状态及决策进度。后向车头时距由车辆位置和速度估计，客流到达率由分时OD计算，系统人数由仿真状态汇总，具体关系见第2.2.1节。",
    ),
    1273: (
        "跨保真策略区分不额外驻站与时长连续可选的附加驻站。以下",
        "沿用第2.2.2节的归一化动作定义，",
        "为策略参数。",
        "是集中在",
        "处的Dirac点质量，",
        "为连续时长分支的动作密度，",
        "表示均值为",
        "、方差为",
        "的高斯密度。",
    ),
    1277: (
        "令",
        "为决定是否额外驻站的门控网络输出，增加驻站时长的概率为",
    ),
    1282: (
        "其中",
        "在归一化动作",
        "处赋予非零概率，对应不额外驻站；",
        "给出连续时长分支的动作密度。归一化动作经",
        "映射为秒。确定性评价时，若",
        "，将高斯均值代入上述变换得到驻站时长，否则选择0秒附加驻站。",
    ),
    1290: (
        "评论家采用两个参数独立的价值估计头，相应目标网络以0.005的系数平滑更新。演员、评论家和价值函数在编码器后均设两个256单元隐藏层。第5.3节进一步说明这些状态、动作和数据条件下的离线初始化与跨保真续训。",
    ),
}


def main():
    assert not BACKUP.exists(), "Keep the existing pre-edit backup."
    with ZipFile(DOCUMENT) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        assert "".join(body[1227].itertext()) == "5.2 模型构建"
        assert "".join(body[1291].itertext()).startswith("5.3 ")
        updated = original
        for index, prose in REVISIONS.items():
            assert 1227 < index < 1291
            updated = replace_paragraph(updated, body[index], prose)

        # Preserve all non-text XML, including native math and field structure.
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
    print({"changed_blocks": changed, "paragraphs": len(changed), "backup": str(BACKUP), "output": str(DOCUMENT)})


if __name__ == "__main__":
    main()
