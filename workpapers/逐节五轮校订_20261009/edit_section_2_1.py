from pathlib import Path
from shutil import copy2
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_1_2_1 import DOCUMENT, W, M, W14, revise_paragraph
from edit_section_1_2_4 import replace_text_nodes


BACKUP = Path(__file__).with_name("校订前_2.1.docx")

REVISIONS = {
    367: (
        "公交系统由道路网络、线路、站点、计划班次、车辆和乘客行程共同构成。同一线路的上、下行分别组织站点与班次，以下将线路与运行方向的组合简称为线路—方向组合。设其集合为",
        "，任一组合",
        "的站点序列为",
        "。序列中的元素是按行驶次序排列的物理站点标识，末项的站点序号表示该方向的站点总数。同一物理站点可被多条线路使用，物理站点标识描述空间位置，线路内站点序号描述车辆在当前班次中的进度，两者需要区分。",
    ),
    368: (
        "班次是按计划执行的一次线路服务，车辆是承担服务的实体，同一车辆可依次执行多个班次。本文以",
        "表示班次，以",
        "表示车辆，并用",
        "和",
        "分别表示计划与实际发车时刻。实际发车受车辆周转和当前运行状态约束，计划中安排了班次，并不意味着执行该班次的车辆已具备发车条件。",
    ),
    369: (
        "车辆执行班次时，依次经历发车、路段行驶、到站、上下客、驻站和离站。必要上下客时间由服务需求决定，附加驻站时间由控制决策决定，策略直接选择的是后者。区分两类时间，才能将正常服务所需的停留与控制额外引入的时间代价分别计算。",
    ),
    370: (
        "仿真运行条件由基础运营资料与客流、路况统计信息共同确定。基础运营资料包括线路及方向、站点序列、路段距离、车辆容量和发车计划；统计信息包括分时客流到达强度及其波动，以及分路段、分时段的速度或运行时间分布。前者规定供给结构，后者描述随机运行背景。均值、方差等统计量还需结合分布形式和样本生成规则，才能用于模拟具体的运行过程。",
    ),
    371: (
        "第3章走廊实验按给定的分时强度生成泊松到达过程，指定区间内到达人数的均值与方差由到达强度和区间长度确定；路段速度以分时均值和随机扰动的标准差描述，设置见第3.4.1节。多线路场景的需求率与速度估计方式见第2.2.1节，路段运行时间的标定与分布检验见第2.4.3节。走廊场景采用既定仿真设定，多线路场景依据已有资料和运行记录标定，基础数据来源见第2.4.1节。这些资料描述运行背景，状态、动作、奖励、后继状态及事件时间组成的转移样本则记录具体控制后果。第3、4章不预先使用目标域转移样本训练策略，但仍使用线路资料和场景统计设置。",
    ),
    373: (
        "设同一线路方向的相邻三个班次在某站的到站时刻分别为",
        "、",
        "和",
        "，当前车辆相对前车与后车的车头时距定义为",
    ),
    376: (
        "上述定义用于描述实际到站间隔。在线决策时，后车若尚未到站，后向间隔需根据当时可获得的位置、运行状态或预测量估计。因此，策略输入中的后向间隔是决策时刻的估计，事后评价中的实际后向间隔则由到站记录计算。",
    ),
    377: (
        "局部延误会改变车辆到站前积累的候车人数，并影响服务时间与后续车头时距。当前车辆与前车的到站间隔增大时，可能接载更多乘客，上客时间随之增长；后车若间隔较小，则可能面对较少候车需求。两车服务时间的差异可能进一步放大间隔失衡。因此，控制不能只依据单车相对时刻表的偏差，还需考虑前后车位置、车内人数和站点需求。",
    ),
    378: (
        "路况与客流变化通过不同途径影响运行。道路拥堵直接延长行驶时间，需求增加首先影响候车和上下客，并可能因车辆满载使部分乘客继续等待。给定线路和时段后，客流与路况按既定随机规律波动，对应第3章的平稳随机运行问题；这些规律发生突变、漂移或反复切换时，则需要识别新的运行条件，对应第4章的非平稳适应问题。",
    ),
    379: (
        "为说明偏差如何传播，考虑同一方向相邻班次在站点之间的演化。以",
        "和",
        "分别表示班次",
        "在第",
        "站的到站与实际离站时刻，以",
        "表示必要上下客时间，",
        "表示附加驻站时长，",
        "表示必要服务与附加驻站结束后，车辆等待实际驶离的时间。各段时间的关系为",
    ),
    381: (
        "其中",
        "表示从实际驶离当前站点至到达下一站的道路运行时间。跨保真仿真需要采用一致的时间划分：若将上下客、驻站和驶离等待计入道路运行时间，又在简易仿真中分别执行这些过程，就会重复计算延误。",
    ),
    385: (
        "控制器直接调节的是附加驻站时长",
        "，而不是任意指定下一站的车头时距",
        "。必要服务时间随上下客人数变化，驶离等待和道路运行时间也受交通条件影响。当前车辆与前车的间隔已过大时，继续驻站通常会使这一间隔进一步增大；是否仍需驻站，还要结合后车位置、下游需求及乘客时间代价判断。",
    ),
    386: (
        "在车辆容量充足、无上次遗留候车乘客，且乘客以近似不变的强度",
        "到达的局部近似下，本次新增上车人数的期望满足",
        "，其中括号内表示本班次在该站新增的上车人数。若每名上车乘客的平均服务时间为",
        "，服务时间对前向间隔的响应约为",
        "。较大的间隔可能带来更多上客和更长服务，进而继续增大间隔，形成正反馈。车辆达到容量上限后，新增需求不再等量增加本车上客人数，而可能使未能上车的乘客继续候车。需求冲击实验因此需要区分容量充足与容量受限两种情况。",
    ),
    387: (
        "车辆传播关系相同，并不意味着驻站控制的目标相同。按时刻表控制以计划离站时刻为参照，按车头时距控制以相邻车辆的服务间隔为参照，乘客导向控制则直接考虑候车与车内延误。三者的优化对象和所需信息不同。以下用典型形式说明这种差别，具体控制律还取决于各项研究对站点、需求和执行条件的假设。",
    ),
    388: (
        "令带上划线的",
        "表示计划离站时刻，",
        "表示参考间隔，",
        "表示将计算出的驻站时长限制在0至允许上限之间，其中上限为最大附加驻站时长。两类典型反馈为",
    ),
    390: (
        "其中",
        "分别为前、后向间隔的反馈增益；两类驻站命令的上标分别表示时刻表反馈和车头时距反馈，后向时距上的帽号表示决策时刻的估计。时刻表反馈通过延后离站纠正提前运行，但无法消除已发生的晚点。车头时距反馈综合前后车关系：前向间隔偏小或后向间隔偏大时，相应反馈项倾向于增加驻站，最终时长由两项合计后限幅确定。这类反馈主要处理间隔失衡，计划偏差、车辆容量与换乘需求仍需结合具体服务目标考虑。",
    ),
    391: (
        "乘客导向模型进一步计算驻站的时间收益与代价。若当前车辆载客",
        "人，附加驻站",
        "秒直接增加约",
        "乘客秒的车内时间；是否减少候车时间，还取决于站点到达率、后续车辆位置和剩余容量。解析随机模型在给定假设下计算这一权衡，滚动预测模型则同时考虑未来多个站点和车辆的演化。",
        "滚动模型根据当前观测预测后续需求与运行状态，选择使预测时域内成本较低的一组动作，只执行当前动作，再根据新观测重新计算。",
    ),
    392: (
        "上述方法提供三种决策方式：直接反馈依据少量实时运行量计算动作，预测优化根据未来运行模型求解动作，强化学习利用已有交互学习价值函数或策略，将主要计算放在训练阶段。预测优化的效果取决于预测能否反映实际运行，学习型控制的效果则取决于训练经验能否支持部署时的动作选择。本文采用学习型驻站控制，沿用车辆传播、间隔反馈与乘客成本的基本关系，进一步研究随机运行、规律变化及仿真与目标环境差异对策略学习的影响。",
    ),
    394: (
        "附加驻站是在必要上下客完成后主动延后离站，不是缩短或取消必要服务。它直接改变前后车辆的时间关系，并可能改变下游乘客在车辆之间的分配。延后离站可能缓解间隔失衡，也会增加车内乘客时间。因此，时距改善与乘客时间收益需要分别评价，不能由其中一项直接推断另一项。",
    ),
    395: (
        "本文各章用于方法比较的公交实验均以附加驻站时长为策略动作，速度倍率固定为1，实际车速仍受道路限速、信号和跟驰条件影响。各算法在相同驻站权限下比较。用于构造标定偏差的道路平均速度是环境设定，而不是策略动作；策略通过调整离站时刻适应这些运行条件。",
    ),
    396: (
        "本文研究给定线路、车辆容量与发车计划下的驻站控制。计划间隔提供服务参照，车辆周转和交通条件影响计划执行。后续章节在同一控制范围内，依次研究随机运行、运行规律变化及目标数据辅助的跨保真学习。",
    ),
}

TEXT_REVISIONS = {
    384: {2: "说明，必要服务、附加驻站、驶离等待和道路运行的时间差，共同影响相邻班次在下一站的车头时距。"},
    397: {
        2: "给出公交事件决策的控制闭环。必要上下客完成后，策略根据当前状态选择附加驻站时长；",
        3: "",
        4: "执行结果通过后续事件中的状态、奖励和乘客成本反馈给学习过程。",
    },
}


def main():
    with ZipFile(DOCUMENT) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        assert "".join(body[365].itertext()) == "2.1 公交运行与控制"
        assert "".join(body[398].itertext()) == "2.2 强化学习基础"
        updated = original
        indices = sorted(REVISIONS.keys() | TEXT_REVISIONS.keys())
        for index in indices:
            paragraph = body[index]
            paragraph_id = paragraph.get(W14 + "paraId")
            pattern = r'<w:p\b[^>]*w14:paraId="' + paragraph_id + r'"[^>]*>.*?</w:p>'
            matches = list(re.finditer(pattern, updated, flags=re.S))
            assert len(matches) == 1, (index, paragraph_id)
            match = matches[0]
            raw = match.group(0)
            if index in TEXT_REVISIONS:
                revised = replace_text_nodes(raw, TEXT_REVISIONS[index])
            else:
                revised = revise_paragraph(raw, paragraph, REVISIONS[index])
            # Change prose only, retaining the original fields and OMML structures.
            mask = lambda text: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", text, flags=re.S)
            assert mask(raw) == mask(revised)
            updated = updated[:match.start()] + revised + updated[match.end():]
        new_body = ET.fromstring(updated).find(W + "body")
        assert len(body) == len(new_body)
        changed = [i for i, (a, b) in enumerate(zip(body, new_body)) if ET.tostring(a) != ET.tostring(b)]
        assert changed == indices, changed
        for index in changed:
            for tag in (W + "hyperlink", M + "oMath", W + "pPr", W + "rPr", W + "instrText", W + "fldChar"):
                assert [ET.tostring(x) for x in body[index].iter(tag)] == [ET.tostring(x) for x in new_body[index].iter(tag)]
        assert not BACKUP.exists(), "Preserve the existing backup before repeating this edit."
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
    print("Only section 2.1 prose changed; equations, fields, citations and formatting retained.")
    for index in changed:
        print(index, "".join(n.text or "" for n in new_body[index].iter() if n.tag in (W + "t", M + "t")))
    print("OUTPUT", DOCUMENT)


if __name__ == "__main__":
    main()
