from html import escape
from pathlib import Path
from shutil import copy2
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
DOCUMENT = ROOT / "output/排版完成_逐节校订_第1轮_20261009.docx"
BACKUP = Path(__file__).with_name("校订前_1.2.1.docx")
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
W14 = "{http://schemas.microsoft.com/office/word/2010/wordml}"

# Each entry supplies prose between the existing citations and native equations.
REVISIONS = {
    264: (
        "公交驻站控制是在既有线网、发车频率、时刻表和车辆安排下实施的运行调整。频率配置研究将乘客时间成本纳入发车间隔优化",
        "；时刻表研究分别考虑双向排班协同",
        "、多模式服务",
        "、高铁接驳方式竞争",
        "及需求响应公交的发车时刻与票价",
        "。线网与站点研究涉及地铁多站接运",
        "和双城公交的碳排放协调",
        "；车辆运用研究包括场站泊位分配",
        "、给定时刻表下的车辆调度",
        "以及共线线路发车协调",
        "。这些决策确定了运行控制所依赖的线路、车辆和班次条件。Ibarra-Rojas等按决策层次梳理公交规划、运营与控制，指出服务水平与运营成本之间的权衡贯穿各层次。",
        "在这些条件给定后，驻站控制通过实时调整离站时刻处理沿途运行偏差。因此，比较驻站方法时，需要保持车辆供给与发车计划一致；允许调整供给的研究则应另外说明其决策范围。",
    ),
    265: (
        "早期驻站研究关注如何在随机运行条件下及时求得控制动作。Hickman将随机运行时间和乘客上下客过程纳入解析模型，把单车控制点的驻站问题表述为单变量凸二次规划。",
        "该方法在明确的运行假设下计算驻站时长，为分析局部控制的时间收益与代价提供了基础。进一步考虑相邻车辆和未来需求时，决策所需的信息及其预测范围随之扩大。",
    ),
    266: (
        "基于车头时距的反馈控制利用相邻车辆的到达或离开时刻调节驻站。Daganzo分析了在多个控制点采用实时车头时距的策略，讨论达到给定间隔容差所需的时刻表预留时间。",
        "这类方法将反馈系数与线路运行稳定性联系起来，其控制依据是车辆之间的相对间隔，而不只是各车相对时刻表的晚点。",
    ),
    267: (
        "车辆间隔均衡与按时刻表运行是两个不同目标。Xuan等引入相对于虚拟时刻表的到达偏差，分析线性动态驻站策略，将计划偏差与间隔规律性共同纳入控制。",
        "对于具有明确班次和终点周转安排的线路，车辆即使保持近似均匀的间隔，也可能整体晚于服务计划。因此，驻站目标需根据运营任务确定，时距均衡不能替代全部服务要求。",
    ),
    268: (
        "滚动优化通过预测后续车辆运行和乘客需求，评价当前驻站的未来影响。赖心荷等以车头时距偏差和乘客车内等待为目标，研究滚动时域下的车队协同驻站。",
        "Delgado等以乘客总延误为目标，联合分析驻站与限制上客，并考虑车辆容量差异及动态服务过程。",
        "Andres和Nair将车头时距预测与动态驻站结合，利用实际线路记录建立预测，再在仿真中评价预测信息的控制作用。",
        "这些研究把决策范围从当前车辆扩展到后续运行，但预测准确程度与最终服务收益仍需分别检验。",
    ),
    269: (
        "除预测未来运行外，控制还可以由只利用前车信息，扩展为同时利用前车和后车信息。Daganzo和Pilachowski研究基于前后车间距的车际协同速度调节；Bartholdi和Eisenstein研究能够自行调节车头时距的线路组织。",
        "Cats等基于车辆定位记录分析按时刻表驻站，并在仿真中评价同时利用前后车间隔的逐站控制，还讨论驾驶员执行和始发站调度对实施效果的影响。",
        "这些方法说明，当前车辆的动作会同时改变前后服务间隔，因而不能脱离相邻车辆评价单车控制。",
    ),
    270: (
        "控制位置与驻站时长同样相互关联。Dai等用合作博弈描述多个控制点的共同作用，根据预测的车头时距评价不同组合，动态判断各控制点的重要性。",
        "与在每个站点采用相同规则相比，该研究强调干预位置及控制点之间的相互作用。Sánchez-Martínez等将随时间变化的需求和运行时间纳入驻站优化，据此更新控制计划。",
        "两项研究分别提示：策略输入需要区分车辆所处位置，控制预测也需要反映需求和道路条件的时间变化。已知的日内变化可以通过时段信息描述，不能将它与尚未识别的运行规律变化混为一谈。",
    ),
    271: (
        "乘客行为与车辆约束也决定了同一动作的运行后果。随机行程时间研究考察运行不确定性对电动公交调度的影响",
        "；车速调控与乘车引导研究同时考虑车辆运行和乘客选择，优化乘客出行成本与车头时距偏差。",
        "Wu等将车辆超越、容量限制及串车时乘客重新选择车辆队列的行为纳入公交运行模型，分析相应条件下的驻站控制。",
        "即使两个仿真的平均行程时间接近，上客分配和车辆相互作用不同，也可能使同一驻站动作产生不同的后续状态与乘客时间。因此，跨保真学习需要比较动作后的运行响应，而不只是平均运行指标。",
    ),
    272: (
        "相关综述进一步归纳了控制方式与服务目标之间的关系。Gkiotsalitis和Cats比较驻站、动态发车和跳站等措施；Rezazada等从需求、供给和决策三个角度梳理串车及其控制权衡。",
        "间隔规律性只是其中一项评价要求，驻站造成的车内延误、容量约束和乘客能否获得服务，也会影响最终控制效果。",
    ),
    273: (
        "公交运营效率与服务质量评价综述也强调从多个方面判断运行效果。",
        "据此，本文在车头时距指标之外，同时考察候车时间、车内时间和行程完成情况。下述国内研究进一步说明，不同的乘客需求和控制方式如何影响这些评价结果。",
    ),
    274: (
        "国内研究结合接驳、共线和换乘等运营任务处理运行失衡。接驳公交共线段站点优化属于供给设计",
        "；网联公交编队引导通过车速协同改善站点附近运行",
        "；多线路换乘协同调度联合考虑发车频率与换乘站驻站。",
        "何胜学研究双线单换乘区的驻站学习控制，将换乘引起的停站不确定性与线路稳定性联系起来。",
        "赵琥等针对重叠线路，通过车速诱导调整车辆进入共同区间的时序；李利华等针对站点群体聚集性客流，联合调整驻站时间和路段速度，权衡车头时距与乘客行程时间。",
        "这些研究的需求条件、线路相互作用和可调节动作不同，比较时需要分别说明。本文以完成必要上下客后的附加驻站为共同动作，保持各组既定线路、班次与供给计划，在相同动作范围内比较学习方法，并以无需训练的驻站规则提供运行参照。",
    ),
    275: (
        "刘鹏杰的公交信号协同优先研究从单交叉口、干线、路网扩展至区域边界，通过信号配时改善车辆通行及动态时刻表的执行。",
        "本文沿用其长沙场景的道路、线路、需求及时刻表资料，但研究的是车辆附加驻站，以及训练环境与目标环境不同条件下的策略学习。信号优先调节道路通行条件，驻站控制调节车辆离站时刻；共同基础资料便于说明应用场景，两类研究的收益仍需按各自控制方式评价。",
    ),
    276: (
        "驻站、跳站、速度调节、信号优先和动态发车作用于不同运行环节，分别调整离站时刻、停靠服务、路段运行、道路通行权和始发供给。",
        "生态车速与信号优先协同研究兼顾延误、燃料消耗及到站偏差",
        "；公交优先配时研究权衡出行者总延误、排放、停车次数和通行能力",
        "；多类型线路研究结合服务模式、速度与发车设置改善可靠性",
        "；干线协调研究兼顾公交与社会车辆的多路径通行。",
        "跳站可能减少部分乘客的服务机会，速度调节和信号优先需要相应的车辆或道路控制权限，动态发车则改变后续供给计划。驻站利用已有停靠过程，在保留线路及基本服务覆盖的条件下调整车辆间隔，适合作为本文不同学习方法的共同动作。它只能延缓离站，并直接增加车内乘客时间，故需同时评价候车收益、车内代价和允许的驻站上限。",
    ),
    277: (
        "在既定驻站动作下，仿真优化与学习方法采用不同的决策求解方式。传统仿真优化常先指定控制规则，例如由反馈系数计算驻站时长，再搜索使平均乘客成本较低的系数。其结构清楚、参数较少，但状态与动作的关系形式预先给定；滚动优化则在每次决策时重新预测并求解。学习型控制直接学习运行状态到动作的映射：给定状态",
        "，由条件动作分布",
        "选择驻站动作，并以后续运行的期望成本评价这一选择。策略由多次仿真的经验更新，使驻站响应能够结合客流、前后间隔和服务阶段等运行信息。已有强化学习驻站研究为这一求解方式提供了基础。",
    ),
    278: (
        "两类方法并非在数学上截然分离。神经网络策略仍由有限参数表示，仿真优化也可以搜索参数化策略；强化学习的特点在于利用动作价值估计和经验复用，把后续运行收益与当前动作联系起来。其适用性取决于训练经验是否覆盖所需状态和动作，以及价值估计能否反映部署条件。本文后续研究据此逐步展开：先在给定随机规律下改进动作评价，再处理运行规律变化后的策略适应，最后利用目标离线数据开展迁移训练与模型修正。",
    ),
}


def revise_paragraph(raw, paragraph, prose):
    text_nodes = list(paragraph.iter(W + "t"))
    positions = {id(node): i for i, node in enumerate(text_nodes)}
    groups, current = [], []
    for child in paragraph:
        if child.tag == W + "r" and child.find(W + "rPr/" + W + "vertAlign") is None:
            current.extend(positions[id(node)] for node in child.iter(W + "t"))
        elif child.tag in (W + "hyperlink", M + "oMath") or child.tag == W + "r":
            if current:
                groups.append(current)
                current = []
    if current:
        groups.append(current)
    assert len(groups) == len(prose), (len(groups), len(prose))
    replacements = {}
    for indices, text in zip(groups, prose):
        replacements[indices[0]] = text
        replacements.update((index, "") for index in indices[1:])
    position = -1

    def replace(match):
        nonlocal position
        position += 1
        if position not in replacements:
            return match.group(0)
        return match.group(1) + escape(replacements[position], quote=False) + match.group(3)

    result = re.sub(r"(<w:t(?:\s[^>]*)?>)(.*?)(</w:t>)", replace, raw, flags=re.S)
    assert position + 1 == len(text_nodes)
    return result


def main():
    with ZipFile(DOCUMENT) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        assert "1.2.1 公交驻站控制" == "".join(body[263].itertext())
        updated = original
        for index, prose in REVISIONS.items():
            paragraph_id = body[index].get(W14 + "paraId")
            pattern = r'<w:p\b[^>]*w14:paraId="' + paragraph_id + r'"[^>]*>.*?</w:p>'
            matches = list(re.finditer(pattern, updated, flags=re.S))
            assert len(matches) == 1, (index, paragraph_id)
            match = matches[0]
            revised = revise_paragraph(match.group(0), body[index], prose)
            # Only ordinary prose changes; all citation, math and formatting XML stays intact.
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
        assert [name for name in before.namelist() if before.read(name) != after.read(name)] == ["word/document.xml"]
    print("Changed body blocks:", changed)
    print("Abstracts and all other sections unchanged; native math, citations and formatting preserved.")
    for index in changed:
        print(index, "".join(n.text or "" for n in new_body[index].iter() if n.tag in (W + "t", M + "t")))
    print("OUTPUT", DOCUMENT)


if __name__ == "__main__":
    main()
