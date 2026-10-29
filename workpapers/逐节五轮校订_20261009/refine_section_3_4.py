from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_3_4 import DOCUMENT, BACKUP, W, M, revise, paragraph_pattern


REVISIONS = {
    816: ("走廊设22个站点，两端为终点。服务窗口06:00至19:00，两方向每360秒发车并错开180秒。车辆完成班次后回到终点，按时刻表承担后续任务。",),
    889: ("滚动均值目标降低绝对误差，但自动调温和固定温度组分别有99.889%和99.998%的均方误差改善来自共同偏移减小，两类回报均未提高。动作差拟合降低训练误差后，留出状态上动作间相对价值的误差反而增加，系统回报也下降。各网络去均值后再聚合，比未经处理的绝对价值模型取得更高后续回报，说明成员间偏移可经最小值聚合影响动作选择；但相对原演员或初始去均值模型，改善方向并不一致。",),
    929: ("累计奖励越高、时距不均衡度越低越好。两项均关闭仍保留集成网络、目标偏移及其他共同设置。",),
    895: ("每组用两个训练种子，固定策略后，在正常与快速切换场景各评价三个共同运行日。正常场景保留日内需求与随机车速；切换场景沿用BAPR配置，每1,800秒依次改变常规、严重拥堵、需求激增、极端天气和局部通行受限模式，区别于第4章的需求倍率测试。每回合至24,000秒，计入未完成者已累积的候车和车内时间。先平均种子内三个评价回合，再汇总两次训练；", "中的点表示训练重复。"),
    923: ("相对尺寸匹配的屏蔽组，嵌入在正常运行减少38.17秒，即2.32%，减少量区间[4.87，71.47]秒；客流激增和快速切换少49.97和39.08秒，区间跨零。", "保留各次训练结果，支持正常运行的类别信息收益，也显示另外两种条件的平均改善。"),
    938: ("比较两类分歧项在不同运行条件下的平均作用。",),
    954: ("同时显示运营指标和计算代价。2、5、10、20个评论家的平均训练耗时为81.97、83.14、82.37、94.17分钟，峰值已分配显存为27.70、38.75、55.76、91.18二进制兆字节（Mebibyte，MiB）。同一种子的四组使用同一节点，实际耗时受并发负载影响，已分配显存不等于进程总显存。部署仅调用演员，中央处理器（Central Processing Unit，CPU）平均单次调用为0.407至0.443毫秒，未随评论家数量持续增加。",),
    960: ("时距行以计划绝对偏差为指标，单位为秒；乘客行以人均广义时间成本为指标，单位为秒/已出发乘客。降幅以同一行集成参照的均值为分母。",),
    962: ("表3-10分析指定正则项，第3.4.6节比较多线路中的RE-SAC与SAC，参照和场景不同。两类目标分别检验相应运营要求；第4章研究非平稳变化，第5、6章再评价目标离线数据辅助训练的效果。",),
    973: ("本章结果支持类别表征、经验共享与保守动作评价的控制作用。多线路时距目标下，RE-SAC比SAC获得更高平均奖励，计划偏差与时距不均衡均下降；同一集成学习器的两项分歧正则比较还显示乘客时间目标下的平均收益。它们检验的对象不同，不能合并为所有目标和场景下的统一排序。",),
}
TABLE_REVISIONS = {
    832: {(7, 0): "仅权重范数惩罚", (8, 0): "仅集成分歧惩罚"},
    887: {(2, 0): "动作差减绝对价值拟合"},
}


def main():
    with ZipFile(DOCUMENT) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        edits = [(body[i], value) for i, value in REVISIONS.items()]
        for i, cells in TABLE_REVISIONS.items():
            for (r, c), value in cells.items():
                paragraph = body[i].findall(W + "tr")[r].findall(W + "tc")[c].find(W + "p")
                edits.append((paragraph, (value,)))
        updated = original
        with ZipFile(BACKUP) as before:
            baseline = before.read("word/document.xml").decode("utf-8")
            initial_body = ET.fromstring(baseline).find(W + "body")
            unchanged_label = initial_body[832].findall(W + "tr")[6].findall(W + "tc")[0].find(W + "p")
            pattern = paragraph_pattern(unchanged_label)
            saved = re.search(pattern, baseline, re.S).group(0)
            current = re.search(pattern, updated, re.S)
            updated = updated[:current.start()] + saved + updated[current.end():]
        for paragraph, values in edits:
            matches = list(re.finditer(paragraph_pattern(paragraph), updated, re.S))
            assert len(matches) == 1
            match = matches[0]
            revised = revise(match.group(0), paragraph, values)
            updated = updated[:match.start()] + revised + updated[match.end():]
        mask = lambda xml: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", xml, flags=re.S)
        assert mask(original) == mask(updated)
        with ZipFile(BACKUP) as before:
            baseline = before.read("word/document.xml").decode("utf-8")
            assert mask(baseline) == mask(updated)
            initial_body = ET.fromstring(baseline).find(W + "body")
            revised_body = ET.fromstring(updated).find(W + "body")
            changed = [i for i, (a, b) in enumerate(zip(initial_body, revised_body)) if ET.tostring(a) != ET.tostring(b)]
            assert len(initial_body) == len(revised_body) and all(812 <= i <= 979 for i in changed)
            for i in range(812, 980):
                assert [ET.tostring(n) for n in initial_body[i].iter(M + "oMath")] == [ET.tostring(n) for n in revised_body[i].iter(M + "oMath")], i
        temporary = DOCUMENT.with_suffix(".section-edit.docx")
        with ZipFile(temporary, "w") as target:
            for item in source.infolist():
                data = updated.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename)
                target.writestr(item, data)
    temporary.replace(DOCUMENT)
    print({"changed_blocks_from_backup": changed, "refined_paragraphs": list(REVISIONS),
           "table_labels": sum(len(cells) for cells in TABLE_REVISIONS.values())})


if __name__ == "__main__":
    main()
