"""Produce a new language revision and a reviewable edit ledger; leave inputs intact."""
from pathlib import Path
from difflib import SequenceMatcher
import json, re, os

A = Path(__file__).resolve().parent
OUTDIR = A.parents[1]
SOURCE = A/'修改前_重构论文.md'
OUT = OUTDIR/'重构论文_语言修订_v2_20261015.md'
original = SOURCE.read_text()
edits = json.loads((A/'edits_ch1_3.json').read_text())
edits += json.loads((A/'edits_ch4_7.json').read_text())['edits']
if (A/'data_table_edits.json').exists():
    data = json.loads((A/'data_table_edits.json').read_text())
    edits += data.get('edits', data) if isinstance(data, dict) else data
intervals = []
for edit in edits:
    assert original.count(edit['old']) == 1, edit.get('id', edit.get('table'))
    i = original.index(edit['old'])
    intervals.append((i, i+len(edit['old']), edit))
intervals.sort(key=lambda item: item[0])
for previous, current in zip(intervals, intervals[1:]):
    assert previous[1] <= current[0], (previous[2].get('id'), current[2].get('id'))
new = original
for start, end, edit in reversed(intervals):
    new = new[:start]+edit['new']+new[end:]

rules = []
def change(old, replacement, reason):
    global new
    count = new.count(old)
    if count:
        new = new.replace(old, replacement)
        rules.append({'old': old, 'new': replacement, 'count': count, 'reason': reason})

change('跨保真运行环境下基于迁移强化学习', '基于跨保真度迁移强化学习', '使跨保真度修饰迁移学习方法')
change('第5、6章跨保真源仿真', '第5、6章源仿真配置', '源环境本身不是跨保真度')
count = len(re.findall(r'跨保真(?!度)', new))
new = re.sub(r'跨保真(?!度)', '跨保真度', new)
rules.append({'old': '跨保真', 'new': '跨保真度', 'count': count, 'reason': '统一术语'})
for old, replacement in [
    ('参考状态归零神经修正', '在参考状态处归零的神经修正'),
    ('参考归零神经修正', '在参考状态处归零的神经修正'),
    ('参考归零基准动作结构', '可学习基准动作与在参考状态处归零的神经修正结构'),
    ('参考状态归零输出结构', '神经修正项在参考状态处归零的输出结构'),
    ('参考状态归零修正结构', '在参考状态处归零的神经修正结构'),
    ('参考归零修正结构', '在参考状态处归零的神经修正结构'),
    ('参考归零差分结构', '在参考状态处归零的差分修正结构'),
]: change(old, replacement, '明确归零的是修正项输出')
change('已出发乘客', '已进入系统的乘客', '避免误解为乘客已经上车离站')
change('已出发的乘客', '已进入系统的乘客', '统一乘客统计口径术语')
change('原源域回归目标', '原源域样本的回归目标', '消除重复搭配')
change('同策略近期交互', '近期在线交互', '近期样本不表示严格同策略采样')
change('近期同策略交互窗口', '近期在线交互窗口', '统一诊断窗口表述')
change('同策略近期状态—动作对', '近期在线状态—动作对', '明确样本来源')
change('最新256条近期在线交互', '最新256条在线交互', '删除重复的近期')
change('最近256条近期在线交互', '最近256条在线交互', '删除重复的近期')
change('最近 10,000 个近期在线状态—动作对', '最近 10,000 个在线状态—动作对', '删除重复的近期')
change('（按未舍入结果计算）', '', '已核实原始差值并增加端点展示精度')
change('按未舍入结果计算，', '', '已核实原始差值并增加端点展示精度')
change(r'\vert{}', r'\vert ', '规范TeX定界符写法，不改变公式含义')
change(r'\Vert{}', r'\Vert ', '规范TeX定界符写法，不改变公式含义')
change('策略仍能取得与完整CS-BAPR接近的乘客成本降幅，完整CS-BAPR相对于RE-SAC的收益因此只可归属于所比较的完整配置。该同结构对照尚未显示SINDy约束带来的独立乘客成本优势。',
       '乘客成本降幅仍与完整CS-BAPR接近。这项对照未显示SINDy约束独立降低乘客成本的优势。', '直接陈述消融结果，保留未显示独立收益的结论')
change('现有对照尚未显示SINDy约束独立降低乘客成本的优势，各组件的独立贡献仍有待区分',
       '同结构消融对照未显示SINDy约束独立降低乘客成本的优势', '减少防御性重复说明')
old_g = ('完整控制器横向对比采用各固定策略在未参与训练的5个新SUMO测试运行日及 '
         '$0.75\\text{、}1.00\\text{、}1.25$ 三档客流需求（合计225个24,000秒完整SUMO回合）下开展评价：')
new_g = ('各固定策略在未参与训练的5个新SUMO测试运行日及 '
         '$0.75\\text{、}1.00\\text{、}1.25$ 三档客流需求下接受评价，合计225个24,000秒完整SUMO回合。结果如下：')
change(old_g, new_g, '修正采用与开展评价的搭配')
change('# 附录F 受控动力学任务模型与跨保真度补充实验设置', '# 附录G 受控动力学任务模型与跨保真度补充实验设置', '与已校核Word的附录字母一致')
change('附录F第五部分', '附录G第五部分', '与既有Word指向一致')
change(r'\tag{附F-', r'\tag{附G-', '与既有Word公式号一致')

# Remove emphasis from ordinary prose and data, retain short paragraph headings and column labels.
lines = new.splitlines(); bold_removed = 0
for i, line in enumerate(lines):
    header = line.lstrip().startswith('|') and i+1 < len(lines) and re.match(r'^\s*\|[ :|\-]+\|\s*$', lines[i+1])
    if header: continue
    def unbold(match):
        global bold_removed
        content = match[1]
        lead = line[:match.start()].strip()
        structured = not lead and re.match(r'^(?:（[一二三四五六七八九十0-9]+）|\([0-9]+\)|[1-9][0-9]*[.、]|定理|引理|命题|证明)', content)
        if structured and len(content) <= 95: return match[0]
        bold_removed += 1
        return content
    lines[i] = re.sub(r'\*\*(.+?)\*\*', unbold, line)
new = '\n'.join(lines)+'\n'
rules.append({'old': '正文中间及表格数据加粗', 'new': '普通正文与数据字重；保留标题和列名', 'count': bold_removed, 'reason': '清理过度强调'})

# The delivery Markdown displays the actual existing figures, not production instructions.
figure_map = json.loads((A.parent/'figure_map_final.json').read_text())['figures']
figures = {entry['figure']: entry for entry in figure_map}
updates = json.loads((A/'figure_language_updates.json').read_text()) if (A/'figure_language_updates.json').exists() else []
for update in updates: figures[update['figure']]['source'] = update['png']
figure_links = []
def figure(match):
    label = re.match(r'【(图\d+-\d+)', match[0])[1]
    path = Path(figures[label]['source']); assert path.exists(), path
    link = os.path.relpath(path, OUTDIR)
    result = '!['+label+']('+link+')'
    figure_links.append({'figure': label, 'old': match[0], 'new': result, 'path': str(path)})
    return result
new = re.sub(r'^【图[^\n]*】[ \t]*$', figure, new, flags=re.M)
assert len(figure_links) == 57

if (A/'final_language_suggestions.json').exists():
    final = json.loads((A/'final_language_suggestions.json').read_text())
    final = final.get('edits', final) if isinstance(final, dict) else final
    for suggestion in final:
        assert new.count(suggestion['old']) == 1, suggestion.get('id')
        change(suggestion['old'], suggestion['new'], suggestion.get('reason', '语法与措辞收尾'))
change('综合表6-3与图6-4至图6-6的结果可以看出，MC-WM在受控动力学偏移下展现出显著且稳健的适应优势：',
       '表6-3与图6-4至图6-6汇总了MC-WM相对于不同基线的配对比较：', '明确比较对象，避免笼统的优越性判断')
change('这证实了在有限目标交互预算下，利用学习到的世界模型生成一步预测样本辅助价值更新，其样本效率与适应性能远超纯无模型学习与分类器奖励重加权。',
       '在相同目标交互预算下，MC-WM在三个任务中的回报均高于仅目标数据SAC和DARC。', '按实际对照陈述回报，不泛化为整体优越性')
change('但完全由第4个训练种子的单次下降', '但均值下降主要来自第4个训练种子的单次下降', '客观表述单种子对均值的贡献')
for suggestion in json.loads((A/'final_review_edits.json').read_text())['edits']:
    assert new.count(suggestion['old']) == 1, suggestion['id']
    change(suggestion['old'], suggestion['new'], suggestion['reason'])

cites = re.compile(r'\[([A-Za-z]+\d{4}[\w-]*(?:\s*,\s*[A-Za-z]+\d{4}[\w-]*)*)\]')
keys = lambda s: [k.strip() for m in cites.finditer(s) for k in m[1].split(',')]
assert keys(original) == keys(new), 'Citation sequence changed'
tags = lambda s: re.findall(r'\\tag\{([^}]+)\}', s.replace('附F-', '附G-'))
assert tags(original) == tags(new), 'Numbered equation sequence changed'
OUT.write_text(new)
(A/'applied_language_edits.json').write_text(json.dumps(edits, ensure_ascii=False, indent=2))
(A/'language_rules.json').write_text(json.dumps(rules, ensure_ascii=False, indent=2))
(A/'markdown_figure_links.json').write_text(json.dumps(figure_links, ensure_ascii=False, indent=2))
print(json.dumps({'output': str(OUT), 'explicit_edits': len(edits), 'bold_spans_removed': bold_removed, 'real_figure_links': len(figure_links), 'citation_sequence_unchanged': True}, ensure_ascii=False))
