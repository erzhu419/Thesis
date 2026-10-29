"""Apply the concrete numbering, notation and appendix fixes found in the review."""
from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree as E
import json, re, shutil

A = Path(__file__).resolve().parent
ROOT = A.parent
BASE = A / '通篇细节检查前.docx'
OUT = A / '通篇细节检查_工作稿.docx'
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
M = '{http://schemas.openxmlformats.org/officeDocument/2006/math}'
NS = {'w': W[1:-1], 'm': M[1:-1]}


def text(e):
    return ''.join(n.text or '' for n in e.iter() if n.tag in (W+'t', M+'t'))


def style(e):
    s = e.find('./'+W+'pPr/'+W+'pStyle')
    return s.get(W+'val') if s is not None else ''


def replace(e, old, new):
    nodes = [n for n in e.iter() if n.tag in (W+'t', M+'t')]
    value = ''.join(n.text or '' for n in nodes)
    assert value.count(old) == 1, (old, value[:160])
    start = value.index(old); end = start+len(old); offset = 0; first = True
    for n in nodes:
        v = n.text or ''; lo = max(start-offset, 0); hi = min(end-offset, len(v))
        if lo < hi:
            assert n.tag == W+'t', ('Attempt to modify mathematics', old)
            n.text = v[:lo]+(new if first else '')+v[hi:]; first = False
        offset += len(v)


with ZipFile(BASE) as z:
    original = z.read('word/document.xml')
    root = E.fromstring(original); body = root.find(W+'body')
    changes = []
    # Keep the former section labels as locators, but use bold paragraph lead-ins.
    appendix = False
    for p in list(body):
        if style(p) == 'chapter' and text(p).startswith('附录'):
            appendix = True
        if not appendix or style(p) not in ('h2', 'h3', 'h4'):
            continue
        following = p.getnext()
        assert following.tag == W+'p' and style(following) == 'body'
        assert not p.findall('./'+W+'pPr/'+W+'sectPr')
        title = text(p)
        prefix = [deepcopy(n) for n in p if n.tag != W+'pPr']
        for n in prefix:
            for run in ([n] if n.tag == W+'r' else list(n.iter(W+'r'))):
                pr = run.find(W+'rPr')
                if pr is None:
                    pr = E.Element(W+'rPr'); run.insert(0, pr)
                b = pr.find(W+'b')
                if b is None: b = E.SubElement(pr, W+'b')
                b.set(W+'val', '1')
        punctuation = E.Element(W+'r')
        pr = E.SubElement(punctuation, W+'rPr'); E.SubElement(pr, W+'b')
        t = E.SubElement(punctuation, W+'t'); t.text = '。'; prefix.append(punctuation)
        at = 1 if following.find(W+'pPr') is not None else 0
        for n in prefix:
            following.insert(at, n); at += 1
        body.remove(p)
        changes.append({'type': 'appendix_lead_in', 'title': title})
    assert len(changes) == 50

    # Repair only the obsolete numeric references, preserving the Chinese wording.
    replacements = [
        ('第3.2.4节从最不利运行后果', '3.2.4', '3.2.5'),
        ('的策略基础目标', '4-17', '4-12'),
        ('第6.3.3节改变即时对数置信', '6.3.3', '6.3.2'),
        ('因此本命题用于解释第6.3.3节', '6.3.3', '6.3.2'),
        ('新增路口沿用第6.4.4节的泊松计数规则', '6.4.4', '6.4.3'),
        ('轨迹等价性与误判误差界证明见附录B定理B.1至B.3', 'B.3', 'B.10'),
    ]
    for locator, old, new in replacements:
        ps = [p for p in body if locator in text(p) and not style(p).startswith('TOC')]
        assert len(ps) == 1, (locator, len(ps))
        replace(ps[0], old, new)
        changes.append({'type': 'numeric_cross_reference', 'locator': locator, 'old': old, 'new': new})

    # Both the tram DQN and VSL papers support state/reward design, not game theory.
    p, = [p for p in body if '状态物理机理的融入与奖励函数的合理塑形' in text(p) and style(p) == 'body']
    links = {n.get(W+'anchor'): n for n in p.iter(W+'hyperlink')}
    insertion = links['latest_ref_91']
    for old_number in (96, 97):
        moved = links['latest_ref_'+str(old_number)]; comma = moved.getnext()
        assert comma.tag == W+'r' and text(comma) == ','
        assert moved.getparent() is p
        p.remove(comma); p.remove(moved)
        insertion.addnext(comma); comma.addnext(moved); insertion = moved
    assert '合理塑形[90,91,96,97]' in text(p) and '合作博弈机制[98]' in text(p)

    remap = {n: (92 if n == 96 else 93 if n == 97 else n+2 if 92 <= n <= 95 else n) for n in range(1, 227)}
    for n in root.iter(W+'hyperlink'):
        anchor = n.get(W+'anchor', '')
        if not anchor.startswith('latest_ref_'): continue
        old = int(anchor.removeprefix('latest_ref_')); new = remap[old]
        if old != new:
            replace(n, str(old), str(new)); n.set(W+'anchor', 'latest_ref_'+str(new))
    bib = [p for p in body if style(p) == 'bibliography']
    assert len(bib) == 226
    by_old = {int(re.match(r'^\[(\d+)\]', text(p))[1]): p for p in bib}
    by_old[91].addnext(by_old[96]); by_old[96].addnext(by_old[97])
    for old, p in by_old.items():
        new = remap[old]
        if old == new: continue
        replace(p, '['+str(old)+']', '['+str(new)+']')
        mark, = [m for m in p.iter(W+'bookmarkStart') if m.get(W+'name') == 'latest_ref_'+str(old)]
        mark.set(W+'name', 'latest_ref_'+str(new))
    changes.extend([
        {'type': 'citation_context', 'paper': '王云鹏等2019有轨电车信号优先', 'old_number': 96, 'new_number': 92},
        {'type': 'citation_context', 'paper': '韩磊等2023可变限速', 'old_number': 97, 'new_number': 93},
    ])

    # Repair fragments are supplied by the mathematical review, independently of the body text.
    repaired = E.parse(str(A/'qa_equation_3_35_corrected.xml')).getroot()
    eq_table, = [e for e in body if e.tag == W+'tbl' and text(e).endswith('(3-35)')]
    existing, = eq_table.findall('.//'+M+'oMath')
    # Preserve the source's minus/prime characters; only delimiter grouping changes.
    for token in repaired.iter(M+'t'):
        token.text = (token.text or '').replace('−', '-').replace('′', "'")
    payload = lambda e: ''.join(n.text or '' for n in e.iter(M+'t')).replace('|', '').replace('∥', '')
    assert payload(existing) == payload(repaired), 'Equation 3-35 algebra changed'
    existing.getparent().replace(existing, repaired)
    changes.append({'type': 'math_delimiters', 'equation': '3-35'})

    aliases = json.loads((A/'qa_symbol_aliases.json').read_text())
    symbol_table = list(body)[42]
    assert symbol_table.tag == W+'tbl' and text(symbol_table).startswith('符号含义')
    for alias in aliases:
        row = symbol_table.findall(W+'tr')[alias['row_index']]
        cell = row.findall(W+'tc')[0]
        para = cell.find(W+'p')
        assert text(cell) == alias['old_text'], (alias, text(cell))
        run = E.SubElement(para, W+'r'); E.SubElement(run, W+'t').text = '；'
        para.append(E.parse(str(A/alias['fragment'])).getroot())
        changes.append({'type': 'symbol_alias', 'old': alias['old_text'], 'added': alias['latex']})

    # Body and appendix Chinese must remain exactly unchanged, including former heading text.
    def main_blocks(r):
        ps = list(r.find(W+'body')); start = next(i for i,e in enumerate(ps) if style(e) == 'chapter' and text(e).startswith('第1章'))
        return [e for e in ps[start:] if style(e) != 'bibliography']
    before = E.fromstring(original)
    han = lambda r: ''.join(re.findall(r'[\u3400-\u9fff]', ''.join(text(e) for e in main_blocks(r))))
    assert han(before) == han(root), 'Chinese wording changed'
    bmath = ''.join(n.text or '' for e in main_blocks(before) if not (e.tag == W+'tbl' and text(e).endswith('(3-35)')) for n in e.iter(M+'t'))
    amath = ''.join(n.text or '' for e in main_blocks(root) if not (e.tag == W+'tbl' and text(e).endswith('(3-35)')) for n in e.iter(M+'t'))
    assert bmath == amath, 'Formula symbols/operators changed'
    ordered_bib = [e for e in body if style(e) == 'bibliography']
    assert [int(re.match(r'^\[(\d+)\]', text(e))[1]) for e in ordered_bib] == list(range(1, 227))
    assert all(style(e) not in ('h2','h3','h4') for e in list(body)[next(i for i,e in enumerate(body) if style(e) == 'chapter' and text(e).startswith('附录')):])
    names = [n.get(W+'name') for n in root.iter(W+'bookmarkStart')]
    assert len(names) == len(set(names))
    assert all(n.get(W+'anchor') in names for n in root.iter(W+'hyperlink') if n.get(W+'anchor'))
    with ZipFile(OUT, 'w', ZIP_DEFLATED) as out:
        for info in z.infolist():
            out.writestr(info, E.tostring(root, encoding='UTF-8', xml_declaration=True, standalone=True) if info.filename == 'word/document.xml' else z.read(info.filename))

# Keep source citation keys and numbering consistent with the edited document.
md = ROOT/'重构论文.md'; mdbackup = A/'通篇细节检查前_重构论文.md'
if not mdbackup.exists(): shutil.copy2(md, mdbackup)
source = mdbackup.read_text()
assert source.count('[xu2025reinforcement, yu2024large]') == 1
source = source.replace('[xu2025reinforcement, yu2024large]', '[xu2025reinforcement, yu2024large, wangyp2019tram, hanl2023vsl]')
source = source.replace('[wangyp2019tram, hanl2023vsl, qin2024signal]', '[qin2024signal]')
assert source.count('附录B定理B.1至B.3') == 1
source = source.replace('附录B定理B.1至B.3', '附录B定理B.1至B.10')
md.write_text(source)
old_number_file = A/'通篇细节检查前_reference_numbers.json'
if not old_number_file.exists(): shutil.copy2(A/'reference_numbers.json', old_number_file)
old_numbers = json.loads(old_number_file.read_text()); numbers = {k: remap[n] for k,n in old_numbers.items()}
(A/'reference_numbers.json').write_text(json.dumps(numbers, ensure_ascii=False, indent=2))
clean = ROOT/'v3版引用文献列表_核对版_20261015.txt'
cleanbackup = A/'通篇细节检查前_v3核对列表.txt'
if not cleanbackup.exists(): shutil.copy2(clean, cleanbackup)
lines = [line for line in cleanbackup.read_text(encoding='utf-8-sig').splitlines() if re.match(r'^\[\d+\]', line)]
rewritten = {remap[int(re.match(r'^\[(\d+)\]', s)[1])]: re.sub(r'^\[\d+\]', '['+str(remap[int(re.match(r'^\[(\d+)\]', s)[1])])+']', s) for s in lines}
clean.write_text('按正文首次出现顺序编号\n\n'+'\n\n'.join(rewritten[n] for n in range(1,227))+'\n', encoding='utf-8-sig')
(A/'qa_applied_changes.json').write_text(json.dumps(changes, ensure_ascii=False, indent=2))
print(json.dumps({'candidate': str(OUT), 'appendix_lead_ins': 50, 'cross_reference_fixes': 6, 'reference_count': 226, 'symbol_aliases': len(aliases), 'Chinese': 'unchanged', 'formula_tokens': 'unchanged'}, ensure_ascii=False))
