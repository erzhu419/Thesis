"""Apply the supplied local replacements to the user's current Word copy."""
from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile
import json
import re
import sys

from lxml import etree as E
import pypandoc

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'workpapers/全文编号符号简称复核_20261011'))
from scan import W, M, NS, content
sys.path.insert(0, str(ROOT / 'workpapers/七章重写稿整合_20261010'))
from integrate import body_spans

BASE = HERE / '修改前_用户已完成1至13项.docx'
OUT = ROOT / 'output/排版完成_残留问题替换_20261011.docx'
SOURCE_MD = ROOT / 'markdown/复核后问题部分_直接替换稿_20261011.md'
XMLSPACE = '{http://www.w3.org/XML/1998/namespace}space'
md = SOURCE_MD.read_text()
changes = []


def section(n):
    s = re.search(rf'^## {n:02d}｜.*?(?=^## \d+｜|^# 二、)', md, re.M | re.S).group()
    return s.split('### 替换正文\n', 1)[1].strip()


def split_fragments(s):
    return [p.strip() for p in re.split(r'\n\s*\n', s) if p.strip()]


def convert_all(fragments):
    s = '\n\n'.join(f'MARKER{n}END\n\n{t}' for n,t in enumerate(fragments))
    target = HERE / '替换片段_公式中间件.docx'
    pypandoc.convert_text(s, 'docx', format='markdown', outputfile=str(target))
    with ZipFile(target) as z:
        body = E.fromstring(z.read('word/document.xml')).find(W+'body')
    result, current = {}, None
    for p in body:
        marker = re.fullmatch(r'MARKER(\d+)END', content(p))
        if marker:
            current = int(marker[1])
            result[current] = []
        elif p.tag != W+'sectPr':
            result[current].append(deepcopy(p))
    assert all(len(v)==1 for v in result.values()), {k:len(v) for k,v in result.items() if len(v)!=1}
    return {s:result[n][0] for n,s in enumerate(fragments)}


def props(p, name):
    return deepcopy(p.find(W+name)) if p.find(W+name) is not None else None


def replace_text(p, old, new):
    nodes = list(p.iter(W+'t'))
    joined = ''.join(n.text or '' for n in nodes)
    assert old in joined, (old, joined[:120])
    start, end = joined.index(old), joined.index(old)+len(old)
    offset, inserted = 0, False
    for n in nodes:
        s = n.text or ''
        right = offset+len(s)
        if right > start and offset < end:
            n.text = s[:max(0,start-offset)] + (new if not inserted else '') + s[min(len(s),end-offset):]
            n.set(XMLSPACE, 'preserve')
            inserted = True
        offset = right


def main():
    with ZipFile(BASE) as z:
        original = z.read('word/document.xml')
        d = E.fromstring(original)
        body = d.find(W+'body')
        old = list(body)
        before = [E.tostring(p) for p in old]
        # The numbered displays retain their original equation cells and bookmarks.
        eq_rows = {}
        for p in old:
            if p.tag == W+'tbl':
                for row in p.findall(W+'tr'):
                    cells = row.findall(W+'tc')
                    if len(cells)==3:
                        label = re.fullmatch(r'\(([A-F0-9]+-\d+[a-z]?)\)', content(cells[-1]))
                        if label:
                            eq_rows[label[1]] = (p,row)
        fragments = []
        for n in range(14,28):
            s = section(n)
            s = re.sub(r'^【[^\n]*】\n?', '', s, flags=re.M)
            s = re.sub(r'\\qquad\\text\{\(([A-F]-\d+[a-z]?)\)\}', '', s)
            fragments.extend(split_fragments(s))
        note9 = section(9).split('【表后说明】\n',1)[1].strip()
        guide = re.search(r'\*\*图前引导句：\*\*(.*?)\n\n',md,re.S)[1]
        fragments.extend([note9,guide])
        generated = convert_all(fragments)
        para_template = old[1002]

        def paragraph(s, template=None):
            p = deepcopy(generated[s])
            template = template if template is not None else para_template
            ppr = p.find(W+'pPr')
            if ppr is not None:
                p.remove(ppr)
            if template.find(W+'pPr') is not None:
                p.insert(0, deepcopy(template.find(W+'pPr')))
            rp = template.find('.//'+W+'rPr')
            for run in p.iter(W+'r'):
                own = run.find(W+'rPr')
                semantic = [] if own is None else [deepcopy(v) for v in own if v.tag in (W+'b',W+'i',W+'vertAlign')]
                if own is not None:
                    run.remove(own)
                fresh = deepcopy(rp) if rp is not None else E.Element(W+'rPr')
                for tag in (W+'b',W+'i'):
                    for child in list(fresh.findall(tag)):
                        fresh.remove(child)
                fresh.extend(semantic)
                run.insert(0,fresh)
            for run in p.iter(M+'r'):
                if run.find(W+'rPr') is None and rp is not None:
                    run.insert(0,deepcopy(rp))
            return p

        def display(s, label=None):
            if label:
                original_table, row = eq_rows[label]
                table = deepcopy(original_table)
                for r in list(table.findall(W+'tr')):
                    table.remove(r)
                fresh_row = deepcopy(row)
                table.append(fresh_row)
                middle = fresh_row.findall(W+'tc')[1]
                p = middle.find(W+'p')
                for child in list(p):
                    if child.tag != W+'pPr':
                        p.remove(child)
                formulas = generated[s].findall('.//'+M+'oMath')
                assert len(formulas)==1, s
                p.append(deepcopy(formulas[0]))
                return table
            p = paragraph(s)
            pp = p.find(W+'pPr')
            if pp is None:
                pp = E.SubElement(p,W+'pPr')
            for child in list(pp):
                if child.tag in (W+'jc',W+'ind',W+'keepNext'):
                    pp.remove(child)
            E.SubElement(pp,W+'jc',{W+'val':'center'})
            E.SubElement(pp,W+'ind',{W+'firstLine':'0'})
            return p

        ranges = {}
        def put(start,end,parts,item):
            assert start not in ranges
            ranges[start]=(end,parts)
            # Preserve any section/heading anchors attached to replaced prose.
            retained = {v.get(W+'name') for p in parts for v in p.iter(W+'bookmarkStart')}
            for source in old[start:end]:
                for bm in source.iter(W+'bookmarkStart'):
                    if bm.get(W+'name') in retained:
                        continue
                    target = parts[0]
                    if target.tag == W+'tbl':
                        target = target.find('.//'+W+'p')
                    target.insert(1 if target.find(W+'pPr') is not None else 0,deepcopy(bm))
                    E.SubElement(target,W+'bookmarkEnd',{W+'id':bm.get(W+'id')})
                    retained.add(bm.get(W+'name'))
            changes.append({'item':item,'old_blocks':[start,end-1],'new_blocks':len(parts)})

        def item_fragments(n):
            s = section(n)
            s = re.sub(r'^【[^\n]*】\n?', '', s, flags=re.M)
            s = re.sub(r'\\qquad\\text\{\(([A-F]-\d+[a-z]?)\)\}', '', s)
            return split_fragments(s)

        f = item_fragments(14)
        replacement14=[display(s) if s.startswith('$$') else paragraph(s) for s in f]
        lead=replacement14[1].find(W+'pPr')
        E.SubElement(lead,W+'keepNext')
        E.SubElement(lead,W+'keepLines')
        put(1001,1003,replacement14,14)
        put(1763,1764,[paragraph(item_fragments(15)[0],old[1763])],15)
        put(1779,1780,[paragraph(item_fragments(16)[0],old[1779])],16)
        f = item_fragments(17)
        put(2215,2220,[display(s,'B-35') if s.startswith('$$') else paragraph(s,old[2215]) for s in f],17)
        put(2225,2226,[display(item_fragments(18)[0],'B-38')],18)
        f = item_fragments(19)
        put(2237,2239,[paragraph(f[0],old[2237]),display(f[1],'B-42'),paragraph(f[2],old[2237])],19)
        put(2241,2242,[paragraph(f[3],old[2241])],19)
        f = item_fragments(20)
        labels = iter(['D-4a','D-4b','D-4c','D-4d','D-4e'])
        put(2410,2422,[display(s,next(labels)) if s.startswith('$$') else paragraph(s,old[2413]) for s in f],20)
        put(2541,2542,[paragraph(s,old[2541]) for s in item_fragments(21)],21)
        put(2554,2555,[paragraph(s,old[2554]) for s in item_fragments(22)],22)
        f = item_fragments(23)
        put(2647,2648,[paragraph(f[0],old[2647])],23)
        put(2651,2652,[paragraph(f[1],old[2651])],23)
        put(2663,2665,[paragraph(s,old[2663]) for s in item_fragments(24)],24)
        put(2682,2683,[paragraph(item_fragments(25)[0],old[2682])],25)
        put(2695,2699,[paragraph(s,old[2695]) for s in item_fragments(26)],26)
        f = item_fragments(27)
        put(2749,2750,[paragraph(f[0],old[2749])],27)
        put(2761,2762,[paragraph(f[1],old[2761])],27)
        # The user rejected the Visio redraw. Keep both the original figure and
        # its original explanatory paragraph; do not describe an unadopted figure.

        # Item 9's row label was pasted, but the separate explanatory note was not.
        assert not any(note9 in content(p) for p in old)
        put(617,618,[paragraph(note9,old[617]),old[617]],'09补全表后说明')
        # Item 10's two definitions were pasted into one row. Restore native rows.
        table = old[623]
        source_row = table.findall(W+'tr')[5]
        targets = [
            ['静态误差信号排序，附录E.6','在指定静态分数定义下比较排序','解释两类分数何时排序一致；不把它们等同于实际MC-WM递推权重或H2O+条件转移比'],
            ['置信尺度归一化，附录E.9','调整当前对数评分在递推目标中的系数','对应第6.3.3节及第6.4.4—6.4.5节；固定策略和评分条件下分析权重尺度，控制效果另由配对实验评价'],
        ]
        def set_cell(cell,text):
            template = cell.find(W+'p')
            for child in list(cell):
                if child.tag != W+'tcPr':
                    cell.remove(child)
            p = E.SubElement(cell,W+'p')
            if template is not None and template.find(W+'pPr') is not None:
                p.append(deepcopy(template.find(W+'pPr')))
            rp=template.find('.//'+W+'rPr') if template is not None else None
            for part in re.split(r'(\$[^$]+\$)',text):
                if not part:
                    continue
                if part.startswith('$'):
                    assert re.fullmatch(r'\$[jbp]\$',part),part
                    formula=E.SubElement(p,M+'oMath')
                    r=E.SubElement(formula,M+'r')
                    if rp is not None:
                        r.append(deepcopy(rp))
                    E.SubElement(r,M+'t').text=part[1:-1]
                else:
                    r=E.SubElement(p,W+'r')
                    if rp is not None:
                        r.append(deepcopy(rp))
                    E.SubElement(r,W+'t').text=part
        for texts in reversed(targets):
            row=deepcopy(source_row)
            for cell,text in zip(row.findall(W+'tc'),texts):
                set_cell(cell,text)
            source_row.addnext(row)
        table.remove(source_row)
        changes.append({'item':'10拆分误合并行','table':'2-8'})
        for ti,ri,ci,sri in [(611,1,2,2),(616,10,0,9)]:
            rows=old[ti].findall(W+'tr')
            cell=rows[ri].findall(W+'tc')[ci]
            sample=rows[sri].findall(W+'tc')[ci].find('.//'+W+'rPr')
            for run in cell.iter(W+'r'):
                rp=run.find(W+'rPr')
                if rp is not None:
                    run.remove(rp)
                if sample is not None:
                    run.insert(0,deepcopy(sample))
        changes.append({'item':'08、09表格格式','format':'与同表原有单元格一致的10.5磅文字'})

        # Symbol definitions change only in the meaning column, never the symbol cell.
        symrows=old[81].findall(W+'tr')
        indexes=[9,15,23,24,37,47,48,49,50,51,63,64,67,68,76,80,81,82,85,91]
        s=md.split('# 二、主要符号说明的替换文字',1)[1].split('# 三、',1)[0]
        definitions=[line.split('|')[-2].strip() for line in s.splitlines() if line.startswith('| $')]
        assert len(definitions)==len(indexes)
        for ix,definition in zip(indexes,definitions):
            if ix==37:
                definition=definition.replace('行为克隆','行为克隆（Behavior Cloning，BC）',1)
            if ix==85:
                definition+='；策略调整敏感性及访问分布敏感性上界'
            set_cell(symrows[ix].findall(W+'tc')[1],definition)
        # A compact table line spacing keeps its closing explanation on the same
        # front-matter spread. Font, font size, and column geometry stay unchanged.
        for p in old[81].iter(W+'p'):
            pp=p.find(W+'pPr')
            if pp is None:
                pp=E.Element(W+'pPr');p.insert(0,pp)
            sp=pp.find(W+'spacing')
            if sp is None:
                sp=E.SubElement(pp,W+'spacing')
            sp.set(W+'line','220');sp.set(W+'lineRule','auto')
        changes.append({'item':'主要符号说明','meaning_cells':indexes})

        # F's labels change, whereas all experimental numbers and signs stay put.
        for i in range(2751,2787):
            if i in ranges:
                continue
            p=old[i]
            pairs=[]
            if i<=2768:
                text=''.join(p.itertext())
                if '人均广义时间' in content(p):
                    pairs.append(('人均广义时间','人均广义时间成本'))
            if 2761<=i<=2778 and '乘客完成率' in content(p):
                pairs.append(('乘客完成率','行程完成率'))
            if i==2754:
                pairs.append(('完成率（%）','行程完成率（%）'))
            if i==2762:
                pairs.append(('共同停站执行规则','共同的附加驻站执行规则'))
            if i in [2773,2778]:
                pairs.append(('停站偏差25%','必要服务时间增加25%'))
            if i==2778:
                pairs.extend([('额外路口25%','新增路口数量比例参数为0.25'),
                    ('正常需求时间下降（%）','正常需求下人均成本降幅（%）'),
                    ('高需求时间下降（%）','1.25倍需求下人均成本降幅（%）'),
                    ('正常需求完成率差（百分点）','正常需求下行程完成率差（百分点）'),
                    ('高需求完成率差（百分点）','1.25倍需求下行程完成率差（百分点）')])
            if i==2780:
                pairs.append(('完整日程下的模型辅助比较','完整混合训练过程中的模型辅助比较'))
            if i==2782:
                pairs.extend([('两个固定模型项系数','两个固定模型损失系数的配置'),
                    ('各续训25000个源事件、20000次更新','各配置继续采集25,000个源事件，并完成20,000次训练更新')])
            for a,b in pairs:
                replace_text(p,a,b)
            if pairs:
                changes.append({'item':28,'block':i,'labels':pairs})

        # Word split this instruction across runs; consolidate it without retargeting.
        instructions=list(old[1676].iter(W+'instrText'))
        for left,right in zip(instructions,instructions[1:]):
            if (left.text or '').endswith('eq6_norm') and (right.text or '').startswith('alized_backup'):
                left.text+=right.text
                right.text=''
        new=[]
        i=0
        while i<len(old):
            if i in ranges:
                end,parts=ranges[i]
                new.extend(parts)
                i=end
            else:
                new.append(old[i]); i+=1

        # Resolve ordinary appendix prose references to the same bookmarks as fields.
        anchors={}
        equations={}
        for p in new:
            starts=list(p.iter(W+'bookmarkStart'))
            heading=re.match(r'^([A-F]\.\d+)\s',content(p))
            if heading and starts:
                anchors[heading[1]]=starts[0].get(W+'name')
            if p.tag==W+'tbl':
                for row in p.findall(W+'tr'):
                    cells=row.findall(W+'tc')
                    if len(cells)==3:
                        label=re.fullmatch(r'\(([A-F0-9]+-\d+[a-z]?)\)',content(cells[-1]))
                        bookmarks=list(cells[-1].iter(W+'bookmarkStart'))
                        if label and bookmarks:
                            equations[label[1]]=bookmarks[0].get(W+'name')
        appanchors={}
        for p in new:
            m=re.match(r'^附录([A-F])\s',content(p))
            marks=list(p.iter(W+'bookmarkStart'))
            if m and marks:
                appanchors[m[1]]=marks[0].get(W+'name')
        refpat=re.compile(r'附录([A-F](?:\.\d+)?)|第([A-F]\.\d+)节|式\(([A-F]-\d+[a-z]?)\)')
        links=[]
        for index,p in enumerate(new):
            if index<=79:
                continue
            for run in list(p.iter(W+'r')):
                if run.getparent().tag in (W+'hyperlink',M+'r') or any(v.tag==W+'fldChar' for v in run):
                    continue
                texts=run.findall(W+'t')
                if len(texts)!=1 or len(run)>2:
                    continue
                t=texts[0].text or ''
                matches=[]
                for m in refpat.finditer(t):
                    key=m[1] or m[2]
                    anchor=equations.get(m[3]) if m[3] else anchors.get(key,appanchors.get(key))
                    if anchor:
                        matches.append((m,anchor))
                if not matches:
                    continue
                parent=run.getparent(); at=parent.index(run)
                cursor=0
                def insert_piece(text,anchor=None):
                    nonlocal at
                    if not text:
                        return
                    r=deepcopy(run);r.find(W+'t').text=text
                    if anchor:
                        h=E.Element(W+'hyperlink',{W+'anchor':anchor,W+'history':'1'});h.append(r)
                        parent.insert(at,h)
                    else:
                        parent.insert(at,r)
                    at+=1
                for m,anchor in matches:
                    insert_piece(t[cursor:m.start()]);insert_piece(m.group(),anchor)
                    links.append({'text':m.group(),'target':anchor})
                    cursor=m.end()
                insert_piece(t[cursor:]);parent.remove(run)
        # Any equation/heading links removed with a replacement are rebuilt above.
        replacements=[]
        spans=body_spans(original)
        for i,p in enumerate(old):
            if i in ranges:
                end,parts=ranges[i]
                replacements.append((spans[i][0],spans[end-1][1],b''.join(E.tostring(v) for v in parts)))
            elif any(start<i<end for start,(end,_) in ranges.items()):
                continue
            elif E.tostring(p)!=before[i]:
                replacements.append((spans[i][0],spans[i][1],E.tostring(p)))
        result=original
        for a,b,data in sorted(replacements,reverse=True):
            result=result[:a]+data+result[b:]
        with ZipFile(OUT,'w') as output:
            for info in z.infolist():
                output.writestr(info,result if info.filename=='word/document.xml' else z.read(info.filename))
    (HERE/'逐项替换记录.json').write_text(json.dumps({'changes':changes,'hyperlinks':links,'source':str(BASE),'output':str(OUT)},ensure_ascii=False,indent=2))
    print(json.dumps({'output':str(OUT),'changed_ranges':len(replacements),'new_appendix_links':len(links)},ensure_ascii=False))


if __name__=='__main__':
    main()
