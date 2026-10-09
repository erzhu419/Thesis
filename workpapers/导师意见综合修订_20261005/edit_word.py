"""Edit the designated Word package in place, preserving native objects and styles."""
from copy import deepcopy
from difflib import SequenceMatcher
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json
from lxml import etree as E

HERE = Path(__file__).resolve().parent
SOURCE = HERE / '修改前.docx'
DEST = HERE / '修订待分页.docx'
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math'}
q = lambda name: '{'+NS[name.split(':')[0]]+'}'+name.split(':')[1]
with ZipFile(SOURCE) as z:
    package = {n: z.read(n) for n in z.namelist()}
doc = E.fromstring(package['word/document.xml'])
body = doc.find('w:body', NS)
paras = body.findall('w:p', NS)
tables = body.findall('w:tbl', NS)
changes = []

def visible(e):
    return ''.join(e.xpath('.//w:t/text()|.//m:t/text()', namespaces=NS))

def span_replace(p, start, end, new):
    nodes = p.xpath('.//w:t', namespaces=NS)
    offset = 0
    for t in nodes:
        s = t.text or ''; stop = offset + len(s)
        if offset <= start < stop:
            prefix = s[:start-offset]
            suffix = s[max(0,end-offset):] if end <= stop else ''
            t.text = prefix + new + suffix
            t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
        elif start < offset < end or (offset == start and start == end):
            t.text = s[max(0,end-offset):] if end <= stop else ''
        offset = stop
    if start == offset:
        nodes[-1].text = (nodes[-1].text or '') + new

def rewrite(p, text):
    assert not p.xpath('.//m:oMath|.//w:drawing', namespaces=NS), visible(p)[:60]
    protected=[];in_field=False;separate=False
    for node in p.iter():
        if node.tag==q('w:fldChar'):
            kind=node.get(q('w:fldCharType'))
            if kind=='begin':in_field=True
            elif kind=='separate':separate=True
            elif kind=='end':in_field=False;separate=False
        elif node.tag==q('w:t') and in_field and separate:protected.append((node,node.text))
    old = visible(p)
    for tag,a,b,c,d in reversed(SequenceMatcher(None,old,text,autojunk=False).get_opcodes()):
        if tag != 'equal': span_replace(p,a,b,text[c:d])
    assert visible(p) == text, (old,visible(p),text)
    assert all(node.text==before for node,before in protected),('Changed field result',old,text)

def setp(i, text, reason='文字与论证修订'):
    old = visible(paras[i])
    fields=paras[i].xpath('.//w:instrText/text()',namespaces=NS)
    if i in (2033,2037,2039) and fields:
        assert all(s.strip().startswith('REF ') for s in fields),fields
        rich(paras[i],[text])
        changes.append({'paragraph':i,'removed_repeated_crossreferences':fields})
    else:rewrite(paras[i],text)
    if old != text: changes.append({'paragraph':i,'reason':reason,'before':old,'after':text})

def replace(p, old, new):
    text = ''.join(p.xpath('.//w:t/text()',namespaces=NS))
    assert old in text, old
    starts=[]; pos=0
    while (at:=text.find(old,pos))>=0:
        starts.append(at);pos=at+len(old)
    for at in reversed(starts):span_replace(p,at,at+len(old),new)

def newp(text, template=1144):
    p=E.Element(q('w:p'))
    pp=paras[template].find('w:pPr',NS)
    if pp is not None:p.append(deepcopy(pp))
    r=E.SubElement(p,q('w:r'));t=E.SubElement(r,q('w:t'));t.text=text
    return p

def insert_after(i,text):
    p=newp(text);paras[i].addnext(p)
    changes.append({'after_paragraph':i,'after':text})
    return p

def text_run(text):
    r=E.Element(q('w:r'));t=E.SubElement(r,q('w:t'));t.text=text
    return r

def rich(p, parts):
    for child in list(p):
        if child.tag not in {q('w:pPr'),q('w:bookmarkStart'),q('w:bookmarkEnd'),q('w:commentRangeStart'),q('w:commentRangeEnd')}:
            p.remove(child)
    for part in parts:p.append(text_run(part) if isinstance(part,str) else part)

# The root editor reviews and integrates the independently checked text drafts.
for name in ['accepted_rewrites.json','detail_rewrites.json']:
    f=HERE/name
    if f.exists():
        for item in json.loads(f.read_text()):setp(item['id'],item['text'],item.get('reason','文字与论证修订'))

# Keep the detailed campaign counts available in Appendix F, after the results.
for i in [831,832,833,834,836]:
    body.remove(paras[i])
heading=newp('F.4 补充实验的训练与评价记录',template=2789)
section=body.find('w:sectPr',NS)
at=list(body).index(section) if section is not None else len(body)
body.insert(at,heading)
previous=heading
for i in [831,832,833,834,836]:previous.addnext(paras[i]);previous=paras[i]
insert_after(830,'各批次训练、初始化复用与评价回合记录集中列于附录F.4。相应算法、模型系数及实验结果在第5、6章分别说明。')
changes.append({'reason':'将五段具体批次记录移至附录F.4，第二章保留共同协议'})

# Individual replacements also work in paragraphs containing native equations.
f=HERE/'replacements.json'
if f.exists():
    for item in json.loads(f.read_text()):
        p=paras[item['id']];old=visible(p);replace(p,item['old'],item['new'])
        changes.append({'paragraph':item['id'],'before':old,'after':visible(p)})

f=HERE/'insertions.json'
if f.exists():
    for item in reversed(json.loads(f.read_text())):insert_after(item['after'],item['text'])

# Use native Word math for the verified training-time evidence calculation.
with ZipFile(HERE/'新增行内公式.docx') as z:
    fragments=E.fromstring(z.read('word/document.xml')).xpath('//m:oMath',namespaces=NS)
for fragment in fragments:
    for run in fragment.xpath('.//m:r',namespaces=NS):
        rp=run.find('w:rPr',NS)
        if rp is None:rp=E.SubElement(run,q('w:rPr'))
        fonts=rp.find('w:rFonts',NS)
        if fonts is None:fonts=E.SubElement(rp,q('w:rFonts'))
        fonts.set(q('w:ascii'),'Cambria Math');fonts.set(q('w:hAnsi'),'Cambria Math')
parts=[
 ['公交配置把近期证据依次转为惊异度、运行长度权重和演员保守系数。惊异度取三个量的最大值：近期奖励均值相对运行均值的标准化偏差、式(4-7)的集成价值标准差上升量、在线与目标评价器权重范数差的一半；再以系数0.1作指数平滑。'],
 ['运行长度索引',deepcopy(fragments[0]),'取0至19，初始权重均匀。对平滑惊异度',deepcopy(fragments[1]),'，构造证据权重',deepcopy(fragments[2]),'，与原权重相乘后归一化；随后按0.02的变化率向零长度分配质量，其余质量向下一长度移位，在有限长度截断后再次归一化。该机制使用规定的证据函数调节运行长度权重，与式(4-2)由显式环境模式似然得到的后验概率分别定义。'],
 ['记归一化运行长度权重为',deepcopy(fragments[3]),'，其加权量为',deepcopy(fragments[4]),'。前100次训练迭代将该加权量置零，之后以',deepcopy(fragments[5]),'确定演员标准差系数，作用于集成均值、集成标准差与熵共同组成的演员目标。评价时按当前观测执行固定演员，不更新上述训练期调节量。']]
prev=paras[1205]
for part in parts:
    p=newp('');rich(p,part);prev.addnext(p);prev=p
    changes.append({'after_paragraph':1205,'reason':'明确公交运行长度证据调节与概率模式后验的区别','after':visible(p)})

# Correct displaced native equation objects; no conversion to images or text.
p=paras[2295];maths=p.findall('m:oMath',NS)
rich(p,['模式',maths[0],'的对角执行器增益为',maths[1],
        '，其对角元素按动作坐标编号。名义信号与加噪后的实际执行信号均为无量纲执行器量，不表示公交驻站秒数。命令',
        maths[2],'先形成名义信号',maths[3],'，再与执行噪声',maths[4],'相加并截断，得到实际执行量'])
changes.append({'paragraph':2295,'reason':'恢复执行器增益定义位置'})
p=paras[1745];maths=p.findall('m:oMath',NS)
rich(p,['每5,000步进行一次固定种子的验证，记录平均回报',maths[0],'与每回合平均违例次数',maths[2],
        '，其中',maths[1],'表示验证序号；peak和final分别标记历史峰值与训练终点。训练过程保存满足',maths[3],
        '的最高回报演员。在最终测试前，若最终验证回报相对历史峰值下降超过',maths[4],
        '，且保存演员回报更高，则恢复该演员。另一种恢复条件是最终验证违例超过1，且保存演员回报不低于最终回报减去',maths[5],
        '。恢复仅针对演员，不重新选择评价器或世界模型。'])
changes.append({'paragraph':1745,'reason':'使违例符号紧接被定义对象'})

# Keep target-network notation consistent with the symbol table.
prime=tables[70].xpath('.//m:t[text()="\'"]',namespaces=NS)
assert len(prime)==1
prime[0].text='−'
changes.append({'table':70,'reason':'式3-24目标参数撇号改上标负号'})

# Break the long KL chain rule across three lines without reducing the font.
eq=tables[268].find('.//m:eqArr',NS)
rows=eq.findall('m:e',NS);assert len(rows)==2
for t in eq.xpath('.//m:t',namespaces=NS):
    if t.text:t.text=t.text.replace('&','')
second=rows[1]
sub=second.findall('m:sSub',NS)[-1]
idx=list(second).index(sub)
third=E.Element(q('m:e'))
mr=E.SubElement(third,q('m:r')); mt=E.SubElement(mr,q('m:t'));mt.text='×'
for node in list(second)[idx:]:second.remove(node);third.append(node)
eq.append(third)
changes.append({'table':268,'reason':'式B-25分三行保留原生公式及独立编号'})

# State the value anchor before the D-58 derivative-to-value bound.
anchor=deepcopy(tables[354].find('.//m:eqArr/m:e',NS))
for t in anchor.xpath('.//m:t',namespaces=NS):
    if t.text=='ref':t.text='e'
    elif t.text:t.text=t.text.replace('&','').rstrip(',')
math=E.Element(q('m:oMath'))
for child in list(anchor):math.append(child)
p=newp('');rich(p,['进一步假设，对每个所考察环境，基准点满足',math,
    '，且式(D-57)的导数误差界在连接该基准点与待评价点的整条线段上成立。上述基准误差上界对所有所考察环境统一成立时，由命题D.4得到'])
old=paras[2654];old.addprevious(p);body.remove(old)
changes.append({'paragraph':2654,'reason':'补充D-58基准函数值误差与线段条件','after':visible(p)})

# Optional reviewed tables are cloned from existing native table formatting.
def insert_table(anchor_id, caption, rows, widths, template_id=207):
    t=deepcopy(tables[template_id])
    for child in list(t):
        if child.tag==q('w:tr'):t.remove(child)
    grid=t.find('w:tblGrid',NS)
    for child in list(grid):grid.remove(child)
    for width in widths:E.SubElement(grid,q('w:gridCol'),{q('w:w'):str(width)})
    tw=t.find('w:tblPr/w:tblW',NS)
    if tw is not None:tw.set(q('w:w'),str(sum(widths)));tw.set(q('w:type'),'dxa')
    templates=tables[template_id].findall('w:tr',NS)
    for row_index,values in enumerate(rows):
        cell_template=templates[0 if row_index==0 else 1].find('w:tc',NS)
        tr=E.SubElement(t,q('w:tr'))
        trp=E.SubElement(tr,q('w:trPr'));E.SubElement(trp,q('w:cantSplit'))
        if row_index==0:E.SubElement(trp,q('w:tblHeader'))
        for value,width in zip(values,widths):
            tc=deepcopy(cell_template)
            for child in list(tc):
                if child.tag!=q('w:tcPr'):tc.remove(child)
            tc.find('w:tcPr/w:tcW',NS).set(q('w:w'),str(width))
            cp=E.Element(q('w:p'))
            original_p=cell_template.find('w:p',NS)
            pp=original_p.find('w:pPr',NS)
            if pp is not None:cp.append(deepcopy(pp))
            rr=E.SubElement(cp,q('w:r'))
            rp=original_p.find('w:r/w:rPr',NS)
            if rp is not None:rr.append(deepcopy(rp))
            for j,line in enumerate(str(value).split('\n')):
                if j:E.SubElement(rr,q('w:br'))
                E.SubElement(rr,q('w:t')).text=line
            tc.append(cp);tr.append(tc)
    cap=newp(caption,template=1143)
    paras[anchor_id].addnext(cap);cap.addnext(t)
    changes.append({'after_paragraph':anchor_id,'table_caption':caption,'rows':rows})

f=HERE/'new_tables.json'
if f.exists():
    for item in json.loads(f.read_text()):insert_table(**item)

# Unify the main actor term while retaining established algorithm names.
for p in doc.xpath('//w:p',namespaces=NS):
    s=''.join(p.xpath('.//w:t/text()',namespaces=NS))
    for old,new in [('行动者','演员'),('无模型骨干','无世界模型混合基线'),
                    ('混合骨干','混合学习基线'),('直接父控制器','同策略未加完成保障的基线控制器')]:
        if old in s:
            before=visible(p);replace(p,old,new)
            changes.append({'reason':'统一术语','before':before,'after':visible(p)})
            s=''.join(p.xpath('.//w:t/text()',namespaces=NS))

# Keep each existing figure with its caption; long body explanations may flow.
for p in body.findall('w:p',NS):
    style=p.find('w:pPr/w:pStyle',NS)
    if style is not None and style.get(q('w:val')) in ('figure','tablecaption'):
        pp=p.find('w:pPr',NS)
        if pp.find('w:keepNext',NS) is None:E.SubElement(pp,q('w:keepNext'))

package['word/document.xml']=E.tostring(doc,xml_declaration=True,encoding='UTF-8',standalone=True)
with ZipFile(DEST,'w',ZIP_DEFLATED) as z:
    for name,data in package.items():z.writestr(name,data)
(HERE/'changes.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2))
print(json.dumps({'output':str(DEST),'edits':len(changes)},ensure_ascii=False))
