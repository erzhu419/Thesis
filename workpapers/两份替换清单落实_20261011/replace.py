"""Apply the two supplied replacement lists without rebuilding the thesis."""
from copy import deepcopy
from difflib import SequenceMatcher
from pathlib import Path
from shutil import copy2
from zipfile import ZipFile
import json
import re
import sys

from lxml import etree as E

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'workpapers/全文编号符号简称复核_20261011'))
from scan import W, M, NS, content
sys.path.insert(0, str(ROOT / 'workpapers/七章重写稿整合_20261010'))
from integrate import body_spans

SOURCE = ROOT / 'output/排版完成_残留问题替换_20261011.docx'
OUTPUT = ROOT / 'output/排版完成_双清单替换_20261011.docx'
LIST1 = ROOT / 'markdown/论文替换清单_章首概括与用词修订_20261011.docx'
LIST2 = ROOT / 'markdown/论文补充替换清单_符号衔接与指标术语_20261011.docx'
R = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'
SPACE = '{http://www.w3.org/XML/1998/namespace}space'


def read_list(path):
    with ZipFile(path) as z:
        r = E.fromstring(z.read('word/document.xml'))
    return r.xpath('/w:document/w:body/w:p | /w:document/w:body/w:tbl', namespaces=NS)


def norm(s):
    return re.sub(r'\s+', '', s).translate(str.maketrans('−–—（）', '---()'))


def find_phrase(text, phrase):
    chars, indexes = [], []
    for i, c in enumerate(text):
        if not c.isspace():
            chars.append(norm(c)); indexes.append(i)
    found = ''.join(chars).find(norm(phrase))
    if found < 0:
        return None
    return indexes[found], indexes[found + len(norm(phrase))-1]+1


class Editor:
    def __init__(self):
        with ZipFile(SOURCE) as z:
            self.raw = z.read('word/document.xml')
            self.doc = E.fromstring(self.raw)
        self.body = self.doc.find(W+'body')
        self.blocks = list(self.body)
        self.spans = body_spans(self.raw)
        self.original = [E.tostring(p) for p in self.blocks]
        self.texts = [content(p) for p in self.blocks]
        self.changes, self.extra, self.new_after = [], {}, {}
        self.fields, self.citations = {}, {}
        for p in self.doc.iter(W+'p'):
            seq, active = [], False
            for child in p:
                marks = child.findall('.//'+W+'fldChar')
                if any(x.get(W+'fldCharType')=='begin' for x in marks):
                    seq, active = [], True
                if active:
                    seq.append(child)
                if any(x.get(W+'fldCharType')=='end' for x in marks) and active:
                    key = ''.join(content(x) for x in seq)
                    self.fields.setdefault(key, deepcopy(seq))
                    active = False
            for link in p.findall('.//'+W+'hyperlink'):
                if link.get(W+'anchor', '').startswith('ref-'):
                    self.citations.setdefault(content(link), deepcopy(link))
        self.reference_anchors = {}
        for p in self.blocks:
            s = content(p)
            bm = p.findall('.//'+W+'bookmarkStart')
            number = re.match(r'^\[(\d+)\]', s)
            anchors = [x.get(W+'name') for x in bm if x.get(W+'name','').startswith('ref-')]
            if number and anchors:
                self.reference_anchors[int(number[1])] = anchors[0]

    def matches(self, phrase, lo=260, hi=2813):
        return [i for i,s in enumerate(self.texts) if lo<=i<hi and find_phrase(s,phrase)]

    def locate(self, phrase, lo=260, hi=2813):
        hits = self.matches(phrase,lo,hi)
        assert len(hits)==1, (phrase, hits)
        return hits[0]

    def log(self, item, index, old, new):
        self.changes.append({'item':item, 'original_block':index, 'old':old, 'new':new})

    def phrase(self, old, new, item, lo=260, hi=2813, all_hits=False):
        hits = self.matches(old,lo,hi)
        assert hits and (all_hits or len(hits)==1), (item,old,hits)
        for i in hits:
            p = self.blocks[i]
            nodes = list(p.iter(W+'t'))
            joined = ''.join(x.text or '' for x in nodes)
            interval = find_phrase(joined,old)
            if not interval:
                self.inline_span(p, old, new)
                self.texts[i] = content(p)
                self.log(item,i,old,new)
                continue
            start,end = interval
            offset, inserted = 0,False
            for n in nodes:
                s = n.text or ''; right = offset+len(s)
                if right>start and offset<end:
                    n.text = s[:max(0,start-offset)]+(new if not inserted else '')+s[min(len(s),end-offset):]
                    n.set(SPACE,'preserve'); inserted=True
                offset=right
            self.texts[i] = content(p)
            self.log(item,i,old,new)
        if all_hits and norm(old) not in norm(new) and self.matches(old,lo,hi):
            self.phrase(old,new,item,lo,hi,all_hits=True)

    def inline_span(self,p,old,new):
        start,end=find_phrase(content(p),old)
        cursor=0;inserted=False
        for child in list(p):
            s=content(child);right=cursor+len(s)
            if right>start and cursor<end:
                a=max(0,start-cursor);b=min(len(s),end-cursor)
                if child.tag!=W+'r':
                    assert a==0 and b==len(s),(old,s)
                at=p.index(child)
                rp=child.find(W+'rPr')
                p.remove(child)
                v=s[:a]+(new if not inserted else '')+s[b:]
                if v:
                    r=E.Element(W+'r')
                    if rp is not None:
                        r.append(deepcopy(rp))
                    E.SubElement(r,W+'t',{SPACE:'preserve'}).text=v
                    p.insert(at,r)
                inserted=True
            cursor=right

    def styled(self, source, template):
        p = deepcopy(source)
        for x in list(p):
            if x.tag in (W+'pPr', W+'bookmarkStart',W+'bookmarkEnd'):
                p.remove(x)
        props = template.find(W+'pPr')
        if props is not None:
            p.insert(0,deepcopy(props))
        candidates = template.xpath('./w:r[w:t]/w:rPr',namespaces=NS)
        rp = next((x for x in candidates if x.find(W+'vertAlign') is None), E.Element(W+'rPr'))
        for run in p.iter(W+'r'):
            own = run.find(W+'rPr')
            if own is not None:
                run.remove(own)
            run.insert(0,deepcopy(rp))
        for run in p.iter(M+'r'):
            own = run.find(W+'rPr')
            if own is not None:
                run.remove(own)
            run.insert(0,deepcopy(rp))
        self.relink(p)
        return p

    def retain_mean_notation(self,p):
        # The thesis uses a bar for an environment-level indicator mean.
        for math in p.iter(M+'oMath'):
            if content(math).startswith('Δ'):
                continue
            for sub in math.iter(M+'sSub'):
                base=sub.find(M+'e')
                if base is not None and content(base)=='M':
                    acc=E.Element(M+'acc')
                    prop=E.SubElement(acc,M+'accPr')
                    E.SubElement(prop,M+'chr',{M+'val':'̅'})
                    expr=E.SubElement(acc,M+'e')
                    for child in list(base):
                        base.remove(child);expr.append(child)
                    base.append(acc)

    def relink(self,p):
        pattern = r'\[\d+(?:,\d+)*\]|[图表][1-7A-F]-\d+|\([1-7A-F]-\d+[a-z]?\)'
        for run in list(p.findall(W+'r')):
            t=run.find(W+'t')
            if t is None:
                continue
            s=t.text or ''; parts=[]; cursor=0
            def literal(v):
                if not v:
                    return
                r=deepcopy(run);r.find(W+'t').text=v;parts.append(r)
            for match in re.finditer(pattern,s):
                key=match[0]
                literal(s[cursor:match.start()])
                if key in self.fields:
                    parts.extend(deepcopy(self.fields[key]))
                elif key in self.citations:
                    parts.append(deepcopy(self.citations[key]))
                elif key.startswith('['):
                    for n in map(int,key[1:-1].split(',')):
                        anchor=self.reference_anchors[n]
                        link=E.Element(W+'hyperlink',{W+'anchor':anchor,W+'history':'1'})
                        r=deepcopy(run);r.find(W+'t').text=f'[{n}]'
                        rp=r.find(W+'rPr')
                        for x in list(rp.findall(W+'vertAlign')):
                            rp.remove(x)
                        E.SubElement(rp,W+'vertAlign',{W+'val':'superscript'})
                        link.append(r);parts.append(link)
                else:
                    literal(key)
                cursor=match.end()
            literal(s[cursor:])
            if cursor:
                at=p.index(run);p.remove(run)
                for j,x in enumerate(parts):
                    p.insert(at+j,x)

    def full(self,index,sources,item):
        template=self.blocks[index]
        new=[self.styled(p,template) for p in sources]
        for bm in template.findall(W+'bookmarkStart'):
            new[0].insert(1,deepcopy(bm))
        for bm in template.findall(W+'bookmarkEnd'):
            new[-1].append(deepcopy(bm))
        self.log(item,index,content(template),'\n'.join(content(p) for p in new))
        self.extra[index]=new
        self.blocks[index]=new[0]
        self.texts[index]=content(new[0])

    def insert(self,index,source,item,template=None):
        p=self.styled(source,template if template is not None else self.blocks[index+1])
        self.new_after.setdefault(index,[]).append(p)
        self.log(item,index,'',content(p))

    def save(self):
        edits=[]
        for i,p in enumerate(self.blocks):
            if i in self.extra or E.tostring(p)!=self.original[i] or i in self.new_after:
                replacements=self.extra.get(i,[p])+self.new_after.get(i,[])
                a,b=self.spans[i]
                edits.append((a,b,b''.join(E.tostring(x) for x in replacements)))
        raw=self.raw
        for a,b,new in reversed(edits):
            raw=raw[:a]+new+raw[b:]
        parts={'word/document.xml':raw}
        with ZipFile(SOURCE) as z:
            for name in z.namelist():
                if re.fullmatch(r'word/header\d+\.xml',name):
                    original=z.read(name)
                    header=E.fromstring(original)
                    if '早期BAPR' in content(header):
                        for text in header.iter(W+'t'):
                            if text.text and '早期' in text.text:
                                text.text=text.text.replace('早期','')
                        parts[name]=E.tostring(header,xml_declaration=True,encoding='UTF-8',standalone=True)
            with ZipFile(OUTPUT,'w') as out:
                for info in z.infolist():
                    out.writestr(info,parts.get(info.filename,z.read(info.filename)))
        (HERE/'逐项替换记录.json').write_text(json.dumps(self.changes,ensure_ascii=False,indent=2))
        print(json.dumps({'changes':len(self.changes),'changed_blocks':len(edits),'output':str(OUTPUT)},ensure_ascii=False))


def main():
    HERE.mkdir(exist_ok=True)
    copy2(SOURCE,HERE/'修改前.docx')
    a,b=read_list(LIST1),read_list(LIST2)
    ed=Editor()
    for number,old_index,new_index in [(2,9,12),(3,19,22),(4,29,32),(5,39,42),(6,49,52)]:
        index=ed.locate(content(a[old_index]))
        ed.full(index,[a[new_index]],f'章首A{number-1:02}')
        ed.full(index+1,[a[new_index+1]],f'章首A{number-1:02}')
    index=ed.locate('第7章 总结与展望')
    ed.insert(index,a[62],'章首A06',ed.blocks[ed.locate(content(a[68]))])
    ed.insert(index,a[63],'章首A06',ed.blocks[ed.locate(content(a[68]))])
    ed.full(ed.locate(content(a[68])),[a[70]],'章首A07')

    # B items are scoped pairs. Table and figure labels are handled separately.
    starts=[i for i,p in enumerate(a) if re.match(r'^B\d\d\s',content(p))]
    for k,start in enumerate(starts):
        end=starts[k+1] if k+1<len(starts) else len(a)
        item=re.match(r'^(B\d\d)',content(a[start]))[1]
        if item in {'B15','B16','B17','B20'}:
            continue
        oldpos=next(i for i in range(start,end) if content(a[i])=='原文')
        newpos=next(i for i in range(oldpos,end) if content(a[i])=='替换为')
        old,new=content(a[oldpos+1]),content(a[newpos+1])
        if item=='B33':
            old,new=old.replace('附录C ',''),new.replace('附录C ','')
        if item=='B09':
            i=ed.locate('式(4-5)列出目标价值聚合与熵项。')
            ed.full(i,[a[newpos+1]],item)
        elif item=='B34':
            i=ed.locate('作者的早期BAPR研究围绕分段平稳环境')
            ed.full(i,[a[newpos+1]],item)
        else:
            ed.phrase(old,new,'章首清单'+item)
    for old,new in [('早期BAPR置信加权算子的数学证明','BAPR置信加权算子的数学证明'),
                    ('F.1 早期接入检验','F.1 共同离线初始化下的续训比较'),
                    ('F.3 早期标定偏差探索','F.3 单次训练下的标定偏差试验')]:
        ed.phrase(old,new,'章首清单-目录同步',0,260)
    ed.phrase('早期 BAPR','BAPR（部分稳定化）','章首清单B15-B16',1050,1085,all_hits=True)
    ed.phrase('ReLU输出，采用早期部分稳定化设计','ReLU输出，采用部分训练稳定化设计','章首清单B15')
    table=ed.locate('运行条件不额外驻站间隔均衡RE-SAC早期BAPR')
    ed.phrase('早期BAPR','BAPR','章首清单B20',table,table+1)

    replacements={1:12,2:19,4:32,5:39,6:46,8:61,10:75,11:81}
    prefixes={1:'其中，ι表示环境',2:'其中，d',4:'作者前期的贝叶斯遗忘分段鲁棒',5:'已有源模型时',
              6:'公交H2O+和MC-WM按式',8:'第5章跨保真任务以必要上下客完成时刻',
              10:'实际应用中，线路、计划及已经发生',11:'第3章原接口直接采用'}
    for number,source_index in replacements.items():
        phrase=prefixes[number]
        if number==2:
            i=ed.locate('为状态动作分布，')
        elif number==6:
            i=ed.locate(phrase,400,430)
        else:
            i=ed.locate(phrase)
        ed.full(i,[b[source_index]],f'补充A{number:02}')
        if number==1:
            ed.retain_mean_notation(ed.blocks[i])
    index=ed.locate('状态设计进一步区分')
    ed.full(index,[b[53]],'补充A07')
    ed.full(index+1,[b[54]],'补充A07')
    index=ed.locate('第3章双向单线路实验使用局部车辆观测')
    ed.extra[index]=[]
    ed.log('补充A09',index,content(ed.blocks[index]),'【整段删除】')
    # First and second A12 fragments contain native inline symbols.
    old='一般决策模型用a表示动作，其取值编码由具体接口确定。'
    ed.phrase(old,content(b[90]).removeprefix('①'),'补充A12-1')
    index=ed.locate('多线路公交离线学习同样以0至60秒的连续附加驻站')
    ed.phrase('多线路公交离线学习同样以0至60秒的连续附加驻站为动作，其他执行命令保持原实验设置。',
              '多线路公交离线学习同样以0至60秒的附加驻站时长为控制动作，不直接调节车速。','补充A12-2')
    source=deepcopy(b[92]); next(source.iter(W+'t')).text=next(source.iter(W+'t')).text.removeprefix('③')
    ed.full(ed.locate('第3、4章的车头时距实验按各方法原有形式'),[source],'补充A12-3')
    index=next(i for i,p in enumerate(ed.blocks) if p.tag==W+'tbl' and content(p).endswith('(2-29)'))
    ed.insert(index,b[25],'补充A03')
    ed.retain_mean_notation(ed.new_after[index][-1])

    # The supplementary B1 table supplies exact text rather than new prose.
    table=b[97]
    for row in table.findall(W+'tr')[1:]:
        cells=row.findall(W+'tc');old,new=content(cells[1]),content(cells[2])
        if '按A04' in new or '上一份A03' in new:
            continue
        if old.startswith('MC-WM采用前期'):
            ed.phrase(old,new,'补充B1')
        else:
            ed.phrase(old,new,'补充B1')
    ed.phrase('前序BAPR','BAPR','补充B2-方法名',all_hits=True)
    ed.phrase('CS-BAPR相对BAPR的收益分解','CS-BAPR相对BAPR（未采用稳定化设计）的收益分解','补充B2-图4-5')
    for old,new in [
        ('前序配置与组件对照则分析','BAPR对照与组件试验分别分析'),
        ('由前序配置的5降为2','由BAPR配置中的5降为2'),
        ('前序稳定化配置','训练稳定化配置'),
        ('前序配置比较则说明了方法形成过程中的不同作用。','训练稳定化与输出结构的对照试验用于区分不同设计的作用。'),
        ('前序驻站改变的到达时刻会影响之后的换乘需求','上一乘车段中的驻站会改变乘客到达换乘站的时刻，进而影响后续乘车段的候车需求'),
        ('前序幅度变换','实施最低驻站保障前的时长调整')]:
        ed.phrase(old,new,'补充B2',all_hits=True)

    # C items explicitly amend the abstract; other abstract wording is retained.
    ed.phrase('以加权的候车与车内时间作为乘客广义时间成本，并','并','补充C01',63,70)
    abstract=ed.locate('（1）针对双向发车环境',63,70)
    ed.phrase('RE-SAC）。','RE-SAC）。'+content(b[118]),'补充C02',abstract,abstract+1)
    ed.phrase(content(b[123]),content(b[125]),'补充C03',70,80)
    ed.phrase(content(b[130]),content(b[132]),'补充C04')
    ed.full(ed.locate('其中，G表示每名已出发乘客'),[b[139]],'补充C05')
    index=ed.locate('表3-4中，可学习嵌入')
    ed.insert(index-1,b[146],'补充C06',ed.blocks[index])
    ed.phrase(content(b[151]),content(b[153]),'补充C07')
    ed.phrase(content(b[158]),content(b[160]),'补充C08')

    for old,new,lo,hi in [
        ('广义时间相对差','人均广义时间成本相对差',540,569),
        ('广义时间差','人均广义时间成本差',540,569),
        ('广义时间和候车时间相对差15%','人均广义时间成本和平均候车时间的相对差15%',540,569),
        ('因此，总成本接近还需结合时间构成判断','因此，人均广义时间成本接近还需结合候车与车内时间的构成判断',540,569),
        ('广义时间响应差','人均广义时间成本响应差',570,583),
        ('其中最大值为：广义时间3.90%','其中最大值为：人均广义时间成本3.90%',570,583),
        ('广义时间归一化响应差','人均广义时间成本的归一化响应差',260,1450),
        ('广义时间差','人均广义时间成本差',1340,1355),
        ('横轴表示人均广义时间减少量','横轴表示人均广义时间成本减少量',1340,1360),
        ('完成保障的候车收益、车内代价与广义时间变化','完成保障的候车收益、车内代价与人均广义时间成本变化',1365,1375),
        ('按权重2和1合计减少94.04秒广义时间','按权重2和1计算，人均广义时间成本减少94.04秒',1380,1420),
        ('广义时间减少988.09秒','该群体的人均广义时间成本减少988.09秒',1400,1450),
        ('广义时间减少9.92秒','该群体的人均广义时间成本减少9.92秒',1400,1450),
        ('其广义时间增加255.12秒','该群体的人均广义时间成本增加255.12秒',1400,1450),
        ('跨保真任务采用广义时间目标后','跨保真任务以乘客广义时间成本为优化目标后',1400,1450),
        ('因而广义时间净减少23.02秒','因而人均广义时间成本净减少23.02秒',2780,2805),
        ('降低人均广义时间','降低人均广义时间成本',2780,2805),
        ('在8组降低人均时间','在8组降低人均广义时间成本',2780,2805)]:
        ed.phrase(old,new,'补充C09',lo,hi,all_hits=True)

    for number,pairs in {
        '5-6':[('广义时间差','人均广义时间成本差'),('候车时间差','平均候车时间差'),('车内时间差','平均车内时间差')],
        '5-10':[('广义时间差/秒','人均广义时间成本差/秒'),('候车差/秒','平均候车时间差/秒'),('车内差/秒','平均车内时间差/秒')],
        '6-6':[('正常需求成本/秒','正常需求人均广义时间成本/秒'),('1.25倍需求成本/秒','1.25倍需求人均广义时间成本/秒')],
        '6-7':[('H2O+成本/秒','H2O+人均广义时间成本/秒'),('MC-WM成本/秒','MC-WM人均广义时间成本/秒'),('时间降幅/%','人均广义时间成本降幅/%')],
        '6-8':[('H2O+成本/秒','H2O+人均广义时间成本/秒'),('混合损失系数0.75成本/秒','混合损失系数0.75人均广义时间成本/秒'),('MC-WM成本/秒','MC-WM人均广义时间成本/秒')],
        '6-11':[('规则成本/秒','规则人均广义时间成本/秒'),('H2O+成本/秒','H2O+人均广义时间成本/秒'),('归一化MC-WM成本/秒','归一化MC-WM人均广义时间成本/秒'),('时间降幅/%','人均广义时间成本降幅/%')],
        '6-16':[('人均成本/秒','人均广义时间成本/秒')],
        'F-2':[('人均成本降幅','人均广义时间成本降幅')],
    }.items():
        cap=next(i for i,p in enumerate(ed.blocks) if p.tag==W+'p'
                 and p.find(W+'pPr/'+W+'pStyle') is not None
                 and p.find(W+'pPr/'+W+'pStyle').get(W+'val')=='tablecaption'
                 and content(p).startswith('表'+number+' '))
        index=cap+1
        assert ed.blocks[index].tag==W+'tbl',(number,index)
        for old,new in pairs:
            ed.phrase(old,new,'补充C10-表'+number,index,index+1,all_hits=True)
    ed.phrase('混合训练基线与Daganzo式规则的成本差','混合训练基线与Daganzo式规则的人均广义时间成本差','补充C10-表6-13')
    ed.phrase('系数调节扩展相对固定0.25配置的平均成本差','系数调节扩展相对固定0.25配置的人均广义时间成本的平均差','补充C10-表6-15')
    ed.phrase('单位：秒/已出发乘客','单位：秒，按已出发乘客平均','补充C10-单位',all_hits=True)
    # Preserve minutes in Figure 3-9; these are only prose units for G.
    for i,s in enumerate(ed.texts):
        if '秒/人' in s and '2.5秒/人' not in s and ('单位' in s or '人均广义时间成本' in s):
            while '秒/人' in ed.texts[i]:
                ed.phrase('秒/人','秒','补充C10-正文单位',i,i+1)
    ed.save()


if __name__=='__main__':
    main()
