"""Integrate the supplied reconstruction while preserving native Word objects."""
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from shutil import copy2
from zipfile import ZipFile
import json
import re
import sys
from lxml import etree as E

from inventory import HERE, ROOT, SOURCE, PLAN, W, M, NS, content, load
sys.path.insert(0,str(ROOT/'workpapers/七章重写稿整合_20261010'))
from integrate import body_spans

OUTPUT=ROOT/'output/排版完成_章节重构_20261012.docx'
SPACE='{http://www.w3.org/XML/1998/namespace}space'


def style(p):
    node=p.find(W+'pPr/'+W+'pStyle')
    return node.get(W+'val') if node is not None else ''


def replace_text(p,old,new):
    # Replace complete field-bearing clauses atomically, then restore retained refs.
    retained=[];nodes=list(p.iter(W+'t'));value=''.join(n.text or '' for n in nodes)
    if old in value:
        start=value.index(old);end=start+len(old);offset=0;group=[];group_start=0
        for child in list(p):
            marks=child.findall('.//'+W+'fldChar')
            if any(n.get(W+'fldCharType')=='begin' for n in marks):group=[];group_start=offset
            active=bool(group) or any(n.get(W+'fldCharType')=='begin' for n in marks)
            if active:group.append(child)
            offset+=len(''.join(child.xpath('.//w:t/text()',namespaces=NS)))
            if active and any(n.get(W+'fldCharType')=='end' for n in marks):
                key=''.join(''.join(c.xpath('.//w:t/text()',namespaces=NS)) for c in group)
                if group_start>=start and offset<=end and (group_start>start or offset<end):
                    pos=p.index(group[0]);copy=deepcopy(group)
                    for node in group:p.remove(node)
                    r=E.Element(W+'r');E.SubElement(r,W+'t').text=key;p.insert(pos,r)
                    if key and key in new:retained.append((key,copy))
                group=[]
    nodes=list(p.iter(W+'t'))
    value=''.join(n.text or '' for n in nodes)
    if old not in value and old in content(p):
        value=content(p);start=value.index(old);end=start+len(old)
        offset=0;inserted=False
        for child in list(p):
            s=content(child);right=offset+len(s)
            if right>start and offset<end:
                a=max(0,start-offset);b=min(len(s),end-offset)
                if child.tag!=W+'r':assert a==0 and b==len(s),(old,s)
                at=p.index(child);p.remove(child)
                v=s[:a]+(new if not inserted else '')+s[b:]
                if v:
                    r=E.Element(W+'r');rp=child.find(W+'rPr')
                    if rp is not None:r.append(deepcopy(rp))
                    E.SubElement(r,W+'t',{SPACE:'preserve'}).text=v;p.insert(at,r)
                inserted=True
            offset=right
        return
    assert old in value,(old,value[:200])
    start=value.index(old);end=start+len(old);offset=0;inserted=False
    for node in nodes:
        v=node.text or '';right=offset+len(v)
        if right>start and offset<end:
            node.text=v[:max(0,start-offset)]+(new if not inserted else '')+v[min(len(v),end-offset):]
            node.set(SPACE,'preserve');inserted=True
        offset=right
    for key,group in retained:
        run=next(r for r in p.findall(W+'r') if r.find(W+'t') is not None and key in (r.find(W+'t').text or ''))
        value=run.find(W+'t').text;before,after=value.split(key,1)
        parts=[]
        for v in [before]:
            if v:r=deepcopy(run);r.find(W+'t').text=v;parts.append(r)
        parts.extend(group)
        if after:r=deepcopy(run);r.find(W+'t').text=after;parts.append(r)
        at=p.index(run);p.remove(run)
        for i,node in enumerate(parts):p.insert(at+i,node)


class Manuscript:
    def __init__(self):
        self.raw,self.root,self.old=load(SOURCE)
        _,_,self.plan=load(PLAN)
        self.spans=body_spans(self.raw)
        self.original={id(p):E.tostring(p) for p in self.old}
        self.index={id(p):i for i,p in enumerate(self.old)}
        self.texts=[content(p) for p in self.old]
        self.replacements={};self.before=defaultdict(list);self.after=defaultdict(list)
        self.eq_orig={};self.cap_orig={};self.new_eq={};self.new_cap={}
        self.reference_anchors={}
        self.next_id=max(int(x) for x in self.root.xpath('.//w:bookmarkStart/@w:id',namespaces=NS))+1
        self.templates={}
        for p in self.old[250:]:
            if style(p) and style(p) not in self.templates and p.find('.//'+W+'sectPr') is None:
                self.templates[style(p)]=p
            if p.tag==W+'tbl':
                rows=p.findall(W+'tr')
                if len(rows)==1 and len(rows[0].findall(W+'tc'))>=2:
                    cells=rows[0].findall(W+'tc')
                    if re.fullmatch(r'\([1-7A-F]-\d+[a-z]?\)',content(cells[-1])):
                        self.eq_orig[id(p)]=content(cells[-1])
            if style(p) in {'figurecaption','tablecaption'}:
                self.cap_orig[id(p)]=re.match(r'^[图表][1-7A-F]-\d+',content(p))[0]
            if style(p)=='bibliography':
                n=re.match(r'^\[(\d+)\]',content(p))
                names=p.xpath('.//w:bookmarkStart/@w:name',namespaces=NS)
                if n:
                    self.reference_anchors[int(n[1])]=next(x for x in names if x.startswith('ref-'))
        self.eq_template=self.old[375]
        self.templates['body']=self.old[364]
        self.logs=[]

    def find(self,phrase,lo=250,hi=None):
        hits=[i for i,t in enumerate(self.texts) if lo<=i<(hi or len(self.old)) and phrase in t]
        assert len(hits)==1,(phrase,hits)
        return hits[0]

    def bookmark(self,p,name):
        n=str(self.next_id);self.next_id+=1
        begin=E.Element(W+'bookmarkStart',{W+'id':n,W+'name':name})
        end=E.Element(W+'bookmarkEnd',{W+'id':n})
        p.insert(1 if p.find(W+'pPr') is not None else 0,begin)
        p.append(end)

    def cite(self,p):
        for run in list(p.findall(W+'r')):
            node=run.find(W+'t')
            if node is None:
                continue
            s=node.text or '';parts=[];at=0
            for m in re.finditer(r'\[(\d+(?:,\d+)*)\]',s):
                if m.start()>at:
                    r=deepcopy(run);r.find(W+'t').text=s[at:m.start()];parts.append(r)
                for n in map(int,m[1].split(',')):
                    link=E.Element(W+'hyperlink',{W+'anchor':self.reference_anchors[n],W+'history':'1'})
                    r=deepcopy(run);r.find(W+'t').text=f'[{n}]'
                    rp=r.find(W+'rPr')
                    if rp is None:
                        rp=E.Element(W+'rPr');r.insert(0,rp)
                    E.SubElement(rp,W+'vertAlign',{W+'val':'superscript'})
                    link.append(r);parts.append(link)
                at=m.end()
            if parts:
                if at<len(s):
                    r=deepcopy(run);r.find(W+'t').text=s[at:];parts.append(r)
                i=p.index(run);p.remove(run)
                for j,x in enumerate(parts): p.insert(i+j,x)

    def formatted(self,source,kind='body'):
        p=deepcopy(source)
        for x in list(p):
            if x.tag in {W+'pPr',W+'bookmarkStart',W+'bookmarkEnd'}:
                p.remove(x)
        template=self.templates[kind]
        pp=template.find(W+'pPr')
        if pp is not None:p.insert(0,deepcopy(pp))
        for r in p.iter(W+'r'):
            rp=r.find(W+'rPr')
            if rp is not None:r.remove(rp)
        for r in p.iter(M+'r'):
            rp=r.find(W+'rPr')
            if rp is not None:r.remove(rp)
            rp=E.Element(W+'rPr')
            fonts={W+'ascii':'Cambria Math',W+'hAnsi':'Cambria Math',W+'eastAsia':'宋体'}
            E.SubElement(rp,W+'rFonts',fonts)
            r.insert(1 if r.find(M+'rPr') is not None else 0,rp)
        self.cite(p)
        return p

    def canonical_theory_symbols(self,paragraphs):
        for p in paragraphs:
            value=content(p)
            mapping={}
            if 'α' in value:mapping['α']=('α','ent')
            if 'CVaR' in value or '概率重加权约束' in value:
                mapping.update({'β':('β','tail'),'η':('η','risk'),'ζ':('ζ','risk'),'C':('C','traj')})
            if '期望分位' in value or '状态价值损失为' in value or '12<τ<1' in value:
                mapping['τ']=('τ','IQL')
            if '隐藏模式为' in value or value.startswith('bt+1z′'):
                mapping['K']=('K','z')
                for r in p.iter(M+'r'):
                    node=r.find(M+'t')
                    if node is not None and node.text=='b':node.text='ρ'
            if value.startswith('inf∥δ∥') or '扰动δ' in value:
                mapping.update({'δ':('δ','dual'),'ε':('ρ','dual'),'x':('f','dual')})
            for r in list(p.iter(M+'r')):
                t=r.find(M+'t')
                if t is None or t.text not in mapping:continue
                base,sub=mapping[t.text]
                container=E.Element(M+'sSub');expr=E.SubElement(container,M+'e')
                copied=deepcopy(r);copied.find(M+'t').text=base;expr.append(copied)
                index=E.SubElement(container,M+'sub');run=E.SubElement(index,M+'r')
                prop=E.SubElement(run,M+'rPr');E.SubElement(prop,M+'sty',{M+'val':'p'})
                E.SubElement(run,M+'t').text=sub
                parent=r.getparent();parent.replace(r,container)

    def paragraph(self,value,kind='body'):
        p=E.Element(W+'p');r=E.SubElement(p,W+'r');E.SubElement(r,W+'t').text=value
        return self.formatted(p,kind)

    def label(self,p,label,sequence,number,marks=()):
        props=deepcopy(p.find(W+'pPr'))
        for child in list(p):p.remove(child)
        if props is not None:p.append(props)
        starts=[deepcopy(x) for x in marks if x.tag==W+'bookmarkStart']
        ends=[deepcopy(x) for x in marks if x.tag==W+'bookmarkEnd']
        p.extend(starts)
        def run(value=None,mark=None,code=None):
            r=E.SubElement(p,W+'r')
            if value is not None:E.SubElement(r,W+'t').text=value
            if mark:E.SubElement(r,W+'fldChar',{W+'fldCharType':mark})
            if code:E.SubElement(r,W+'instrText',{SPACE:'preserve'}).text=code
        run(label);run(mark='begin');run(code=f'SEQ {sequence} \\* ARABIC')
        run(mark='separate');run(str(number));run(mark='end')
        if label.startswith('('):run(')')
        p.extend(ends)

    def equation(self,index,key):
        source=self.plan[index]
        table=deepcopy(self.eq_template)
        for n in list(table.iter(W+'bookmarkStart'))+list(table.iter(W+'bookmarkEnd')):
            n.getparent().remove(n)
        cells=table.find(W+'tr').findall(W+'tc')
        p=cells[-2].find(W+'p')
        props=deepcopy(p.find(W+'pPr'))
        for n in list(p):p.remove(n)
        if props is not None:p.append(props)
        mp=deepcopy(source.find(M+'oMathPara'))
        assert mp is not None,(index,content(source))
        om=mp.find(M+'oMath')
        if index<101:
            tag=om[-1]
            assert tag.tag==M+'d' and re.fullmatch(r'2−\d+',content(tag)),(index,content(tag))
            om.remove(tag)
            if om[-1].tag==M+'r' and not content(om[-1]).strip():om.remove(om[-1])
        for rp in mp.iter(W+'rPr'):
            for el in list(rp):rp.remove(el)
            E.SubElement(rp,W+'rFonts',{W+'ascii':'Cambria Math',W+'hAnsi':'Cambria Math',W+'eastAsia':'宋体'})
        p.append(mp)
        label=cells[-1].find(W+'p')
        self.label(label,'(0-','Eq0',0)
        self.bookmark(label,'restruct_eq_'+key)
        self.new_eq[id(table)]=key
        return table

    def plan_blocks(self,indices,prefix):
        result=[]
        for index in indices:
            p=self.plan[index]
            if not content(p).strip():continue
            if p.find(M+'oMathPara') is not None:
                result.append(self.equation(index,f'{prefix}_{index}'))
            else:
                result.append(self.formatted(p,'h2' if style(p)=='Heading2' else 'body'))
        return result

    def replace(self,a,b,new,item):
        self.replacements[a]=new
        for i in range(a+1,b):self.replacements[i]=[]
        self.logs.append({'item':item,'old_indices':[a,b],'new_blocks':len(new)})

    def append(self,index,blocks,item,before=False):
        (self.before if before else self.after)[index].extend(blocks)
        self.logs.append({'item':item,'at':index,'new_blocks':len(blocks)})

    def rewrite(self,old,new):
        i=self.find(old);p=self.old[i]
        replace_text(p,old,new)

    def merged(self,indices):
        p=self.paragraph('')
        for i,index in enumerate(indices):
            if i:
                r=E.SubElement(p,W+'r');E.SubElement(r,W+'t').text=''
            for node in self.old[index]:
                if node.tag not in {W+'pPr',W+'bookmarkStart',W+'bookmarkEnd'}:
                    p.append(deepcopy(node))
        return p

    def budget(self,col,key,title):
        source=self.old[616]
        table=deepcopy(source)
        for n in list(table.iter(W+'bookmarkStart'))+list(table.iter(W+'bookmarkEnd')):
            n.getparent().remove(n)
        total=sum(int(n.get(W+'w')) for n in source.find(W+'tblGrid'))
        grid=table.find(W+'tblGrid')
        for n in list(grid):grid.remove(n)
        widths=[int(total*.28),total-int(total*.28)]
        for width in widths:E.SubElement(grid,W+'gridCol',{W+'w':str(width)})
        for ri,row in enumerate(table.findall(W+'tr')):
            cells=row.findall(W+'tc')
            for cell in list(cells):
                if cell not in [cells[0],cells[col]]:row.remove(cell)
            for cell,width in zip(row.findall(W+'tc'),widths):
                cw=cell.find(W+'tcPr/'+W+'tcW');cw.set(W+'w',str(width))
            if ri==0:
                p=row.findall(W+'tc')[1].find(W+'p')
                for run in list(p):
                    if run.tag!=W+'pPr':p.remove(run)
                run=E.SubElement(p,W+'r');E.SubElement(run,W+'t').text='配置'
        if col<=3:
            for row in table.findall(W+'tr')[:-1]:
                for p in row.iter(W+'p'):
                    props=p.find(W+'pPr')
                    if props is None:props=E.Element(W+'pPr');p.insert(0,props)
                    if props.find(W+'keepNext') is None:E.SubElement(props,W+'keepNext')
        cap=self.paragraph(title,'tablecaption')
        self.new_cap[id(cap)]=key
        self.bookmark(cap,'restruct_table_'+key)
        return [cap,table]

    def add_references(self):
        names={215:'ref-Luo2022ESCP',216:'ref-Pinto2017RARL',217:'ref-Sutton2018Introduction'}
        self.reference_anchors.update(names)
        values={215:content(self.plan[218]).replace('\u00a0',' ').replace('–','-'),
                216:'[216] PINTO L, DAVIDSON J, SUKTHANKAR R, et al. Robust Adversarial Reinforcement Learning[EB/OL]. arXiv, 2017[2026-10-12]. https://arxiv.org/abs/1703.02702.',
                217:content(self.plan[220]).replace('\u00a0',' ')}
        end=self.find('[214]',1814,2050)
        new=[]
        for n,value in values.items():
            p=self.paragraph(value,'bibliography');self.bookmark(p,names[n]);new.append(p)
        self.append(end,new,'新增核实的文献215-217')

    def build(self):
        self.add_references()
        # The chapter-two section break remains in its original position.
        new2=[self.formatted(self.plan[12],'chapter')]+self.plan_blocks(range(13,101),'theory2')
        self.canonical_theory_symbols(new2)
        self.theory_paragraphs={id(p) for p in new2}
        for bm in self.old[363]:
            if bm.tag in {W+'bookmarkStart',W+'bookmarkEnd'}:new2[0].append(deepcopy(bm))
        self.new2=new2
        summary=next(p for p in new2 if content(p).startswith('本章介绍了传统公交在途控制'))
        E.SubElement(summary.find(W+'pPr'),W+'keepLines')
        self.replace(363,632,new2,'第二章完整替换')
        self.rewrite(content(self.old[354]),content(self.plan[199]))
        self.replace(645,646,self.plan_blocks([109,110],'3A'),'3A')
        for bm in self.old[645]:
            if bm.tag in {W+'bookmarkStart',W+'bookmarkEnd'}:self.replacements[645][0].append(deepcopy(bm))
        self.replace(663,664,self.plan_blocks([112,113,114],'3B')+[self.old[398],self.old[402],self.old[403]],'3B及事件图迁入')
        self.append(698,self.plan_blocks([116,117,118],'3C'),'3C')
        self.append(930,self.plan_blocks([120],'3D'),'3D')
        self.replace(964,965,self.plan_blocks([125],'4A'),'4A')
        self.append(972,self.plan_blocks(range(127,133),'4B'),'4B')
        self.replace(1045,1046,self.plan_blocks([134],'4C'),'4C')
        self.replace(1208,1217,self.plan_blocks(range(141,155),'5A')+[self.old[417],self.old[418]],'5A及输入来源表迁入')
        state_paragraph=next(p for p in self.replacements[1208] if content(p).startswith('状态同时描述当前车辆'))
        for bm in self.old[1214]:
            if bm.tag in {W+'bookmarkStart',W+'bookmarkEnd'}:state_paragraph.append(deepcopy(bm))
        replacement_g=next(p for p in self.replacements[1208] if self.new_eq.get(id(p))=='5A_153')
        self.equation_aliases={'(5-5)':'5A_153','(2-8)':'5A_145','(2-9)':'5A_147','(2-12)':'5A_150','(2-15)':'5A_153','(2-30)':'5B_160','(2-31)':'5B_160','(2-2)':'3B_113'}
        label=replacement_g.find(W+'tr').findall(W+'tc')[-1].find(W+'p')
        original_label=self.old[1211].find(W+'tr').findall(W+'tc')[-1].find(W+'p')
        for bm in original_label:
            if bm.tag in {W+'bookmarkStart',W+'bookmarkEnd'}:label.append(deepcopy(bm))
        self.append(1188,[self.old[542],self.old[543],self.old[544]],'跨保真数据分工图迁入')
        # Keep empirical detail beside each migrated figure, without duplicate summaries.
        calibration=self.plan_blocks([156,157],'5B')
        calibration += [self.merged([527,528]),self.old[529],self.old[530],self.old[538],self.old[539],self.old[540]]
        calibration += self.plan_blocks([158],'5B')
        calibration += [self.merged([548,549]),self.old[550],self.old[551],self.merged([552,553]),self.old[558],self.old[559]]
        calibration += [self.merged([555])]
        # Existing mean-error definition is retained in its native equation.
        details=self.merged([560,561,562,563,564])
        replace_text(details,'按0.75、1.00和1.25倍需求排列，人均广义时间成本相对差分别为1.21%、0.25%和−1.85%，','同一需求顺序下，')
        calibration += [self.old[556],self.formatted(self.old[557]),details,self.old[565],self.old[566]]
        response=self.plan_blocks([159,160,161],'5B')
        for p in response:
            for math in p.iter(M+'oMath'):
                for sub in math.iter(M+'sSub'):
                    base=sub.find(M+'e')
                    if base is not None and content(base)=='M':
                        acc=E.Element(M+'acc');prop=E.SubElement(acc,M+'accPr')
                        E.SubElement(prop,M+'chr',{M+'val':'̅'})
                        expr=E.SubElement(acc,M+'e')
                        for node in list(base):base.remove(node);expr.append(node)
                        base.append(acc)
        calibration += response
        response_detail=self.merged([569,575,576])
        old='对策略π和指标M，先在各环境内扣除基线，定义控制响应为'
        if old in content(response_detail):replace_text(response_detail,old,'')
        calibration += [response_detail,self.old[578],self.old[579],self.old[582],self.old[583],self.merged([584,585])]
        calibration += self.plan_blocks([162],'5B')
        self.append(1291,calibration,'5B及原标定图表迁入')
        self.append(1323,self.budget(1,'budget_formal','正式控制器的训练与评价安排')+[self.paragraph('评价回合数、独立运行日数和独立训练次数分别统计，不相互替代。')],'正式控制器配置表')
        self.append(1379,self.budget(2,'budget_initial','初步因素比较的训练与评价安排'),'初步因素配置表')
        self.append(1407,self.budget(3,'budget_matched','更新次数匹配比较的训练与评价安排'),'更新次数匹配配置表')
        self.replace(1468,1469,self.plan_blocks(range(168,176),'6A'),'6A')
        self.append(1529,self.plan_blocks(range(177,180),'6B'),'6B')
        self.append(1550,self.plan_blocks(range(181,185),'6C'),'6C')
        self.append(1644,self.plan_blocks([186,187],'6D')+self.budget(4,'budget_model','公交模型辅助比较的训练与评价安排')+[self.old[617],self.old[619]],'6D及模型配置表')
        self.replace(1645,1646,[self.paragraph('本节先分析一次拟合和原置信递推的结果，说明为何需要进一步调整预测样本的使用方式。')],'合并6D重复概括')
        self.append(self.find('最后，采用车头时距奖励补充比较训练目标与执行方式的影响'),self.plan_blocks([189,190,191],'6E'),'6E')
        # Only the listed short replacements and necessary migration links change.
        for old,new in [
            ('本章的平稳性假设沿用第2章的定义，约束给定条件下的转移与奖励规律','本章的平稳性假设是：给定线路、位置、方向和时段后，状态转移与奖励的生成规律保持不变'),
            ('完整策略仍需依据第2章的运行关系评价其对后续站点、车辆周转和乘客服务的影响','完整策略的评价仍需考虑后续站点、车辆周转和乘客服务，不能仅依据上述局部关系判断'),
            ('风险敏感与鲁棒评价的基本定义已见第2.2.4节，以下直接讨论其与本章目标的联系','以下分别从最不利动作评价和局部乘客成本出发，说明本章学习目标的依据'),
            ('沿用第2章的记号，',''),
            ('控制效果沿用前两章的评价方式，分别报告车头时距与人均广义时间成本','本章分别报告计划时距偏差、前后车头时距不均衡度、人均广义时间成本和行程完成率'),
            ('本节在第3章的共享策略与集成价值评价基础上，说明CS-BAPR的训练调节及策略表示','本节采用车队共享策略与价值集成，说明CS-BAPR的训练调节、动作表示及符号响应约束'),
            ('采用第3.2.3节的计划间隔奖励训练策略','采用本章第4.2.1节给出的计划间隔奖励训练策略'),
            ('惩罚值取第3.2.3节负奖励的绝对值','惩罚值取本章计划间隔奖励的负值'),
            ('需求变化对控制的影响可与第2章的服务时间关系对应','需求变化通过上客人数和服务时间影响后续车头时距'),
            ('沿用第2.3.1节的分布分解，',''),
            ('第2.3.1节已说明联合转移分类器与状态动作分类器的关系，DARC及H2O据此处理两类环境的动力学差异','联合转移分类器和状态动作分类器分别估计两类记录的域概率，其对数几率之差用于估计条件转移比率；DARC及H2O采用相关思想处理动力学差异'),
            ('第2章的30回合自然运行检验和210回合驻站响应检验','本章用于标定的30回合自然运行与210回合驻站响应记录'),
            ('源仿真的自然运行与驻站响应已按第2章完成检验','本章已检验源仿真的自然运行与驻站响应'),
            ('第2章的独立标定检验','本章的独立标定检验'),
            ('逐人时间累计之和与第2.2.2节的系统人数积分一致','逐人时间累计之和与本章第5.2.2节定义的系统人数积分一致'),
            ('第5章已经区分状态动作分布差异、条件转移差异和采样误差，本章重点研究如何利用目标记录修正运行预测，减少不合适的源预测对动作评价的影响','源目标经验的差异既可能来自访问状态与动作的频率，也可能来自条件转移规律及有限采样；本章利用目标记录修正运行预测，减少不合适的源预测对动作评价的影响'),
            ('本章公交实验延续第5章的数据条件','本章公交实验使用固定目标域离线数据'),
            ('以下在前文共同模型的基础上，说明两类任务的数据条件与预测对象','以下分别定义受控连续任务和公交事件任务的数据条件、决策过程与预测对象'),
            ('公交模型沿用第5章的状态、动作和成本定义','公交模型采用本章第6.2.1节给出的事件状态、驻站动作和乘客成本'),
            ('公交MC-WM在第5章混合学习的基础上加入事件预测','公交MC-WM在固定目标记录与新增源经验的混合更新中加入事件预测'),
            ('一次拟合比较采用表2-7中更新次数匹配的设置','一次拟合比较使各组分别完成相同的策略网络与动作价值网络更新次数'),
            ('式(4-3)对应第2章介绍的POMDP','式(4-3)对应运行状态可观测而模式不可观测的部分可观测决策过程（POMDP）'),
            ('本章保持前文规定的线路、班次与车辆供给','本章保持线路、班次与车辆供给不变'),
            ('规则参数沿用第5章，不根据本组结果调整','车头时距规则把前向计划间隔与实际间隔之差乘以0.75，再减去后向计划间隔与实际间隔之差的0.35倍，并截断至0至60秒；参数不根据本组结果调整'),
            ('另一车头时距规则沿用第5章','另一车头时距规则把前向计划间隔与实际间隔之差乘以0.75，再减去后向计划间隔与实际间隔之差的0.35倍，并截断至0至60秒'),
        ]:
            self.rewrite(old,new)
        self.replace(1184,1185,[self.paragraph('本节分别定义多线路离线学习和跨保真混合训练的数据条件、观测状态、驻站动作与学习目标，说明同时到站车辆的决策组织及乘客成本的计算方式。两项任务的运行时域与乘客统计对象分别保留。')],'第五章引导段')
        self.replace(1196,1197,[],'删除建模部分重复标定结果')
        i=self.find('沿用式(4-6)的未加权动作平方误差')
        p=self.old[i]
        self.rewrite('LBC沿用式(4-6)的未加权动作平方误差，比较经验动作与当前策略通过重参数化采样得到的动作，再由βBC统一加权。','')
        self.append(i,self.plan_blocks([204],'BC'),'行为克隆局部定义')
        # Assemble once, so all location instructions refer to the original draft.
        self.final=[]
        for i,p in enumerate(self.old):
            self.final.extend(self.before[i])
            self.final.extend(self.replacements.get(i,[p]))
            self.final.extend(self.after[i])
        self.update_toc_structure()
        self.preserve_bookmarks()

    def update_toc_structure(self):
        entries=[]
        for p in self.new2:
            if style(p) not in {'chapter','h2'}:continue
            heading=content(p)
            if style(p)=='chapter':
                anchor='_Toc242183017';template=self.old[94]
            else:
                n=int(re.match(r'2\.(\d+)',heading)[1]);anchor=f'restruct_chapter2_section_{n}'
                self.bookmark(p,anchor);template=self.old[95]
            entry=deepcopy(template)
            nodes=entry.findall('.//'+W+'t')
            old=''.join(n.text or '' for n in nodes[:-1])
            replace_text(entry,old,heading)
            for link in entry.iter(W+'hyperlink'):link.set(W+'anchor',anchor)
            for field in entry.iter(W+'instrText'):
                field.text=re.sub(r'PAGEREF\s+\S+',f'PAGEREF {anchor}',field.text or '')
            entries.append(entry)
        a=self.final.index(self.old[94]);b=self.final.index(self.old[115])
        self.final[a:b]=entries

    def preserve_bookmarks(self):
        starts={n.get(W+'id') for p in self.final for n in p.iter(W+'bookmarkStart')}
        ends={n.get(W+'id') for p in self.final for n in p.iter(W+'bookmarkEnd')}
        original_starts={n.get(W+'id'):n for n in self.root.iter(W+'bookmarkStart')}
        original_ends={n.get(W+'id'):n for n in self.root.iter(W+'bookmarkEnd')}
        anchor=self.new2[0]
        # Preserve existing named ranges that cross replaced paragraph boundaries.
        for n,node in original_starts.items():
            if n not in starts:
                target=anchor
                name=node.get(W+'name')
                if name=='由采样分布到目标分布的重加权':
                    target=next(p for p in self.new2 if content(p).startswith('重要性采样通过'))
                elif name=='随机性运行规律变化与目标信息':
                    target=next(p for p in self.new2 if content(p).startswith('不确定性可以来自'))
                target.append(deepcopy(node))
                if n not in ends:target.append(deepcopy(original_ends[n]));ends.add(n)
                starts.add(n)
        for n,node in original_ends.items():
            if n not in ends:anchor.append(deepcopy(node));ends.add(n)

    def renumber(self):
        self.mapping={};self.anchor_labels={};self.chapter=0
        eq=defaultdict(int);fig=defaultdict(int);tab=defaultdict(int)
        for p in self.final[250:]:
            if style(p)=='chapter':
                hit=re.match(r'第(\d+)章',content(p))
                self.chapter=int(hit[1]) if hit else 0
            if self.chapter not in range(2,7):continue
            if id(p) in self.eq_orig or id(p) in self.new_eq:
                eq[self.chapter]+=1;n=eq[self.chapter]
                old=self.eq_orig.get(id(p));new=f'({self.chapter}-{n})'
                self.mapping[old or '@'+self.new_eq[id(p)]]=new
                cell=p.find(W+'tr').findall(W+'tc')[-1]
                label=cell.find(W+'p');marks=list(label.findall(W+'bookmarkStart'))+list(label.findall(W+'bookmarkEnd'))
                self.label(label,f'({self.chapter}-',f'Eq{self.chapter}',n,marks)
                for bm in label.findall(W+'bookmarkStart'):self.anchor_labels[bm.get(W+'name')]=new
            if style(p) in {'figurecaption','tablecaption'}:
                isfig=style(p)=='figurecaption';counter=fig if isfig else tab
                counter[self.chapter]+=1;n=counter[self.chapter]
                old=self.cap_orig.get(id(p));key=old or '@'+self.new_cap[id(p)]
                new=f"{'图' if isfig else '表'}{self.chapter}-{n}"
                self.mapping[key]=new
                value=content(p)
                tail=value[len(old):] if old else ' '+value
                marks=list(p.findall(W+'bookmarkStart'))+list(p.findall(W+'bookmarkEnd'))
                self.label(p,f"{'图' if isfig else '表'}{self.chapter}-",f"{'Fig' if isfig else 'Tab'}{self.chapter}",n,marks)
                r=E.SubElement(p,W+'r');E.SubElement(r,W+'t').text=tail
                for bm in p.findall(W+'bookmarkStart'):self.anchor_labels[bm.get(W+'name')]=new
        self.counts={'equations':dict(eq),'figures':dict(fig),'tables':dict(tab)}
        for old,key in self.equation_aliases.items():
            self.mapping[old]=self.mapping['@'+key]
        # Replace original visible references in one pass, never cascade new labels.
        pattern=re.compile(r'\([2-6]-\d+[a-z]?\)|[图表][2-6]-\d+')
        for p in self.final:
            if p.tag!=W+'p' or style(p) in {'figurecaption','tablecaption'} or id(p) in self.theory_paragraphs:continue
            nodes=list(p.iter(W+'t'));value=''.join(n.text or '' for n in nodes)
            changes=[(m.start(),m.end(),self.mapping[m[0]]) for m in pattern.finditer(value) if m[0] in self.mapping]
            for a,b,new in reversed(changes):
                offset=0;inserted=False
                for node in nodes:
                    s=node.text or '';right=offset+len(s)
                    if right>a and offset<b:
                        node.text=s[:max(0,a-offset)]+(new if not inserted else '')+s[min(len(s),b-offset):]
                        node.set(SPACE,'preserve');inserted=True
                    offset=right
        # Newly inserted theory refers to its own numbering, not the removed equations.
        self.rewrite_local_budget_links()
        self.relink_local_sections()

    def relink_local_sections(self):
        headings={}
        for p in self.final:
            if style(p) not in {'h2','h3'}:continue
            hit=re.match(r'([1-7]\.\d+(?:\.\d+)?)\s',content(p))
            names=p.xpath('.//w:bookmarkStart/@w:name',namespaces=NS)
            if hit and names:headings[hit[1]]=next((n for n in names if n.startswith('_Toc')),names[0])
        for p in self.final:
            for link in p.iter(W+'hyperlink'):
                hit=re.fullmatch(r'第?([1-7]\.\d+(?:\.\d+)?)(?:节)?',content(link).strip())
                if hit and hit[1] in headings:link.set(W+'anchor',headings[hit[1]])

    def rewrite_local_budget_links(self):
        matched=self.mapping['@budget_matched'];initial=self.mapping['@budget_initial'];model=self.mapping['@budget_model']
        for i,replacement in [(1384,initial),(2810,model)]:
            p=self.old[i]
            if '表2-7' in ''.join(p.itertext()):replace_text(p,'表2-7',replacement)
        chapter=0
        for p in self.final:
            if style(p)=='chapter':
                m=re.match(r'第(\d+)章',content(p));chapter=int(m[1]) if m else 0
            target='restruct_table_budget_model' if chapter==6 or p is self.old[2810] else 'restruct_table_budget_matched'
            if p is self.old[1384]:target='restruct_table_budget_initial'
            for field in p.iter(W+'instrText'):
                if re.search(r'\bREF\s+tbl2_7\b',field.text or ''):
                    field.text=re.sub(r'\bREF\s+tbl2_7\b',f'REF {target}',field.text)
        # The original matching text and reused-comparison disclosure remain.
        for p in self.final:
            if p is self.old[1384]:continue
            v=content(p)
            if p is self.old[618]:continue
            if '表2-7' in v:replace_text(p,'表2-7',matched)

    def save(self):
        body_start=self.spans[0][0];body_end=self.spans[-1][1]
        chunks=[]
        for p in self.final:
            i=self.index.get(id(p))
            if i is not None and E.tostring(p)==self.original[id(p)]:
                a,b=self.spans[i];chunks.append(self.raw[a:b])
            else:chunks.append(E.tostring(p))
        raw=self.raw[:body_start]+b''.join(chunks)+self.raw[body_end:]
        with ZipFile(SOURCE) as z:
            entries=[]
            for info in z.infolist():
                data=raw if info.filename=='word/document.xml' else z.read(info.filename)
                if re.fullmatch(r'word/header\d+\.xml',info.filename):
                    header=E.fromstring(data)
                    if '第2章 公交在途控制建模基础理论与方法介绍' in content(header):
                        replace_text(header,'第2章 公交在途控制建模基础理论与方法介绍','第2章 公交控制与强化学习基础')
                        data=E.tostring(header,xml_declaration=True,encoding='UTF-8',standalone=True)
                entries.append((info,data))
        with ZipFile(OUTPUT,'w') as z:
            for info,data in entries:z.writestr(info,data)
        (HERE/'迁移与编号.json').write_text(json.dumps({'operations':self.logs,'labels':self.mapping,'counts':self.counts},ensure_ascii=False,indent=2))
        print(json.dumps({'output':str(OUTPUT),'counts':self.counts},ensure_ascii=False))


if __name__=='__main__':
    copy2(SOURCE,HERE/'修改前.docx')
    m=Manuscript();m.build();m.renumber();m.save()
