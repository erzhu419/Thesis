from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree as E
import json
W=Path(__file__).resolve().parent
T=W.parents[1]/'output/排版完成_导师意见复核_20261005.docx'
N={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
PID='{http://schemas.microsoft.com/office/word/2010/wordml}paraId'
with ZipFile(T) as z: data={k:z.read(k) for k in z.namelist()}
r=E.fromstring(data['word/document.xml']); ps=r.find('w:body',N).findall('w:p',N)
locked=[E.tostring(p) for p in ps[61:80]]
math=[E.tostring(p) for p in r.xpath('//m:oMath',namespaces=N)]
byid={p.get(PID):p for p in ps}; changes=[]
def txt(p):return ''.join(p.xpath('.//w:t/text()',namespaces=N))
def replace(p,old,new):
    nodes=p.xpath('.//w:t',namespaces=N); s=txt(p)
    assert s.count(old)==1,(old,s)
    start=s.index(old);end=start+len(old);pos=0;done=False
    for n in nodes:
        t=n.text or '';q=pos+len(t)
        if pos<end and q>start:
            a=max(0,start-pos); b=min(len(t),end-pos)
            n.text=t[:a]+(new if not done else '')+t[b:];done=True
            if n.text.startswith(' ') or n.text.endswith(' '):n.set('{http://www.w3.org/XML/1998/namespace}space','preserve')
        pos=q
    assert txt(p)==s.replace(old,new)
    changes.append({'paragraph_id':p.get(PID),'old':old,'new':new})
replace(byid['0FB0B855'],'作者前期提出的RE-SAC由这一思路出发','作者前期提出的鲁棒集成软演员评论家方法（Robust Ensemble Soft Actor-Critic，RE-SAC）由这一思路出发')
replace(byid['5BF7E1E5'],'本文在此基础上进一步研究CS-BAPR，其中公交实验采用训练稳定化和神经算术输出配置。','本文在此基础上研究贝叶斯自适应策略正则化（Bayesian Adaptive Policy Regularization，BAPR），并进一步提出因果符号贝叶斯自适应策略正则化（Causal-Symbolic Bayesian Adaptive Policy Regularization，CS-BAPR），其中公交实验采用训练稳定化和神经算术输出配置。')
replace(byid['22E2D092'],'作者提出的MC-WM','作者提出的基于模型修正的世界模型方法（Model-Corrected World Model，MC-WM）')
replace(byid['46CB9442'],'的置信调节进一步研究CS-BAPR的训练稳定化与策略输出结构。','的置信调节进一步研究CS-BAPR的训练稳定化与策略输出结构。前序BAPR论文使用Bayesian Amnesic Piecewise-Robust这一名称；本章研究的贝叶斯自适应策略正则化及其因果符号扩展沿此基础发展。')
replace(byid['14EC903E'],'仿真到现实（simulation-to-reality，sim2real）','仿真到现实（sim-to-real transfer）')
for p in ps[80:]:
    if p.xpath('w:pPr/w:pStyle[@w:val="Bibliography"]',namespaces=N):continue
    s=txt(p)
    if 'sim2real' in s and ('沿用' in s or '技术路线' in s or 'Zhao等的' in s):replace(p,'sim2real','仿真到现实')
    if '采用相同学习策略但未加入完成保障的基线控制器' in s:replace(p,'采用相同学习策略但未加入完成保障的基线控制器','直接父控制器（采用相同学习策略、未加入完成保障）')
    if '共享停站决策映射' in s:replace(p,'共享停站决策映射','共享驻站决策映射')
assert [E.tostring(p) for p in ps[61:80]]==locked
assert [E.tostring(p) for p in r.xpath('//m:oMath',namespaces=N)]==math
# These edits must leave all table contents and structures untouched.
oldr=E.fromstring(data['word/document.xml'])
assert [E.tostring(p) for p in r.xpath('//w:tbl',namespaces=N)]==[E.tostring(p) for p in oldr.xpath('//w:tbl',namespaces=N)]
data['word/document.xml']=E.tostring(r,encoding='UTF-8',xml_declaration=True,standalone=True)
with ZipFile(T,'w',ZIP_DEFLATED) as z:
    for k,v in data.items():z.writestr(k,v)
(W/'正文术语修改.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2))
print(json.dumps({'term_changes':len(changes),'abstract_xml_unchanged':True,'equations_unchanged':len(math),'tables_unchanged':True}))
