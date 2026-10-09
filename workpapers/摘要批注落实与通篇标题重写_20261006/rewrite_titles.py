from pathlib import Path
from copy import deepcopy
from lxml import etree as E
from collections import Counter
import zipfile, json, difflib
ROOT=Path('/home/erzhu419/mine_code/Thesis');WORK=Path(__file__).parent
SRC=WORK/'修改前.docx';OUT=ROOT/'排版完成_摘要指定句与标题重写_20261006.docx'
N={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'};W='{'+N['w']+'}'
z=zipfile.ZipFile(SRC);infos=z.infolist();data={i.filename:z.read(i.filename) for i in infos}
r=E.fromstring(data['word/document.xml']);b=r.find('w:body',N);original=list(b)
def text(p):return ''.join(p.xpath('.//w:t/text()',namespaces=N))
def style(p):
 q=p.find('w:pPr/w:pStyle',N);return q.get(W+'val','') if q is not None else ''
TITLES={
'1.1':'研究背景与意义','1.2':'国内外研究综述','1.2.1':'公交驻站控制','1.2.2':'强化学习控制','1.2.3':'仿真迁移','1.2.4':'现有研究总结','1.3':'研究目标与内容','1.4':'技术路线','1.5':'论文结构',
'2.1':'公交运行与控制','2.1.1':'运行条件','2.1.2':'运行偏差','2.1.3':'控制范围',
'2.2':'强化学习基础','2.2.1':'决策过程','2.2.2':'动作与目标','2.2.3':'价值与策略','2.2.4':'不确定性与鲁棒性',
'2.3':'仿真平台','2.3.1':'基础数据','2.3.2':'平台构建','2.3.3':'参数标定',
'2.4':'迁移学习基础','2.4.1':'分布差异','2.4.2':'运行预测',
'2.5':'平台检验与实验设置','2.5.1':'运行特征','2.5.2':'控制响应','2.5.3':'评价方法','2.5.4':'适用范围','2.6':'本章小结',
'3.1':'问题描述','3.2':'模型构建','3.2.1':'基本条件','3.2.2':'决策过程','3.2.3':'优化目标','3.2.4':'目标分析',
'3.3':'求解方法','3.3.1':'价值评价','3.3.2':'策略优化','3.3.3':'理论分析',
'3.4':'仿真实验与结果分析','3.4.1':'实验设计','3.4.2':'服务水平对比','3.4.3':'训练稳定性','3.4.4':'控制机理分析','3.4.5':'影响因素分析','3.4.6':'多线路验证','3.4.7':'结果讨论','3.5':'本章小结',
'4.1':'问题描述','4.2':'模型构建','4.2.1':'运行条件','4.2.2':'环境识别',
'4.3':'求解方法','4.3.1':'训练设计','4.3.2':'策略表示','4.3.3':'理论分析',
'4.4':'仿真实验与结果分析','4.4.1':'实验设计','4.4.2':'需求强度的影响','4.4.3':'需求变化的影响','4.4.4':'适应效果分析','4.4.5':'影响因素分析','4.4.6':'结果讨论','4.5':'本章小结',
'5.1':'问题描述','5.2':'模型构建','5.2.1':'基本条件','5.2.2':'状态与目标','5.2.3':'决策变量',
'5.3':'求解方法','5.3.1':'离线与在线学习','5.3.2':'参照策略','5.3.3':'混合学习','5.3.4':'控制实施',
'5.4':'仿真实验与结果分析','5.4.1':'训练方式对比','5.4.2':'服务水平对比','5.4.3':'影响因素分析','5.4.4':'训练过程分析','5.4.5':'结果讨论','5.5':'本章小结',
'6.1':'问题描述','6.2':'模型构建','6.2.1':'基本条件','6.2.2':'预测关系',
'6.3':'求解方法','6.3.1':'预测方案选择','6.3.2':'置信度评价','6.3.3':'策略训练','6.3.4':'理论分析',
'6.4':'仿真实验与结果分析','6.4.1':'对照实验','6.4.2':'影响因素分析','6.4.3':'控制效果分析','6.4.4':'改进效果分析','6.4.5':'服务水平对比','6.4.6':'结果讨论','6.5':'本章小结',
'7.1':'研究总结','7.2':'主要创新','7.3':'研究展望',
'A.1':'基本条件','A.2':'动作评价','A.3':'基本性质','A.4':'熵正则化','A.5':'收敛性','A.6':'反馈影响','A.7':'风险评价','A.8':'适用范围',
'B.1':'基本条件','B.2':'补偿条件','B.3':'补偿效果','B.4':'模式识别','B.5':'动作误差','B.6':'路径分析','B.7':'期望分析','B.8':'回报损失','B.9':'适用范围',
'C.1':'基本定义','C.2':'等价表示','C.3':'基本性质','C.4':'更新性质','C.5':'保守性','C.6':'共享表示','C.7':'反馈影响','C.8':'识别速度','C.9':'适用范围',
'D.1':'基本条件','D.2':'标量误差','D.3':'向量误差','D.4':'输出结构','D.5':'复合结构','D.6':'基本性质','D.7':'反馈影响','D.8':'误差关系','D.9':'环境一致性','D.10':'误差传播','D.11':'适用范围',
'E.1':'基本条件','E.2':'收敛性','E.3':'误差关系','E.4':'数据覆盖','E.5':'更新条件','E.6':'置信排序','E.7':'理论解释','E.8':'适用范围','E.9':'尺度分析',
'F.1':'初步实验','F.2':'训练预算','F.3':'环境偏差','F.4':'实验记录',
}
# The latest clarification identifies the MC-WM passage and joint-mismatch definition.
# All other abstract paragraphs, including its ending, remain as the supplied baseline.
mcwm_explanation='公交应用中，利用预先采集的同一批SUMO运行记录，学习执行驻站动作后公交运行状态与乘客服务的变化。模型既预测后续到站事件的线路、站点等类别信息，也预测候车与载客人数、车头时距、事件间隔和乘客时间成本等数值信息。将这些预测结果作为补充样本用于驻站策略训练，并让预测可靠的样本获得较大权重。针对置信度在多步更新中累积后使样本权重过小的问题，对置信更新进行归一化，避免有用的模型预测在训练后期被过度削弱。'
joint_definition='将训练仿真中的道路平均速度、乘客OD强度、路口通行影响和必要停站时间同时改变，定义为联合源参数偏差。联合50%和联合75%两档设置相对于标定条件，分别将速度降低50%和75%，将OD强度与必要停站时间提高50%和75%，并将表征额外路口影响的遭遇密度设为0.5和0.75。'
paragraph=text(original[69]);prefix=paragraph.split('公交应用中，',1)[0]
result='单一驻站动作、乘客时间目标下，'+paragraph.split('单一驻站动作、乘客时间目标下，',1)[1]
result=result.replace('在正常标定及上述两档联合偏差条件下，','在正常标定、联合50%和联合75%三种条件下，')
ABSTRACT={69:prefix+mcwm_explanation+joint_definition+result}

def edit_range(p,start,end,repl):
 spans=[];pos=0
 for n in p.xpath('.//w:t',namespaces=N):
  s=n.text or '';spans.append((n,pos,pos+len(s)));pos+=len(s)
 if start==end:
  for n,a,c in spans:
   if a<=start<=c and c>a:
    t=n.text or '';n.text=t[:start-a]+repl+t[start-a:];n.set('{http://www.w3.org/XML/1998/namespace}space','preserve');return
  raise AssertionError(('insert',start))
 hits=[(n,a,c) for n,a,c in spans if c>start and a<end];assert hits
 for j,(n,a,c) in enumerate(hits):
  t=n.text or '';n.text=t[:max(start-a,0)]+(repl if j==0 else '')+t[min(end-a,len(t)):];n.set('{http://www.w3.org/XML/1998/namespace}space','preserve')
changes=[]
def settext(p,new,kind):
 old=text(p)
 if old==new:return
 for op,a,c,d,f in reversed(difflib.SequenceMatcher(None,old,new,autojunk=False).get_opcodes()):
  if op!='equal':edit_range(p,a,c,new[d:f])
 assert text(p)==new
 changes.append({'kind':kind,'before':old,'after':new})
for i,new in ABSTRACT.items():
 assert style(original[i])=='abstractcnbody'
 settext(original[i],new,'摘要指定句与联合偏差定义')
seen=set()
for p in original:
 if style(p) in ('h2','h3','h4'):
  num=text(p).split()[0];assert num in TITLES,num;seen.add(num)
  settext(p,num+' '+TITLES[num],'二级及以下标题')
assert seen==set(TITLES)
# Preserve all other manuscript content, including author headings and data.
base=E.fromstring(z.read('word/document.xml')).find('w:body',N)
for i,p in enumerate(original):
 if i not in ABSTRACT and style(p) not in ('h2','h3','h4'):assert E.tostring(p)==E.tostring(base[i]),i
assert not [p for p in original if style(p)=='h4']
assert Counter(E.tostring(v) for v in r.xpath('//m:oMath',namespaces=N))==Counter(E.tostring(v) for v in base.xpath('.//m:oMath',namespaces=N))
data['word/document.xml']=E.tostring(r,xml_declaration=True,encoding='UTF-8',standalone=True)
with zipfile.ZipFile(OUT,'w') as o:
 for info in infos:o.writestr(info,data[info.filename])
(WORK/'修改记录.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2))
(WORK/'新目录.txt').write_text('\n'.join(text(p) for p in original if style(p) in ('chapter','h2','h3')))
print(json.dumps({'output':str(OUT),'titles_reviewed':len(seen),'titles_rewritten':sum(c['kind']=='二级及以下标题' for c in changes),'abstract_paragraphs_rewritten':1,'requested_passages_rewritten':2,'all_other_content_unchanged':True},ensure_ascii=False))
