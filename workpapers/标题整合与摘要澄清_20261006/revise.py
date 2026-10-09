from pathlib import Path
from copy import deepcopy
from collections import Counter
from lxml import etree as E
import zipfile,json,re,difflib
from heading_plan import KEEP,APPENDIX
ROOT=Path('/home/erzhu419/mine_code/Thesis')
WORK=Path(__file__).parent
SRC=WORK/'修改前.docx'
OUT=ROOT/'排版完成_标题整合与摘要澄清_20261006.docx'
N={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
W='{'+N['w']+'}'; RN='{http://schemas.openxmlformats.org/package/2006/relationships}'
z=zipfile.ZipFile(SRC);infos=z.infolist();data={i.filename:z.read(i.filename) for i in infos}
r=E.fromstring(data['word/document.xml']);body=r.find('w:body',N);original=list(body)
def txt(p):return ''.join(p.xpath('.//w:t/text()',namespaces=N))
def style(p):
 e=p.find('w:pPr/w:pStyle',N);return e.get(W+'val','') if e is not None else ''
def edit_range(p,start,end,repl):
 spans=[];pos=0
 for n in p.xpath('.//w:t',namespaces=N):
  t=n.text or '';spans.append((n,pos,pos+len(t)));pos+=len(t)
 if start==end:
  for n,a,b in spans:
   if a<=start<=b and b>a:
    n.text=(n.text or '')[:start-a]+repl+(n.text or '')[start-a:];n.set('{http://www.w3.org/XML/1998/namespace}space','preserve');return
  raise ValueError(('insertion',start,txt(p)))
 affected=[(n,a,b) for n,a,b in spans if b>start and a<end];assert affected
 for j,(n,a,b) in enumerate(affected):
  t=n.text or '';n.text=t[:max(0,start-a)]+(repl if j==0 else '')+t[min(end-a,len(t)):]
  n.set('{http://www.w3.org/XML/1998/namespace}space','preserve')
changes=[]
def settext(p,new,reason):
 old=txt(p)
 if old==new:return
 for op,a,b,c,d in reversed(difflib.SequenceMatcher(None,old,new,autojunk=False).get_opcodes()):
  if op!='equal':edit_range(p,a,b,new[c:d])
 assert txt(p)==new
 changes.append({'reason':reason,'before':old,'after':new})
def replace(i,a,b,reason='段落衔接'):
 p=original[i];assert a in txt(p),(i,a);settext(p,txt(p).replace(a,b),reason)
# Only the two explicitly requested Chinese abstract passages are changed.
replace(69,'公交扩展采用固定目标离线数据与混合事件模型，并以归一化置信备份缓解模型作用衰减。',
'公交应用中，利用预先采集的固定SUMO运行数据，学习下一次到站事件的车辆与站点类别、候车与载客人数、车头时距、事件间隔及乘客时间成本，形成同时预测不同类型运行信息的混合事件模型。将模型生成的预测样本加入简易仿真中的策略训练，并按预测可信度确定其训练权重。为避免置信度反复累积后使模型样本权重过小，采用归一化置信更新，使可靠预测能够持续参与价值学习。','摘要指定句解释')
replace(69,'单一驻站动作、乘客时间目标下，在正常标定、联合50%和联合75%源参数偏差三种条件下，',
'将源仿真的道路平均速度、乘客OD强度、必要停站时间及额外路口遭遇密度同时调整，构造联合参数偏差。联合50%和联合75%分别将速度降低50%和75%、OD强度与必要停站时间提高50%和75%，并将额外路口遭遇密度设为0.5和0.75。单一驻站动作、乘客时间目标下，在正常标定及上述两档联合偏差条件下，','摘要联合偏差前置定义')
replace(69,'联合偏差同时改变道路平均速度、乘客OD、路口延误和基本停站服务时间等转移参数。','','摘要重复定义删除')
# Research content joins research objectives; the technical route begins afterwards.
KEEP.pop(569);KEEP[578]='1.4 技术路线'
for i,p in enumerate(original):
 if style(p)=='h2' and i>2440:
  num=txt(p).split()[0];KEEP[i]=num+' '+APPENDIX[num]
headings={i:p for i,p in enumerate(original) if style(p) in ('h2','h3','h4')}
sectionmap={};headinglog=[];current=None
for i,p in headings.items():
 old=txt(p);oldnum=old.split()[0]
 if i in KEEP:current=KEEP[i].split()[0]
 assert current
 sectionmap[oldnum]=current
 headinglog.append({'index':i,'before':old,'after':KEEP.get(i),'merged_into':current})
# Broad references spanning newly separated result groups remain broad.
sectionmap.update({'5.4.3':'5.4.3至5.4.4','6.4.3':'6.4.3至6.4.5','3.4.8':'3.4.5至3.4.6','1.4':'1.3','1.4.5':'1.4'})
# Keep existing chapter titles and settings byte-for-byte.
chapter_xml=[E.tostring(p) for p in original if style(p)=='chapter']
# Preserve bookmarks and comments while physically removing redundant headings.
for i,p in headings.items():
 if i in KEEP:
  settext(p,KEEP[i],'标题概括')
  s='h2' if KEEP[i].split()[0].count('.')==1 else 'h3'
  pp=p.find('w:pPr',N);pp.find('w:pStyle',N).set(W+'val',s)
  ol=pp.find('w:outlineLvl',N)
  if ol is not None:ol.set(W+'val','1' if s=='h2' else '2')
  # Direct size formatting must follow the retained heading level.
  if style(original[i])!=s:raise AssertionError('style assignment')
 else:
  assert not p.xpath('.//m:oMath|.//w:drawing|.//w:instrText|.//w:sectPr',namespaces=N),(i,txt(p))
  target=p.getnext()
  while target is not None and target.tag!=W+'p':target=target.getnext()
  assert target is not None
  insert=1 if target.find('w:pPr',N) is not None else 0
  for c in list(p):
   if c.tag in (W+'bookmarkStart',W+'bookmarkEnd',W+'commentRangeStart',W+'commentRangeEnd') or (c.tag==W+'r' and c.find('w:commentReference',N) is not None):
    p.remove(c);target.insert(insert,c);insert+=1
  body.remove(p)
# Merge the short conclusion and research-content fragments into coherent lists.
for group in ([571,573,575,577],[2243,2246,2248,2252],[2259,2261,2263,2265]):
 for j,i in enumerate(group,1):settext(original[i],f'（{j}）'+txt(original[i]),'合并后的研究内容或结论分项')
# Explicit transitions replace title fragments only where the next paragraph needs context.
for i,prefix in {1030:'在完成价值评价后，进一步更新驻站策略与探索程度。',1313:'驻站策略通过状态特征提取与动作输出两部分构建。',1438:'为进一步分析环境识别与动作补偿的关系，采用可明确控制执行器变化的连续控制任务。'}.items():
 settext(original[i],prefix+txt(original[i]),'合并后的内容衔接')
# Use section references only, never numeric experimental measurements or theorem numbers.
NUM=r'[1-7]\.\d+(?:\.\d+){0,2}'
REF=re.compile(r'第'+NUM+r'(?:[至及、和与到]第?'+NUM+r')*节')
NP=re.compile(NUM)
def mapref(m):
 raw=m.group(0);nums=NP.findall(raw)
 mapped=[sectionmap.get(v,v) for v in nums]
 if len(set(mapped))==1:return '第'+mapped[0]+'节'
 out=NP.sub(lambda q:sectionmap.get(q.group(),q.group()),raw)
 return out
for top in list(body):
 if top in headings.values() or top in original[:467]:continue
 for p in ([top] if top.tag==W+'p' else top.xpath('.//w:p',namespaces=N)):
  if style(p) in ('bibliography','chapter'):continue
  old=txt(p);new=REF.sub(mapref,old)
  new=re.sub(r'(?<![A-Za-z\d.])[A-F]\.\d+\.\d+(?![\d.])',lambda m:sectionmap.get(m.group(),m.group()),new)
  if old!=new:settext(p,new,'章节引用更新')
# Section introductions must describe the new organization, not obsolete subsection counts.
settext(original[1871],
'MC-WM算法[132]包含四个共同环节：划分目标数据并拟合候选模型；在留出数据上选择残差或直接模型；从已有经验状态出发生成一步预测并计算可信权重；把加权模型项加入价值更新，再由价值评价改进策略。源模型在拟合后冻结，目标模型的更新日程由允许获得的数据决定，部署时仍由演员网络输出动作。第6.3.1至6.3.2节给出允许目标在线交互的受控连续任务实现，第6.3.3节给出固定目标离线数据下的公交实现，第6.3.4节分析模型误差与置信尺度。','整合后的方法导读')
settext(original[1998],
'实验先在受控动力学任务中比较完整MC-WM与主要参照，再通过模型类别和组件比较分析预测选择的作用；随后在固定SUMO离线数据下评价公交驻站控制，依次考察模型辅助效果、置信权重调整及不同运行条件下的乘客服务。受控任务采用第6.3.1至6.3.2节的连续模型算法，公交任务采用第6.3.3节的事件模型与混合训练流程，各组按相应的数据条件和训练预算报告结果。','整合后的实验导读')
replace(2082,'前六小节分析模型类别','以下先分析模型类别') if '前六小节分析模型类别' in txt(original[2082]) else None
replace(1035,'第3.4.5至3.4.6及第3.4.6节','第3.4.5至3.4.6节','合并后的引用去重')
# Remove stale relative subsection pointers when both studies are now within one section.
for i,a,b in [(2096,'前一节','前述同预算比较'),(2115,'前节完整配置','上述完整训练配置'),(2126,'前节九组模型训练','上述九组模型训练'),(2160,'前节相同的十个评价场景','上述相同的十个评价场景'),(2160,'前节已经固定的策略','上述已经固定的策略')]:
 if a in txt(original[i]):replace(i,a,b)
# Add the new self-citation and its software source with native hyperlinks.
rels=E.fromstring(data['word/_rels/document.xml.rels'])
ridmax=max(int(e.get('Id')[3:]) for e in rels if e.get('Id','').startswith('rId'))
bidmax=max(int(v) for v in r.xpath('//w:bookmarkStart/@w:id',namespaces=N))
def text_run(text,sup=False):
 rr=E.Element(W+'r')
 if sup:
  pp=E.SubElement(rr,W+'rPr');E.SubElement(pp,W+'vertAlign',{W+'val':'superscript'})
 t=E.SubElement(rr,W+'t');t.text=text;t.set('{http://www.w3.org/XML/1998/namespace}space','preserve');return rr
def citation(p,needle,number,anchor):
 full=txt(p);pos=full.index(needle)+len(needle);nxt=0
 for n in p.xpath('.//w:t',namespaces=N):
  t=n.text or ''
  if nxt<pos<=nxt+len(t):
   rr=n.getparent();assert rr.tag==W+'r';parent=rr.getparent();assert parent is p
   suffix=t[pos-nxt:];n.text=t[:pos-nxt]
   h=E.Element(W+'hyperlink',{W+'anchor':anchor});h.append(text_run(f'[{number}]',True))
   ix=parent.index(rr)+1;parent.insert(ix,h)
   if suffix:
    tail=deepcopy(rr)
    for q in tail.xpath('.//w:t',namespaces=N):q.text=suffix
    parent.insert(ix+1,tail)
   return
  nxt+=len(t)
 raise AssertionError('citation position')
for i,needle in [(521,'CS-BAPR）'),(573,'CS-BAPR'),(1256,'CS-BAPR'),(1295,'CS-BAPR')]:citation(original[i],needle,158,'ref-zhang2026csbapr')
p=original[1256];p.append(text_run('相关代码实现见公开仓库'));h=E.SubElement(p,W+'hyperlink',{W+'anchor':'ref-csbapr-code'});h.append(text_run('[159]',True));p.append(text_run('。'))
def addbib(number,anchor,text,url):
 global ridmax,bidmax
 p=E.Element(W+'p');p.append(deepcopy(original[2439].find('w:pPr',N)))
 bidmax+=1;ident=str(bidmax);E.SubElement(p,W+'bookmarkStart',{W+'id':ident,W+'name':anchor})
 p.append(text_run(f'[{number}] '+text+' '))
 ridmax+=1;rid=f'rId{ridmax}'
 E.SubElement(rels,RN+'Relationship',Id=rid,Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink',Target=url,TargetMode='External')
 h=E.SubElement(p,W+'hyperlink',{'{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id':rid});h.append(text_run(url))
 p.append(text_run('.'));E.SubElement(p,W+'bookmarkEnd',{W+'id':ident})
 body.insert(body.index(original[2440]),p)
addbib(158,'ref-zhang2026csbapr','ZHANG Y, ZHENG L. Arithmetic Actor Heads and Training Stabilization for Out-of-Distribution Reinforcement Learning[EB/OL]. arXiv, 2026[2026-10-06].','https://arxiv.org/abs/2610.05143v1')
addbib(159,'ref-csbapr-code','ZHANG Y. CS-BAPR: source code[CP/OL]. GitHub[2026-10-06].','https://github.com/erzhu419/CS-BAPR')
# Word will regenerate its existing automatic contents field to three levels.
for n in r.xpath('//w:instrText',namespaces=N):
 if n.text and 'TOC ' in n.text:n.text=n.text.replace('"1-4"','"1-3"')
# Preserve first-level titles exactly; abstracts other than paragraph 69 exactly.
assert [E.tostring(p) for p in body if style(p)=='chapter']==chapter_xml
oldr=E.fromstring(z.read('word/document.xml'));oldb=oldr.find('w:body',N)
for i in range(63,82):
 if i!=69:assert E.tostring(original[i])==E.tostring(oldb[i]),('abstract',i)
assert not [p for p in body if style(p)=='h4']
assert Counter(E.tostring(m) for m in r.xpath('//m:oMath',namespaces=N))==Counter(E.tostring(m) for m in oldr.xpath('//m:oMath',namespaces=N))
assert len(r.xpath('//w:drawing',namespaces=N))==len(oldr.xpath('//w:drawing',namespaces=N))
data['word/document.xml']=E.tostring(r,xml_declaration=True,encoding='UTF-8',standalone=True)
data['word/_rels/document.xml.rels']=E.tostring(rels,xml_declaration=True,encoding='UTF-8',standalone=True)
with zipfile.ZipFile(OUT,'w') as out:
 for info in infos:out.writestr(info,data[info.filename])
(WORK/'修改记录.json').write_text(json.dumps({'edits':changes,'headings':headinglog,'sectionmap':sectionmap},ensure_ascii=False,indent=2))
(WORK/'新目录.txt').write_text('\n'.join(txt(p) for p in body if style(p) in ('chapter','h2','h3')))
print(json.dumps({'output':str(OUT),'old_headings':len(headings),'new_level2_3':len(KEEP),'math':len(r.xpath('//m:oMath',namespaces=N)),'figures':len(r.xpath('//w:drawing',namespaces=N)),'first_level_unchanged':True,'bibliography':159},ensure_ascii=False))
