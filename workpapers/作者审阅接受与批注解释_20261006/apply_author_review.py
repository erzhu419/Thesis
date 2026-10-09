from pathlib import Path
from copy import deepcopy
from collections import Counter
from lxml import etree as E
import zipfile,json,re,difflib
ROOT=Path('/home/erzhu419/mine_code/Thesis')
WORK=ROOT/'workpapers/作者审阅接受与批注解释_20261006'
SRC=WORK/'作者审阅原稿.docx'
OUT=ROOT/'排版完成_接受作者修订_全文统一_20261006.docx'
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
W='{'+NS['w']+'}'
with zipfile.ZipFile(SRC) as z:
 infos=z.infolist(); data={i.filename:z.read(i.filename) for i in infos}
r=E.fromstring(data['word/document.xml']); body=r.find('w:body',NS)
revision_counts={n:len(r.xpath('//w:'+n,namespaces=NS)) for n in ['ins','del']}
for n in list(r.xpath('//w:del',namespaces=NS)):n.getparent().remove(n)
for n in list(r.xpath('//w:ins',namespaces=NS)):
 parent=n.getparent();i=parent.index(n)
 for c in list(n):parent.insert(i,c);i+=1
 parent.remove(n)
settings=E.fromstring(data['word/settings.xml'])
for n in settings.findall('w:trackRevisions',NS):settings.remove(n)
data['word/settings.xml']=E.tostring(settings,xml_declaration=True,encoding='UTF-8',standalone=True)
def xml():return E.tostring(r,xml_declaration=True,encoding='UTF-8',standalone=True)
def save(path):
 data['word/document.xml']=xml()
 with zipfile.ZipFile(path,'w') as z:
  for i in infos:z.writestr(i,data[i.filename])
save(WORK/'仅接受作者修订.docx')
original_nodes=list(body)
def wt(n):return ''.join(n.xpath('.//w:t/text()',namespaces=NS))
def st(n):
 q=n.find('w:pPr/w:pStyle',NS)
 return q.get(W+'val','') if q is not None else ''
protected=[n for i,n in enumerate(original_nodes) if 63<=i<=81 or st(n) in ['chapter','references','appendixtitle']]
protected_xml=[E.tostring(n) for n in protected]
math_before=Counter(E.tostring(n) for n in r.xpath('//m:oMath',namespaces=NS))
changes=[]
def edit_range(p,start,end,repl):
 nodes=p.xpath('.//w:t',namespaces=NS); pos=0; spans=[]
 for n in nodes:
  txt=n.text or '';spans.append((n,pos,pos+len(txt)));pos+=len(txt)
 if start==end:
  for n,a,b in spans:
   if a<=start<=b and (b>a or len(spans)==1):n.text=(n.text or '')[:start-a]+repl+(n.text or '')[start-a:];n.set('{http://www.w3.org/XML/1998/namespace}space','preserve');return
  raise ValueError(('insert',start,wt(p)))
 affected=[(n,a,b) for n,a,b in spans if b>start and a<end]
 assert affected,(start,end,wt(p))
 first=affected[0][0]
 for n,a,b in affected:
  txt=n.text or '';n.text=txt[:max(start-a,0)]+(repl if n is first else '')+txt[min(end-a,len(txt)):]
  n.set('{http://www.w3.org/XML/1998/namespace}space','preserve')
def settext(p,new):
 old=wt(p)
 if old==new:return
 assert p not in protected
 for tag,a,b,c,d in reversed(difflib.SequenceMatcher(None,old,new,autojunk=False).get_opcodes()):
  if tag!='equal':edit_range(p,a,b,new[c:d])
 assert wt(p)==new,(old,new,wt(p))
def replace(i,a,b):
 p=original_nodes[i];old=wt(p);assert a in old,(i,a)
 settext(p,old.replace(a,b));changes.append({'index':i,'before':old,'after':wt(p)})
def rewrite(i,new):
 p=original_nodes[i];assert not p.xpath('.//m:oMath',namespaces=NS),i
 old=wt(p);settext(p,new);changes.append({'index':i,'before':old,'after':new})
# Resolve the author's wording consistently, while retaining mathematical terms where defined.
rewrite(567,'第二，公交环境的随机性随时间变化时，如何利用历史信息调整驻站策略？ 相同星期和时段的客流强度、道路速度及其波动程度可能随月份缓慢变化，也可能受到活动与道路管制冲击。策略需要从近期运行记录中识别当前条件，并在历史样本不再代表当前环境时调整驻站决策。')
replace(558,'运行规律本身的变化','客流与路况的统计特征随时间变化')
replace(591,'运行规律变化中的适应','运行环境变化中的适应')
replace(614,'运行规律改变后','运行环境改变后')
rewrite(719,'2.2.8 随机性、环境变化与目标信息')
replace(1502,'随机波动与运行规律变化','随机波动与运行条件变化')
replace(2262,'运行规律变化后的','运行环境变化后的')
rewrite(1847,'6.1 问题：源域与目标域环境失配下的模型辅助驻站控制')
rewrite(1852,'6.2 模型：源域与目标域的状态转移关系及公交事件预测')
replace(576,'源目标转移规律不一致','源域与目标域的运行环境不一致')
replace(1835,'第3章早期走廊环境','第3章早期RE-SAC实验环境')
replace(1690,'直接父控制器','调整前的对照控制器')
for i in [1840,2251]:replace(i,'1350次配对评价','1350次组合评价')
replace(1699,'同一外生单元内，各控制器共享相同需求与交通实现。','每次组合评价对应一个完整运行回合；同一外生单元内，各控制器共享相同需求与交通实现，因此控制器间的差异按配对结果统计。')
replace(1799,'同更新预算下的完整训练方案','使用同一批目标离线数据、更新预算相同条件下的完整训练方案')
replace(1841,'五训练种子的同更新预算比较','使用同一批目标离线数据并保持更新预算相同的五训练种子比较')
replace(2252,'五种子同更新预算比较中','在使用同一批目标离线数据、更新预算相同的五种子比较中')
rewrite(578,'针对源仿真对车辆运行和乘客服务的预测不符合目标环境的问题，提出MC-WM[132]。方法利用未参与拟合的目标数据，比较在源预测上学习误差修正与直接学习目标预测两种方案，再按预测可信度确定模型样本参与价值更新的权重。公交扩展同时预测下一事件的车辆与站点类别、连续运行状态、事件间隔及乘客成本，使用预先采集的SUMO数据拟合模型。在此基础上，对置信更新作归一化，避免累计对数置信使模型样本权重过小，并检验这一调整能否改善乘客服务。')
rewrite(1846,'本章以作者提出的MC-WM[132]为基础，利用未参与拟合的目标数据，选择源模型加误差修正或直接目标建模，再将所选模型生成的一步预测样本用于价值学习。公交事件既含车辆与站点等类别信息，也含人数、时间和成本等数值信息，因此模型分别预测这些不同类型的量。针对累计对数置信可能使模型样本权重过小的问题，进一步引入归一化置信更新。受控动力学任务用于分析模型类别选择，固定SUMO离线数据下的公交实验用于评价模型辅助训练的服务效果和计算代价。')
replace(1869,'公交世界模型需要处理离散事件身份、连续运行状态与不等长物理时间。','公交世界模型需要同时预测离散事件身份、连续运行状态与不等长物理时间，本文将这种联合预测称为混合事件模型。“混合”指输出变量的类型不同：下一辆车属于哪条线路、位于哪一站是类别，候车人数是计数，车头时距、持续时间和乘客成本则是连续量。')
replace(1942,'公交实现将第6.2.3节的混合事件模型接入第5章的混合学习骨干。','公交实现将第6.2.3节的事件预测模型接入第5章的混合学习骨干。这里的“混合学习”指共同使用目标离线数据与源仿真交互数据，与模型同时预测不同类型的事件变量是两个概念。')
rewrite(1955,'置信备份是用当前预测的可靠程度与后续状态的预计可靠程度，更新模型样本权重的计算过程。原方法反复累积非正的对数置信：即使单步预测质量没有变差，累计值也可能越来越负，取指数后的权重便趋近于零，模型样本对价值学习的作用随之减弱。公交归一化配置用传播折扣的补因子缩放当前对数置信，再与后续置信加权相加，使权重尺度不再随累计长度被过度压低。具体尺度关系见第6.3.8节，训练作用及乘客服务效果见第6.4.3节。')
rewrite(2139,'这一尺度问题可用恒定置信的情形说明。若每一步的局部置信均为0.5、传播折扣为0.9且不终止，式(6-30)给出的原模型权重为0.5的10次方，约为0.001；归一化更新对应的权重则为0.5。模型预测质量没有改变，仅置信的累计方式就能使训练权重相差很大。这个例子解释权重衰减的原因，实际权重仍取决于各状态的预测误差与后续置信。原权重的数量级与图6-11中的末期记录接近。')
replace(2084,'正常标定、联合50%和75%源参数偏差下','正常标定及四类源参数同时偏离标定值的50%、75%联合偏差下')
replace(2084,'前六小节分析模型类别','联合偏差同时改变道路平均速度、乘客OD强度、额外路口遭遇密度和必要停站时间，具体设置见第6.4.3.5节。前六小节分析模型类别')
rewrite(2142,'新增比较沿用完整训练日程，取两个训练种子，各自复用50000次IQL初始化，再完成20000次策略与评论家更新。源交互保留同时到站批次，共25000至25002个事件；策略学习率、网络结构、离线数据、停站变换及完成保障均相同。源条件包括正常标定、联合50%失配和联合75%失配。“联合”表示同时改变道路平均速度、乘客OD强度、额外路口遭遇密度和必要停站时间四类参数。50%条件将速度设为标定值的0.5倍，OD强度与必要停站时间设为1.5倍，额外路口遭遇密度设为0.5；75%条件对应0.25倍、1.75倍和0.75。额外路口遭遇密度是直接设定的参数值，其余三项是相对标定值的倍率。50%和75%是这两组联合干预的名称，不表示所有变量都向同一方向变化，也不表示转移概率具有统一的百分比误差。')
rewrite(2174,'本节沿用第6.4.3.5节的正常标定、联合50%失配和联合75%失配三种源条件。两档联合失配均同时改变道路平均速度、乘客OD强度、额外路口遭遇密度和必要停站时间：50%条件下速度降至标定值的0.5倍，OD强度和必要停站时间增至1.5倍，额外路口遭遇密度设为0.5；75%条件对应0.25倍、1.75倍和0.75。上述设置只改变训练用简易仿真，目标SUMO路网和测试需求分布保持一致，因此比较的是不同源训练偏差下向同一目标任务迁移的表现。')
rewrite(2236,'本章针对源域与目标域的运行环境不一致问题，提出MC-WM。方法先固定源模型，再利用未参与拟合的目标数据，比较修正源预测与直接学习目标预测两种方案，并根据可信度使用一步预测样本。公交实现同时预测事件类别、连续状态、持续时间与乘客成本；归一化置信更新用于避免累计对数置信使模型样本权重过小。固定SUMO离线数据因而既用于策略训练，也用于学习和检验预测模型。')
replace(2256,'归一化置信备份保留未来置信传播，并以相应折扣的补因子缩放即时对数置信。','归一化置信更新对当前与后续对数置信加权求和，避免模型预测质量基本稳定时，累计置信仍使模型样本的训练权重过度减小。')
replace(2256,'正常标定、联合50%和75%源参数偏差下','正常标定及速度、OD强度、路口遭遇密度与停站时间同时变化的50%、75%联合偏差下')
rewrite(2266,'针对源模型预测不符合目标环境的问题，提出MC-WM，在保留源模型并学习误差修正与直接建立目标模型之间进行选择，再利用预测误差确定模型样本的训练权重。针对对数置信累积使模型作用减弱的问题，引入归一化置信更新，保持模型样本对价值学习的有效贡献；公交模型则共同预测下一事件的类别、人数、连续运行状态、持续时间和乘客成本。')
# Move the categorical representation and its native equation into the existing Chapter 3 method subsection.
oldh,newh=original_nodes[651],original_nodes[912]
rewrite(912,'3.2.3 类别特征、连续状态与共享策略')
# Preserve both existing bookmarks and the author's comment anchor.
starts={n.get(W+'id') for n in oldh.findall('w:bookmarkStart',NS)}
for n in list(oldh):
 if n.tag==W+'bookmarkStart' or (n.tag==W+'bookmarkEnd' and n.get(W+'id') in starts):
  oldh.remove(n);newh.append(n)
 elif n.tag==W+'bookmarkEnd':
  oldh.remove(n);original_nodes[657].insert(1,n)
for tag in ['commentRangeStart','commentRangeEnd']:
 for n in list(oldh.findall('w:'+tag,NS)):
  oldh.remove(n)
  if tag=='commentRangeStart':newh.insert(1,n)
  else:newh.append(n)
for n in list(oldh.findall('w:r',NS)):
 if n.find('w:commentReference',NS) is not None:oldh.remove(n);newh.append(n)
# The old section bookmark must end beside its moved content, not back in Chapter 2.
for ident in starts:
 ends=r.xpath('//w:bookmarkEnd[@w:id="'+ident+'"]',namespaces=NS)
 for n in ends:
  if n.getparent() is not newh:
   n.getparent().remove(n);original_nodes[655].append(n)
body.remove(oldh)
moved=[original_nodes[i] for i in range(652,656)]
for n in moved:body.remove(n)
insert_at=body.index(newh)+1
for n in moved:body.insert(insert_at,n);insert_at+=1
replace(655,'第3章讨论该组织方式对公交强化学习的作用。','以下进一步给出本章四类身份信息的嵌入形式及其对共享公交控制的作用。')
# Keep H2O+ observation sizes with the general interface information, not within RE-SAC.
n=original_nodes[656];body.remove(n);body.insert(body.index(original_nodes[658])+1,n)
replace(909,'具体信息来源见第2.2.3节。','输入采用后车最近保存的前向间隔，并使用下一站当前候车量与路段距离、速度参数构成的时间代理；缓存间隔可能滞后于当前决策。共用信息接口见第2.2.3节。')
rewrite(668,'早期RE-SAC的局部输入与上述多线路接口有所不同，具体车辆编码、类别嵌入及连续输入见第3.2.2至3.2.3节。各实验中的对照方法共用相应输入，跨章节的输入差异不归因于某一价值学习算法。')
# Add the requested synthesis to the conclusion, leaving both abstracts untouched.
p=E.Element(W+'p');p.append(deepcopy(original_nodes[2243].find('w:pPr',NS)))
rr=E.SubElement(p,W+'r');tt=E.SubElement(rr,W+'t');tt.text='上述研究围绕同一驻站控制任务逐步增加可利用的信息。类别共享策略与RE-SAC先利用仿真经验处理车辆差异和随机价值波动；BAPR及CS-BAPR进一步利用近期运行信息应对环境变化；H2O+在获得目标离线数据后，将其与简易仿真交互结合训练；MC-WM再利用同一类目标数据学习并选择预测模型。实验分别支持类别表征、训练稳定化、混合训练及模型辅助在所测条件下的作用，同时表明时距改善不等同于乘客时间改善，执行保障的收益也需与学习方法的收益分别识别。这些结果共同说明，从仿真到现实的驻站控制既需要利用目标环境信息，也需要根据实际服务目标检验训练与执行环节的贡献。'
body.insert(body.index(original_nodes[2258]),p)
changes.append({'index':'7.1末尾','before':'','after':tt.text})
# Renumber moved equation, chapter-local equation sequences, section references and cached fields.
number_map={'(2-8)':'(3-2)','（2-8）':'（3-2）'}
for c,start,last,delta in [(2,9,34,-1),(3,2,41,1)]:
 for j in range(start,last+1):
  number_map[f'({c}-{j})']=f'({c}-{j+delta})';number_map[f'（{c}-{j}）']=f'（{c}-{j+delta}）'
for j in range(3,11):number_map[f'2.2.{j}']=f'2.2.{j-1}'
# References to the removed general section now point to Chapter 3.
number_map['2.2.2']='3.2.3'
pat=re.compile('|'.join(sorted(map(re.escape,number_map),key=len,reverse=True))+r'(?!\d)')
number_changes=0
for top in list(body):
 if top in protected:continue
 for para in ([top] if top.tag==W+'p' else top.xpath('.//w:p',namespaces=NS)):
  if any('SEQ ' in v for v in para.xpath('.//w:instrText/text()',namespaces=NS)):continue
  old=wt(para);new=pat.sub(lambda m:number_map[m.group(0)],old)
  if new!=old:settext(para,new);number_changes+=1
original_nodes[654].xpath('.//w:tc[last()]//w:t',namespaces=NS)[0].text='(3-'
for n in original_nodes[654].xpath('.//w:instrText',namespaces=NS):
 if 'SEQ Eq2' in (n.text or ''):n.text=n.text.replace('SEQ Eq2','SEQ Eq3')
# Keep complete figures near their discussion, letting following prose fill the preceding page.
for caption,next_heading in [('图2-2 公交H2O+场景的SUMO路网与方向性服务','2.3.4 平台构建与源环境标定'),('图3-1 计划对齐奖励与指数时距均衡奖励的响应曲面及截面','3.2.8 风险评价、鲁棒目标与正则化学习'),('图3-3 早期双向走廊实验的小时需求与路段速度输入','3.4.2 类别共享策略的早期结果')]:
 cap=next(n for n in body if wt(n)==caption);fig=cap.getprevious();assert fig.xpath('.//w:drawing',namespaces=NS)
 anchor=next(n for n in body if wt(n)==next_heading)
 for n in [fig,cap]:body.remove(n);body.insert(body.index(anchor),n)
# The old heading's cached TOC entry is regenerated by native Word, and is not a new heading.
assert [E.tostring(n) for n in protected]==protected_xml,'Protected section changed'
assert Counter(E.tostring(n) for n in r.xpath('//m:oMath',namespaces=NS))==math_before,'Native math changed'
assert not r.xpath('//w:ins|//w:del',namespaces=NS)
assert len(r.xpath('//w:commentRangeStart',namespaces=NS))==5
save(OUT)
(WORK/'正文修改记录.json').write_text(json.dumps({'accepted':revision_counts,'edits':changes,'numbering_paragraphs':number_changes,'output':str(OUT)},ensure_ascii=False,indent=2))
print(json.dumps({'output':str(OUT),'accepted':revision_counts,'editorial_changes':len(changes),'numbering_paragraphs':number_changes,'native_math_unchanged':sum(math_before.values()),'comments':5,'protected_elements':len(protected)},ensure_ascii=False))
