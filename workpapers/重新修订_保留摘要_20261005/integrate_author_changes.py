"""Integrate substantive review points into the author's changed paragraphs."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree as E
from copy import deepcopy
import json

HERE = Path(__file__).resolve().parent
TARGET = Path('/mnt/d/Paper/Thesis/排版完成_重新修订_保留摘要_20261005.docx')
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
PID='{http://schemas.microsoft.com/office/word/2010/wordml}paraId'
with ZipFile(HERE/'用户指定底稿_修改前.docx') as z:
    original=E.fromstring(z.read('word/document.xml'))
base=original.find('w:body',NS).findall('w:p',NS)
with ZipFile(TARGET) as z:package={n:z.read(n) for n in z.namelist()}
doc=E.fromstring(package['word/document.xml']);before=deepcopy(doc)
byid={p.get(PID):p for p in doc.find('w:body',NS).findall('w:p',NS)}
changes=[]
def visible(p):return ''.join(p.xpath('.//w:t/text()|.//m:t/text()',namespaces=NS))
def edit(i,old,new):
    assert i>79
    p=byid[base[i].get(PID)];nodes=p.xpath('.//w:t',namespaces=NS)
    s=''.join(t.text or '' for t in nodes);assert s.count(old)==1,(i,old,s[:150])
    start=s.index(old);end=start+len(old);offset=0
    for t in nodes:
        value=t.text or '';stop=offset+len(value)
        if offset<=start<stop:t.text=value[:start-offset]+new+(value[end-offset:] if end<=stop else '')
        elif start<offset<end:t.text=value[end-offset:] if end<=stop else ''
        offset=stop
    assert ''.join(t.text or '' for t in nodes)==s.replace(old,new)
    changes.append({'author_paragraph':i,'old':old,'new':new})

edit(552,'其中，目标离线数据与源交互的联合使用已有H2O/H2O+方法基础，短模型滚动也不是本文首次提出。',
     'H2O/H2O+为目标离线数据与源交互的联合使用提供方法基础，短模型滚动则支持利用预测样本辅助策略更新。')
edit(554,'这些是具体的建模和实验差异，不意味着通用方法无法采用相同接口，也不把文献未报告的公交效果解释为其能力不足。',
     '这些公交任务中的建模与评价设置界定了本文在通用学习方法基础上的扩展。')
edit(578,'首先，从固定随机规律到未知运行变化。','首先，区分运行规律的变化与不确定性的来源。')
edit(578,'已知且均匀的骰子每次投掷仍有随机性，但其概率规律不随时间变化；若骰子的各面概率未知，平稳环境也存在认知不确定性。','给定运行规律下仍存在随机波动，对固定但未知的规律也可能存在认识不足。')
edit(845,'两者通过共同任务接口、标定及动作响应比较相联系。',
     '两者通过共同任务接口、标定及动作响应比较相联系。21个非零策略与需求组合中，广义时间归一化响应差的最大绝对95%区间端点为3.90%，完成率响应差的对应值为0.574个百分点。上述结果支持简易仿真在所测条件下承担策略训练，所学策略的服务效果则在SUMO中评价。')
edit(994,'本节保留研究初期的完整方案及集成规模比较。','本节比较研究初期的完整控制方案与不同集成规模。')
edit(1136,'原项目优化配方在多线路环境中的复核','原RE-SAC训练设置在多线路环境中的对照')
protocol=next(x['text'] for x in json.loads((HERE.parent/'导师意见综合修订_20261005/detail_rewrites.json').read_text()) if x['id']==1137)
protocol=protocol.replace('本节在多线路环境中比较RE-SAC及其参照配置，','为检验原RE-SAC关键训练设置在多线路环境中的表现，本节比较RE-SAC及其参照配置，')
edit(1137,visible(byid[base[1137].get(PID)]),protocol)
edit(1146,'原项目配方下','相应训练设置下')
edit(1156,'并显示中等规模RE-SAC集成在所考察公交场景中的优势。','集成规模的相对表现则随训练设置变化。')
edit(1341,'并扩展奖励和动作范围','并扩展至两种运营目标和六种需求条件')
edit(1362,'两种配置的服务成本接近。','该组未显示乘客时间改善。')
edit(1363,'同一sim2real问题','同一仿真到现实问题')
edit(1394,'奈奎斯特采样思想提示重建需要明确条件，但其带限信号与采样频率条件不能直接推广为任意概率分布的重建定理。','')
edit(1394,'；不以采样次数、保真度名称或反复复制的类比替代这一判断。','。')
edit(1672,'以及修正后的预测应以多大强度参与策略更新。','以及修正后的预测应以多大强度参与策略更新。本章据此分别研究模型类别选择与模型使用强度，并在共同执行条件下评价公交服务结果。')
edit(1814,'这些比较构成本章完整模型适应方法的主要性能结果。','这些比较构成受控动力学任务下完整模型适应方法的主要性能结果，公交驻站效用则由第6.4.3节在固定目标数据与共同执行条件下检验。')
edit(2011,'首轮180次评价','180次评价')
edit(2011,'稳定模型增益仍未分辨。','模型辅助相对无世界模型混合基线的稳定额外收益仍不确定。')
edit(2012,'额外修正源目标没有形成稳定增量。','额外源目标修正与上下文系数调节未形成一致的附加收益。')
edit(2013,'进一步的单一驻站多种子确认表明，以乘客时间训练、固定模型项系数0.25时，在正常标定、联合50%和75%源参数偏差下，相对共同保障的Daganzo式规则分别减少57.67、70.67和76.26秒人均广义时间。',
     '进一步的单一驻站比较包含20个独立初始化。以乘客时间训练、固定模型项系数0.25时，按测试情形等权汇总，在正常标定、联合50%和75%源参数偏差下，相对无世界模型的混合学习基线分别减少51.14、69.93和87.05秒人均广义时间；相对采用相同完成保障的Daganzo式规则分别减少57.67、70.67和76.26秒。')
edit(2013,'另以时距目标、模型项系数0.125复核两个独立初始化','另以时距目标、模型项系数0.125评价两个独立初始化')
# Accurate method expansion belongs in the body; the user's abstracts stay untouched.
edit(521,'作者的早期BAPR研究用环境识别与模式置信更新调节策略学习。',
     '作者提出贝叶斯遗忘分段鲁棒SAC（Bayesian Amnesic Piecewise-Robust SAC，BAPR），用环境识别与模式置信更新调节策略学习。')

for xp in ['//m:oMath','//w:drawing','//w:sectPr']:
    assert [E.tostring(e) for e in before.xpath(xp,namespaces=NS)]==[E.tostring(e) for e in doc.xpath(xp,namespaces=NS)]
assert [E.tostring(byid[p.get(PID)]) for p in base[61:80]]==[E.tostring(p) for p in base[61:80]]
package['word/document.xml']=E.tostring(doc,encoding='UTF-8',xml_declaration=True,standalone=True)
with ZipFile(TARGET,'w',ZIP_DEFLATED) as z:
    for n,v in package.items():z.writestr(n,v)
(HERE/'作者段落整合记录.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2))
print(json.dumps({'saved':str(TARGET),'integrated_author_edits':len(changes),'abstract_xml_identical':True},ensure_ascii=False))
