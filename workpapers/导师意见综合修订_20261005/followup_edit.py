"""Apply the independently verified follow-up corrections to the saved Word."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from copy import deepcopy
from lxml import etree as E
import json,shutil,re

HERE=Path(__file__).resolve().parent
TARGET=HERE.parents[1]/'output/排版完成_导师意见复核_20261005.docx'
BACKUP=HERE/'再次复核_修改前.docx'
if not BACKUP.exists():shutil.copy2(TARGET,BACKUP)
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
w=lambda k:'{'+ns['w']+'}'+k
with ZipFile(BACKUP) as z:package={n:z.read(n) for n in z.namelist()}
doc=E.fromstring(package['word/document.xml']);body=doc.find('w:body',ns)
paras=body.findall('w:p',ns);tables=body.findall('w:tbl',ns);changes=[]
def visible(p):return ''.join(p.xpath('.//w:t/text()|.//m:t/text()',namespaces=ns))
def replace(p,old,new):
    nodes=p.xpath('.//w:t',namespaces=ns);s=''.join(t.text or '' for t in nodes)
    assert s.count(old)==1,(old,s[:100])
    start=s.index(old);end=start+len(old);offset=0
    for t in nodes:
        value=t.text or '';stop=offset+len(value)
        if offset<=start<stop:t.text=value[:start-offset]+new+(value[end-offset:] if end<=stop else '')
        elif start<offset<end:t.text=value[end-offset:] if end<=stop else ''
        offset=stop
    assert ''.join(t.text or '' for t in nodes)==s.replace(old,new)
def insert_after(i,text):
    p=E.Element(w('p'));pp=paras[i].find('w:pPr',ns)
    if pp is not None:p.append(deepcopy(pp))
    r=E.SubElement(p,w('r'));E.SubElement(r,w('t')).text=text
    paras[i].addnext(p);changes.append({'after_paragraph':i,'text':text})

replace(paras[465],'目标网络参数统一以上标负号标记','第3、4章目标网络参数以上标负号标记，第6章目标置信评价器采用其首次定义的横线记号')
changes.append({'paragraph':465,'reason':'总记号约定承认第6章目标置信评价器的横线定义'})
replace(paras[1150],'第3.4.9节沿用持续过程的同车续接','第3.4.8.8节的原始记录保留了广义时间合计和完成率，未保留候车与车内时间的独立积分，因此不报告该组的时间分解。第3.4.9节沿用持续过程的同车续接')
changes.append({'paragraph':1150,'reason':'在本章局限中说明无法从现存记录恢复时间分解'})
insert_after(1199,'式(4-5)突出最小值聚合与熵项。公交主配置对每个目标评论家还保留系数0.01的正参数范数偏移；该范数为全部权重和非输出层偏置的L1范数之和，并与聚合值一起乘以折扣及非终止掩码。评论家回归损失另加入系数0.1的当前批动作价值集成标准差均值。因而，式(4-5)所示聚合、固定目标偏移和集成分歧正则共同构成实际价值更新。')
for i in [2070,2074,2108]:
    links=paras[i].findall('w:hyperlink',ns);assert len(links)==1
    assert 'arxiv.org/abs/' in visible(links[0])
    assert not links[0].xpath('.//w:bookmarkStart|.//w:bookmarkEnd|.//m:oMath',namespaces=ns)
    paras[i].remove(links[0]);changes.append({'paragraph':i,'reason':'移除DOI尾接的旧预印本链接'})
for ti,ri,ci,old,new in [(209,1,5,'+0.054','+0.053'),(210,1,4,'2263.59','2263.60')]:
    cell=tables[ti].findall('w:tr',ns)[ri].findall('w:tc',ns)[ci]
    replace(cell,old,new);changes.append({'table':ti,'row':ri,'col':ci,'before':old,'after':new,'reason':'按未舍入实验汇总值统一舍入'})
for item in json.loads((HERE/'复核_精简措辞.json').read_text()):
    i=item['最终Word段落索引'];new=item['替换为'].replace('20个独立初始化的训练结果表明','20个独立初始化的策略评价结果表明')
    replace(paras[i],item['原文片段'],new);changes.append({'paragraph':i,'before':item['原文片段'],'after':new})

f=HERE/'再次复核_补充数据段落.json'
if f.exists():
    for item in json.loads(f.read_text()):insert_after(item['after'],item['text'])

# Preserve every existing native equation, image, bookmark and section.
old=E.fromstring(package['word/document.xml'])
for query in ['//m:oMath','//w:drawing','//w:sectPr','//w:bookmarkStart','//w:bookmarkEnd']:
    assert [E.tostring(x) for x in old.xpath(query,namespaces=ns)]==[E.tostring(x) for x in doc.xpath(query,namespaces=ns)],query
package['word/document.xml']=E.tostring(doc,xml_declaration=True,encoding='UTF-8',standalone=True)
with ZipFile(TARGET,'w',ZIP_DEFLATED) as z:
    for n,v in package.items():z.writestr(n,v)
(HERE/'再次复核_补修记录.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2))
print(json.dumps({'saved':str(TARGET),'changes':len(changes)},ensure_ascii=False))
