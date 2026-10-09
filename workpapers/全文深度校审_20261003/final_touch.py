"""Fix remaining native-Word typography and discrete indices after pagination."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
from zipfile import ZipFile,ZIP_DEFLATED

from lxml import etree as E
from extract_word import NS,content
from revise_word import q,mr,sub

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
base=ROOT/'output/排版完成.docx'
backup=HERE/'排版完成_原生分页后.docx'
if not backup.exists(): shutil.copy2(base,backup)
with ZipFile(base) as z:
    parts={n:z.read(n) for n in z.namelist()}
doc=E.fromstring(parts['word/document.xml'])
body=doc.find('w:body',NS)
blocks=list(body)
assert len(blocks)==3164
paragraphs=doc.findall('.//w:p',NS)
failures=json.loads((HERE/'format_native.json').read_text())['failures']
changed=set()
for failure in failures:
    if failure['category']!='table': continue
    p=paragraphs[failure['p_index']]
    assert ''.join(p.xpath('.//w:t/text()',namespaces=NS))[:75]==failure['text']
    for r in p.findall('.//w:r',NS):
        pr=r.find('w:rPr',NS)
        if pr is None:
            pr=E.Element(q('w:rPr'));r.insert(0,pr)
        tag=q('w:sz' if failure['property']=='run.size' else 'w:b')
        n=pr.find(tag)
        if n is None: n=E.SubElement(pr,tag)
        n.set(q('w:val'),'21' if failure['property']=='run.size' else '0')
    changed.add(failure['p_index'])

# Source-residual model predictions must use the x_k defined in this section.
for i,old,new in [(1781,'t','k'),(1188,'t','n_stat'),(1189,'t','n_stat')]:
    for t in list(blocks[i].xpath('.//m:t',namespaces=NS)):
        if t.text==old:
            if '_' in new:
                r=t.getparent();r.getparent().replace(r,sub(*new.split('_',1)))
            else: t.text=new

def append_text(i,text):
    r=E.SubElement(blocks[i],q('w:r'));E.SubElement(r,q('w:t')).text=text

append_text(1188,'，其中统计序号按相邻诊断窗口计，而非物理时刻：')
append_text(1221,'符号参考如何约束完整策略，还取决于拟合导数误差及跨环境共享关系：附录D.8用三角不等式连接两层误差，附录D.9将共享关系的估计误差与环境特有部分分开，附录D.10再给出已知基准泛化误差时的传播界。三者为符号参考适配提供条件分析，与本章具体公交结果分别对应。')
# Definitions are already present in E.6; make its role explicit at first use.
append_text(1874,'其中静态排序使用给定高斯误差模型的密度比定义，附录E.6分别列出均匀拟合误差与有界拟合误差下的证明；它为后续研究状态相关置信提供分析起点。')

parts['word/document.xml']=E.tostring(doc,xml_declaration=True,encoding='UTF-8',standalone=True)
stage=HERE/'排版完成_最终校订.docx'
with ZipFile(stage,'w',ZIP_DEFLATED) as z:
    for name,value in parts.items(): z.writestr(name,value)
shutil.copy2(stage,base)
print('Typography corrected:',len(changed),'paragraphs; discrete-model indices and appendix connections updated.')
