"""Keep physical-time integration distinct from holding and ride-time symbols."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree as E
from extract_word import NS,content
from revise_word import q,mr,run

root=Path(__file__).resolve().parents[2]
base=root/'output/排版完成.docx'
with ZipFile(base) as z: parts={n:z.read(n) for n in z.namelist()}
doc=E.fromstring(parts['word/document.xml'])
body=doc.find('w:body',NS)
for block in body:
    text=content(block)
    if '(2-16)' in text or '(5-5)' in text:
        for t in block.xpath('.//m:t',namespaces=NS):
            if t.text=='u': t.text='τ'
            elif t.text=='du': t.text='dτ'
        if '(2-16)' in text:
            from revise_word import sub
            for t in list(block.xpath('.//m:t',namespaces=NS)):
                if t.text=='T':
                    r=t.getparent();r.getparent().replace(r,sub('T','hor'))
    if '(3-21)' in text:
        for t in block.xpath('.//m:t',namespaces=NS):
            if t.text=='u': t.text='a'
            elif t.text=='du': t.text='da'
    if text.startswith('在配分函数有限时'):
        pr=block.find('w:pPr',NS)
        pos=block.index(pr)+1
        math=E.Element(q('m:oMath'));math.append(mr('A'))
        for n in reversed([run('上述积分在可行动作空间'),math,run('上进行。')]): block.insert(pos,n)
    if text.startswith('实验采用经SUMO场景校准的多线路公交事件仿真环境'):
        for t in block.xpath('.//w:t',namespaces=NS):
            if t.text: t.text=t.text.replace('12条城市线路','6条线路的12条方向性服务')
    # Only the horizon in the superscript changes; the ride-time variable stays T^ride.
    for s in list(block.xpath('.//m:sSub',namespaces=NS)):
        if content(s)=='T_{hor}':
            parent=s.getparent()
            if parent.tag==q('m:e') and parent.getparent().tag in [q('m:sSup'),q('m:sSubSup')]:
                outer=parent.getparent()
                sup=outer.find('m:sup',NS)
                if sup is not None and content(sup)=='ride': parent.replace(s,mr('T'))
parts['word/document.xml']=E.tostring(doc,xml_declaration=True,encoding='UTF-8',standalone=True)
stage=Path(__file__).resolve().parent/'排版完成_积分收尾.docx'
with ZipFile(stage,'w',ZIP_DEFLATED) as z:
    for name,value in parts.items():z.writestr(name,value)
import shutil
shutil.copy2(stage,base)
print('Cost integrals, action integration, ride-time notation, and directional service count corrected.')
