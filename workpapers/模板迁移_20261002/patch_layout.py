"""Repair the two layout defects observed in the Word preview, on a copy."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json
from lxml import etree as E
from PIL import ImageFont

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'ref_format_ready.docx'
DEST = ROOT / 'layout_ready.docx'
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
NS = {'w': W[1:-1]}

def set_prop(parent, name, **attrs):
    node = parent.find(W + name)
    if node is None:
        node = E.SubElement(parent, W + name)
    node.attrib.clear()
    for key, value in attrs.items():
        node.set(W + key, str(value))
    return node

with ZipFile(SOURCE) as z:
    parts = {n: z.read(n) for n in z.namelist()}
font = ImageFont.truetype('/mnt/c/Windows/Fonts/times.ttf', 420)
school = '中南大学博士学位论文'
changes = []
for name, data in list(parts.items()):
    if not (name.startswith('word/header') and name.endswith('.xml')):
        continue
    root = E.fromstring(data)
    text = ''.join(root.xpath('//w:t/text()', namespaces=NS))
    if not text.startswith(school):
        continue
    title = text[len(school):]
    width = 105 + sum(10.5 if ord(c) > 255 else font.getlength(c) / 40 for c in title)
    two_lines = width + 10.5 > 415.3
    p = root.find(W + 'p')
    pr = p.find(W + 'pPr')
    for child in list(p):
        if child is not pr:
            p.remove(child)
    tabs = set_prop(pr, 'tabs')
    for child in list(tabs):
        tabs.remove(child)
    E.SubElement(tabs, W+'tab', {W+'val':'clear', W+'pos':'4153'})
    E.SubElement(tabs, W+'tab', {W+'val':'right', W+'pos':'8306'})
    set_prop(pr, 'jc', val='left')
    set_prop(pr, 'ind', left=0, right=0, firstLine=0, firstLineChars=0)
    set_prop(pr, 'spacing', before=0, after=0, line=240 if two_lines else 400, lineRule='exact')
    border = pr.find(W+'pBdr')
    set_prop(border, 'bottom', val='single', sz=6, space=3, color='auto')
    def run(value=None, control=None):
        r = E.SubElement(p, W+'r')
        rp = E.SubElement(r, W+'rPr')
        set_prop(rp, 'rFonts', ascii='Times New Roman', hAnsi='Times New Roman', eastAsia='宋体')
        set_prop(rp, 'sz', val=21)
        set_prop(rp, 'szCs', val=21)
        if value is not None:
            t = E.SubElement(r, W+'t')
            t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
            t.text = value
        if control:
            E.SubElement(r, W+control)
    run(school)
    if two_lines:
        run(control='br')
    run(control='tab')
    run(title)
    parts[name] = E.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
    changes.append({'part':name, 'title':title, 'lines':2 if two_lines else 1,
                    'font_pt':10.5, 'estimated_combined_width_pt':round(width,2),
                    'bottom_border_space_pt':3, 'clear_inherited_center_tab_twips':4153})

# Native PAGE fields keep template 9-pt TNR instead of the source theme's Cambria.
for name,data in list(parts.items()):
    if not (name.startswith('word/footer') and name.endswith('.xml')):
        continue
    root=E.fromstring(data)
    if 'PAGE' not in ''.join(root.xpath('//w:instrText/text()', namespaces=NS)):
        continue
    for r in root.findall('.//w:r', NS):
        rp=r.find(W+'rPr')
        if rp is None:
            rp=E.Element(W+'rPr');r.insert(0,rp)
        set_prop(rp,'rFonts',ascii='Times New Roman',hAnsi='Times New Roman',cs='Times New Roman',eastAsia='宋体')
        set_prop(rp,'sz',val=18)
        set_prop(rp,'szCs',val=18)
    parts[name]=E.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)

root = E.fromstring(parts['word/document.xml'])
paragraphs = root.findall('w:body/w:p', NS)
for i,p in enumerate(paragraphs):
    if ''.join(p.xpath('.//w:t/text()', namespaces=NS)) != '主要符号说明':
        continue
    close = paragraphs[i-1]
    assert not close.xpath('.//w:t/text()', namespaces=NS)
    assert close.xpath('.//w:fldChar[@w:fldCharType="end"]', namespaces=NS)
    pr = close.find(W+'pPr')
    set_prop(pr, 'spacing', before=0, after=0, line=20, lineRule='exact')
    set_prop(pr, 'keepNext', val=0)
    rp = set_prop(pr, 'rPr')
    set_prop(rp, 'sz', val=2)
    set_prop(rp, 'szCs', val=2)
    for r in close.findall(W+'r'):
        rp = r.find(W+'rPr')
        if rp is None:
            rp = E.Element(W+'rPr'); r.insert(0, rp)
        set_prop(rp, 'sz', val=2)
        set_prop(rp, 'szCs', val=2)
    break
else:
    raise RuntimeError('TOC closing paragraph not found')
parts['word/document.xml'] = E.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
with ZipFile(DEST, 'w', ZIP_DEFLATED) as z:
    for name,data in parts.items():
        z.writestr(name,data)
(ROOT/'layout_repairs.json').write_text(json.dumps({'output':str(DEST), 'headers':changes,
    'toc_closing_paragraph':'Retain field end; set empty paragraph to 1 pt to remove overflow blank page'},ensure_ascii=False,indent=2)+'\n')
print(DEST)
