from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
import json

work = Path(__file__).resolve().parent
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
q = lambda name: '{' + NS['w'] + '}' + name
def ensure(parent, name):
    child = parent.find(q(name))
    if child is None:
        child = E.SubElement(parent, q(name))
    return child
def set_prop(parent, name, value):
    ensure(parent, name).set(q('val'), value)
def ppr(p):
    pr = p.find(q('pPr'))
    if pr is None:
        pr = E.Element(q('pPr')); p.insert(0, pr)
    return pr
def style(p):
    s = p.find('w:pPr/w:pStyle', NS)
    return s.get(q('val')) if s is not None else ''
def normal_header(rpr):
    fonts = ensure(rpr, 'rFonts')
    for key in list(fonts.attrib):
        if 'Theme' in key or 'theme' in key: del fonts.attrib[key]
    for key, val in [('eastAsia','宋体'),('ascii','Times New Roman'),('hAnsi','Times New Roman'),('cs','Times New Roman')]:
        fonts.set(q(key), val)
    set_prop(rpr, 'b', '0'); set_prop(rpr, 'bCs', '0')

with ZipFile(work / '修改前.docx') as z:
    doc = E.fromstring(z.read('word/document.xml'))
    body = doc.find('w:body', NS)
    toc_title = next(p for p in body if style(p) == 'toctitle')
    symbol_title = next(p for p in body if style(p) == 'symboltitle')
    symbol_index = body.index(symbol_title)
    symbol_nodes = list(body)[symbol_index:symbol_index+4]
    for node in symbol_nodes:
        body.remove(node); body.insert(body.index(toc_title), node)

    # The TOC is permanently limited to the manuscript after the preliminary pages.
    first_chapter = next(p for p in body if style(p) == 'chapter')
    last_p = [p for p in body if p.tag == q('p')][-1]
    bid = str(max(map(int, doc.xpath('.//w:bookmarkStart/@w:id', namespaces=NS))) + 1)
    start = E.Element(q('bookmarkStart'), {q('id'):bid, q('name'):'ThesisMainText'})
    first_chapter.insert(1, start)
    last_p.append(E.Element(q('bookmarkEnd'), {q('id'):bid}))
    for instr in doc.findall('.//w:instrText', NS):
        if (instr.text or '').lstrip().startswith('TOC '):
            instr.text = (instr.text or '').rstrip() + ' \\b ThesisMainText '

    tables = []
    for index, table in enumerate(doc.findall('.//w:tbl', NS)):
        if table.find('.//w:tblHeader', NS) is None: continue
        rows = table.findall('w:tr', NS)
        for p in rows[0].findall('.//w:p', NS):
            normal_header(ensure(ppr(p), 'rPr'))
        for run in rows[0].findall('.//w:r', NS):
            rp = run.find(q('rPr'))
            if rp is None: rp = E.Element(q('rPr')); run.insert(0, rp)
            normal_header(rp)
        long_table = len(rows) > 40
        for ri, row in enumerate(rows):
            set_prop(ensure(row, 'trPr'), 'cantSplit', '1')
            for p in row.findall('.//w:p', NS):
                keep = ri == 0 if long_table else ri < len(rows)-1
                set_prop(ppr(p), 'keepNext', '1' if keep else '0')
        tables.append({'index':index+1,'rows':len(rows),'long_table':long_table})

    for p in body:
        st = style(p)
        if st == 'figure':
            set_prop(ppr(p), 'keepNext', '1'); set_prop(ppr(p), 'keepLines', '1')
        elif st in ('figurecaption','tablecaption'):
            set_prop(ppr(p), 'keepNext', '1' if st == 'tablecaption' else '0')
            set_prop(ppr(p), 'keepLines', '1')
            # Existing regular KaiTi captions are already non-bold; remove direct bold overrides.
            for rp in p.findall('.//w:rPr', NS):
                set_prop(rp, 'b', '0'); set_prop(rp, 'bCs', '0')
                fonts = rp.find('w:rFonts', NS)
                if fonts is not None and fonts.get(q('eastAsia')) in ('黑体','SimHei'):
                    fonts.set(q('eastAsia'), '宋体')
    target = work / '候选01.docx'
    with ZipFile(target, 'w') as out:
        for member in z.infolist():
            data = E.tostring(doc, xml_declaration=True, encoding='UTF-8', standalone=True) if member.filename == 'word/document.xml' else z.read(member.filename)
            out.writestr(member, data)
(work / '首次排版修改.json').write_text(json.dumps({'tables':tables,'symbols_moved_before_toc':True,'toc_bookmark':'ThesisMainText'},ensure_ascii=False,indent=2))
print(target)
