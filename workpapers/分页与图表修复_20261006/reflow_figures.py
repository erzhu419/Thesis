"""Move intact figure/caption units later to let following prose fill a short page."""
from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
import json, re, sys, math

source, report_path, target = map(Path, sys.argv[1:4])
ns = {'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm':'http://schemas.openxmlformats.org/officeDocument/2006/math',
      'wp':'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'}
q = lambda name: '{'+ns['w']+'}'+name
def text(node): return ''.join(node.xpath('.//w:t/text()|.//m:t/text()',namespaces=ns))
def style(node):
    s=node.find('w:pPr/w:pStyle',ns)
    return s.get(q('val')) if s is not None else ''
def height(node):
    if node.tag==q('tbl'):
        return 27 * len(node.findall('w:tr',ns))
    st=style(node)
    chars=sum(1 if ord(c)>255 else .49 for c in text(node))
    width=34.4 if st=='body' else 32
    lines=max(1,math.ceil((chars+(2 if st=='body' else 0))/width))
    return lines*20 + (18 if st in ['h2','h3','h4'] else 0)

report=json.loads(report_path.read_text())
moves=[]
with ZipFile(source) as z:
    doc=E.fromstring(z.read('word/document.xml'));body=doc.find('w:body',ns)
    for gap in report['gaps']:
        if gap['kind']!='章内': continue
        label=gap['next_caption']
        if not label or not label.startswith('图'): continue
        cap=next((p for p in body if style(p)=='figurecaption' and re.match(r'^'+re.escape(label)+r'(?!\d)',re.sub(r'\s+','',text(p)))),None)
        if cap is None: continue
        fig=cap.getprevious()
        if fig is None or style(fig)!='figure': continue
        following=list(body)[body.index(cap)+1:]
        inserted_after=None; estimate=0; count=0
        for node in following:
            st=style(node)
            if st in ['chapter','bibliographytitle','figure','tablecaption'] or node.tag==q('sectPr') or node.find('w:pPr/w:sectPr',ns) is not None:
                break
            if node.tag==q('tbl') and node.find('.//w:tblHeader',ns) is not None: break
            estimate+=height(node);inserted_after=node;count+=1
            if st in ['h2','h3','h4']:continue
            nxt=node.getnext()
            # Never insert a figure between a formula introduction and its equation.
            if nxt is not None and nxt.tag==q('tbl') and nxt.find('.//m:oMath',ns) is not None: continue
            if estimate>=gap['gap']-220:break
        if inserted_after is None:continue
        body.remove(fig);body.remove(cap)
        pos=body.index(inserted_after)+1;body.insert(pos,fig);body.insert(pos+1,cap)
        moves.append({'figure':label,'old_pdf_page':gap['pdf_page'],'following_blocks_brought_forward':count,'estimated_height':estimate,'after':text(inserted_after)[:110]})
    with ZipFile(target,'w') as out:
        for part in z.infolist():
            data=E.tostring(doc,encoding='UTF-8',xml_declaration=True,standalone=True) if part.filename=='word/document.xml' else z.read(part.filename)
            out.writestr(part,data)
target.with_suffix('.调整.json').write_text(json.dumps(moves,ensure_ascii=False,indent=2))
print(json.dumps(moves,ensure_ascii=False,indent=2))
