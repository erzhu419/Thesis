"""Check native objects, missing references, and render the changed layouts."""
from pathlib import Path
from zipfile import ZipFile
from collections import Counter
import json
from lxml import etree as E
import fitz

P = Path(__file__).resolve().parent
NS = {'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
def read(name):
    with ZipFile(P/name) as z:
        return E.fromstring(z.read('word/document.xml')), {n:z.read(n) for n in z.namelist() if n.startswith('word/media/')}
old, media0 = read('修改前.docx')
new, media1 = read('修订待分页.docx')
counts = {}
for label, path in [('equations','//m:oMath'),('drawings','//w:drawing'),('sections','//w:sectPr'),('bookmarks','//w:bookmarkStart')]:
    counts[label] = [len(old.xpath(path,namespaces=NS)),len(new.xpath(path,namespaces=NS))]
assert counts['equations'] == [2080,2086], counts
assert counts['drawings'][0] == counts['drawings'][1], counts
assert counts['sections'] == [16,16], counts
assert media0 == media1, 'A source image changed'
def geometry(root):
    return [[(E.QName(e).localname,dict(e.attrib)) for e in s if E.QName(e).localname in ('pgSz','pgMar','cols','pgNumType')]
            for s in root.xpath('//w:sectPr',namespaces=NS)]
assert geometry(old) == geometry(new), 'Section geometry changed'
text = ''.join(new.xpath('//w:t/text()|//m:t/text()',namespaces=NS))
for missing in ('错误!未定义书签','错误！未定义书签','Error! Reference source not found','Error! Bookmark not defined'):
    assert missing not in text, missing
pdf = fitz.open(P/'修订预览.pdf')
texts = [page.get_text() for page in pdf]
for missing in ('错误!未定义书签','错误！未定义书签','Error! Reference source not found','Error! Bookmark not defined'):
    assert all(missing not in t for t in texts), missing
queries = ['表6-16','B-25','D-58','对角执行器增益','运行长度索引','F.4 补充实验','3.4.9','4.4.7','6.4.3.9']
hits = {q:[i+1 for i,t in enumerate(texts) if q.replace(' ','') in t.replace(' ','').replace('\n','')] for q in queries}
report = {'counts':counts,'images_unchanged':True,'section_geometry_unchanged':True,'pages':len(pdf),'target_pages':hits}
(P/'render_check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False,indent=2))
for q in ['表6-16','B-25','D-58','对角执行器增益','运行长度索引','F.4 补充实验']:
    for page_num in hits[q][-2:]:
        pdf[page_num-1].get_pixmap(matrix=fitz.Matrix(1.25,1.25)).save(P/f'preview_{page_num}.png')
