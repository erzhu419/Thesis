"""Read native Word blocks before applying the supplied chapter supplement."""
from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
M = '{http://schemas.openxmlformats.org/officeDocument/2006/math}'
NS = {'w': W[1:-1], 'm': M[1:-1]}
SOURCE = ROOT / 'output/排版完成_章节重构_20261012.docx'
PLAN = ROOT / 'markdown/章节重构复核与补充替换稿_20261012.docx'

def text(p):
    return ''.join(p.xpath('.//w:t/text() | .//m:t/text()', namespaces=NS))

def read(path):
    raw = ZipFile(path).read('word/document.xml')
    root = E.fromstring(raw)
    return raw, root, list(root.find(W + 'body'))

content = text
load = read

def style(p):
    node = p.find(W + 'pPr/' + W + 'pStyle')
    return node.get(W + 'val') if node is not None else E.QName(p).localname

if __name__ == '__main__':
    for path, name in [(SOURCE, '现稿'), (PLAN, '补充稿')]:
        _, root, blocks = read(path)
        rows = [{'i': i, 'style': style(p), 'text': text(p),
                 'math': bool(p.xpath('.//m:oMath', namespaces=NS))}
                for i, p in enumerate(blocks)]
        (HERE / (name + '.txt')).write_text('\n'.join(
            f"[{r['i']}] [{r['style']}] {r['text']}" for r in rows))
        (HERE / (name + '.json')).write_text(json.dumps(rows, ensure_ascii=False, indent=2))
        print(name, len(rows), 'blocks')
