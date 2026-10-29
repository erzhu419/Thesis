from collections import Counter
from pathlib import Path
from zipfile import ZipFile
import json

import fitz
from lxml import etree as E

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'


def text(p):
    return ''.join(t.text or '' for t in p.iter(W + 't'))


def compact(e):
    if e is None:
        return None
    return {E.QName(c).localname: {E.QName(k).localname: v for k, v in c.attrib.items()}
            for c in e}


def inspect_docx(path):
    with ZipFile(path) as z:
        body = E.fromstring(z.read('word/document.xml')).find(W + 'body')
    heading = [i for i, p in enumerate(body) if text(p).strip() == '参考文献'][-1]
    refs = []
    for i in range(heading + 1, len(body)):
        p = body[i]
        if not text(p).startswith('['):
            break
        refs.append({'index': i, 'text': text(p), 'format': compact(p.find(W + 'pPr')),
                     'tabs': len(list(p.iter(W + 'tab'))),
                     'breaks': len(list(p.iter(W + 'br'))),
                     'run_formats': [compact(r) for r in p.findall('.//' + W + 'rPr')]})
    (WORK / (path.stem[:24] + '_参考文献.json')).write_text(
        json.dumps(refs, ensure_ascii=False, indent=2))
    print(path.name, 'references', len(refs), 'lengths', Counter(len(r['text']) // 50 for r in refs))
    print('paragraph_formats', Counter(json.dumps(r['format'], sort_keys=True) for r in refs))
    print('manual_breaks', sum(r['breaks'] for r in refs))
    return refs


def render(path, pages, prefix):
    doc = fitz.open(path)
    for n in pages:
        page = doc[n - 1]
        page.get_pixmap(matrix=fitz.Matrix(1.35, 1.35)).save(str(WORK / f'{prefix}_{n}.png'))
        lines = []
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines', []):
                spans = line['spans']
                lines.append({'bbox': line['bbox'], 'text': ''.join(s['text'] for s in spans),
                              'fonts': [(s['font'], s['size']) for s in spans]})
        (WORK / f'{prefix}_{n}.json').write_text(json.dumps(lines, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    inspect_docx(ROOT / 'output/排版完成_摘要与附录终校_20261011.docx')
    inspect_docx(next(ROOT.glob('中南大学博士学位论文_gty*.docx')))
    render(ROOT / 'output/排版完成_摘要与附录终校_20261011.pdf', [222, 225, 233], '原稿')
    render(next(ROOT.glob('不确定环境下多线路*pdf')), [190, 194, 200], '刘鹏杰')
