from pathlib import Path
from zipfile import ZipFile
import json
from lxml import etree as E

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / 'output/排版完成_七章整合_文献核验_20261011.docx'
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math'}

with ZipFile(SOURCE) as z:
    body = E.fromstring(z.read('word/document.xml')).find('w:body', NS)
    records = []
    for i, node in enumerate(body):
        st = node.find('w:pPr/w:pStyle', NS)
        records.append(dict(index=i, tag=E.QName(node).localname,
                            style=st.get('{'+NS['w']+'}val') if st is not None else '',
                            text=''.join(node.xpath('.//w:t/text()|.//m:t/text()', namespaces=NS))))
    (HERE / '原稿块.json').write_text(json.dumps(records, ensure_ascii=False, indent=2))
    (HERE / '原稿附录.txt').write_text('\n\n'.join(f"{r['index']} [{r['style']}] {r['text']}" for r in records if r['index'] >= 2002))
    (HERE / '原稿关联正文.txt').write_text('\n\n'.join(f"{r['index']} [{r['style']}] {r['text']}" for r in records if r['index'] < 83 or 950 <= r['index'] <= 1030 or 1450 <= r['index'] <= 1570))
    (HERE / '页眉.txt').write_text('\n'.join(name+': '+''.join(E.fromstring(z.read(name)).xpath('.//w:t/text()',namespaces=NS)) for name in z.namelist() if name.startswith('word/header') and name.endswith('.xml')))
print(len(records))
