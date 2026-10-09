from pathlib import Path
import json
import re
import zipfile

import fitz
from lxml import etree

ROOT = Path('/home/erzhu419/mine_code')
HERE = Path(__file__).resolve().parent
PAPERS = {
    'RE-SAC': ROOT / 'RE-SAC/paper/paper.pdf',
    'BAPR': ROOT / 'BAPR/paper/paper.pdf',
    'CS-BAPR': ROOT / 'CS-BAPR/paper/paper.pdf',
    'H2Oplus': ROOT / 'sumo-rl/H2Oplus/paper/main.pdf',
    'MC-WM': ROOT / 'MC-WM/paper/main.pdf',
    'Categorical': ROOT / 'Catetorical-Feature-SAC-in-bus-simulation/Single Agent Robust Deep Reinforcement Learning for Bus Fleet Control.pdf',
    'TransitDuet': ROOT / 'TransitDuet/paper/main.pdf',
    'offline-sumo': ROOT / 'offline-sumo/paper/main.pdf',
}
for key, path in PAPERS.items():
    with fitz.open(path) as pdf:
        text = '\n\n'.join(f'=== PAGE {i + 1} ===\n' + page.get_text() for i, page in enumerate(pdf))
        (HERE / f'{key}.txt').write_text(text, encoding='utf-8')
        print(key, len(pdf), pdf[0].get_text()[:200].replace('\n', ' '))

docx = ROOT / 'Thesis/output/排版完成_参考文献复核_20261007.docx'
ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math'}
with zipfile.ZipFile(docx) as archive:
    document = etree.fromstring(archive.read('word/document.xml'))
equations = {}
for paragraph in document.xpath('//w:body//w:p', namespaces=ns):
    maths = paragraph.xpath('.//m:oMath[not(ancestor::m:oMath)]', namespaces=ns)
    words = ''.join(paragraph.xpath('.//w:t/text()', namespaces=ns))
    number = re.search(r'\(([2-6]-\d+)\)', words)
    if number and maths:
        equations[number.group(1)] = [etree.tostring(m, encoding='unicode') for m in maths]
(HERE / 'equations.json').write_text(json.dumps(equations, ensure_ascii=False, indent=2), encoding='utf-8')
print('Numbered equations extracted:', len(equations))
