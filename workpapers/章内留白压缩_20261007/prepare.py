"""Freeze the latest Word and list the existing figure/table bookmarks."""
import json
import re
import shutil
from pathlib import Path
from xml.etree import ElementTree as E
from zipfile import ZipFile

ROOT = Path('/home/erzhu419/mine_code/Thesis')
WORK = Path(__file__).resolve().parent
SOURCE = ROOT / 'output/排版完成_六图替换_20261007.docx'
OUTPUT = ROOT / 'output/排版完成_章内留白压缩_20261007.docx'
shutil.copy2(SOURCE, WORK / '修改前.docx')
shutil.copy2(SOURCE, OUTPUT)
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
with ZipFile(SOURCE) as z:
    d = E.fromstring(z.read('word/document.xml'))
    styles = E.fromstring(z.read('word/styles.xml'))
names = {e.get(W + 'name') for e in d.iter(W + 'bookmarkStart')}
targets = {'Figures': [], 'Tables': [], 'ChapterTitles': [],
           'Styles': {s.get(W + 'styleId'): s.find(W + 'name').get(W + 'val')
                      for s in styles.findall(W + 'style') if s.find(W + 'name') is not None}}
for p in d.find(W + 'body'):
    s = p.find(W + 'pPr/' + W + 'pStyle')
    style = s.get(W + 'val') if s is not None else ''
    text = ''.join(t.text or '' for t in p.iter(W + 't'))
    if style in ['figurecaption', 'tablecaption']:
        number = re.match(r'^[图表]([\dA-F]+-\d+)\s', text).group(1)
        name = ('fig' if style == 'figurecaption' else 'tbl') + number.replace('-', '_')
        targets['Figures' if style == 'figurecaption' else 'Tables'].append({'Bookmark': name, 'Label': text.split()[0], 'Caption': text})
    if style in ['chapter', 'bibliographytitle']:
        targets['ChapterTitles'].append(text)
assert len(targets['Figures']) == 63 and len(targets['Tables']) == 61
(WORK / '目标.json').write_text(json.dumps(targets, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'source': str(SOURCE), 'output': str(OUTPUT), 'figures': 63, 'tables': 61}, ensure_ascii=False))
