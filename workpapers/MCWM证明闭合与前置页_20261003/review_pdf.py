"""Inspect final pagination, discover TOC links, and render changed proof pages."""
import json
from pathlib import Path
import re
import sys
import fitz

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'workpapers/模板迁移_20261002'))
from render_full_preview import inspect_page

pdf = ROOT / 'output/排版完成_20261003_MCWM证明闭合与前置页_交付.pdf'
doc = fitz.open(pdf)
records, toc, changed_pages = [], [], []
for page in doc:
    data = inspect_page(page)
    data.pop('lines')
    records.append(data)
    lines = [l for b in page.get_text('dict')['blocks'] if b.get('type') == 0 for l in b['lines']]
    for link in page.get_links():
        if link.get('kind') != fitz.LINK_GOTO:
            continue
        if not 0 <= link.get('page', -1) < len(doc):
            raise AssertionError(link)
        rect = fitz.Rect(link['from'])
        line = min(lines, key=lambda l: abs((l['bbox'][1]+l['bbox'][3])/2-(rect.y0+rect.y1)/2))
        shown = ''.join(s['text'] for s in line['spans']).strip()
        pieces = re.split(r'\.{2,}', shown)
        if len(pieces) < 2 or not re.fullmatch(r'[IVXLCDM]+|\d+', pieces[-1].strip()):
            continue
        dest = doc[link['page']]
        footer = [l for b in dest.get_text('dict')['blocks'] if b.get('type') == 0
                  for l in b['lines'] if l['bbox'][1] > dest.rect.height-100 and
                  re.fullmatch(r'\s*(?:[IVXLCDM]+|\d+)\s*', ''.join(s['text'] for s in l['spans']))]
        printed = ''.join(s['text'] for s in max(footer, key=lambda l: l['bbox'][1])['spans']).strip() if footer else ''
        expected = re.sub(r'\s+', '', pieces[0])
        text = re.sub(r'\s+', '', dest.get_text())
        toc.append({'from': page.number+1, 'to': link['page']+1, 'shown': shown,
                    'heading_matches': expected.casefold() in text.casefold(),
                    'page_matches': pieces[-1].strip() == printed})
    if page.number < 3 or '固定策略下，先对动作概率求和' in page.get_text() or re.search(r'\(E-(?:1[18]|2[34569]|3[0269]|4[3-6])\)', page.get_text()):
        changed_pages.append(page.number+1)
        page.get_pixmap(matrix=fitz.Matrix(1.2, 1.2), alpha=False).save(HERE/f'page_{page.number+1:03d}.png')

abstract_page = next(p.number+1 for p in doc if '随机客流和路况变化使公交车头时距失衡' in p.get_text())
result = {'pdf': str(pdf), 'pages': len(doc), 'abstract_physical_page': abstract_page,
          'toc_entries': len(toc), 'toc_errors': [v for v in toc if not v['heading_matches'] or not v['page_matches']],
          'blank_pages': [v['physical_page'] for v in records if v['body_blank_candidate']],
          'text_outside_page': [v['physical_page'] for v in records if v['text_outside_page']],
          'images_outside_page': [v['physical_page'] for v in records if any(im['outside_page'] for im in v['images'])],
          'rendered_changed_pages': changed_pages, 'toc': toc}
(HERE/'pdf_review.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps({k: v for k, v in result.items() if k != 'toc'}, ensure_ascii=False))
assert '张异凡' in doc[0].get_text() and '郑亮' in doc[0].get_text()
assert not result['toc_errors'] and not result['text_outside_page'] and not result['images_outside_page']
