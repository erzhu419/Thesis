"""Check the current Word export's TOC destinations, footers and page bounds."""
import json
from pathlib import Path
import re
import sys

import fitz

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'workpapers/模板迁移_20261002'))
from render_full_preview import inspect_page

pdf = fitz.open(sys.argv[1] if len(sys.argv) > 1 else HERE / 'Word原生终验.pdf')
toc, errors, footers = [], [], []

def page_footer(page):
    candidates = [line for block in page.get_text('dict')['blocks'] if block.get('type') == 0
                  for line in block['lines'] if line['bbox'][1] > page.rect.height-100
                  and re.fullmatch(r'\s*(?:[IVXLCDM]+|\d+)\s*',
                                   ''.join(v['text'] for v in line['spans']))]
    return ''.join(v['text'] for v in max(candidates, key=lambda x: x['bbox'][1])['spans']).strip() if candidates else ''

for page in pdf:
    geometry = inspect_page(page)
    for key in ('text_outside_page', 'body_blank_candidate'):
        if geometry[key]:
            errors.append({'page': page.number + 1, 'failure': key})
    if any(v['outside_page'] for v in geometry['images']):
        errors.append({'page': page.number + 1, 'failure': 'images_outside_page'})
    lines = [line for block in page.get_text('dict')['blocks'] if block.get('type') == 0
             for line in block['lines']]
    for link in page.get_links():
        if link.get('kind') != fitz.LINK_GOTO:
            continue
        if not 0 <= link.get('page', -1) < len(pdf):
            errors.append({'page': page.number + 1, 'failure': 'link_target', 'link': link})
            continue
        rect = fitz.Rect(link['from'])
        line = min(lines, key=lambda v: abs((v['bbox'][1] + v['bbox'][3])/2 - (rect.y0+rect.y1)/2))
        shown = ''.join(v['text'] for v in line['spans']).strip()
        pieces = re.split(r'\.{2,}', shown)
        if len(pieces) < 2 or not re.fullmatch(r'[IVXLCDM]+|\d+', pieces[-1].strip()):
            continue
        target = pdf[link['page']]
        footer = page_footer(target)
        expected = re.sub(r'\s+', '', pieces[0]).casefold()
        actual = re.sub(r'\s+', '', target.get_text()).casefold()
        toc.append({'from': page.number+1, 'to': target.number+1, 'text': shown,
                    'heading_matches': expected in actual,
                    'page_matches': pieces[-1].strip() == footer})
    footer = page_footer(page)
    footers.append(footer)
    if page.number >= 22 and footer != str(page.number-21):
        errors.append({'page': page.number+1, 'failure': 'body_page_sequence', 'shown': footer})
    if re.search(r'错误!|Error! Reference source|Bookmark not defined', page.get_text()):
        errors.append({'page': page.number+1, 'failure': 'field_error'})

result = {'pages': len(pdf), 'toc_entries': len(toc),
          'toc_errors': [v for v in toc if not v['heading_matches'] or not v['page_matches']],
          'errors': errors, 'front_page_numbers': footers[:22], 'toc': toc}
report = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / 'pdf_acceptance.json'
report.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({k: v for k, v in result.items() if k != 'toc'}, ensure_ascii=False))
assert len(toc) == 329 and not result['toc_errors'] and not errors
