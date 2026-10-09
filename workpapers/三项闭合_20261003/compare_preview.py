"""Reuse prior full-page inspection only for pages with identical rendered pixels."""
import json
from pathlib import Path
import sys
import fitz
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
baseline = (Path(sys.argv[1]) if len(sys.argv) > 1 else
            ROOT / 'workpapers/最终验收与人工扩展证明_20261003/Word原生终验.pdf')
stem = sys.argv[2] if len(sys.argv) > 2 else 'visual_acceptance'
old = fitz.open(baseline)
new = fitz.open(ROOT / 'output/排版预览.pdf')
assert len(old) == len(new) == 317
changed = []
for i, (a, b) in enumerate(zip(old, new)):
    pa, pb = a.get_pixmap(), b.get_pixmap()
    if (pa.width, pa.height, pa.samples) != (pb.width, pb.height, pb.samples):
        changed.append(i + 1)
out = HERE / ('visual' if stem == 'visual_acceptance' else stem)
out.mkdir(exist_ok=True)
sheets = []
for start in range(0, len(changed), 4):
    pages = changed[start:start + 4]
    sheet = Image.new('RGB', (1240, 1760), 'white')
    draw = ImageDraw.Draw(sheet)
    for j, number in enumerate(pages):
        pix = new[number - 1].get_pixmap()
        im = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
        x, y = (j % 2) * 620, (j // 2) * 880
        draw.text((x + 12, y + 5), f'Page {number}', fill='black')
        sheet.paste(im, (x + 12, y + 28))
    name = out / f'changed_{start // 4 + 1:02}.png'
    sheet.save(name)
    sheets.append(str(name))
result = {'pages': len(new), 'changed_pages': changed, 'unchanged_pages': len(new) - len(changed),
          'baseline': str(baseline),
          'comparison': 'Direct equality of every 72-dpi rendered page; no file hashes.',
          'sheets': sheets, 'changed_pages_visually_reviewed': False}
(HERE / (stem + '.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False))
