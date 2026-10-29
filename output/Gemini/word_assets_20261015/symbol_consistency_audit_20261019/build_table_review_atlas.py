"""Join the 104 completed native row screenshots without resizing source pixels."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import json
import sys

A = Path(__file__).resolve().parent
ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else A
D = ROOT / 'table_ui'
meta = json.loads((ROOT / 'native_table_review.json').read_text(encoding='utf-8-sig'))
assert len(meta['rows']) == 104 and not meta['failures']
assert {x['row'] for x in meta['rows']} == set(range(1, 105))
dest = D / 'atlas'
dest.mkdir(exist_ok=True)
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
index = []
for group, start in enumerate(range(1, 105, 18), 1):
    rows = list(range(start, min(105, start + 18)))
    images = [Image.open(D / f'row_{i:03d}.png').convert('RGB') for i in rows]
    sheet = Image.new('RGB', (max(im.width for im in images) + 76,
                             sum(im.height for im in images) + 8 * (len(rows) + 1)), '#dddddd')
    draw = ImageDraw.Draw(sheet)
    y, placements = 8, []
    for i, im in zip(rows, images):
        sheet.paste(im, (68, y))
        draw.text((6, y + 4), f'r{i:03d}', fill='black', font=font)
        placements.append({'row': i, 'file': f'row_{i:03d}.png', 'native_size': list(im.size),
                           'atlas_box': [68, y, 68 + im.width, y + im.height]})
        y += im.height + 8
    filename = f'table_group_{group:02d}.png'
    sheet.save(dest / filename)
    index.append({'file': filename, 'rows': rows, 'native_scale': 1, 'placements': placements})
(dest / 'index.json').write_text(json.dumps({'sheets': index, 'all_104_covered': True}, indent=2) + '\n')
print(json.dumps({'sheets': len(index), 'rows': 104, 'resized': False}))
