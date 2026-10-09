"""Arrange native Word page renders for local visual inspection only."""
import json
from pathlib import Path

from PIL import Image, ImageDraw

WORK = Path(__file__).resolve().parent
folder = WORK / 'Word页面复查'
record = json.loads((WORK / 'Word复查记录.json').read_text(encoding='utf-8-sig'))
pages = record['PreviewPages']
for i in range(0, len(pages), 4):
    selected = pages[i:i+4]
    sheet = Image.new('RGB', (1680, 2436), '#dddddd')
    draw = ImageDraw.Draw(sheet)
    for j, number in enumerate(selected):
        with Image.open(folder / f'page_{number:03d}.png') as page:
            page = page.convert('RGB').resize((830, 1174), Image.Resampling.LANCZOS)
            x, y = (j % 2) * 840, (j // 2) * 1218
            draw.text((x+12, y+8), f'Word page {number}', fill='black')
            sheet.paste(page, (x+5, y+30))
    target = folder / f'contact_{i//4+1:02d}.jpg'
    sheet.save(target, quality=94)
    print(target.name, selected)
