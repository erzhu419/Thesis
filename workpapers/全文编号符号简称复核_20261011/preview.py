"""Render affected notation pages and detect clipped text or broken PDF links."""
from pathlib import Path
import json

import fitz
from PIL import Image, ImageDraw

from scan import HERE
from repair import OUT


def main():
    pdf = fitz.open(OUT.with_suffix('.pdf'))
    pages = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
             39, 44, 50, 55,
             82, 111, 116, 123, 125, 178, 248, 249, 253, 268, 271,
             272, 276, 277, 282, 285, 145]
    destination = HERE / '页面预览'
    destination.mkdir(exist_ok=True)
    outside, bad_links = [], []
    for index, page in enumerate(pdf):
        for block in page.get_text('dict')['blocks']:
            if block['type'] != 0:
                continue
            for line in block['lines']:
                x0, y0, x1, y1 = line['bbox']
                if x0 < -1 or y0 < -1 or x1 > page.rect.width + 1 or y1 > page.rect.height + 1:
                    outside.append({'page': index + 1, 'bbox': line['bbox'], 'text': ''.join(s['text'] for s in line['spans'])})
        for link in page.get_links():
            if link['kind'] == fitz.LINK_GOTO and not 0 <= link.get('page', -1) < len(pdf):
                bad_links.append({'page': index + 1, 'link': link})
    for number in pages:
        pixmap = pdf[number - 1].get_pixmap(matrix=fitz.Matrix(1.15, 1.15))
        pixmap.save(destination / f'p{number:03d}.png')
    for index in range(0, len(pages), 4):
        canvas = Image.new('RGB', (1420, 2070), '#e5e5e5')
        draw = ImageDraw.Draw(canvas)
        for offset, number in enumerate(pages[index:index + 4]):
            image = Image.open(destination / f'p{number:03d}.png').convert('RGB')
            image.thumbnail((690, 994))
            x, y = (offset % 2) * 710 + 10, (offset // 2) * 1035 + 30
            draw.text((x, y - 20), f'PDF p.{number}', fill='black')
            canvas.paste(image, (x, y))
        canvas.save(destination / f'contact_{index // 4 + 1}.png')
    report = {'pages': len(pdf), 'rendered_for_visual_review': pages,
              'text_outside_page': outside, 'invalid_pdf_goto_links': bad_links}
    (HERE / 'PDF页面检查.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
