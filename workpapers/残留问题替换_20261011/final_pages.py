"""Render the final pages affected by spacing and replacement fixes."""
from pathlib import Path
import fitz
from PIL import Image, ImageDraw

here = Path(__file__).resolve().parent
pdf = Path('/mnt/c/Users/erzhu/AppData/Local/Temp/thesis_replacements_20261011/preview.pdf')
document = fitz.open(pdf)
pages = [11, 12, 13, 68, 69, 70, 114, 115, 218, 221]
for offset in range(0, len(pages), 2):
    selected = pages[offset:offset + 2]
    images = []
    for page in selected:
        pixmap = document[page - 1].get_pixmap(matrix=fitz.Matrix(1.3, 1.3), alpha=False)
        images.append(Image.frombytes('RGB', (pixmap.width, pixmap.height), pixmap.samples))
    canvas = Image.new('RGB', (sum(im.width for im in images), max(im.height for im in images) + 30), '#dddddd')
    draw = ImageDraw.Draw(canvas)
    x = 0
    for page, im in zip(selected, images):
        canvas.paste(im, (x, 30))
        draw.text((x + 10, 8), f'PDF {page}', fill='black')
        x += im.width
    canvas.save(here / '页面预览' / ('最终_' + '_'.join(map(str, selected)) + '.png'))
print({'pages': len(document), 'final_review_pages': pages})
