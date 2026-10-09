from pathlib import Path
import json
import fitz
from PIL import Image, ImageDraw, ImageFont

root = Path(__file__).resolve().parents[2]
work = Path(__file__).resolve().parent
pdf_path = root / '排版完成_最终核对_五图重绘_20261005.pdf'
mapping = json.loads((work / '图与正文对应.json').read_text())
pdf = fitz.open(pdf_path)
texts = [''.join(page.get_text().split()) for page in pdf]
preview_dir = work / '排版预览'
preview_dir.mkdir(exist_ok=True)
excerpts = fitz.open()
records = []
for figure, info in mapping.items():
    caption = ''.join(info['caption'].split())
    pages = [i for i, text in enumerate(texts) if caption in text]
    assert len(pages) == 1, (figure, pages)
    index = pages[0]
    dest = preview_dir / f'图{figure}_PDF第{index+1}页.png'
    pdf[index].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).save(dest)
    excerpts.insert_pdf(pdf, from_page=index, to_page=index)
    records.append({'figure': figure, 'pdf_page': index + 1, 'render': str(dest)})
excerpts.save(work / '五图排版预览.pdf')
(work / '页面核对.json').write_text(json.dumps({'total_pages': len(pdf), 'figures': records}, ensure_ascii=False, indent=2))

# A contact sheet for navigation; the five delivery PNGs remain unmodified.
canvas = Image.new('RGB', (1920, 760), 'white')
draw = ImageDraw.Draw(canvas)
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 25)
for column, (figure, title) in enumerate([('4-1', 'CS-BAPR'), ('5-2', 'H2O+'), ('6-1', 'MC-WM')]):
    im = Image.open(work / '交付图片' / f'图{figure}.png').convert('RGB')
    im.thumbnail((620, 680), Image.Resampling.LANCZOS)
    x = column * 640 + (640 - im.width) // 2
    canvas.paste(im, (x, 55 + (680 - im.height) // 2))
    draw.text((column * 640 + 20, 16), title, fill='#153449', font=font)
canvas.save(work / '三种方法预览.png')
print(json.dumps({'total_pages': len(pdf), 'figures': records}, ensure_ascii=False, indent=2))
