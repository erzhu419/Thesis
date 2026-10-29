"""Render actual changed pages to detect clipping, broken fields and table pagination."""
from pathlib import Path
import argparse
import json
import re
import fitz
from PIL import Image, ImageDraw
from inventory import HERE

parser = argparse.ArgumentParser()
parser.add_argument('pdf')
args = parser.parse_args()
doc = fitz.open(args.pdf)
texts = [re.sub(r'\s+', '', p.get_text()) for p in doc]
needles = ['2.6公交仿真平台与场景资料', '2.7本章小结',
           '3.2.5共享策略与鲁棒价值模型', '4.2.3近期反馈与符号约束的策略价值模型',
           '5.2.4混合训练的策略与价值模型', '6.2.3模型修正与策略价值协同模型',
           '本章15维公交观测的组成', '多线路验证采用的15项运行观测',
           '公交批次状态的局部特征与归一化', '跨保真驻站控制中的主要动力学差异',
           '更新次数匹配的无世界模型训练与评价安排', '公交世界模型各类训练安排',
           '公交模型实验的重复与评价统计对象', '固定附加驻站下同车相邻决策的时间间隔',
           '大偏差项分别为', '不同运行日仍可能产生不同的候车队列',
           '大型活动、临时通行限制或持续拥堵', '两类环境可以使用相同的线路',
           '其生成的下一事件、车头时距和乘客时间', '局部近似', '主要软符号配置采用']
locations = {needle: [i+1 for i, value in enumerate(texts) if needle in value]
             for needle in needles}
pages = set()
for i, page in enumerate(doc):
    if 'PAGEREF' in page.get_text() or 'Error! Reference' in page.get_text() or '错误!未找到' in texts[i]:
        raise ValueError(f'field rendering error on {i+1}')
for needle, values in locations.items():
    for p in values:
        if p <= 18:
            continue
        pages.update([p, min(p+1, len(doc))])
for needle in needles[:6]:
    for p in locations[needle]:
        if p > 18:
            pages.update(range(p, min(p+4, len(doc)+1)))
pages.update([6, 7, 9, 14, 15, 16])
folder = HERE / ('页面预览_' + Path(args.pdf).stem)
folder.mkdir(exist_ok=True)
pages = sorted(pages)
for offset in range(0, len(pages), 2):
    pair = pages[offset:offset+2]
    views = []
    for page in pair:
        pix = doc[page-1].get_pixmap(matrix=fitz.Matrix(1.2, 1.2), alpha=False)
        views.append(Image.frombytes('RGB', [pix.width, pix.height], pix.samples))
    out = Image.new('RGB', [sum(im.width for im in views), max(im.height for im in views)+25], '#eee')
    drawer = ImageDraw.Draw(out)
    x = 0
    for page, im in zip(pair, views):
        out.paste(im, [x, 25])
        drawer.text([x+10, 5], f'PDF {page}', fill='black')
        x += im.width
    out.save(folder / ('Word_' + '_'.join(map(str, pair)) + '.png'))
(HERE / '分页定位.json').write_text(json.dumps(
    {'pdf': args.pdf, 'pages': len(doc), 'locations': locations, 'rendered_pages': pages},
    ensure_ascii=False, indent=2))
print(json.dumps({'pages': len(doc), 'locations': locations, 'rendered_pages': pages}, ensure_ascii=False))
