"""Locate changed equations, references and notation in the Word-native PDF."""
from pathlib import Path
import json
import re
import fitz
from PIL import Image, ImageDraw
from audit import HERE

path = Path('/mnt/c/Users/erzhu/AppData/Local/Temp/thesis_refs_20261012/preview.pdf')
doc = fitz.open(path)
pages = [re.sub(r'\s+', '', p.get_text()).replace('−','-') for p in doc]
needles = ['主要符号说明', '5.4.3服务水平对比', '(2-5)', '(2-6)', '(2-17)', '(3-52)',
           '(4-2)', '(4-19)', '(4-20)', '(4-21)', '图4-620倍', '图4-6',
           '(6-3)', '(6-14)', '(6-16)', '(6-36)', '(D-4c)', '(D-4e)', '表F-2']
locations = {s: [i+1 for i,t in enumerate(pages) if s in t] for s in needles}
errors = [i+1 for i,p in enumerate(doc) if any(s in p.get_text() for s in ['Error!','错误!','错误！未找到'])]
assert not errors, errors
definitions = json.loads((HERE / '编号与引用检查.json').read_text())
missing_equations = [key for key in definitions['equations'] if not any('('+key+')' in p for p in pages)]
missing_captions = [key for key in definitions['captions'] if not any(key in p for p in pages)]
assert not missing_equations, missing_equations
assert not missing_captions, missing_captions
chosen = {p for values in locations.values() for p in values if p>18}
chosen.update(range(9,19))
folder = HERE / '页面预览'
folder.mkdir(exist_ok=True)
selection = sorted(chosen)
for offset in range(0,len(selection),2):
    pair = selection[offset:offset+2]
    images = []
    for p in pair:
        pix = doc[p-1].get_pixmap(matrix=fitz.Matrix(1.2,1.2), alpha=False)
        images.append(Image.frombytes('RGB',(pix.width,pix.height),pix.samples))
    image = Image.new('RGB',(sum(i.width for i in images),max(i.height for i in images)+22),'#eee')
    draw = ImageDraw.Draw(image)
    x = 0
    for p,im in zip(pair,images):
        image.paste(im,(x,22)); draw.text((x+12,5),'PDF '+str(p),fill='black'); x += im.width
    image.save(folder / ('Word_'+'_'.join(map(str,pair))+'.png'))
result = {'pages': len(doc), 'rendered_pages': selection, 'locations': locations,
          'equation_labels_present': len(definitions['equations']),
          'caption_labels_present': len(definitions['captions']), 'field_errors': errors}
(HERE / '页面定位与标签验收.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=False))
