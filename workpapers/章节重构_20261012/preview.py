"""Locate the changed sections and render their actual native Word pagination."""
from pathlib import Path
import json
import re
import fitz
from PIL import Image,ImageDraw
from inventory import HERE

PDF=Path('/mnt/c/Users/erzhu/AppData/Local/Temp/thesis_restructure_20261012/preview_deliver.pdf')
if not PDF.exists():PDF=PDF.with_name('preview_v2.pdf')
doc=fitz.open(PDF)
texts=[re.sub(r'\s+','',p.get_text()) for p in doc]
needles=['第2章公交控制与强化学习基础','第3章平稳随机环境',
         '本章保持线路、车辆供给和发车计划不变','车辆执行一次班次时','乘客时间目标用于检查间隔调节',
         '本章研究运行规律变化时的公交驻站控制','本章分别采用车头时距目标',
         '本章分别设置多线路离线学习任务','状态同时描述当前车辆',
         '本章多线路场景采用刘鹏杰提供','不同线路方向的站间道路运行时间分布',
         '源目标响应差','正式控制器的训练与评价安排','初步因素比较的训练与评价安排',
         '更新次数匹配比较的训练与评价安排','公交任务使用固定的SUMO目标域离线记录',
         '公交模型辅助学习以固定目标数据','公交策略同时表示不额外驻站的概率',
         '公交模型辅助比较的训练与评价安排','该补充比较重新以计划间隔奖励']
locations={s:[i+1 for i,t in enumerate(texts) if s in t] for s in needles}


def chapter_page(prefix):
    for i,page in enumerate(doc):
        if i<18:continue
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines',[]):
                spans=line['spans']
                value=re.sub(r'\s+','',''.join(s['text'] for s in spans))
                if value.startswith(prefix) and max(s['size'] for s in spans)>15:
                    return i+1
    raise ValueError(prefix)


ch2=chapter_page(needles[0]);ch3=chapter_page(needles[1])
locations[needles[0]]=[ch2];locations[needles[1]]=[ch3]
pages=set(range(ch2,ch3))
for needle,values in locations.items():
    for p in values:
        if p>20:pages.update([p,min(p+1,len(doc))])
pages.update([9,10])
pages=sorted(pages)
folder=HERE/'交付版页面预览';folder.mkdir(exist_ok=True)
for offset in range(0,len(pages),2):
    selected=pages[offset:offset+2];images=[]
    for p in selected:
        pix=doc[p-1].get_pixmap(matrix=fitz.Matrix(1.3,1.3),alpha=False)
        images.append(Image.frombytes('RGB',(pix.width,pix.height),pix.samples))
    canvas=Image.new('RGB',(sum(im.width for im in images),max(im.height for im in images)+26),'#eee')
    draw=ImageDraw.Draw(canvas);x=0
    for p,im in zip(selected,images):
        canvas.paste(im,(x,26));draw.text((x+8,6),f'PDF {p}',fill='black');x+=im.width
    canvas.save(folder/('Word_'+'_'.join(map(str,selected))+'.png'))
(HERE/'页面定位.json').write_text(json.dumps({'preview':str(PDF),'pages':len(doc),'chapter_2_pages':[ch2,ch3-1],'locations':locations,'review_pages':pages},ensure_ascii=False,indent=2))
print(json.dumps({'pages':len(doc),'chapter_2_pages':[ch2,ch3-1],'review_pages':pages},ensure_ascii=False))
