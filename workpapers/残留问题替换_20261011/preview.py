"""Locate changed pages and render readable paired previews from Word's PDF."""
from pathlib import Path
import json
import fitz
from PIL import Image,ImageDraw

HERE=Path(__file__).resolve().parent
PDF=Path('/mnt/c/Users/erzhu/AppData/Local/Temp/thesis_replacements_20261011/preview.pdf')
doc=fitz.open(PDF)
needles=['主要符号说明','表2-6','表2-7','表2-8','本行列出','4.3.2 策略表示',
         '此处','符号模型先预测','7.1 研究总结','7.2 主要创新','B.6 路径分析','B.7 期望分析',
         '(B-35)','(B-38)','(B-42)','最终公交软符号配置','(D-4a)','(D-4b)','(D-4c)','(D-4d)','(D-4e)',
         'D.6','D.8 误差关系','将式(E-20)','回到本附录前述有限','E.5 更新条件','E.6 置信排序',
         '附录F 公交世界模型','F.2 最终控制器比较','表F-2','F.4 实验记录','图6-15']
locations={}
texts=[p.get_text() for p in doc]
for term in needles:
    hit=[i+1 for i,text in enumerate(texts) if term in text and i>18]
    locations[term]=hit
pages=sorted({p for k,v in locations.items() for p in v if not 12<=p<=20})
# Front symbol pages and continuation pages are checked as well.
pages=sorted(set(pages+[11,12,13,14]))
folder=HERE/'页面预览'
folder.mkdir(exist_ok=True)
for index in range(0,len(pages),2):
    selected=pages[index:index+2]
    imgs=[]
    for p in selected:
        pix=doc[p-1].get_pixmap(matrix=fitz.Matrix(1.3,1.3),alpha=False)
        im=Image.frombytes('RGB',[pix.width,pix.height],pix.samples)
        imgs.append(im)
    canvas=Image.new('RGB',(sum(i.width for i in imgs),max(i.height for i in imgs)+30),'#dddddd')
    draw=ImageDraw.Draw(canvas)
    x=0
    for p,im in zip(selected,imgs):
        canvas.paste(im,(x,30));draw.text((x+10,8),f'PDF {p}',fill='black');x+=im.width
    canvas.save(folder/('页_'+ '_'.join(map(str,selected))+'.png'))
(HERE/'页面定位.json').write_text(json.dumps({'pages':len(doc),'terms':locations,'rendered_pages':pages},ensure_ascii=False,indent=2))
print(json.dumps({'pages':len(doc),'terms':locations,'rendered_pages':pages},ensure_ascii=False))
