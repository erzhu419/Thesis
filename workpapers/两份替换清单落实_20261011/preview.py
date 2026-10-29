"""Build readable review sheets for modified labels and sensitive Word pages."""
from pathlib import Path
import json
import re
import sys
from PIL import Image,ImageDraw
import fitz

HERE=Path(__file__).resolve().parent
FOLDER=HERE/'页面预览'
FOLDER.mkdir(exist_ok=True)


def sheets(images,prefix):
    for offset in range(0,len(images),6):
        group=images[offset:offset+6]
        canvas=Image.new('RGB',(1600,650*((len(group)+1)//2)),'#eeeeee')
        draw=ImageDraw.Draw(canvas)
        for j,(label,im) in enumerate(group):
            im.thumbnail((785,615),Image.Resampling.LANCZOS)
            x=(j%2)*800+(800-im.width)//2;y=(j//2)*650+30
            canvas.paste(im,(x,y));draw.text(((j%2)*800+12,(j//2)*650+8),label,fill='black')
        canvas.save(FOLDER/f'{prefix}_{offset//6+1:02}.png')


if sys.argv[1]=='figures':
    rows=json.loads((HERE/'图内标签映射.json').read_text())
    sheets([(r['figure'],Image.open(HERE/f"图内标签/图{r['figure']}.png").convert('RGB')) for r in rows],'figures')
else:
    path=HERE.parents[1]/'output/排版完成_双清单替换_20261011.pdf'
    doc=fitz.open(path)
    needles=['（1）针对双向发车环境','Generalized passenger time cost is defined',
             '公交驻站控制需要将异步到站','在客流与路况的条件统计规律保持不变',
             '客流突增、反复需求冲击','目标环境中已有的驻站运行记录',
             '在跨保真驻站训练中','公交驻站策略由仿真训练走向实际应用',
             '输入维度是指状态向量','两环境的响应差归一化为',
             '表2-7','表3-4','表5-6','表5-10','表6-6','表6-7','表6-8',
             '表6-11','表6-13','表6-15','表6-16','附录C BAPR','F.1 共同离线',
             'F.3 单次训练','表F-2']
    texts=[re.sub(r'\s+','',p.get_text()) for p in doc]
    locations={s:[i+1 for i,t in enumerate(texts) if re.sub(r'\s+','',s) in t] for s in needles}
    pages=sorted({p for v in locations.values() for p in v if p<=9 or p>19})
    (HERE/'Word页面定位.json').write_text(json.dumps({'pages':len(doc),'terms':locations,'review_pages':pages},ensure_ascii=False,indent=2))
    for offset in range(0,len(pages),2):
        selected=pages[offset:offset+2];images=[]
        for page in selected:
            pix=doc[page-1].get_pixmap(matrix=fitz.Matrix(1.3,1.3),alpha=False)
            images.append(Image.frombytes('RGB',[pix.width,pix.height],pix.samples))
        canvas=Image.new('RGB',(sum(i.width for i in images),max(i.height for i in images)+28),'#eeeeee')
        draw=ImageDraw.Draw(canvas);x=0
        for page,im in zip(selected,images):
            canvas.paste(im,(x,28));draw.text((x+10,8),f'PDF {page}',fill='black');x+=im.width
        canvas.save(FOLDER/('Word_'+'_'.join(map(str,selected))+'.png'))
    print(json.dumps({'pages':len(doc),'review_pages':pages,'locations':locations},ensure_ascii=False))
