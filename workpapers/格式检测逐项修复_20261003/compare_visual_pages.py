"""Compare every new page with the prior accepted preview, then render changes."""
from pathlib import Path
import json,re
import fitz
from PIL import Image,ImageDraw

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=ROOT/'output/排版预览.pdf'
NEW=HERE/'排版预览_修复候选.pdf'
OUT=HERE/'visual_changes'
OUT.mkdir(exist_ok=True)

def body_start(doc):
    return next(i for i,p in enumerate(doc) if re.search(r'中南大学博士学位论文\s*第1\s*章\s*绪论',p.get_text()))

with fitz.open(OLD) as a,fitz.open(NEW) as b:
    old_start,new_start=body_start(a),body_start(b)
    old_pages,new_pages=len(a),len(b)

changed=[];records=[]
# Reopen in small batches so decoded images do not accumulate for 318 pages.
for start in range(0,new_pages,16):
    with fitz.open(OLD) as a,fitz.open(NEW) as b:
        for index in range(start,min(start+16,new_pages)):
            reference=index if index<9 else old_start+(index-new_start) if index>=new_start else None
            identical=False
            if reference is not None and 0<=reference<old_pages:
                pa=a[reference].get_pixmap();pb=b[index].get_pixmap()
                identical=(pa.width,pa.height,pa.samples)==(pb.width,pb.height,pb.samples)
                del pa,pb
            if not identical:changed.append(index+1)
            records.append({'new_page':index+1,'prior_page':None if reference is None else reference+1,
                            'rendered_pixels_identical':identical})

sheets=[]
for start in range(0,len(changed),4):
    pages=changed[start:start+4]
    sheet=Image.new('RGB',(1600,2336),'#D8D8D8')
    draw=ImageDraw.Draw(sheet)
    with fitz.open(NEW) as document:
        for offset,number in enumerate(pages):
            page=document[number-1]
            scale=min(96/72,800/page.rect.width,1131/page.rect.height)
            pix=page.get_pixmap(matrix=fitz.Matrix(scale,scale),alpha=False)
            x=(offset%2)*800;y=(offset//2)*1168
            draw.text((x+8,y+5),f'NEW physical page {number}',fill='black')
            sheet.paste(Image.frombytes('RGB',(pix.width,pix.height),pix.samples),(x+(800-pix.width)//2,y+30))
            del pix
    destination=OUT/f'changed_{start//4+1:03}.png'
    sheet.save(destination)
    sheets.append({'path':str(destination),'pages':pages})

result={'old_pdf':str(OLD),'new_pdf':str(NEW),'old_pages':old_pages,'new_pages':new_pages,
        'old_body_start':old_start+1,'new_body_start':new_start+1,
        'comparison':'Direct equality of 72-dpi rendered pixels, with body pages aligned after the enlarged TOC; no hashes.',
        'identical_pages':new_pages-len(changed),'changed_pages':changed,
        'pages':records,'sheets':sheets,'all_changed_pages_visually_reviewed':False}
(HERE/'逐页图像比对.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('old_pages','new_pages','old_body_start','new_body_start','identical_pages','changed_pages')},ensure_ascii=False))
