from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
from PIL import Image,ImageDraw
import fitz,json,re
ROOT=Path(__file__).resolve().parents[2];WORK=Path(__file__).resolve().parent
stem=ROOT/'排版完成_接受作者修订_全文统一_20261006'
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
norm=lambda t:re.sub(r'\s+','',t)
with ZipFile(stem.with_suffix('.docx')) as z:r=E.fromstring(z.read('word/document.xml'))
captions=[''.join(p.xpath('.//w:t/text()',namespaces=ns)) for p in r.xpath('.//w:p[w:pPr/w:pStyle/@w:val="figurecaption"]',namespaces=ns)]
pdf=fitz.open(stem.with_suffix('.pdf'));texts=[norm(p.get_text()) for p in pdf]
audit=json.loads(stem.with_suffix('.留白.json').read_text());native=json.loads((WORK/'Word分页记录.json').read_text(encoding='utf-8-sig'))
figures=[];issues=[];images=[]
for i,p in enumerate(pdf):
 if i+1<audit['body_start']:continue
 for im in sorted(p.get_image_info(),key=lambda x:x['bbox'][1]):images.append({'page':i+1,'bbox':list(im['bbox'])})
assert len(images)==len(captions)==63,(len(images),len(captions))
for cap,im in zip(captions,images):
 hit=[i+1 for i,t in enumerate(texts) if norm(cap) in t]
 if hit!=[im['page']]:issues.append({'caption':cap,'image_page':im['page'],'caption_pages':hit})
 if not (im['bbox'][0]>=85 and im['bbox'][2]<=515 and im['bbox'][1]>=70 and im['bbox'][3]<=771):issues.append({'image_bounds':im,'caption':cap})
 figures.append({'caption':cap,**im})
for row in native['TableSpans']:
 if row['Caption']=='主要符号说明':continue
 hit=[i+1 for i,t in enumerate(texts) if norm(row['Caption']) in t]
 if hit!=[row['StartPage']] or row['StartPage']!=row['EndPage']:issues.append({'table':row,'caption_pages':hit})
errors=[i+1 for i,t in enumerate(texts) if re.search('错误[!！]|Error!|未定义书签|未找到引用源',t)]
assert not errors,errors
pages=set()
for key in ['针对源仿真对车辆运行和乘客服务的预测不符合目标环境的问题','本章以作者提出的MC-WM','3.2.3类别特征、连续状态与共享策略','6.3.7公交归一化置信与源目标修正','6.4.3.5归一化置信的同预算公交比较','上述研究围绕同一驻站控制任务逐步增加可利用的信息']:
 hit=[i+1 for i,t in enumerate(texts) if i+1>=audit['body_start'] and key in t];pages.update(hit)
 if key.startswith('3.2.3'):pages.update([a+1 for a in hit])
for fig in figures:
 if any(fig['caption'].startswith('图'+label+' ') for label in ['2-2','2-3','3-1','3-3']):pages.update([fig['page']-1,fig['page']])
# A page overview catches visual overlap that text extraction cannot detect.
out=WORK/'页面预览';out.mkdir(exist_ok=True);sheets=[];pages=sorted(pages)
for k in range(0,len(pages),6):
 canvas=Image.new('RGB',(1800,1760),'#ddd');draw=ImageDraw.Draw(canvas)
 for slot,n in enumerate(pages[k:k+6]):
  pm=pdf[n-1].get_pixmap(matrix=fitz.Matrix(.97,.97),alpha=False);x=slot%3*600;y=slot//3*880
  draw.text((x+12,y+8),f'PDF {n}',fill='black');canvas.paste(Image.frombytes('RGB',(pm.width,pm.height),pm.samples),(x+10,y+25))
 path=out/f'检查{k//6+1:02}.png';canvas.save(path);sheets.append(str(path))
result={'pages':len(pdf),'body_start':audit['body_start'],'interior_gaps':[x for x in audit['gaps'] if x['kind']=='章内'],'figure_count':len(figures),'ordinary_table_count':len(native['TableSpans'])-1,'layout_issues':issues,'preview_sheets':sheets,'preview_pages':pages}
(WORK/'版面核对.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2))
assert not issues,issues
assert not result['interior_gaps'],result['interior_gaps']
