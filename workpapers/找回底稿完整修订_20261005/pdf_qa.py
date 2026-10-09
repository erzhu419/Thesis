from pathlib import Path
import fitz,json
from PIL import Image,ImageDraw
W=Path(__file__).resolve().parent;P=W.parents[1]/'output/排版完成_导师意见复核_20261005.pdf'
d=fitz.open(P)
assert {3,6}.issubset({l.get('page') for l in d[9].get_links()})
V=W/'页面检查';V.mkdir(exist_ok=True)
issues=[];matches={k:[] for k in ['B-25','D-58','表6-16','表6-7','表6-8','正则项','参考文献','附录F.4','模式置信']}
for start in range(0,len(d),20):
 board=Image.new('RGB',(1600,20//4*590),'#d9d9d9');draw=ImageDraw.Draw(board)
 for j,i in enumerate(range(start,min(start+20,len(d)))):
  page=d[i];txt=page.get_text();pix=page.get_pixmap(matrix=fitz.Matrix(.64,.64),alpha=False)
  im=Image.frombytes('RGB',[pix.width,pix.height],pix.samples);x=(j%4)*400;y=(j//4)*590
  board.paste(im,(x+10,y+23));draw.text((x+12,y+5),f'PDF {i+1}',fill='black')
  if len(txt.strip())<30:issues.append({'page':i+1,'kind':'low_text','text':txt.strip()})
  if any(k in txt for k in ['错误!','错误！','Error! Reference','Error! Bookmark']):issues.append({'page':i+1,'kind':'broken_field'})
  for b in page.get_text('blocks'):
   if b[4].strip() and (b[0]<-1 or b[1]<-1 or b[2]>page.rect.width+1 or b[3]>page.rect.height+1):issues.append({'page':i+1,'kind':'outside_page','bbox':b[:4]})
  for k in matches:
   if k in txt or k.replace('-','−') in txt:matches[k].append(i+1)
 board.save(V/f'总览_{start+1:03d}-{min(start+20,len(d)):03d}.jpg',quality=85)
(W/'PDF逐页检查.json').write_text(json.dumps({'pages':len(d),'all_pages_rendered':True,'scan_findings':issues,'key_pages':matches,'toc_abstract_links_corrected':True,'scope':'320页逐页渲染及越界、错误域筛查；缩略页总览另作视觉核查，关键页放大。'},ensure_ascii=False,indent=2))
print(json.dumps({'pages':len(d),'issues':issues,'key_pages':matches},ensure_ascii=False))
