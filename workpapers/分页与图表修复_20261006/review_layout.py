from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
from PIL import Image, ImageDraw
import fitz, sys, json, re

base=Path(sys.argv[1]); work=base.parent
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
q=lambda n:'{'+ns['w']+'}'+n
norm=lambda t:re.sub(r'\s+','',t)
with ZipFile(base.with_suffix('.docx')) as z:
    d=E.fromstring(z.read('word/document.xml'))
    captions=[''.join(p.xpath('.//w:t/text()',namespaces=ns)) for p in d.xpath('.//w:p[w:pPr/w:pStyle/@w:val="figurecaption"]',namespaces=ns)]
pdf=fitz.open(base.with_suffix('.pdf'))
texts=[norm(p.get_text()) for p in pdf]
audit=json.loads(base.with_suffix('.留白.json').read_text())
assert not [g for g in audit['gaps'] if g['kind']=='章内'],'Interior large blanks remain'
images=[]
for i,p in enumerate(pdf):
    if i+1<audit['body_start']:continue
    for im in sorted(p.get_image_info(),key=lambda x:x['bbox'][1]):
        images.append({'page':i+1,'bbox':list(im['bbox'])})
assert len(images)==len(captions),(len(images),len(captions))
figures=[]
for caption,im in zip(captions,images):
    hit=[i+1 for i,t in enumerate(texts) if norm(caption) in t]
    assert hit==[im['page']],(caption,hit,im)
    assert im['bbox'][0]>=85 and im['bbox'][2]<=515 and im['bbox'][1]>=70 and im['bbox'][3]<=771,(caption,im)
    figures.append({'caption':caption,**im})

native=json.loads(base.with_name(base.name+'_Word.json').read_text(encoding='utf-8-sig'))
splits=[r for r in native['TableSpans'] if r['StartPage']!=r['EndPage']]
assert len(splits)==1 and splits[0]['Index']==3,splits
for table in native['TableSpans']:
    if table['Index']==3:continue
    hits=[i+1 for i,t in enumerate(texts) if norm(table['Caption']) in t]
    assert hits==[table['StartPage']],('Table caption detached',table,hits)
assert not any('错误!未定义书签' in t or 'Error!Bookmarknotdefined' in t for t in texts)
moved=set()
for p in work.glob('候选*.调整.json'):
    for item in json.loads(p.read_text()):moved.add(item['figure'])
pages={10,11,12,13,audit['body_start']}
for fig in figures:
    label=re.match(r'^图\s*[0-9A-Z]+-[0-9]+',fig['caption']).group().replace(' ','')
    if label in moved:pages.update([fig['page']-1,fig['page']])
for t in native['TableSpans']:
    if t['Index'] in [119,127]:pages.add(t['StartPage'])
pages=sorted(pages)
out=work/'最终页面检查';out.mkdir(exist_ok=True)
sheets=[]
for k in range(0,len(pages),6):
    group=pages[k:k+6]
    canvas=Image.new('RGB',(1800,1760),'#ddd');draw=ImageDraw.Draw(canvas)
    for slot,n in enumerate(group):
        pix=pdf[n-1].get_pixmap(matrix=fitz.Matrix(.97,.97),alpha=False)
        x=slot%3*600;y=slot//3*880
        draw.text((x+12,y+10),f'PDF {n}',fill='black')
        canvas.paste(Image.frombytes('RGB',(pix.width,pix.height),pix.samples),(x+10,y+30))
    dest=out/f'页面检查{k//6+1:02}.png';canvas.save(dest);sheets.append(str(dest))
result={'pages':len(pdf),'interior_blanks_over_one_third':0,'figures_and_captions_on_same_page':len(figures),'single_page_data_tables':61,'long_symbol_table_pages':[splits[0]['StartPage'],splits[0]['EndPage']],'review_pages':pages,'contact_sheets':sheets,'figures':figures}
(work/'最终版面核对.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='figures'},ensure_ascii=False,indent=2))
