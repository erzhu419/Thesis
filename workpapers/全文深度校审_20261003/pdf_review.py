"""Render the full PDF and check page boundaries, TOC destinations and captions."""
import json
from pathlib import Path
import re
import sys
import fitz
from PIL import Image,ImageDraw,ImageFont

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'workpapers/模板迁移_20261002'))
from render_full_preview import inspect_page

pdf=ROOT/'output/排版完成_20261003_全文深度校审_终校.pdf'
out=HERE/'visual'
out.mkdir(exist_ok=True)
font=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',16)
doc=fitz.open(pdf)
records=[]
for first in range(0,len(doc),20):
    sheet=Image.new('RGB',(1280,2320),'#d8d8d8')
    draw=ImageDraw.Draw(sheet)
    for offset in range(min(20,len(doc)-first)):
        page=doc[first+offset]
        data=inspect_page(page)
        lines=data.pop('lines')
        links=page.get_links()
        data['links']=len(links)
        data['invalid_link_destinations']=[l for l in links if l.get('kind')==fitz.LINK_GOTO and not 0<=l.get('page',-1)<len(doc)]
        data['captions']=[l['text'] for l in lines if re.match(r'^\s*[图表]\s*[A-F0-9]+-\d+\s',l['text'])]
        data['equation_numbers']=re.findall(r'\(([A-F0-9]+-\d+)\)',page.get_text())
        records.append(data)
        pix=page.get_pixmap(matrix=fitz.Matrix(.51,.51),alpha=False)
        x=(offset%4)*320;y=(offset//4)*464
        draw.text((x+6,y+3),str(page.number+1),font=font,fill='black')
        sheet.paste(Image.frombytes('RGB',(pix.width,pix.height),pix.samples),(x+8,y+24))
    sheet.save(out/f'pages_{first+1:03d}-{min(first+20,len(doc)):03d}.png')
    print('Rendered',first+1,'to',min(first+20,len(doc)),flush=True)
toc=[]
for page in doc:
    if not 7 <= page.number <= 17: continue
    lines=[l for b in page.get_text('dict')['blocks'] if b.get('type')==0 for l in b['lines']]
    for link in page.get_links():
        if link.get('kind')!=fitz.LINK_GOTO: continue
        rect=fitz.Rect(link['from'])
        line=min(lines,key=lambda l:abs((l['bbox'][1]+l['bbox'][3])/2-(rect.y0+rect.y1)/2))
        shown=''.join(s['text'] for s in line['spans']).strip()
        dest=doc[link['page']]
        pieces=re.split(r'\.{2,}',shown)
        expected=pieces[0].strip()
        shown_number=pieces[-1].strip() if len(pieces)>1 else ''
        footer_lines=[l for b in dest.get_text('dict')['blocks'] if b.get('type')==0
                      for l in b['lines'] if l['bbox'][1]>dest.rect.height-100
                      and re.fullmatch(r'\s*(?:[IVXLCDM]+|\d+)\s*', ''.join(s['text'] for s in l['spans']))]
        footer_line=max(footer_lines,key=lambda l:l['bbox'][1]) if footer_lines else None
        printed_number=''.join(s['text'] for s in footer_line['spans']).strip() if footer_line else ''
        destination=re.sub(r'\s+','',dest.get_text())
        title=re.sub(r'\s+','',expected)
        toc.append({'from':page.number+1,'to':link['page']+1,'shown':shown,
                    'heading_found':title.casefold() in destination.casefold(),
                    'printed_page_matches':shown_number==printed_number})
summary={
    'pages':len(doc),'rendered_pages':len(records),'toc_links':toc,
    'blank_pages':[r['physical_page'] for r in records if r['body_blank_candidate']],
    'text_outside_page':[r['physical_page'] for r in records if r['text_outside_page']],
    'images_outside_page':[r['physical_page'] for r in records if any(im['outside_page'] for im in r['images'])],
    'invalid_pdf_links':[r['physical_page'] for r in records if r['invalid_link_destinations']],
    'toc_heading_mismatches':[r for r in toc if not r['heading_found']],
    'toc_page_number_mismatches':[r for r in toc if not r['printed_page_matches']],
    'page_records':records
}
(HERE/'pdf_geometry.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k not in ['toc_links','page_records']},ensure_ascii=False))
