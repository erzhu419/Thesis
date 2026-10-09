#!/usr/bin/env python3
"""Render every page and collect review candidates; no automatic layout verdict."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import sys

import fitz
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PDF = ROOT / 'output/排版预览.pdf'
DEFAULT_OUT = Path(__file__).resolve().parent / 'visual'
CAPTION = re.compile(r'^\s*(?:图|Figure|Fig\.)\s*([0-9A-Z]+)\s*[-–]\s*(\d+)(?:\s|$)')

def box(values):
    return [round(float(v), 2) for v in values]

def outside(rect, width, height, tolerance=.5):
    return rect[0] < -tolerance or rect[1] < -tolerance or rect[2] > width+tolerance or rect[3] > height+tolerance

def inspect_page(page):
    width, height = page.rect.width, page.rect.height
    # Infinite clip includes off-page text, unlike the default page rectangle.
    text = page.get_text('dict', flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES,
                         clip=fitz.INFINITE_RECT())
    lines, spans, fonts, headers, footers, overflows, captions = [], [], Counter(), [], [], [], []
    body_chars = 0
    for block in text['blocks']:
        if block.get('type') != 0:
            continue
        for line in block.get('lines', []):
            clean = [{k:s[k] for k in ('text','font','size','flags','bbox')} for s in line['spans']]
            for s in clean:
                s['bbox'] = box(s['bbox']); s['size'] = round(s['size'], 2)
                fonts[(s['font'], s['size'])] += len(s['text'].strip())
                if s['text'].strip() and outside(s['bbox'], width, height):
                    overflows.append(s)
            value = ''.join(s['text'] for s in clean)
            entry = {'bbox':box(line['bbox']), 'text':value, 'spans':clean}
            lines.append(entry); spans.extend(clean)
            center_y = (line['bbox'][1] + line['bbox'][3]) / 2
            if center_y < 60:
                headers.append({'text':value,'bbox':entry['bbox']})
            elif center_y > height-60:
                footers.append({'text':value,'bbox':entry['bbox']})
            else:
                body_chars += len(value.strip())
                if CAPTION.match(value):
                    captions.append({'text':value,'bbox':entry['bbox']})
    images = []
    for item in page.get_image_info():  # No image hashes or pixel extraction.
        images.append({'bbox':box(item['bbox']), 'width_pixels':item['width'],
                       'height_pixels':item['height'],
                       'outside_page':outside(item['bbox'],width,height)})
    body_images = [x for x in images if x['bbox'][3] > 60 and x['bbox'][1] < height-60]
    return {'physical_page':page.number+1, 'pdf_label':page.get_label(),
            'page_size_points':[round(width,2),round(height,2)], 'rotation':page.rotation,
            'body_characters':body_chars, 'body_blank_candidate':body_chars==0 and not body_images,
            'body_sparse_candidate':body_chars<80 and not body_images,
            'text_outside_page':overflows,
            'fonts':[{'font':font,'size':size,'characters':count} for (font,size),count in sorted(fonts.items())],
            'header_candidates':headers, 'footer_candidates':footers,
            'images':images, 'figure_caption_candidates':captions, 'lines':lines}

def render_chunk(pdf, out, start, stop):
    out.mkdir(parents=True,exist_ok=True)
    records=[]; tile_width,tile_height,label_height=800,1168,30
    font=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',18)
    with fitz.open(pdf) as doc:
        stop=min(stop,len(doc))
        for first in range(start,stop,4):
            sheet=Image.new('RGB',(tile_width*2,tile_height*2),'#D8D8D8')
            draw=ImageDraw.Draw(sheet)
            for offset in range(min(4,stop-first)):
                page=doc[first+offset]
                record=inspect_page(page);records.append(record)
                scale=min(96/72,800/page.rect.width,1131/page.rect.height)
                pix=page.get_pixmap(matrix=fitz.Matrix(scale,scale),alpha=False,colorspace=fitz.csRGB)
                x=(offset%2)*tile_width; y=(offset//2)*tile_height
                draw.text((x+8,y+4),f'PDF physical page {page.number+1:04d}',font=font,fill='black')
                sheet.paste(Image.frombytes('RGB',(pix.width,pix.height),pix.samples),
                            (x+(tile_width-pix.width)//2,y+label_height))
                del pix
            name=f'sheet_{first+1:04d}-{min(first+4,stop):04d}.png'
            sheet.save(out/name)
            del sheet
    dest=out/f'geometry_{start+1:04d}-{stop:04d}.json'
    dest.write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'start':start+1,'stop':stop,'geometry':str(dest)},ensure_ascii=False),flush=True)

def cross_page_candidates(pages):
    result=[]
    for i,page in enumerate(pages):
        for caption in page['figure_caption_candidates']:
            preceding=[im for im in page['images'] if im['bbox'][3] <= caption['bbox'][1]+20 and im['bbox'][3]-im['bbox'][1] > 35]
            if preceding:
                continue
            neighbors=[]
            for j in (i-1,i+1):
                if 0 <= j < len(pages) and pages[j]['images']:
                    neighbors.append({'physical_page':pages[j]['physical_page'],'images':pages[j]['images']})
            result.append({'physical_page':page['physical_page'],'caption':caption,
                           'reason':'No same-page image above this caption candidate; inspect this and adjacent pages.',
                           'adjacent_image_pages':neighbors})
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pdf',nargs='?',type=Path,default=DEFAULT_PDF)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--chunk',nargs=2,type=int,metavar=('START_ZERO_BASED','STOP_EXCLUSIVE'))
    args=parser.parse_args()
    if args.chunk:
        render_chunk(args.pdf,args.output,*args.chunk);return
    with fitz.open(args.pdf) as doc:
        total=len(doc)
    args.output.mkdir(parents=True,exist_ok=True)
    pages=[]
    for start in range(0,total,16):
        stop=min(start+16,total)
        subprocess.run([sys.executable,__file__,str(args.pdf),'--output',str(args.output),
                        '--chunk',str(start),str(stop)],check=True)
        pages.extend(json.loads((args.output/f'geometry_{start+1:04d}-{stop:04d}.json').read_text()))
    summary={'status':'CANDIDATES_FOR_VISUAL_REVIEW_NOT_A_PASS_VERDICT','pdf':str(args.pdf.resolve()),
             'physical_pages':total,'contact_sheets':(total+3)//4,'chunk_size':16,
             'rendering':'96 dpi; A4 pages about 794 by 1123 pixels; 2 by 2 contact sheets.',
             'body_blank_candidates':[p['physical_page'] for p in pages if p['body_blank_candidate']],
             'body_sparse_candidates':[p['physical_page'] for p in pages if p['body_sparse_candidate']],
             'text_page_boundary_candidates':[{'physical_page':p['physical_page'],'spans':p['text_outside_page']} for p in pages if p['text_outside_page']],
             'image_page_boundary_candidates':[{'physical_page':p['physical_page'],'images':[im for im in p['images'] if im['outside_page']]} for p in pages if any(im['outside_page'] for im in p['images'])],
             'cross_page_figure_caption_candidates':cross_page_candidates(pages),
             'header_footer_by_page':[{'physical_page':p['physical_page'],'pdf_label':p['pdf_label'],'headers':p['header_candidates'],'footers':p['footer_candidates']} for p in pages]}
    (args.output/'review_candidates.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    with (args.output/'page_geometry.jsonl').open('w') as f:
        for p in pages:f.write(json.dumps(p,ensure_ascii=False)+'\n')
    print(json.dumps({k:summary[k] for k in ['status','physical_pages','contact_sheets','body_blank_candidates','body_sparse_candidates']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
