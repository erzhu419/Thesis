from pathlib import Path
import sys, re, json
import fitz

path = Path(sys.argv[1])
offset = int(sys.argv[2]) if len(sys.argv)>2 else 0
doc = fitz.open(path)
pages = []
for i, page in enumerate(doc):
    blocks = page.get_text('blocks')
    content = [b for b in blocks if b[1] >= 71 and b[3] < 772]
    ends = [b[3] for b in content]
    ends += [im['bbox'][3] for im in page.get_image_info() if im['bbox'][1] >= 71 and im['bbox'][3] < 772]
    ends += [draw['rect'].y1 for draw in page.get_drawings() if draw['rect'].y0 >= 71 and draw['rect'].y1 < 772]
    text = '\n'.join(b[4].strip() for b in content if b[6] == 0)
    footer = [b[4].strip() for b in blocks if b[1] > 772 and b[6] == 0]
    chapter_start = any(re.match(r'^(第[1-7]章|附录[A-F]|参考文献$)', re.sub(r'\s+', '', b[4])) for b in content if b[1] < 115 and b[6] == 0)
    pages.append({'pdf_page':i+1, 'printed':footer[-1] if footer else '', 'bottom':round(max(ends,default=72),2), 'text':text, 'chapter_start':chapter_start})
start = 0 if offset else next(i for i,p in enumerate(pages) if p['printed']=='1' and p['chapter_start'])
gaps = []
for i,p in enumerate(pages):
    gap = 769.9 - p['bottom']
    if i < start or gap <= (769.9-72)/3: continue
    kind = '末页' if i==len(pages)-1 else '章末' if pages[i+1]['chapter_start'] else '章内'
    following = pages[i+1]['text'] if i+1<len(pages) else ''
    first_caption = re.search(r'(图\s*[1-7]-\d+|表\s*[1-7]-\d+)', following)
    gaps.append({'pdf_page':i+1+offset,'printed':p['printed'],'gap':round(gap,2),'kind':kind,'last':p['text'][-180:],'next':following[:240], 'next_caption':re.sub(r'\s+','',first_caption.group(1)) if first_caption else None})
report = {'pdf':str(path), 'pages':len(doc), 'body_start':start+1, 'gaps':gaps}
dest = path.with_suffix('.留白.json')
dest.write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({'pages':len(doc),'body_start':start+1,'gaps':[ {k:v for k,v in g.items() if k not in ['last','next']} for g in gaps]},ensure_ascii=False,indent=2))
