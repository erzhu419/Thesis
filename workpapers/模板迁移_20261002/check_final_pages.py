"""Check every updated TOC target and printed page number in the Word PDF."""
from pathlib import Path
import json, re
import fitz

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
info = json.loads((HERE/'final_word.json').read_text(encoding='utf-8-sig'))
pdf = fitz.open(ROOT/'output/排版预览.pdf')
norm = lambda s: re.sub(r'\s+', '', s).casefold()
def roman(n):
    out=''
    for value,letter in [(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),(90,'XC'),(50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]:
        while n>=value:
            out+=letter;n-=value
    return out

sections=info['Sections']
body_first=sections[2]['FirstPhysicalPage']
front_first=sections[1]['FirstPhysicalPage']
page_text=[p.get_text() for p in pdf]
rows=[]
for line in info['TocText'].split('\r'):
    if '\t' not in line:continue
    title,label=line.rsplit('\t',1)
    if label.isdigit():physical=body_first+int(label)-1
    else:
        candidates=[n for n in range(1,body_first) if roman(n)==label]
        assert len(candidates)==1,(title,label)
        physical=front_first+candidates[0]-1
    rows.append({'title':title,'printed_page':label,'physical_page':physical,
                 'title_present_on_target':norm(title) in norm(page_text[physical-1])})
page_checks=[]
for i,p in enumerate(pdf,1):
    expected='' if i==1 else roman(i-front_first+1) if i<body_first else str(i-body_first+1)
    # Front matter and body footer baselines differ; take the lowest printed
    # central line instead of cutting through the last body line's glyph box.
    words=[w for w in p.get_text('words') if w[1]>p.rect.height-100
           and abs((w[0]+w[2])/2-p.rect.width/2)<75]
    lowest=max((w[3] for w in words),default=0)
    footer=' '.join(w[4] for w in sorted(words,key=lambda w:w[0]) if abs(w[3]-lowest)<1)
    body_rect=fitz.Rect(0,72,p.rect.width,p.rect.height-75)
    text=p.get_text(clip=body_rect).strip()
    images=[x for x in p.get_image_info() if fitz.Rect(x['bbox']).intersects(body_rect)]
    errors=[t for t in ['错误!','错误！','Error!','未定义书签','未找到引用源'] if t in page_text[i-1]]
    page_checks.append({'physical_page':i,'expected_page_number':expected,
                        'footer_text':footer.strip(),'number_matches':norm(footer)==norm(expected),
                        'body_blank':not text and not images,'field_errors':errors})
result={'pages':len(pdf),'toc_entries':len(rows),'toc_target_passed':all(r['title_present_on_target'] for r in rows),
        'page_numbers_passed':all(p['number_matches'] for p in page_checks),
        'blank_pages':[p['physical_page'] for p in page_checks if p['body_blank']],
        'field_errors':[p for p in page_checks if p['field_errors']],
        'toc_targets':rows,'page_checks':page_checks}
(HERE/'final_pages_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['toc_targets','page_checks']},ensure_ascii=False))
print('TOC mismatches:',[r for r in rows if not r['title_present_on_target']])
print('Footer mismatches:',[p for p in page_checks if not p['number_matches']])
