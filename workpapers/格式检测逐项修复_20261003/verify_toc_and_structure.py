"""Check cached TOC targets against the actual final Word-rendered PDF pages."""
from pathlib import Path
from zipfile import ZipFile
from collections import Counter
import json,re
import fitz
from lxml import etree

HERE=Path(__file__).resolve().parent
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'm':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
W='{'+NS['w']+'}'
with ZipFile(HERE/'排版完成_修复候选.docx') as package:
    document=etree.fromstring(package.read('word/document.xml'))
    settings=etree.fromstring(package.read('word/settings.xml'))
paragraphs=document.xpath('//w:p',namespaces=NS)
def text(p):return ''.join(p.xpath('.//w:t/text()|.//m:t/text()',namespaces=NS))
def style(p):return p.xpath('string(./w:pPr/w:pStyle/@w:val)',namespaces=NS)
def compact(s):return re.sub(r'[\s\u200b\ufeff]+','',s)
headings=[p for p in paragraphs if style(p)=='h4']
toc4=[p for p in paragraphs if style(p)=='TOC4']
names=set(document.xpath('//w:bookmarkStart/@w:name',namespaces=NS))
errors=[]
for link in document.xpath('//w:hyperlink[@w:anchor]',namespaces=NS):
    if link.get(W+'anchor') not in names:errors.append({'missing_hyperlink_target':link.get(W+'anchor')})
for code in document.xpath('//w:instrText/text()',namespaces=NS):
    match=re.search(r'\b(?:REF|PAGEREF)\s+([^\s\\]+)',code)
    if match and match[1] not in names:errors.append({'missing_field_target':match[1]})
with fitz.open(HERE/'排版预览_修复候选.pdf') as pdf:
    body_start=next(i for i,p in enumerate(pdf) if re.search(r'中南大学博士学位论文\s*第1\s*章\s*绪论',p.get_text()))
    page_text=[compact(p.get_text()) for p in pdf]
    matches=[]
    for heading in headings:
        title=text(heading)
        entry=[p for p in toc4 if compact(text(p)).startswith(compact(title))]
        actual=[i+1 for i,t in enumerate(page_text) if i>=body_start and compact(title) in t]
        if len(entry)!=1 or len(actual)!=1:
            errors.append({'heading':title,'toc_matches':len(entry),'pdf_matches':actual})
            continue
        cached=entry[0].xpath('.//w:t/text()',namespaces=NS)[-1]
        printed=pdf[actual[0]-1].get_text(clip=fitz.Rect(0,pdf[actual[0]-1].rect.height-55,pdf[actual[0]-1].rect.width,pdf[actual[0]-1].rect.height)).strip()
        passed=cached==printed
        if not passed:errors.append({'heading':title,'toc_page':cached,'printed_page':printed})
        matches.append({'heading':title,'physical_page':actual[0],'toc_page':cached,'printed_page':printed,'passed':passed})
    roman=[p.get_text().strip().splitlines()[0].strip() for p in pdf[3:body_start]]
    arabic=[p.get_text(clip=fitz.Rect(0,p.rect.height-55,p.rect.width,p.rect.height)).strip() for p in pdf[body_start:]]
    if arabic!=list(map(str,range(1,len(arabic)+1))):errors.append({'arabic_page_sequence_failed':arabic})
    pages=len(pdf)
result={'passed':not errors and len(matches)==len(headings)==len(toc4)==53,
        'physical_pages':pages,'body_start':body_start+1,'h4_count':len(headings),'toc4_count':len(toc4),
        'toc_entries_by_level':dict(Counter(style(p) for p in paragraphs if style(p) in {'TOC1','TOC2','TOC3','TOC4'})),
        'toc_code':document.xpath('//w:instrText[contains(text(),"TOC")]/text()',namespaces=NS),
        'sections':len(document.xpath('//w:sectPr',namespaces=NS)),
        'native_math_objects':len(document.xpath('//m:oMath',namespaces=NS)),
        'arabic_pages':len(arabic),'roman_page_labels':roman,
        'heading_page_checks':matches,'errors':errors}
(HERE/'目录页码与结构验收.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('passed','physical_pages','h4_count','toc4_count','toc_entries_by_level','sections','native_math_objects','errors')},ensure_ascii=False,indent=2))
