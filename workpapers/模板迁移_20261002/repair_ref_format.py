#!/usr/bin/env python3
"""Keep verified field results; make REF refresh use their current local format."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from copy import deepcopy
from lxml import etree as E
import json,re
from verify_format import Doc,props,blend,normr
HERE=Path(__file__).resolve().parent
SOURCE=HERE/'review_ready.docx';OUT=HERE/'ref_format_ready.docx'
W='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
z=ZipFile(SOURCE);root=E.fromstring(z.read('word/document.xml'));stack=[];changed=[]
resolved=Doc(SOURCE);paragraphs=root.findall('.//'+W+'p');paragraph_indexes={p:i for i,p in enumerate(paragraphs)}
values=json.load(open(HERE/'template_analysis_values.json'))
def effective_run_format(run):
 p=run.getparent()
 while p is not None and p.tag!=W+'p':p=p.getparent()
 sid,_,base,_,_=resolved.resolve(resolved.ps[paragraph_indexes[p]])
 name=resolved.styles[sid].find(W+'name').get(W+'val','');role=name.replace('模板迁移 ','') if name.startswith('模板迁移 ') else 'body'
 ancestor=p.getparent();table=None;cell=None;row=None
 while ancestor is not None:
  if ancestor.tag==W+'tc':cell=ancestor
  if ancestor.tag==W+'tr':row=ancestor
  if ancestor.tag==W+'tbl':table=ancestor;break
  ancestor=ancestor.getparent()
 if table is not None:
  codes=''.join(t.text or '' for t in table.iter(W+'instrText'))
  role='equation' if re.search(r'\bSEQ\s+Eq',codes) else 'table'
 eff=deepcopy(values[role]['effective_rPr'] if role in values else base)
 eff.pop('rStyle',None);eff['rFonts']=resolved.fonts(eff)
 # REF inherits the typography of its current paragraph/cell, never the
 # heading being referenced or a previously contaminated cached result.
 eff['b']={'val':'1' if role=='table' and row is table.find(W+'tr') else '0'}
 if role in {'chapter','cover_title','cover_degree','abstract_cn_title','abstract_en_title','bibliography_title','symbol_title','toc_title'}:eff['b']={'val':'1'}
 eff['bCs']=deepcopy(eff['b']);eff['i']={'val':'0'};eff['iCs']={'val':'0'};eff['vertAlign']={'val':'baseline'}
 eff['color']={'val':'auto'};eff['spacing']={'val':'0'}
 for k in ['b','i']:eff.setdefault(k,{'val':'0'})
 desired=E.Element(W+'rPr')
 for k,attributes in eff.items():
  child=E.SubElement(desired,W+k)
  for key,value in attributes.items():child.set(W+key,str(value))
 return desired
for e in list(root.iter()):
 if e.tag==W+'fldChar':
  typ=e.get(W+'fldCharType')
  if typ=='begin':stack.append({'phase':'code','code':[],'results':[]})
  elif typ=='separate' and stack:stack[-1]['phase']='result'
  elif typ=='end' and stack:
   field=stack.pop();code=''.join(x.text or '' for x in field['code'])
   if not re.match(r'^\s*REF\b',code,re.I):continue
   if not field['results']:raise RuntimeError('REF has no displayed result: '+code)
   first=next(x for x in field['code'] if re.search('[A-Za-z]',x.text or ''));code_run=first.getparent()
   result_run=field['results'][0];desired=effective_run_format(result_run)
   before=code_run.find(W+'rPr');different=E.tostring(before)!=E.tostring(desired) if before is not None else True
   if before is not None:code_run.remove(before)
   code_run.insert(0,deepcopy(desired))
   for result in field['results']:
    current=result.find(W+'rPr')
    if current is not None:result.remove(current)
    result.insert(0,deepcopy(desired))
   if not re.search(r'\\\*\s+CHARFORMAT\b',code,re.I):field['code'][-1].text=(field['code'][-1].text or '').rstrip()+' \\* CHARFORMAT '
   # Word requires the field-instruction whitespace to survive XML loading.
   # A missing preserved trailing delimiter makes CHARFORMAT an invalid switch.
   field['code'][0].text=' '+(field['code'][0].text or '').lstrip()
   field['code'][-1].text=(field['code'][-1].text or '').rstrip()+' '
   for instruction in field['code']:instruction.set('{http://www.w3.org/XML/1998/namespace}space','preserve')
   changed.append({'code_before':code.strip(),'code_after':''.join(x.text or '' for x in field['code']).strip(),
                   'display':''.join(t.text or '' for r in field['results'] for t in r.iter(W+'t')),
                   'first_code_run_format_differed':different})
 elif e.tag==W+'instrText' and stack and stack[-1]['phase']=='code':stack[-1]['code'].append(e)
 elif e.tag==W+'t' and stack and stack[-1]['phase']=='result':
  r=e.getparent()
  if (e.text or '') and r not in stack[-1]['results']:stack[-1]['results'].append(r)
with ZipFile(OUT,'w',ZIP_DEFLATED) as out:
 for name in z.namelist():out.writestr(name,E.tostring(root,encoding='UTF-8',xml_declaration=True,standalone=True) if name=='word/document.xml' else z.read(name))
report={'source':str(SOURCE),'output':str(OUT),'patched_REF_fields':len(changed),'first_code_run_formats_changed':sum(x['first_code_run_format_differed'] for x in changed),
        'method':'Set REF results and first code-letter run to the referencing paragraph/cell baseline typography (body 12 pt regular; data cells 10.5 pt, header bold), then add CHARFORMAT. Retain every cached result and bookmark argument.','fields':changed}
(HERE/'ref_format_repair.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='fields'},ensure_ascii=False))
