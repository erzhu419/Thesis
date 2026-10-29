"""Typeset the supplied manuscript without rewriting its Chinese text."""
from pathlib import Path
from copy import deepcopy
from io import BytesIO
from collections import Counter
import json, re, sys
import pypandoc
from docx import Document
from docx.text.paragraph import Paragraph
from docx.table import Table
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, Inches, Emu
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT

ROOT=Path('/home/erzhu419/mine_code/Thesis')
ASSETS=Path(__file__).resolve().parent
SOURCE=ROOT/'output/Gemini/重构论文.md'
BASE=ROOT/'output/排版完成_冗余表达与用词校订_20261012.docx'
OUT=ROOT/'output/Gemini/重构论文_排版稿_20261015.docx'
MODE=sys.argv[1] if len(sys.argv)>1 else 'draft'
W=qn('w:p').split('}')[0]+'}'
M=qn('m:oMath').split('}')[0]+'}'
R=qn('r:id').split('}')[0]+'}'
TITLE=SOURCE.read_text().splitlines()[0].removeprefix('# ')

def text(e):
 return ''.join(n.text or '' for n in e.iter() if n.tag in (qn('w:t'),qn('m:t')))

def sty(e):
 n=e.find(qn('w:pPr'))
 if n is None:return ''
 n=n.find(qn('w:pStyle'))
 return n.get(qn('w:val'),'') if n is not None else ''

def pnew(value='',style='body'):
 e=OxmlElement('w:p');p=Paragraph(e,base._body);p.style=base.styles[style]
 if value:p.add_run(value)
 return e

def field(p,code,result=''):
 n=OxmlElement('w:fldSimple');n.set(qn('w:instr'),code)
 if result:
  r=OxmlElement('w:r');t=OxmlElement('w:t');t.text=result;r.append(t);n.append(r)
 p.append(n)
 return n

base=Document(BASE)
old=list(base.element.body)
old_sections=[deepcopy(s._sectPr) for s in base.sections]
annex_start=next(i for i,e in enumerate(old) if sty(e)=='chapter' and text(e).startswith('附录A '))
annex=deepcopy(old[annex_start:])
old_refs={}
for e in old:
 if sty(e)=='bibliography':
  m=re.match(r'\[(\d+)\]',text(e))
  if m:
   anchor=next((n.get(qn('w:name')) for n in e.iter(qn('w:bookmarkStart')) if n.get(qn('w:name'),'').startswith('ref-')),None)
   old_refs[int(m[1])]=anchor
source=SOURCE.read_text()
source=source.split('\n# 附录F')[0].rstrip()+'\n'
# Figure notes are production instructions. Captions and body prose are retained.
source=re.sub(r'^【图[^\n]*】[ \t]*$',lambda m:'\n\nFIGUREPLACEHOLDER'+re.match(r'【(图\d+-\d+)',m[0])[1]+'\n\n',source,flags=re.M)
source=re.sub(r'^---\s*$','',source,flags=re.M)
cm=json.loads((ASSETS/'citation_key_map.json').read_text())
matched=cm['matched']
for extra in ASSETS.glob('citation_additions_*.json'):
 if extra.name.endswith('_unresolved.json'):continue
 data=json.loads(extra.read_text());matched.update(data.get('matched',data))
canon={k:v.get('old_id',k) for k,v in matched.items()}
records={v.get('old_id',k):v for k,v in matched.items()}
order=[];unresolved=set()
CITE=re.compile(r'\[([A-Za-z]+\d{4}[\w-]*(?:\s*,\s*[A-Za-z]+\d{4}[\w-]*)*)\]')
def cite(m):
 keys=[x.strip() for x in m[1].split(',')]
 values=[]
 for k in keys:
  c=canon.get(k,k)
  if k not in matched:unresolved.add(k)
  if c not in order:order.append(c)
  values.append(c)
 return '^\\['+','.join('['+str(order.index(c)+1)+'](#latest_ref_'+str(order.index(c)+1)+')' for c in values)+'\\]^'
source=CITE.sub(cite,source)
# Original annexes cite exactly these three existing bibliography entries.
annex_numbers={}
for n in [66,69,80]:
 anchor=old_refs[n]
 c=anchor.removeprefix('ref-') if anchor else None
 candidate=next((v.get('old_id',k) for k,v in matched.items() if v.get('old_number')==n),c)
 if candidate not in order:order.append(candidate)
 annex_numbers[n]=order.index(candidate)+1
 if candidate not in records:
  olde=next(e for e in old if sty(e)=='bibliography' and re.match(r'\['+str(n)+r'\]',text(e)))
  records[candidate]={'bibliography_text':text(olde),'old_id':candidate}
if MODE=='final' and unresolved:raise RuntimeError('Missing bibliography: '+', '.join(sorted(unresolved)))
# Preserve explicit equation numbers while converting TeX to native OMML.
eqnums=[]
def equation(m):
 raw=m[1];tag=re.search(r'\\tag\{([^}]+)\}',raw)
 if not tag:raise ValueError('Untagged display equation')
 eqnums.append(tag[1]);formula=re.sub(r'\\tag\{[^}]+\}','',raw).strip()
 return '\n\n$$'+formula+'$$\n\nEQUATIONNUMBER'+str(len(eqnums))+'END\n\n'
source=re.sub(r'\$\$(.*?)\$\$',equation,source,flags=re.S)
front,chapters=source.split('# 第1章',1)
front=front+'\n\n'+(ASSETS/'abstract_en.md').read_text()+'\n'
source=front+'\n# 第1章'+chapters
intermediate=ASSETS/'body_native_math.docx'
pypandoc.convert_text(source,'docx',format='markdown+tex_math_dollars-smart',outputfile=str(intermediate),extra_args=['--reference-doc='+str(BASE)])
new=Document(intermediate)
newbody=[deepcopy(e) for e in new.element.body if e.tag!=qn('w:sectPr')]
figuremap=json.loads((ASSETS/('figure_map_final.json' if (ASSETS/'figure_map_final.json').exists() else 'figure_map_source.json')).read_text())
figs={x['figure']:x for x in figuremap['figures']}
stylemap={'Heading 1':'chapter','Heading 2':'h2','Heading 3':'h3','Heading 4':'h4','Body Text':'body','First Paragraph':'body','Normal':'body'}
newstyle={s.style_id:s.name for s in new.styles}
# Import relationships for all supported generated images/links.
relmap={}
for e in newbody:
 for n in e.iter():
  for attr in [qn('r:embed'),qn('r:link'),qn('r:id')]:
   rid=n.get(attr)
   if not rid:continue
   if rid not in relmap:
    rel=new.part.rels[rid]
    if rel.is_external:relmap[rid]=base.part.relate_to(rel.target_ref,rel.reltype,is_external=True)
    elif rel.reltype.endswith('/image'):relmap[rid]=base.part.get_or_add_image(BytesIO(rel.target_part.blob))[0]
    else:raise ValueError('Unexpected generated relationship: '+rel.reltype)
   n.set(attr,relmap[rid])
# Lists need numbering definitions from the converter.
numroot=base.part.numbering_part.element
newnumroot=new.part.numbering_part.element
abstract_offset=max([int(n.get(qn('w:abstractNumId'))) for n in numroot.findall(qn('w:abstractNum'))]+[-1])+1
num_offset=max([int(n.get(qn('w:numId'))) for n in numroot.findall(qn('w:num'))]+[0])
for n in newnumroot:
 n=deepcopy(n)
 if n.tag==qn('w:abstractNum'):n.set(qn('w:abstractNumId'),str(int(n.get(qn('w:abstractNumId')))+abstract_offset))
 elif n.tag==qn('w:num'):
  n.set(qn('w:numId'),str(int(n.get(qn('w:numId')))+num_offset))
  a=n.find(qn('w:abstractNumId'));a.set(qn('w:val'),str(int(a.get(qn('w:val')))+abstract_offset))
 numroot.append(n)
for e in newbody:
 for n in e.iter(qn('w:numId')):n.set(qn('w:val'),str(int(n.get(qn('w:val')))+num_offset))
# Generated bookmark ids and names are separated from preserved annex fields.
bookmark_offset=max([int(n.get(qn('w:id'))) for n in base.element.iter(qn('w:bookmarkStart'))]+[0])+100
for e in newbody:
 for n in e.iter():
  if n.tag in (qn('w:bookmarkStart'),qn('w:bookmarkEnd')):
   n.set(qn('w:id'),str(int(n.get(qn('w:id')))+bookmark_offset))
   if n.tag==qn('w:bookmarkStart'):n.set(qn('w:name'),'latest_'+n.get(qn('w:name')))
  if n.tag==qn('w:hyperlink') and n.get(qn('w:anchor')) and not n.get(qn('w:anchor')).startswith('latest_ref_'):n.set(qn('w:anchor'),'latest_'+n.get(qn('w:anchor')))

width=Emu(base.sections[2].page_width-base.sections[2].left_margin-base.sections[2].right_margin)
figcount=0;numbered=0;abstract='cn';chapter_index=1
converted=[]
for e in newbody:
 if e.tag==qn('w:tbl'):
  table=Table(e,base._body);table.autofit=False;table.alignment=WD_TABLE_ALIGNMENT.CENTER
  oldwidths=[int(g.get(qn('w:w'))) for g in e.findall('.//'+qn('w:gridCol'))]
  total=sum(oldwidths) or 1
  target=width.twips
  tw=table._tbl.tblPr.find(qn('w:tblW'))
  if tw is None:tw=OxmlElement('w:tblW');table._tbl.tblPr.append(tw)
  tw.set(qn('w:w'),str(target));tw.set(qn('w:type'),'dxa')
  for tcw in e.iter(qn('w:tcW')):
   if tcw.get(qn('w:type'))=='dxa':tcw.set(qn('w:w'),str(round(int(tcw.get(qn('w:w')))/total*target)))
  for g,ow in zip(e.findall('.//'+qn('w:gridCol')),oldwidths):g.set(qn('w:w'),str(round(ow/total*target)))
  borders=OxmlElement('w:tblBorders')
  for side in ['top','bottom','left','right','insideH','insideV']:
   n=OxmlElement('w:'+side);n.set(qn('w:val'),'single' if side in ['top','bottom'] else 'nil');n.set(qn('w:sz'),'8');borders.append(n)
  for existing in list(table._tbl.tblPr.findall(qn('w:tblBorders'))):table._tbl.tblPr.remove(existing)
  table._tbl.tblPr.append(borders)
  for rowidx,row in enumerate(table.rows):
   row._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
   if rowidx==0:row._tr.get_or_add_trPr().append(OxmlElement('w:tblHeader'))
   for cell in row.cells:
    for p in cell.paragraphs:
     p.style=base.styles['body'];p._p.get_or_add_pPr().get_or_add_ind().set(qn('w:firstLineChars'),'0');pf=p.paragraph_format;pf.first_line_indent=Pt(0);pf.left_indent=Pt(0);pf.line_spacing=Pt(15);pf.line_spacing_rule=WD_LINE_SPACING.AT_LEAST;pf.space_before=Pt(2);pf.space_after=Pt(2);pf.keep_with_next=False
     for run in p.runs:run.font.size=Pt(10.5)
  converted.append(e);continue
 if e.tag!=qn('w:p'):
  converted.append(e);continue
 p=Paragraph(e,base._body);value=text(e);name=newstyle.get(sty(e),'')
 if value.startswith('EQUATIONNUMBER'):
  idx=int(re.fullmatch(r'EQUATIONNUMBER(\d+)END',value)[1])-1
  formula=converted.pop()
  assert formula.find('.//'+qn('m:oMath')) is not None
  table=base.add_table(rows=1,cols=3);table.autofit=False;table.alignment=WD_TABLE_ALIGNMENT.CENTER
  widths=[Pt(30),width-Pt(60),Pt(30)]
  for c,col,cw in zip(table.rows[0].cells,table.columns,widths):
   c.width=col.width=cw;c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
   mar=OxmlElement('w:tcMar')
   for side in ['top','bottom','left','right']:
    n=OxmlElement('w:'+side);n.set(qn('w:w'),'0');n.set(qn('w:type'),'dxa');mar.append(n)
   c._tc.get_or_add_tcPr().append(mar)
   for pp in c.paragraphs:
    pp._p.get_or_add_pPr().get_or_add_ind().set(qn('w:firstLineChars'),'0')
    pp.style=base.styles['body'];pp.paragraph_format.first_line_indent=Pt(0);pp.paragraph_format.line_spacing_rule=WD_LINE_SPACING.AT_LEAST;pp.paragraph_format.line_spacing=Pt(20)
  c=table.cell(0,1);c._tc.remove(c.paragraphs[0]._p);c._tc.append(formula)
  fp=Paragraph(formula,base._body);fp._p.get_or_add_pPr().get_or_add_ind().set(qn('w:firstLineChars'),'0');fp.style=base.styles['body'];fp.paragraph_format.first_line_indent=Pt(0);fp.paragraph_format.alignment=WD_ALIGN_PARAGRAPH.CENTER;fp.paragraph_format.line_spacing_rule=WD_LINE_SPACING.AT_LEAST;fp.paragraph_format.line_spacing=Pt(20)
  right=table.cell(0,2).paragraphs[0];right.paragraph_format.alignment=WD_ALIGN_PARAGRAPH.RIGHT;right.add_run('('+eqnums[idx]+')').font.size=Pt(10.5)
  # Numeric labels remain fixed to the authoritative manuscript's explicit tags.
  b=OxmlElement('w:bookmarkStart');b.set(qn('w:id'),str(bookmark_offset+10000+idx));b.set(qn('w:name'),'latest_eq_'+eqnums[idx].replace('-','_'));right._p.insert(1,b)
  b=OxmlElement('w:bookmarkEnd');b.set(qn('w:id'),str(bookmark_offset+10000+idx));right._p.append(b)
  bd=OxmlElement('w:tblBorders')
  for side in ['top','bottom','left','right','insideH','insideV']:
   n=OxmlElement('w:'+side);n.set(qn('w:val'),'nil');bd.append(n)
  table._tbl.tblPr.append(bd);table.rows[0]._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
  table._tbl.getparent().remove(table._tbl)
  converted.append(table._tbl);numbered+=1;continue
 if value.startswith('FIGUREPLACEHOLDER'):
  label=value.removeprefix('FIGUREPLACEHOLDER');rec=figs[label];path=rec.get('source')
  if not path or not Path(path).exists():
   if MODE=='final':raise ValueError('Missing figure '+label)
   p.text=label+'（制图中）';p.style=base.styles['figure'];converted.append(e);continue
  p.text='';p.style=base.styles['figure'];pf=p.paragraph_format;pf.line_spacing_rule=WD_LINE_SPACING.SINGLE;pf.first_line_indent=Pt(0)
  from PIL import Image
  im=Image.open(path);ratio=im.height/im.width
  fw=min(width.inches,7.45/ratio)
  p.add_run().add_picture(path,width=Inches(fw));figcount+=1
 elif name.startswith('Heading'):
  if value==TITLE:p.style=base.styles['abstractcntitle']
  elif value=='摘要':p.style=base.styles['abstractcntitle'];p.paragraph_format.space_before=Pt(0);p.paragraph_format.space_after=Pt(12)
  elif value=='Abstract':abstract='en';p.style=base.styles['abstractentitle'];p.paragraph_format.page_break_before=True
  else:p.style=base.styles[stylemap.get(name,'h4')]
 elif re.match(r'^图\d+-\d+\s',value):p.style=base.styles['figurecaption']
 elif re.match(r'^表\d+-\d+\s',value):p.style=base.styles['tablecaption']
 elif abstract in ['cn','en'] and not any(text(x).startswith('第1章') for x in converted):
  p.style=base.styles['keywordscn' if abstract=='cn' else 'keywordsen'] if value.startswith(('关键词','Keywords:')) else base.styles['abstractcnbody' if abstract=='cn' else 'abstractenbody']
 else:p.style=base.styles['body']
 if e.find('.//'+qn('m:oMath')) is not None:
  p.paragraph_format.line_spacing_rule=WD_LINE_SPACING.AT_LEAST;p.paragraph_format.line_spacing=Pt(20)
 for link in e.iter(qn('w:hyperlink')):
  for run in link.iter(qn('w:r')):
   rp=run.find(qn('w:rPr'))
   if rp is None:rp=OxmlElement('w:rPr');run.insert(0,rp)
   color=OxmlElement('w:color');color.set(qn('w:val'),'000000');rp.append(color)
   u=OxmlElement('w:u');u.set(qn('w:val'),'none');rp.append(u)
 converted.append(e)

# Retain original declarations, notation table and annex objects in the same package.
for e in list(base.element.body):base.element.body.remove(e)
for e in deepcopy(old[:19]):base.element.body.append(e)
chapstarts=[i for i,e in enumerate(converted) if sty(e)=='chapter' and re.match(r'^第[1-7]章',text(e))]
assert len(chapstarts)==7
for e in converted[:chapstarts[0]]:base.element.body.append(e)
for e in deepcopy(old[34:38]):base.element.body.append(e)
toc=pnew('目录','toctitle');Paragraph(toc,base._body).paragraph_format.page_break_before=True;base.element.body.append(toc)
tocfield=pnew();field(tocfield,'TOC \\h \\z \\t "模板迁移 chapter,1,模板迁移 h2,2,模板迁移 h3,3,模板迁移 h4,4,模板迁移 bibliography_title,1"','目录将在Word中更新');base.element.body.append(tocfield)
sec=pnew();sec.find(qn('w:pPr')).append(deepcopy(old_sections[1]));base.element.body.append(sec)
for ci,start in enumerate(chapstarts):
 end=chapstarts[ci+1] if ci+1<len(chapstarts) else len(converted)
 for e in converted[start:end]:base.element.body.append(e)
 section=deepcopy(old_sections[ci+2]);pg=section.find(qn('w:pgNumType'))
 if ci>0 and pg is not None and qn('w:start') in pg.attrib:del pg.attrib[qn('w:start')]
 sec=pnew();sec.find(qn('w:pPr')).append(section);base.element.body.append(sec)
 # Update only the chapter text in its existing header.
 rid=section.find(qn('w:headerReference')).get(qn('r:id'))
 header=base.part.rels[rid].target_part.element
 ts=list(header.iter(qn('w:t')))
 if ts:
  ts[0].text='中南大学博士学位论文'
  if len(ts)>1:
   ts[1].text=text(converted[start])
   for t in ts[2:]:t.text=''
  else:ts[0].text='中南大学博士学位论文\t'+text(converted[start])
base.element.body.append(pnew('参考文献','bibliographytitle'))
for i,c in enumerate(order,1):
 rec=records.get(c)
 if rec:
  value=re.sub(r'^\[\d+\]\s*','',rec['bibliography_text'])
 else:value=c
 e=pnew('['+str(i)+']\t'+value,'bibliography');p=Paragraph(e,base._body);p.paragraph_format.left_indent=Pt(30);p.paragraph_format.first_line_indent=Pt(-30);p.paragraph_format.tab_stops.add_tab_stop(Pt(30))
 b=OxmlElement('w:bookmarkStart');b.set(qn('w:id'),str(bookmark_offset+11000+i));b.set(qn('w:name'),'latest_ref_'+str(i));e.insert(1,b)
 b=OxmlElement('w:bookmarkEnd');b.set(qn('w:id'),str(bookmark_offset+11000+i));e.append(b)
 base.element.body.append(e)
sec=pnew();sec.find(qn('w:pPr')).append(deepcopy(old_sections[9]));base.element.body.append(sec)
for e in annex:
 for n in e.iter(qn('w:t')):
  if n.text:n.text=re.sub(r'\[(66|69|80)\]',lambda m:'['+str(annex_numbers[int(m[1])])+']',n.text)
 for n in e.iter(qn('w:hyperlink')):
  a=n.get(qn('w:anchor'))
  for oldnum,newnum in annex_numbers.items():
   if a==old_refs[oldnum]:
    n.set(qn('w:anchor'),'latest_ref_'+str(newnum))
    for t in n.iter(qn('w:t')):
     if t.text==str(oldnum):t.text=str(newnum)
 base.element.body.append(e)
# The original appendix refers to the training-budget table by this bookmark.
budget=next(e for e in base.element.body if text(e).startswith('表6-5 ') and sty(e)=='tablecaption')
sys.path.insert(0,str(ROOT/'scripts'))
from thesis_crossrefs import bookmark_prefix
bookmark_prefix(Paragraph(budget,base._body),'表6-5','restruct_table_budget_model',bookmark_offset+12500)
base.settings.odd_and_even_pages_header_footer=True
base.core_properties.title=TITLE
base.core_properties.subject=TITLE
base.save(OUT)
# This detects missing or altered Chinese in conversion and blocks final delivery.
def han(s):return ''.join(re.findall(r'[\u3400-\u9fff]',s))
expected=SOURCE.read_text().split('\n# 附录F')[0]
expected=re.sub(r'^【图[^\n]*】\s*$','',expected,flags=re.M)
expected=han(expected)
actual=han(''.join(text(e) for e in converted if not text(e).endswith('（制图中）')))
if actual!=expected:
 from difflib import SequenceMatcher
 diffs=[]
 for tag,a,b,c,d in SequenceMatcher(None,expected,actual,autojunk=False).get_opcodes():
  if tag!='equal':diffs.append({'expected':expected[max(0,a-15):b+15],'actual':actual[max(0,c-15):d+15],'tag':tag})
 (ASSETS/'chinese_differences.json').write_text(json.dumps(diffs,ensure_ascii=False,indent=2))
 raise RuntimeError('Chinese text mismatch: '+str(diffs[:5]))
report={'output':str(OUT),'chinese_text_unchanged':True,'native_math':len(base.element.xpath('.//m:oMath')),'numbered_display_equations':numbered,'embedded_main_figures':figcount,'bibliography_count':len(order),'unresolved_citation_keys':sorted(unresolved),'annexes':'Original Word A-F','annex_number_map':annex_numbers,'word_finalized':False}
(ASSETS/'build_result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
(ASSETS/'reference_numbers.json').write_text(json.dumps({k:order.index(canon.get(k,k))+1 for k in matched if canon.get(k,k) in order},ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False,indent=2))
