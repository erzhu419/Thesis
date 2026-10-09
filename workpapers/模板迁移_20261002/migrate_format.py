from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from copy import deepcopy
from collections import Counter
from lxml import etree as E
import json,re
ROOT=Path(__file__).resolve().parents[2]; WORK=Path(__file__).resolve().parent
SOURCE=ROOT/'output/博士论文_批注核对补充稿_20261002.docx'
TEMPLATE=WORK/'template_readonly.docx'; OUTPUT=ROOT/'output/排版完成.docx'
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships','m':'http://schemas.openxmlformats.org/officeDocument/2006/math','wp':'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing','a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
W='{'+NS['w']+'}';R='{'+NS['r']+'}';REL='{http://schemas.openxmlformats.org/package/2006/relationships}'; CT='{http://schemas.openxmlformats.org/package/2006/content-types}'
def elt(name,**attrs):
 e=E.Element(W+name)
 for k,v in attrs.items():e.set(W+k,str(v))
 return e
def ensure(parent,name):
 x=parent.find('w:'+name,NS)
 if x is None:x=elt(name);parent.append(x)
 return x
def replace(parent,name,node=None,**attrs):
 for c in parent.findall('w:'+name,NS):parent.remove(c)
 parent.append(deepcopy(node) if node is not None else elt(name,**attrs))
def value(p,path,default=None):
 e=p.find(path,NS);return e.get(W+'val',default) if e is not None else default
def text(p):return ''.join(p.xpath('.//w:t/text()',namespaces=NS))
def props_xml(name,d):
 e=elt(name)
 for tag,attrs in d.items():
  if tag in ['pStyle','rStyle','sectPr','numPr']:continue
  child=elt(tag)
  for k,v in attrs.items():
   if k=='children':
    for entry in v:child.append(elt(entry['tag'],**{a:b for a,b in entry.items() if a!='tag'}))
   else:child.set(W+k,str(v))
  e.append(child)
 return e
sz=ZipFile(SOURCE); tz=ZipFile(TEMPLATE)
parts={n:sz.read(n) for n in sz.namelist()}
doc=E.fromstring(parts['word/document.xml']);body=doc.find('w:body',NS)
tdoc=E.fromstring(tz.read('word/document.xml'));tbody=tdoc.find('w:body',NS)
styles=E.fromstring(parts['word/styles.xml']); tstyles=E.fromstring(tz.read('word/styles.xml'))
analysis=json.load(open(WORK/'template_analysis.json'))
# Keep all original character/list styles and import a distinct template style family.
for st in tstyles.findall('w:style',NS):
 st=deepcopy(st);st.set(W+'styleId','GTY_'+st.get(W+'styleId'));st.attrib.pop(W+'default',None)
 nm=st.find('w:name',NS)
 if nm is not None:nm.set(W+'val','GTY '+nm.get(W+'val'))
 for child in st:
  if child.tag in {W+'basedOn',W+'next',W+'link'}:child.set(W+'val','GTY_'+child.get(W+'val'))
 styles.append(st)
# Paragraph defaults must also match: source otherwise contributes after=10pt/1.15 lines.
sdef=styles.find('w:docDefaults',NS);tdef=tstyles.find('w:docDefaults',NS)
for old in sdef.findall('w:pPrDefault',NS):sdef.remove(old)
sdef.append(deepcopy(tdef.find('w:pPrDefault',NS)))
# Original document defaults retained for original semantic character styles;
# mapped paragraph styles below are explicit and have no inherited source layout.
values=json.load(open(WORK/'template_analysis_values.json'))
specs={k:{'p':deepcopy(v['effective_pPr']),'r':deepcopy(v['effective_rPr']),'template_location':v['location'],'evidence':v['evidence']} for k,v in values.items() if k not in ['_notes','table','equation']}
specs['h4']['p']['spacing'].update(line='400',lineRule='exact')
# Remove incidental sample attributes, make unchanged implicit defaults explicit.
for k,v in specs.items():
 p,r=v['p'],v['r'];p.pop('pStyle',None);p.pop('sectPr',None);p.pop('numPr',None);r.pop('rStyle',None)
 for tag in ['vertAlign','position','highlight','shd']:r.pop(tag,None)
 r.setdefault('b',{'val':'0'});r.setdefault('i',{'val':'0'})
 r.setdefault('color',{'val':'auto'});r.setdefault('spacing',{'val':'0'})
 if 'sz' in r:r['szCs']=deepcopy(r['sz'])
 # A paragraph mark's font does not override content runs.
 if k.startswith('toc') and k!='toc_title':
  r['sz']=r['szCs']={'val':'24'};r['b']={'val':'0'}
  r['rFonts']={'ascii':'Times New Roman','hAnsi':'Times New Roman','eastAsia':'黑体' if k=='toc1' else '宋体','cs':'Times New Roman'}
  p['tabs']={'children':[{'tag':'tab','val':'right','pos':'8296','leader':'dot'}]}
 p.setdefault('ind',{'firstLine':'0','firstLineChars':'0'})
 p.setdefault('jc',{'val':'left'})
 if k in ['chapter','h2','h3','h4','abstract_cn_title','abstract_en_title','symbol_title','bibliography_title','toc_title']:
  p['keepNext']={'val':'1'};p['keepLines']={'val':'1'};p['pageBreakBefore']={'val':'0'}
 if k=='toc_title':p['outlineLvl']={'val':'9'}
 if k=='figure':
  p['spacing'].update(line='240',lineRule='auto');p['keepNext']={'val':'1'};p['keepLines']={'val':'1'};p['snapToGrid']={'val':'0'}
 if k=='table_caption':p['keepNext']={'val':'1'};p['keepLines']={'val':'1'}
 if k=='figure_caption':p['keepLines']={'val':'1'}
# Store semantic categories in explicit paragraph styles. TOC uses the built-ins.
style_ids={k:('TOC'+k[-1] if re.fullmatch('toc[123]',k) else 'Migration_'+k) for k in specs}
for k,v in specs.items():
 sid=style_ids[k];old=styles.find('w:style[@w:styleId="'+sid+'"]',NS)
 if old is not None:styles.remove(old)
 st=elt('style',type='paragraph',styleId=sid);st.append(elt('name',val=('toc '+k[-1] if re.fullmatch('toc[123]',k) else '模板迁移 '+k)));st.append(elt('qFormat'))
 st.append(props_xml('pPr',v['p']));st.append(props_xml('rPr',v['r']));styles.append(st)
count=Counter();mapping=[];exception=[]

def format_p(p,category):
 old=p.find('w:pPr',NS);new=elt('pPr');new.append(elt('pStyle',val=style_ids[category]))
 if old is not None:
  for node in old:
   if node.tag in {W+'numPr',W+'sectPr'}:new.append(deepcopy(node))
  p.remove(old)
 p.insert(0,new)
 # Only typography is replaced; semantic emphasis, links, superscripts remain.
 rbase=specs[category]['r']; heading=category in {'cover_title','cover_degree','chapter','h2','h3','h4','abstract_cn_title','abstract_en_title','toc_title','symbol_title','bibliography_title'}
 for run in p.iter(W+'r'):
  if any(a.tag.startswith('{'+NS['m']+'}') for a in run.iterancestors()):continue
  rp=run.find('w:rPr',NS)
  if rp is None:rp=elt('rPr');run.insert(0,rp)
  for tag in ['rFonts','sz','szCs','color','spacing','kern']:
   if tag in rbase:replace(rp,tag,props_xml('rPr',{tag:rbase[tag]})[0])
   elif tag in ['spacing','kern']:
    for x in rp.findall('w:'+tag,NS):rp.remove(x)
  if heading:
   for tag in ['b','bCs','i','iCs']:
    replace(rp,tag,val=rbase.get(tag,rbase.get(tag[0],{'val':'0'})).get('val','0'))
 if category in ['keywords_cn','keywords_en']:
  for run in p.findall('w:r',NS):
   if text(run) in ['关键词：','Keywords:']:
    rp=ensure(run,'rPr');replace(rp,'b',val='1')
    if category=='keywords_cn':replace(rp,'rFonts',ascii='黑体',eastAsia='黑体',hAnsi='Times New Roman',cs='Times New Roman')
 if category=='body' and p.xpath('.//m:f|.//m:limLow|.//m:nary',namespaces=NS):
  replace(new,'spacing',line='400',lineRule='atLeast');exception.append({'kind':'tall_inline_math','text':text(p)[:80]})
 count[category]+=1

original=list(body);chapter_starts=[]
for i,p in enumerate(original):
 if p.tag!=W+'p':continue
 sid=value(p,'w:pPr/w:pStyle','');txt=text(p)
 if i==0:cat='cover_title'
 elif i==1:cat='cover_degree'
 elif i==3:cat='abstract_cn_title'
 elif i==4:cat='abstract_cn_first'
 elif i==11:cat='keywords_cn'
 elif 5<=i<=10:cat='abstract_cn_body'
 elif i==12:cat='abstract_en_title'
 elif i==13:cat='abstract_en_first'
 elif i==20:cat='keywords_en'
 elif 14<=i<=19:cat='abstract_en_body'
 elif i==21:cat='toc_title'
 elif sid in ['TOC1','TOC2','TOC3']:cat='toc'+sid[-1]
 elif i==347:cat='symbol_title'
 elif sid=='1':
  cat='bibliography_title' if txt=='参考文献' else 'chapter';chapter_starts.append((p,txt))
 elif sid in ['21','31','4']:cat={'21':'h2','31':'h3','4':'h4'}[sid]
 elif sid=='14':cat='bibliography'
 elif sid=='ThesisFigureCaption' or re.match(r'^图[\dA-F]+-\d+\s',txt):cat='figure_caption'
 elif re.match(r'^表[\dA-F]+-\d+\s',txt):cat='table_caption'
 elif p.find('.//w:drawing',NS) is not None:cat='figure'
 else:cat='body'
 format_p(p,cat);mapping.append({'body_index':i,'category':cat,'style':style_ids[cat],'text':txt[:100]})
 # Preserve current abstract keywords semantic bold, disable abstract list first indent to keep list labels.
 if p.find('w:pPr/w:numPr',NS) is not None:replace(p.find('w:pPr',NS),'ind',left='420',hanging='420')
 if i in [12,21,347]:replace(p.find('w:pPr',NS),'pageBreakBefore',val='1')
 # No explicit source layout page-break nodes should force a second break.
 if sid=='1' and i>=352:
  for br in p.xpath('.//w:br[@w:type="page"]',namespaces=NS):br.getparent().remove(br)
# Retain source cover order, use whitespace only to place its two existing blocks on a cover.
replace(original[0].find('w:pPr',NS),'spacing',before='2000',after='600',line='560',lineRule='atLeast')
replace(original[1].find('w:pPr',NS),'spacing',before='0',after='0',line='400',lineRule='atLeast')
# Restore protected tables with their own actual template mapping.
from migrate_tables import format_tables
table_report=format_tables(doc,tdoc,NS)
# Fit existing images to physical writing area without changing crop, media, or aspect ratio.
scaled=[];maxcx=int(8306*635);maxcy=int(610*12700)
for draw in doc.findall('.//w:drawing',NS):
 for ext in draw.findall('.//wp:extent',NS):
  cx,cy=int(ext.get('cx')),int(ext.get('cy'));scale=min(1,maxcx/cx,maxcy/cy)
  if scale<1:
   nx,ny=round(cx*scale),round(cy*scale);ext.set('cx',str(nx));ext.set('cy',str(ny))
   for ae in draw.findall('.//a:xfrm/a:ext',NS):ae.set('cx',str(nx));ae.set('cy',str(ny))
   scaled.append({'before':[cx,cy],'after':[nx,ny]})
# Rebuild section boundaries only, retaining all content nodes and order.
for sec in doc.findall('.//w:sectPr',NS):sec.getparent().remove(sec)
rels=E.fromstring(parts['word/_rels/document.xml.rels']);types=E.fromstring(parts['[Content_Types].xml'])
serial=100;newparts={}
def add_part(kind,root):
 global serial
 serial+=1;filename=kind+str(serial)+'.xml';rid='rIdMigration'+str(serial)
 newparts['word/'+filename]=E.tostring(root,encoding='UTF-8',xml_declaration=True,standalone=True)
 re=E.Element(REL+'Relationship');re.set('Id',rid);re.set('Type',NS['r']+'/'+kind);re.set('Target',filename);rels.append(re)
 ce=E.Element(CT+'Override');ce.set('PartName','/word/'+filename);ce.set('ContentType','application/vnd.openxmlformats-officedocument.wordprocessingml.'+kind+'+xml');types.append(ce)
 return rid

def remap_styles(root):
 for x in root.xpath('.//w:pStyle|.//w:rStyle',namespaces=NS):x.set(W+'val','GTY_'+x.get(W+'val'))
 return root
blankheader=add_part('header',remap_styles(E.fromstring(tz.read('word/header2.xml'))))
blankfooter=add_part('footer',remap_styles(E.fromstring(tz.read('word/footer2.xml'))))
frontfooter=add_part('footer',remap_styles(E.fromstring(tz.read('word/footer4.xml'))))
normalfooter=add_part('footer',remap_styles(E.fromstring(tz.read('word/footer5.xml'))))
# Template's fixed school title and current source title; long title uses two compact header lines.
def header(title):
 root=remap_styles(E.fromstring(tz.read('word/header4.xml')));p=root.find('w:p',NS);pr=p.find('w:pPr',NS)
 for c in list(p):
  if c.tag!=W+'pPr':p.remove(c)
 replace(pr,'tabs',node=props_xml('pPr',{'tabs':{'children':[{'tag':'tab','val':'right','pos':'8306'}]}})[0])
 replace(pr,'jc',val='left');replace(pr,'ind',firstLine='0',firstLineChars='0')
 wide=len(title)>25
 if wide:replace(pr,'spacing',before='0',after='0',line='240',lineRule='exact')
 def run(s=None,control=None):
  r=elt('r');rp=elt('rPr');rp.append(elt('rFonts',ascii='Times New Roman',hAnsi='Times New Roman',eastAsia='宋体'));rp.append(elt('sz',val='21'));r.append(rp)
  if control:r.append(elt(control))
  if s is not None:t=elt('t');t.text=s;t.set('{http://www.w3.org/XML/1998/namespace}space','preserve');r.append(t)
  p.append(r)
 run('中南大学博士学位论文')
 if wide:run(control='br')
 run(control='tab');run(title)
 if wide:exception.append({'kind':'two_line_header','title':title,'line_pt':12,'font_pt':10.5})
 return add_part('header',root)
sections=[]
def section_props(template_index,title,roman=False,start=None,cover=False):
 s=deepcopy(tdoc.findall('.//w:sectPr',NS)[template_index])
 for c in list(s):
  if c.tag in {W+'headerReference',W+'footerReference',W+'pgNumType',W+'titlePg',W+'type'}:s.remove(c)
 h=elt('headerReference',type='default');h.set(R+'id',blankheader if cover or roman else header(title));s.insert(0,h)
 f=elt('footerReference',type='default');f.set(R+'id',blankfooter if cover else frontfooter if roman else normalfooter);s.insert(1,f)
 s.insert(2,elt('type',val='nextPage'))
 pg=elt('pgNumType',fmt='upperRoman' if roman else 'decimal')
 if start is not None:pg.set(W+'start',str(start))
 s.append(pg)
 return s
# Cover / front matter / each source chapter and appendix; references stay in source position.
starts=[(original[0],'封面',0),(original[3],'摘要',1)]+[(p,t,2 if t!='参考文献' else 9) for p,t in chapter_starts]
for idx,(p,title,ti) in enumerate(starts):
 s=section_props(ti,title,roman=idx==1,start=1 if idx in [1,2] else None,cover=idx==0)
 if idx+1<len(starts):
  nxt=starts[idx+1][0];loc=list(body).index(nxt);previous=body[loc-1]
  if previous.tag==W+'p' and not text(previous) and previous.find('.//m:oMath',NS) is None:boundary=previous
  else:
   boundary=elt('p');pp=elt('pPr');pp.append(elt('spacing',before='0',after='0',line='20',lineRule='exact'));pp.append(elt('rPr'));boundary.append(pp);body.insert(loc,boundary)
  ensure(boundary,'pPr').append(s)
 else:body.append(s)
 sections.append({'section':idx+1,'title':title,'template_section':ti+1,'page_restart':1 if idx in [1,2] else None,'number_format':'upperRoman' if idx==1 else 'decimal'})
settings=E.fromstring(parts['word/settings.xml'])
for k in ['evenAndOddHeaders','updateFields']:
 for x in settings.findall('w:'+k,NS):settings.remove(x)
# Keep mathematical OOXML and original numbering definitions unchanged.
for name,root in [('word/document.xml',doc),('word/styles.xml',styles),('word/settings.xml',settings),('word/_rels/document.xml.rels',rels),('[Content_Types].xml',types)]:parts[name]=E.tostring(root,encoding='UTF-8',xml_declaration=True,standalone=True)
parts.update(newparts)
with ZipFile(OUTPUT,'w',ZIP_DEFLATED) as out:
 for name,data in parts.items():out.writestr(name,data)
report={'source':str(SOURCE),'output':str(OUTPUT),'method':'Preserve source OOXML object/content topology; map resolved template formatting; rebuild correct source chapter boundaries.','mapped_counts':dict(count),'mapping':mapping,'specs':specs,'tables':table_report,'resized_images':scaled,'sections':sections,'layout_exceptions':exception}
(WORK/'migration_log.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({'output':str(OUTPUT),'categories':dict(count),'sections':len(sections),'images_resized':len(scaled),'layout_exceptions':len(exception)},ensure_ascii=False))
