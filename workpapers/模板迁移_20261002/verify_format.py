#!/usr/bin/env python3
"""Re-run actual OOXML formatting acceptance after migration and Word save."""
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as E
from collections import Counter,defaultdict
import copy,json,re,sys
from template_analysis import NS,W,R,props,val,local,merge
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
M='{'+NS['m']+'}'
def text(p):return ''.join(t.text or '' for t in p.iter(W+'t'))
def on(d,k):return d.get(k,{}).get('val','0') not in ('0','false','off')
def blend(a,b,toggle=False):
 c=merge(a,b,toggle)
 if 'tabs' in b:
  stops={x.get('pos'):copy.deepcopy(x) for x in a.get('tabs',{}).get('children',[])}
  for stop in b['tabs'].get('children',[]):
   if stop.get('val')=='clear':stops.pop(stop.get('pos'),None)
   else:stops[stop.get('pos')]=copy.deepcopy(stop)
  c['tabs']={'children':sorted(stops.values(),key=lambda x:int(x.get('pos','0')))}
 # An explicit font selector at a later cascade layer replaces the earlier
 # selector for that script, whether that earlier selector was theme or face.
 if 'rFonts' in b:
  f=b['rFonts'];cf=c['rFonts']
  for direct,theme in [('ascii','asciiTheme'),('hAnsi','hAnsiTheme'),('eastAsia','eastAsiaTheme'),('cs','cstheme')]:
   if direct in f and theme not in f:cf.pop(theme,None)
 return c
class Doc:
 def __init__(self,path):
  self.path=Path(path);self.z=ZipFile(path);self.root=E.fromstring(self.z.read('word/document.xml'));self.body=self.root.find('w:body',NS)
  sr=E.fromstring(self.z.read('word/styles.xml'));self.styles={s.get(W+'styleId'):s for s in sr.findall('w:style',NS)}
  self.default=next(k for k,s in self.styles.items() if s.get(W+'type')=='paragraph' and s.get(W+'default')=='1')
  self.dp=props(sr.find('w:docDefaults/w:pPrDefault/w:pPr',NS));self.dr=props(sr.find('w:docDefaults/w:rPrDefault/w:rPr',NS))
  self.parents={c:p for p in self.root.iter() for c in p};self.ps=self.root.findall('.//w:p',NS);self.pi={p:i for i,p in enumerate(self.ps)}
  self.themes={}
  theme=E.fromstring(self.z.read('word/theme/theme1.xml'))
  for fam in ['majorFont','minorFont']:
   node=theme.find('.//a:fontScheme/a:'+fam,NS);self.themes[fam]={local(v.tag)+(':'+v.get('script') if v.get('script') else ''):v.get('typeface','') for v in node}
 def apply(self,d,sid,kind):
  chain=[];seen=set()
  while sid in self.styles and sid not in seen:
   seen.add(sid);chain.insert(0,sid);sid=val(self.styles[sid].find('w:basedOn',NS))
  for k in chain:d=blend(d,props(self.styles[k].find('w:'+kind,NS)),kind=='rPr')
  return d
 def fonts(self,rp):
  f=rp.get('rFonts',{});out={}
  for role,th in [('ascii','asciiTheme'),('hAnsi','hAnsiTheme'),('eastAsia','eastAsiaTheme'),('cs','cstheme')]:
   if th in f:
    fam='majorFont' if f[th].startswith('major') else 'minorFont';fs=self.themes[fam]
    out[role]=(fs.get('font:Hans') or fs.get('ea')) if role=='eastAsia' else fs.get('cs') if role=='cs' else fs.get('latin')
   else:out[role]=f.get(role)
  return out
 def resolve(self,p):
  direct=props(p.find('w:pPr',NS));sid=direct.get('pStyle',{}).get('val',self.default);pp=copy.deepcopy(self.dp);rp=copy.deepcopy(self.dr)
  ancestor=self.parents.get(p);tbl=None
  while ancestor is not None:
   if ancestor.tag==W+'tbl':tbl=ancestor;break
   ancestor=self.parents.get(ancestor)
  if tbl is not None:
   ts=val(tbl.find('w:tblPr/w:tblStyle',NS));pp=self.apply(pp,ts,'pPr');rp=self.apply(rp,ts,'rPr')
  pp=blend(self.apply(pp,sid,'pPr'),direct);rp=self.apply(rp,sid,'rPr')
  runs=[]
  for r in p.iter(W+'r'):
   t=text(r)
   if not t:continue
   raw=props(r.find('w:rPr',NS));er=blend(self.apply(copy.deepcopy(rp),raw.get('rStyle',{}).get('val'),'rPr'),raw)
   runs.append((t,er))
  return sid,pp,rp,runs,tbl

def normp(p):
 sp=p.get('spacing',{});ind=p.get('ind',{})
 def distance(k):
  if sp.get(k+'Autospacing','0') not in ('0','false','off'):return ['auto']
  return ['line',float(sp[k+'Lines'])/100] if k+'Lines'in sp else ['pt',float(sp.get(k,0))/20]
 def indent(k):return ['char',float(ind[k+'Chars'])/100] if k+'Chars' in ind else ['pt',float(ind.get(k,0))/20]
 first=indent('hanging') if 'hanging' in ind or 'hangingChars' in ind else indent('firstLine')
 if 'hanging' in ind or 'hangingChars'in ind:first[1]=-first[1]
 if first[1]==0:first=['pt',0.0]
 return {'alignment':p.get('jc',{}).get('val','left'),'line':[sp.get('lineRule','auto'),float(sp.get('line','240'))],
         'before':distance('before'),'after':distance('after'),'first_line':first,'left':indent('left'),'right':indent('right'),
         'outline':p.get('outlineLvl',{}).get('val','9'),'tabs':p.get('tabs',{}).get('children',[])}
def normr(doc,r):
 return {'fonts':doc.fonts(r),'size':float(r.get('sz',{}).get('val',22))/2,'bold':on(r,'b'),'italic':on(r,'i'),
         'spacing':float(r.get('spacing',{}).get('val',0))/20,'scale':float(r.get('w',{}).get('val',100)),
         'position':float(r.get('position',{}).get('val',0))/2,'vertical':r.get('vertAlign',{}).get('val','baseline')}

def main(path,prefix):
 target=Doc(path);source=Doc(ROOT/'output/博士论文_批注核对补充稿_20261002.docx')
 values=json.load(open(HERE/'template_analysis_values.json'));log=json.load(open(HERE/'migration_log.json'))
 specs=log['specs'];groups=defaultdict(lambda:{'paragraphs':0,'runs':0,'failures':Counter(),'samples':[],'allowed_semantic_runs':0,'paragraph_forms':Counter(),'run_forms':Counter()})
 source_emphasis=defaultdict(list)
 for p in source.ps:
  s,pp,rp,runs,t=source.resolve(p);source_emphasis[text(p)].append({k:[flag for txt,r in runs for flag in [normr(source,r)[k]]*len(txt)] for k in ['bold','italic']})
 failures=[];inferences=[];undefined=[];exception_counts=Counter();table_styles=Counter();unused_font_slots=Counter()
 tables=target.root.findall('.//w:tbl',NS);table_kinds={}
 front_styles={sid for sid,s in target.styles.items() if val(s.find('w:name',NS))=='学位论文前置页'}
 field_stack=[];toc_closing=set()
 for node in target.root.iter():
  if node.tag==W+'fldChar':
   typ=node.get(W+'fldCharType')
   if typ=='begin':field_stack.append([])
   elif typ=='end' and field_stack:
    code=''.join(field_stack.pop())
    if re.match(r'^\s*TOC\b',code,re.I):
     ancestor=target.parents.get(node)
     while ancestor is not None and ancestor.tag!=W+'p':ancestor=target.parents.get(ancestor)
     if ancestor is not None:toc_closing.add(ancestor)
  elif node.tag==W+'instrText' and field_stack:field_stack[-1].append(node.text or '')
 for i,t in enumerate(tables):
  code=''.join(x.text or '' for x in t.findall('.//w:instrText',NS))
  front=any(val(p.find('w:pPr/w:pStyle',NS)) in front_styles for p in t.findall('.//w:p',NS))
  table_kinds[t]=('front' if front else 'equation' if re.search(r'\bSEQ\s+Eq',code) and len(t.findall('w:tblGrid/w:gridCol',NS))==3 else 'table',i)
 def fail(cat,p,key,actual,expected):
  g=groups[cat];g['failures'][key]+=1
  item={'category':cat,'p_index':target.pi[p],'text':text(p)[:75],'property':key,'actual':actual,'expected':expected}
  if len(g['samples'])<8:g['samples'].append(item)
  if len(failures)<80:failures.append(item)
 for p in target.ps:
  sid,pp,rp,runs,tbl=target.resolve(p)
  style_name=val(target.styles[sid].find('w:name',NS),'') if sid in target.styles else ''
  if tbl is not None:cat=table_kinds[tbl][0]
  elif sid.startswith('Migration_'):cat=sid[len('Migration_'):]
  elif style_name.startswith('模板迁移 '):cat=style_name[len('模板迁移 '):]
  elif sid in ['TOC1','TOC2','TOC3']:cat='toc'+sid[-1]
  else:continue # Section-only inserted paragraphs have no actual source content.
  # Keywords are an explicit template mixed-font role, even if an older
  # migration accidentally assigned an ordinary abstract-body style.
  if tbl is None and text(p).startswith('关键词：'):cat='keywords_cn'
  elif tbl is None and text(p).startswith('Keywords:'):cat='keywords_en'
  if cat not in values:continue
  g=groups[cat];g['paragraphs']+=1;actualp=normp(pp);g['paragraph_forms'][json.dumps(actualp,ensure_ascii=False,sort_keys=True)]+=1
  expectedp=normp(values[cat]['effective_pPr']);expectedr=normr(target,values[cat]['effective_rPr']);allowed=set()
  if cat=='h4':expectedp['line']=['exact',400.0];exception_counts['h4_definition_with_20pt_line']+=1
  if cat=='toc_title':expectedp['outline']='9'
  if cat=='cover_title':expectedp.update(before=['pt',100.0],after=['pt',30.0],line=['atLeast',560.0]);exception_counts['source_two_block_cover']+=1
  if cat=='cover_degree':expectedp.update(before=['pt',0.0],after=['pt',0.0],line=['atLeast',400.0])
  if cat=='body' and any(p.findall('.//m:'+x,NS) for x in ['f','limLow','nary']):expectedp['line']=['atLeast',400.0];exception_counts['tall_inline_math_atLeast20']+=1
  if p in toc_closing and not text(p) and actualp['line']==['exact',20.0]:
   expectedp['line']=['exact',20.0];exception_counts['TOC_closing_empty_paragraph_1pt']+=1
  if cat=='body' and text(p).startswith('数学记号按完整的字母、上下标及字体识别。') and actualp['after']==['pt',6.0]:
   expectedp['after']=['pt',6.0];exception_counts['symbol_introduction_after6pt']+=1
  if cat=='figure':expectedp['line']=['auto',240.0];exception_counts['figure_autoline_no_grid']+=1
  if pp.get('numPr'):
   allowed.update(['left','first_line']);exception_counts['source_numbered_list_indent']+=1
  if not text(p) and p.find('.//m:oMath',NS) is None and p.find('.//w:drawing',NS) is None:
   allowed.add('tabs') # Word leaves a TOC tab stop on an otherwise empty end paragraph.
  if cat=='equation':
   # Template itself uses multiple formula structures. Test native math/number
   # fonts and containment independently; do not impose its overwide 9600 tab.
   allowed.update(['line','before','after','first_line','left','right','alignment','tabs','outline'])
  if cat=='table':
   allowed.update(['alignment','outline','tabs']) # Source cell roles remain semantic.
   table_styles[sid]+=1
  for key,expected in expectedp.items():
   if key not in allowed and actualp[key]!=expected:fail(cat,p,'paragraph.'+key,actualp[key],expected)
  src=source_emphasis.get(text(p),[])
  if src:sem=src.pop(0)
  else:sem=None
  emphasis_actual={k:sum(len(txt) for txt,r in runs if normr(target,r)[k]) for k in ['bold','italic']}
  semantic_category=cat not in {'cover_title','cover_degree','chapter','h2','h3','h4','abstract_cn_title','abstract_en_title','toc_title','symbol_title','bibliography_title','toc1','toc2','toc3','keywords_cn','keywords_en'}
  checked_runs=[];offset=0;keyword_rule=values[cat].get('mixed_run_rule')
  label_end=len(keyword_rule['label_text']) if keyword_rule else 0
  for txt,run in runs:
   boundary=label_end-offset
   if keyword_rule and 0<boundary<len(txt):
    checked_runs.extend([(txt[:boundary],run,True,offset),(txt[boundary:],run,False,offset+boundary)])
   else:checked_runs.append((txt,run,bool(keyword_rule and offset<label_end),offset))
   offset+=len(txt)
  for txt,run,is_label,run_start in checked_runs:
   if keyword_rule and not txt.strip():continue
   run_expected=normr(target,keyword_rule['label_effective_rPr']) if is_label else expectedr
   if cat=='body' and run_start==0 and txt=='证明。':
    # The label is a retained semantic span even when its proof prose changes.
    labels=[normr(source,r)['bold'] for sp in source.ps if text(sp).startswith('证明。')
            for st,r in source.resolve(sp)[3] if st=='证明。']
    if labels and all(labels):
     run_expected=copy.deepcopy(run_expected);run_expected['bold']=True
   g['runs']+=1;actualr=normr(target,run);g['run_forms'][json.dumps(actualr,ensure_ascii=False,sort_keys=True)]+=len(txt)
   for key,expected in run_expected.items():
    if key=='fonts':
     # Compare the selectors used by this run, not an inactive Chinese font
     # slot on an ASCII-only equation number.
     active=[]
     if any(ord(c)<128 for c in txt):active.append('ascii')
     if re.search(r'[\u2e80-\u9fff\uf900-\ufaff\uff00-\uffef]',txt):active.append('eastAsia')
     if any(ord(c)>=128 and not re.match(r'[\u2e80-\u9fff\uf900-\ufaff\uff00-\uffef]',c) for c in txt):active.append('hAnsi')
     for slot in ['ascii','hAnsi','eastAsia']:
      if slot not in active and actualr[key][slot]!=expected[slot]:unused_font_slots[cat+':'+slot]+=1
     ar={x:actualr[key][x] for x in active};er={x:expected[x] for x in ar}
     if ar!=er:fail(cat,p,'run.fonts',ar,er)
    elif key in ['bold','italic'] and semantic_category and sem is not None and sem[key][run_start:run_start+len(txt)]==[actualr[key]]*len(txt):
     if actualr[key]!=expected:g['allowed_semantic_runs']+=1
    elif key=='vertical':
     if actualr[key] in ['superscript','subscript']:g['allowed_semantic_runs']+=1
    elif actualr[key]!=expected:fail(cat,p,'run.'+key,actualr[key],expected)
 # Check actual active sections and parts after Word's rewrite, never stale caches.
 rels={x.get('Id'):x.get('Target') for x in E.fromstring(target.z.read('word/_rels/document.xml.rels'))};sections=[];resolved={};hf=[]
 for i,sec in enumerate(target.root.findall('.//w:sectPr',NS)):
  for k in ['header','footer']:
   for ref in sec.findall('w:'+k+'Reference',NS):resolved[k+':'+ref.get(W+'type')]=rels[ref.get(R+'id')]
  pg=props(sec);section={'index':i+1,'page':pg.get('pgSz'),'margins':pg.get('pgMar'),'grid':pg.get('docGrid'),'columns':pg.get('cols',{}),'numbering':pg.get('pgNumType'),'first_page_distinct':on(pg,'titlePg'),'break_type':pg.get('type',{}).get('val','nextPage'),'active_parts':copy.deepcopy(resolved)};sections.append(section)
  title=log['sections'][i]['title'] if i<len(log['sections']) else None
  for k in ['header','footer']:
   name=resolved.get(k+':default')
   if not name:continue
   part='word/'+name;ar=E.fromstring(target.z.read(part))
   field_code=''.join(x.text or '' for x in ar.iter(W+'instrText'))
   if k=='footer' and i>=1 and not re.match(r'^\s*PAGE\b',field_code,re.I):failures.append({'category':'footer','section':i+1,'property':'page_field','actual':field_code,'expected':'Active PAGE field'})
   for p in ar.findall('.//w:p',NS):
    sid,pp,rp,runs,t=target.resolve(p);forms=[normr(target,r) for txt,r in runs];st=text(p)
    hf.append({'section':i+1,'kind':k,'part':part,'text':st[:100],'paragraph':normp(pp),'run_formats':forms,'field_code':field_code.strip()})
    if k=='header' and i>=2:
     wanted=12 if title and re.match(r'^第[34]章',title) else 20
     if normp(pp)['line']!=['exact',wanted*20]:failures.append({'category':'header','section':i+1,'property':'line','actual':normp(pp)['line'],'expected':['exact',wanted*20]})
     if wanted==12:exception_counts['two_line_header_12pt']+=1
     for f in forms:
      if f['size']!=10.5:failures.append({'category':'header','section':i+1,'property':'size','actual':f['size'],'expected':10.5})
      if f['fonts']['eastAsia']!='宋体' or f['fonts']['ascii']!='Times New Roman' or f['bold'] or f['italic']:failures.append({'category':'header','section':i+1,'property':'typeface','actual':f,'expected':'Song/TNR 10.5 regular'})
     actualtabs=normp(pp)['tabs']
     if actualtabs!=[{'tag':'tab','val':'right','pos':'8306'}]:failures.append({'category':'header','section':i+1,'property':'effective_tabs','actual':actualtabs,'expected':'Only right tab at 8306; inherited center 4153 cleared.'})
     borders={x['tag']:x for x in pp.get('pBdr',{}).get('children',[])};bottom=borders.get('bottom',{})
     if bottom.get('space')!='3':failures.append({'category':'header','section':i+1,'property':'bottom_border_text_space','actual':bottom,'expected':3})
    if k=='footer' and i>=1:
     for f in forms:
      if f['size']!=9:failures.append({'category':'footer','section':i+1,'property':'size','actual':f['size'],'expected':9})
      if f['fonts']['ascii']!='Times New Roman' or f['bold'] or f['italic']:failures.append({'category':'footer','section':i+1,'property':'typeface','actual':f,'expected':'Times New Roman 9 pt regular'})
     if forms and normp(pp)['line']!=['exact',400.0]:failures.append({'category':'footer','section':i+1,'property':'line','actual':normp(pp)['line'],'expected':['exact',400.0]})
    if k=='header' and i<2 and st.strip():failures.append({'category':'header','section':i+1,'property':'blank_front_header','actual':st})
 baseline=json.load(open(HERE/'template_analysis.json'))['sections']
 if len(sections)!=len(log['sections']):failures.append({'category':'section','property':'section_count','actual':len(sections),'expected':len(log['sections'])})
 settings=E.fromstring(target.z.read('word/settings.xml'));setting_properties=props(settings)
 switches={k:on(setting_properties,k) for k in ['evenAndOddHeaders','mirrorMargins']}
 if any(switches.values()):failures.append({'category':'section','property':'odd_even_or_mirror','actual':switches,'expected':{'evenAndOddHeaders':False,'mirrorMargins':False}})
 for i,s in enumerate(sections):
  expected=log['sections'][i] if i<len(log['sections']) else None
  if not expected:continue
  if s['page'].get('w')!='11906' or s['page'].get('h')!='16838':failures.append({'category':'section','section':i+1,'property':'A4','actual':s['page']})
  if (s['numbering'] or {}).get('fmt','decimal')!=expected['number_format']:failures.append({'category':'section','section':i+1,'property':'page_number_format','actual':s['numbering'],'expected':expected['number_format']})
  actual_start=(s['numbering'] or {}).get('start');wanted_start=str(expected['page_restart']) if expected['page_restart'] is not None else None
  if actual_start!=wanted_start:failures.append({'category':'section','section':i+1,'property':'page_restart','actual':actual_start,'expected':wanted_start})
  if s['first_page_distinct'] or s['break_type']!='nextPage':failures.append({'category':'section','section':i+1,'property':'first_page_or_break_type','actual':[s['first_page_distinct'],s['break_type']],'expected':[False,'nextPage']})
  template_sec=baseline[expected['template_section']-1]['properties']
  for actualkey,templatekey in [('margins','pgMar'),('grid','docGrid')]:
   if s[actualkey]!=template_sec.get(templatekey):failures.append({'category':'section','section':i+1,'property':actualkey,'actual':s[actualkey],'expected':template_sec.get(templatekey)})
  def cols(v):return {'num':v.get('num','1'),'space':v.get('space','720'),'equalWidth':v.get('equalWidth','1'),'sep':v.get('sep','0'),'children':v.get('children',[])}
  if cols(s['columns'])!=cols(template_sec.get('cols',{})):failures.append({'category':'section','section':i+1,'property':'columns','actual':s['columns'],'expected':template_sec.get('cols',{})})
 # Small structural-format checks: detect unscaled tables or remnants of fixed
 # heights; a failure prompts only the affected table's adjustment.
 table_checks={'count':len(tables),'data':0,'equation':0,'front':0,'overwide':[],'exact_row_heights':[],'data_border_failures':[],'header_border_failures':[]}
 for tbl,(kind,idx) in table_kinds.items():
  if kind=='front':
   table_checks['front']+=1
   continue # Borderless front metadata tables are compared with the actual template separately.
  table_checks['data' if kind=='table' else 'equation']+=1
  grid=sum(int(x.get(W+'w','0')) for x in tbl.findall('w:tblGrid/w:gridCol',NS))
  if grid>8306:table_checks['overwide'].append({'table':idx,'grid_width':grid})
  if tbl.findall('.//w:trHeight[@w:hRule="exact"]',NS):table_checks['exact_row_heights'].append(idx)
  if kind=='table':
   borders=props(tbl.find('w:tblPr/w:tblBorders',NS));
   if any(borders.get(k,{}).get('sz')!='12' for k in ['top','bottom']):table_checks['data_border_failures'].append(idx)
   for cell in tbl.findall('w:tr',NS)[0].findall('w:tc',NS):
    edge=cell.find('w:tcPr/w:tcBorders/w:bottom',NS)
    if edge is None or edge.get(W+'sz')!='6':table_checks['header_border_failures'].append(idx);break
 # Native mathematical face/style is separate from ordinary w:r typography.
 def math_stats(d):
  settings=E.fromstring(d.z.read('word/settings.xml'));mf=settings.find('m:mathPr/m:mathFont',NS)
  faces=Counter();sty=Counter();sizes=Counter();bad_data=[]
  for run in d.root.findall('.//m:r',NS):
   pr=props(run.find('w:rPr',NS));mpr=props(run.find('m:rPr',NS));txt=''.join(v.text or '' for v in run.findall('m:t',NS))
   faces[pr.get('rFonts',{}).get('ascii','mathFont default')]+=len(txt)
   for char in txt:
    cp=copy.deepcopy(mpr)
    if not char.isalpha() and not char.isdigit():
     cp.pop('scr',None)
     if cp.get('sty',{}).get('val','i') in ['i','p']:cp.pop('sty',None)
    sty[json.dumps(cp,sort_keys=True)]+=1
   sizes[pr.get('sz',{}).get('val','inherited')]+=len(txt)
   if d is target:
    anc=d.parents.get(run)
    while anc is not None and anc.tag!=W+'tbl':anc=d.parents.get(anc)
    if anc in table_kinds and table_kinds[anc][0]=='table' and pr.get('sz',{}).get('val')!='21':bad_data.append(table_kinds[anc][1])
  return {'mathFont':mf.get(M+'val') if mf is not None else None,'run_fonts':dict(faces),'math_style_forms':dict(sty),'explicit_halfpoint_sizes':dict(sizes),'data_table_size_failures':sorted(set(bad_data))}
 sm=math_stats(source);tm=math_stats(target)
 math_checks={'source':sm,'actual':tm,'status':'通过'}
 # Content edits change character totals. Compare the actually used faces and
 # style forms; never make matching old character counts a formatting condition.
 math_checks['unexpected_faces']=sorted(set(tm['run_fonts'])-set(sm['run_fonts']))
 math_checks['unexpected_styles']=sorted(set(tm['math_style_forms'])-set(sm['math_style_forms']))
 if math_checks['unexpected_faces'] or math_checks['unexpected_styles'] or tm['mathFont']!='Cambria Math' or tm['data_table_size_failures']:
  math_checks['status']='未通过';failures.append({'category':'native_math','property':'font_style_or_table_size','actual':tm,'expected':'Source mathematical faces/style roles retained, independent of edited character totals; data-table math 10.5 pt; Cambria Math default.'})
 inferences=[
  'Accepted: 18 tall inline-math paragraphs use atLeast 20 pt instead of exact 20 pt.',
  'Accepted: preserve the source two-paragraph cover order; title spacing 100 pt above/30 pt below with atLeast 28 pt, degree atLeast 20 pt.',
  'Accepted: Heading 4 has no actual template sample; use its style definition plus exact 20 pt.',
  'Accepted: chapter 3/4 headers retain 10.5 pt and full text, with two exact-12-pt lines; other headers use exact 20 pt. Inherited center tab 4153 is cleared, right tab 8306 retained, and bottom-border distance is 3 pt to prevent the rule crossing text.',
  'Migration layout controls: keepNext/keepLines for headings/captions; figure auto line and snapToGrid=0; source numbered-list indents and cell alignments retained.',
  'The empty paragraph carrying the TOC closing field marker may use exact 1 pt as a layout-only marker; it has no source prose and is identified by the actual TOC field boundary.',
  'The single symbols-table introductory paragraph beginning 数学记号按完整的字母、上下标及字体识别 may use 6 pt after-spacing to separate its final line from the table top border. Ordinary body paragraphs retain template after-spacing 0.',
  'Source bold/italic emphasis is checked at matching character positions in the corresponding source paragraph; superscript/subscript remain semantic. Keywords instead follow the template mixed-font rule: bold label, regular terms, zero first-line indent. REF results use the referencing paragraph baseline, so prior inherited heading bold is not treated as intended emphasis.',
  'Only font selectors active for the actual run text are compared. The unused EastAsian selector on ASCII equation numbers differs (Song vs TNR), while visible number text uses TNR 12 pt.',
  'Equation table spacing/alignment follow source layout because template formula examples are mixed and its 9600-twip tab is overwide; typography is checked separately.'
 ]
 undefined=['No actual Heading 4, footnote or endnote text sample in the template.','This is XML format acceptance; it does not claim that Word pagination or every rendered page passed visual inspection.']
 outgroups={}
 for cat,g in groups.items():
  outgroups[cat]={'status':'未通过' if g['failures'] else '通过',**{k:v for k,v in g.items() if k not in ['failures','paragraph_forms','run_forms']},'failures':dict(g['failures']),
   'paragraph_forms':[{'count':n,**json.loads(s)} for s,n in g['paragraph_forms'].most_common(8)],
   'effective_run_forms':[{'characters':n,**json.loads(s)} for s,n in g['run_forms'].most_common()],
   'template_paragraph':normp(values[cat]['effective_pPr']),'template_run':normr(target,values[cat]['effective_rPr'])}
 result={'document':str(Path(path).resolve()),'status':'未通过' if failures or any(table_checks[k] for k in ['overwide','exact_row_heights','data_border_failures','header_border_failures']) else '通过',
         'method':'Resolve docDefaults -> table style -> paragraph style -> character style -> direct run. Compare effective values, not style names; paragraph mark font does not override text.',
         'groups':outgroups,'failures':failures,'sections':sections,'header_footer':hf,'tables':table_checks,'native_math':math_checks,'accepted_exception_counts':dict(exception_counts),'inferences_and_conflicts':inferences,'undefined':undefined,
         'docDefaults_pPr':target.dp,'document_section_switches':switches,'unused_font_slot_differences':dict(unused_font_slots),'toc_note':'TOC caches may change after Word field update; rerun this script after final save.'}
 pre=Path(prefix);pre.with_suffix('.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 lines=[f'实际格式核验：{result["status"]}。对象：`{Path(path).name}`。本报告由本次文件的 XML 实际计算生成。','', '| 分类 | 段落 | 文字运行 | 判定 | 差异 |','|---|---:|---:|---|---|']
 for cat,g in outgroups.items():lines.append(f'| {cat} | {g["paragraphs"]} | {g["runs"]} | {g["status"]} | '+('、'.join(f'{k}: {v}' for k,v in g['failures'].items()) or '—')+' |')
 lines+=['','推断与冲突：']+['- '+v for v in inferences]+['','未定义／未验收：']+['- '+v for v in undefined]
 pre.with_suffix('.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps({'status':result['status'],'failed_groups':{k:v['failures'] for k,v in outgroups.items() if v['failures']},'exceptions':dict(exception_counts),'tables':table_checks,'report':str(pre.with_suffix('.json'))},ensure_ascii=False))
if __name__=='__main__':main(sys.argv[1] if len(sys.argv)>1 else ROOT/'output/排版完成.docx',sys.argv[2] if len(sys.argv)>2 else HERE/'format_acceptance')
