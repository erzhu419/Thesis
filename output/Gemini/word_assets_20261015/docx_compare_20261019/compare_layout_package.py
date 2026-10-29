from pathlib import Path, PurePosixPath
from zipfile import ZipFile
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from lxml import etree as E
import json,re,posixpath

T=Path(__file__).resolve().parent
BASE=Path('/mnt/d/Paper/Thesis/Gemini/重构论文_导言小结同步_20261018.docx')
CAND=Path('/home/erzhu419/mine_code/Thesis/output/Gemini/调整完格式的.docx')
N={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
W='{'+N['w']+'}'
def local(k):return E.QName(k).localname
def tx(p):return ''.join(p.xpath('.//w:t/text()|.//m:t/text()',namespaces=N))
def clean(s):return re.sub(r'\s+','',s)
def node(e,sort=False):
 if e is None:return None
 ch=[node(c,sort)for c in e if local(c.tag)not in('rsid',)]
 if sort:ch=sorted(ch,key=lambda x:json.dumps(x,ensure_ascii=False))
 return {'tag':local(e.tag),'attrs':{local(k):v for k,v in sorted(e.attrib.items())if not local(k).startswith('rsid')},'text':e.text if len(e)==0 and e.text else '', 'children':ch}
def props(e,styles=None):
 if e is None:return {}
 out={}
 for c in e:
  name=local(c.tag)
  if name in('rsid','sectPr'):continue
  val=node(c,True)
  if name in('pStyle','rStyle','basedOn','next','link')and styles:
   v=c.get(W+'val');val['attrs']['val']=styles.get(v,{}).get('name',v)
  out[name]=val
 return out
def diff(a,b):return {k:{'before':a.get(k),'after':b.get(k)}for k in sorted(set(a)|set(b))if a.get(k)!=b.get(k)}
def flat(e,prefix=''):
 if e is None:return {}
 out={}
 for c in e:
  k=prefix+local(c.tag)
  if k in('rsid','sectPr'):continue
  if len(c):out.update(flat(c,k+'/'))
  elif c.attrib:
   out.update({k+'/'+local(a):v for a,v in c.attrib.items()})
  else:out[k] = True
 return out
def chain(styles,sid):
 out=[];seen=set()
 while sid in styles and sid not in seen:
  seen.add(sid);s=styles[sid]['xml'];out.append(s);b=s.find('w:basedOn',N);sid=b.get(W+'val')if b is not None else None
 return list(reversed(out))
def effective_p(p,styles):
 ss=p.find('w:pPr/w:pStyle',N);sid=ss.get(W+'val')if ss is not None else'a0';v={'keepNext':False,'keepLines':False,'pageBreakBefore':False}
 bkeys=['keepNext','keepLines','pageBreakBefore','widowControl','wordWrap','snapToGrid','adjustRightInd']
 def apply(e):
  f=flat(e)
  for k in bkeys:
   if k in f or k+'/val'in f:v[k]=f.get(k,f.get(k+'/val'))not in('0','false','off')
   f.pop(k,None);f.pop(k+'/val',None)
  v.update(f)
 for s in chain(styles,sid):apply(s.find('w:pPr',N))
 apply(p.find('w:pPr',N));return{k:a for k,a in v.items()if not k.startswith(('pStyle','rPr','numPr'))}
def effective_runs(p,styles,root):
 ss=p.find('w:pPr/w:pStyle',N);sid=ss.get(W+'val')if ss is not None else'a0';base=flat(root.find('w:docDefaults/w:rPrDefault/w:rPr',N));bkeys=['b','bCs','i','iCs','caps','smallCaps','strike','dstrike','vanish'];
 for k in bkeys:base[k]=False
 def apply(v,e,styled=False):
  f=flat(e)
  for k in bkeys:
   if k in f or k+'/val'in f:
    yes=f.get(k, f.get(k+'/val'))not in('0','false','off')
    v[k]=(not v.get(k,False))if styled and yes else(v.get(k,False)if styled else yes)
   f.pop(k,None);f.pop(k+'/val',None)
  # A direct font name supersedes lower-priority theme for that script.
  for key,theme in [('ascii','asciiTheme'),('hAnsi','hAnsiTheme'),('eastAsia','eastAsiaTheme'),('cs','cstheme')]:
   if 'rFonts/'+key in f and 'rFonts/'+theme not in f:v.pop('rFonts/'+theme,None)
  v.update(f)
 for s in chain(styles,sid):apply(base,s.find('w:rPr',N),True)
 chars=[]
 for r in p.xpath('.//w:r[not(ancestor::m:oMath)]',namespaces=N):
  v=dict(base);c=r.find('w:rPr/w:rStyle',N)
  if c is not None:
   for s in chain(styles,c.get(W+'val')):apply(v,s.find('w:rPr',N),True)
  apply(v,r.find('w:rPr',N));v={k:x for k,x in v.items()if k in('b','i','bCs','iCs','sz/val','szCs/val','vertAlign/val','spacing/val')or k.startswith('rFonts/')and not k.endswith('hint')}
  for char in ''.join(r.xpath('./w:t/text()',namespaces=N)):
   if not char.isspace():chars.append((char,v))
 return chars
def label(p,i,styles):
 s=p.find('w:pPr/w:pStyle',N);sid=s.get(W+'val')if s is not None else'a0'
 return {'paragraph_index_0based':i,'style_id':sid,'style_name':styles.get(sid,{}).get('name',sid),'text':tx(p)[:160]}
def get_styles(z):
 r=E.fromstring(z.read('word/styles.xml'));out={}
 for s in r.findall('w:style',N):
  name=s.find('w:name',N);sid=s.get(W+'styleId')
  out[sid]={'id':sid,'type':s.get(W+'type'),'name':name.get(W+'val')if name is not None else sid,'xml':s}
 return r,out
def rels(z,p='word/_rels/document.xml.rels'):
 r=E.fromstring(z.read(p));out={}
 for e in r:out[e.get('Id')]={'type':e.get('Type').split('/')[-1],'target':e.get('Target'),'mode':e.get('TargetMode','Internal')}
 return out
def main():
 with ZipFile(BASE)as a,ZipFile(CAND)as b:
  namesa=set(a.namelist());namesb=set(b.namelist());changed=[p for p in sorted(namesa&namesb)if a.read(p)!=b.read(p)]
  da=E.fromstring(a.read('word/document.xml'));db=E.fromstring(b.read('word/document.xml'));sta,sa=get_styles(a);stb,sb=get_styles(b)
  ra,rb=rels(a),rels(b)
  out={'baseline':str(BASE),'candidate':str(CAND),'baseline_authority':'用户最后明确以D盘版本为唯一基准；WSL同名文件不用于最终比较。','read_only':True,'scope':'格式、分页相关属性、包结构和关系。未运行Word原生分页或PDF转换。',
       'package':{'before_parts':len(namesa),'after_parts':len(namesb),'added':sorted(namesb-namesa),'removed':sorted(namesa-namesb),'byte_changed_parts':changed,'unchanged_parts':sorted(p for p in namesa&namesb if p not in changed)}}
  # Compare relationship destinations, ignoring harmless rId renumbering.
  def rc(r):return Counter((v['type'],v['target'],v['mode'])for v in r.values())
  ca,cb=rc(ra),rc(rb)
  out['relationships']={'before_count':len(ra),'after_count':len(rb),'removed_semantic':[{'type':k[0],'target':k[1],'mode':k[2],'count':v}for k,v in (ca-cb).items()], 'added_semantic':[{'type':k[0],'target':k[1],'mode':k[2],'count':v}for k,v in(cb-ca).items()]}
  extern=[];broken=[]
  for z,which in [(a,'before'),(b,'after')]:
   for rp in (p for p in z.namelist()if p.endswith('.rels')):
    base=posixpath.dirname(posixpath.dirname(rp))
    for e in E.fromstring(z.read(rp)):
     item={'part':rp,'id':e.get('Id'),'type':e.get('Type').split('/')[-1],'target':e.get('Target'),'mode':e.get('TargetMode','Internal')}
     if item['mode']=='External':extern.append(dict(file=which,**item))
     else:
      target=posixpath.normpath(posixpath.join(base,item['target'].replace('\\','/'))).lstrip('/')
      if target not in z.namelist():broken.append(dict(file=which,resolved=target,**item))
  out['relationships']['all_external']=extern;out['relationships']['missing_internal_targets']=broken
  out['embedded_parts']={'before':[p for p in sorted(namesa)if any(t in p.lower()for t in ['embeddings/','vbaproject','activex/','externalLink'.lower()])], 'after':[p for p in sorted(namesb)if any(t in p.lower()for t in ['embeddings/','vbaproject','activex/','externalLink'.lower()])]}
  out['objects']={which:{tag:len(d.xpath('.//'+tag,namespaces=N))for tag in ['w:object','w:altChunk','w:ins','w:del','w:fldSimple','w:hyperlink','w:bookmarkStart','w:bookmarkEnd','m:oMath','w:tbl','w:tr','w:p']}for which,d in [('before',da),('after',db)]}
  # Style IDs are not stable after editor resave; compare by type/name.
  sna={(v['type'],v['name']):v for v in sa.values()};snb={(v['type'],v['name']):v for v in sb.values()}
  usea=Counter(da.xpath('.//w:pStyle/@w:val|.//w:rStyle/@w:val|.//w:tblStyle/@w:val',namespaces=N));useb=Counter(db.xpath('.//w:pStyle/@w:val|.//w:rStyle/@w:val|.//w:tblStyle/@w:val',namespaces=N))
  sc=[];ren=[]
  for k in sorted(set(sna)&set(snb)):
   x,y=sna[k],snb[k]
   if x['id']!=y['id']:ren.append({'name':k[1],'type':k[0],'before_id':x['id'],'after_id':y['id'],'before_usage':usea[x['id']],'after_usage':useb[y['id']]})
   dd=diff(props(x['xml'],sa),props(y['xml'],sb))
   if dd:sc.append({'name':k[1],'type':k[0],'before_id':x['id'],'after_id':y['id'],'before_usage':usea[x['id']],'after_usage':useb[y['id']],'differences':dd})
  out['styles']={'counts':[len(sa),len(sb)],'added':[{'id':snb[k]['id'],'name':k[1],'type':k[0],'usage':useb[snb[k]['id']]}for k in sorted(set(snb)-set(sna))], 'removed':[{'id':sna[k]['id'],'name':k[1],'type':k[0],'usage':usea[sna[k]['id']]}for k in sorted(set(sna)-set(snb))], 'renamed_ids':ren,'definition_changes':sc,'docDefaults':diff(props(sta.find('w:docDefaults',N)),props(stb.find('w:docDefaults',N)))}
  # Actual sections: resolve header/footer targets instead of rIds.
  sect=[]
  def sectnode(s,r):
   p={}
   for c in s:
    if local(c.tag)in ('headerReference','footerReference'):
     p[local(c.tag)+':'+c.get(W+'type')]=r[c.get('{'+N['r']+'}id')]['target']
    else:p[local(c.tag)]=node(c,True)
   return p
  secsA=da.xpath('.//w:sectPr',namespaces=N);secsB=db.xpath('.//w:sectPr',namespaces=N)
  for i,(s,t)in enumerate(zip(secsA,secsB)):
   x,y=sectnode(s,ra),sectnode(t,rb);dd=diff(x,y)
   if dd:sect.append({'section_index_1based':i+1,'differences':dd})
  out['sections']={'counts':[len(secsA),len(secsB)],'differences':sect}
  out['settings_changes']=diff(props(E.fromstring(a.read('word/settings.xml'))),props(E.fromstring(b.read('word/settings.xml'))))
  # Headers/footers can be re-numbered: compare by section slot.
  hs=[]
  for i,(s,t)in enumerate(zip(secsA,secsB)):
   x,y=sectnode(s,ra),sectnode(t,rb)
   for key in sorted(set(x)|set(y)):
    if not key.startswith(('headerReference','footerReference')):continue
    pa=x.get(key);pb=y.get(key)
    aa=E.fromstring(a.read('word/'+pa))if pa else None;bb=E.fromstring(b.read('word/'+pb))if pb else None
    if node(aa,True)!=node(bb,True):hs.append({'section_index_1based':i+1,'slot':key,'before_part':pa,'after_part':pb,'before_text':tx(aa)if aa is not None else None,'after_text':tx(bb)if bb is not None else None,'before_xml':node(aa,True),'after_xml':node(bb,True)})
  out['header_footer_changes']=hs
  # Paragraph properties on exact aligned texts. Lists of unmatched blocks retain real boundary changes.
  psA=da.xpath('.//w:p',namespaces=N);psB=db.xpath('.//w:p',namespaces=N);textsA=[clean(tx(p))for p in psA];textsB=[clean(tx(p))for p in psB]
  pairs=[];unmatched=[]
  for op,i,j,k,l in SequenceMatcher(None,textsA,textsB,autojunk=False).get_opcodes():
   if op=='equal':pairs.extend(zip(range(i,j),range(k,l)))
   else:unmatched.append({'operation':op,'before_range_0based':[i,j],'after_range_0based':[k,l],'before':[label(psA[q],q,sa)for q in range(i,j)],'after':[label(psB[q],q,sb)for q in range(k,l)]})
  pchanges=[];rchanges=[];pc=Counter();rcount=Counter();styleapply=Counter()
  def runs(p):
   chunks=[]
   for r in p.xpath('.//w:r[not(ancestor::m:oMath)]',namespaces=N):
    text=''.join(r.xpath('./w:t/text()',namespaces=N));rp=props(r.find('w:rPr',N))
    for ch in text:
     if not ch.isspace():chunks.append((ch,rp))
   return chunks
  for i,j in pairs:
   p,q=psA[i],psB[j];x,y=props(p.find('w:pPr',N),sa),props(q.find('w:pPr',N),sb);dd=diff(x,y)
   if dd:
    for k in dd:pc[k]+=1
    if 'pStyle'in dd:styleapply[(label(p,i,sa)['style_name'],label(q,j,sb)['style_name'])]+=1
    pchanges.append({'before':label(p,i,sa),'after':label(q,j,sb),'differences':dd})
   xs,ys=runs(p),runs(q)
   if len(xs)!=len(ys)or ''.join(c for c,_ in xs)!=''.join(c for c,_ in ys):continue
   seg=[];last=None
   for (ch,rx),(ch2,ry)in zip(xs,ys):
    # rStyle can be renumbered; resolve by style name.
    for z,styles in[(rx,sa),(ry,sb)]:
     if 'rStyle'in z:
      v=z['rStyle']['attrs'].get('val');z['rStyle']['attrs']['val']=styles.get(v,{}).get('name',v)
    dr=diff(rx,ry)
    if dr:
     for key in dr:rcount[key]+=1
     sign=json.dumps(dr,ensure_ascii=False,sort_keys=True)
     if sign==last:seg[-1]['text']+=ch
     else:seg.append({'text':ch,'differences':dr});last=sign
    else:last=None
   if seg:rchanges.append({'before':label(p,i,sa),'after':label(q,j,sb),'segments':seg})
  out['paragraph_format']={'aligned_exact_text_paragraphs':len(pairs),'changed_pPr_count':len(pchanges),'changed_properties_counts':dict(pc),'style_application_changes':[{'before':k[0],'after':k[1],'count':v}for k,v in styleapply.items()],'changes':pchanges,'unmatched_boundary_or_text_blocks':unmatched}
  out['run_direct_format']={'changed_paragraphs':len(rchanges),'changed_character_property_counts':dict(rcount),'changes':rchanges}
  ep=[];erc=Counter();erc_loc=defaultdict(list)
  for i,j in pairs:
   p,q=psA[i],psB[j];dd=diff(effective_p(p,sa),effective_p(q,sb))
   if dd:ep.append({'before':label(p,i,sa),'after':label(q,j,sb),'differences':dd})
   xs,ys=effective_runs(p,sa,sta),effective_runs(q,sb,stb)
   if ''.join(c for c,_ in xs)!=''.join(c for c,_ in ys):continue
   for (c,x),(c2,y)in zip(xs,ys):
    for k,d in diff(x,y).items():
     sig=json.dumps([k,d['before'],d['after']],ensure_ascii=False);erc[sig]+=1
     if len(erc_loc[sig])<4:erc_loc[sig].append({'before_p':i,'after_p':j,'character':c,'paragraph_text':tx(p)[:100]})
  out['effective_format']={'method':'按docDefaults、段落样式basedOn链、字符样式链和直接格式合成；样式b/i等遵循OOXML切换语义；不读取pPr/rPr作为既有文字格式。未原生渲染，不将字体名别名等价性猜成相同。','paragraph_changes':ep,'run_character_transitions':[{'property':json.loads(k)[0],'before':json.loads(k)[1],'after':json.loads(k)[2],'characters':v,'example_locations':erc_loc[k]}for k,v in erc.most_common()]}
  # MS Mincho inspection: math control/run properties vs ordinary prose runs.
  fontdetails=[]
  for i,p in enumerate(psB):
   for f in p.xpath('.//w:rFonts[@w:eastAsia="MS Mincho"]',namespaces=N):
    math=bool(f.xpath('ancestor::m:oMath',namespaces=N));r=f.getparent().getparent()
    fontdetails.append({'paragraph_index_0based':i,'paragraph_text':tx(p)[:130],'in_native_math':math,'parent':local(r.tag),'run_text':tx(r)[:120]})
  out['ms_mincho']={'occurrences':len(fontdetails),'inside_native_math':sum(x['in_native_math']for x in fontdetails),'outside_native_math':sum(not x['in_native_math']for x in fontdetails),'locations':fontdetails}
  # Exact text matched tables: compare geometry and row split rules.
  tsA=da.xpath('.//w:tbl',namespaces=N);tsB=db.xpath('.//w:tbl',namespaces=N);tabchanges=[];tabun=[]
  for op,i,j,k,l in SequenceMatcher(None,[clean(tx(t))for t in tsA],[clean(tx(t))for t in tsB],autojunk=False).get_opcodes():
   if op!='equal':tabun.append({'operation':op,'before_range':[i,j],'after_range':[k,l],'before_texts':[tx(t)[:160]for t in tsA[i:j]],'after_texts':[tx(t)[:160]for t in tsB[k:l]]});continue
   for ai,bi in zip(range(i,j),range(k,l)):
    x,y=tsA[ai],tsB[bi];dd={}
    for typ in ['tblPr','tblGrid']:
     u,v=node(x.find('w:'+typ,N),True),node(y.find('w:'+typ,N),True)
     if u!=v:dd[typ]={'before':u,'after':v}
    row=[]
    for ri,(u,v)in enumerate(zip(x.findall('w:tr',N),y.findall('w:tr',N))):
     if props(u.find('w:trPr',N))!=props(v.find('w:trPr',N)):row.append({'row_index_0based':ri,'differences':diff(props(u.find('w:trPr',N)),props(v.find('w:trPr',N)))})
    cell=[]
    for ci,(u,v)in enumerate(zip(x.xpath('.//w:tc',namespaces=N),y.xpath('.//w:tc',namespaces=N))):
     if props(u.find('w:tcPr',N))!=props(v.find('w:tcPr',N)):cell.append({'cell_index_0based':ci,'differences':diff(props(u.find('w:tcPr',N)),props(v.find('w:tcPr',N)))})
    if dd or row or cell:tabchanges.append({'before_table_index_0based':ai,'after_table_index_0based':bi,'text':tx(x)[:160],'properties':dd,'row_properties':row,'cell_properties':cell})
  out['tables']={'counts':[len(tsA),len(tsB)],'rows':[len(da.xpath('.//w:tr',namespaces=N)),len(db.xpath('.//w:tr',namespaces=N))],'changed_exact_text_tables':tabchanges,'unmatched_tables':tabun}
  out['pagination_markers']={which:{'explicit_page_breaks':len(d.xpath('.//w:br[@w:type="page"]',namespaces=N)),'pageBreakBefore_true':len(d.xpath('.//w:pPr/w:pageBreakBefore[not(@w:val="0")]',namespaces=N)),'lastRenderedPageBreak':len(d.xpath('.//w:lastRenderedPageBreak',namespaces=N)),'cantSplit':len(d.xpath('.//w:trPr/w:cantSplit',namespaces=N)),'keepNext':len(d.xpath('.//w:pPr/w:keepNext',namespaces=N)),'keepLines':len(d.xpath('.//w:pPr/w:keepLines',namespaces=N))}for which,d in [('before',da),('after',db)]}
  out['numbering_changes']={'before':node(E.fromstring(a.read('word/numbering.xml')),True),'after':node(E.fromstring(b.read('word/numbering.xml')),True)}
  out['font_table_changes']={'before':node(E.fromstring(a.read('word/fontTable.xml')),True),'after':node(E.fromstring(b.read('word/fontTable.xml')),True)}
  for part,key in [('docProps/app.xml','app_metadata'),('docProps/core.xml','core_metadata'),('docProps/custom.xml','custom_metadata'),('customXml/item1.xml','custom_bibliography_store'),('customXml/itemProps1.xml','custom_bibliography_store_properties'),('word/webSettings.xml','web_settings')]:out[key]=diff(props(E.fromstring(a.read(part))),props(E.fromstring(b.read(part))))
  (T/'layout_package_independent.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
  print(json.dumps({'parts':[len(namesa),len(namesb)],'paragraphs':[len(psA),len(psB)],'pPr':len(pchanges),'direct_run_p':len(rchanges),'msmincho':out['ms_mincho']['inside_native_math'],'style_changes':len(sc),'tablechanges':len(tabchanges),'sectionchanges':len(sect)},ensure_ascii=False))
if __name__=='__main__':main()
