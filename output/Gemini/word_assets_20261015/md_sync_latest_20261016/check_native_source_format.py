"""Read-only source completeness and native ordinary-text emphasis check."""
from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
from collections import Counter
import json,re,sys
import native_helpers as h
A=Path(__file__).resolve().parent;R=A.parents[1];W=h.W;M=h.M
path=Path(sys.argv[1])if len(sys.argv)>1 else R/'重构论文_MD同步更新_20261016.docx'
with ZipFile(path)as z:
 root=E.fromstring(z.read('word/document.xml'));styles=E.fromstring(z.read('word/styles.xml'))
body=list(root.find(W+'body'));new=[e for e in h.read(A/'word_sync_after_native.docx').find(W+'body')if h.norm(h.text(e))]
norm=lambda s:h.norm(s).replace('ϕ','φ')
texts=[norm(h.text(e))for e in body];cursor=0;matched={};missing=[]
equation_positions={}
for k,block in enumerate(body):
 if block.tag!=W+'tbl':continue
 for row in block.findall(W+'tr'):
  m=re.fullmatch(r'\s*[（(]([1-7A-F]-\d+[a-z]*)[）)]\s*',h.text(row.findall(W+'tc')[-1]))
  if m:equation_positions[m[1]]=k
for j,src in enumerate(new):
 wanted=norm(h.text(src))
 label=re.fullmatch(r'DISPLAY_EQUATION_(.+)',h.text(src))
 if label:
  found=[equation_positions[label[1]]]if label[1]in equation_positions else[]
  assert found,('formula label',label[1]);cursor=found[0]+1;continue
 found=[k for k in range(cursor,len(body))if body[k].tag==src.tag and texts[k]==wanted]
 if not found:missing.append({'source_flow':j,'text':h.text(src)[:100]});continue
 k=found[0];matched[j]=k;cursor=k+1
style_map={s.get(W+'styleId'):s for s in styles.findall(W+'style')}
def props(rp):
 if rp is None:return{}
 result={}
 for e in rp:
  key=E.QName(e).localname
  if key in('b','bCs','i','iCs'):result[key]=e.get(W+'val','1')not in('0','false','off')
  elif key in('sz','szCs'):result[key]=e.get(W+'val')
  elif key=='rFonts':result[key]=dict(e.attrib)
 return result
cache={}
def overlay(out,more):
 for key,value in more.items():
  if key=='rFonts':out[key]={**out.get(key,{}),**value}
  else:out[key]=value
 return out
def styleprops(sid):
 if sid in cache:return cache[sid]
 s=style_map.get(sid)
 if s is None:return{}
 base=s.find(W+'basedOn');out=styleprops(base.get(W+'val')) .copy()if base is not None else{}
 overlay(out,props(s.find(W+'rPr')));cache[sid]=out;return out
base=props(styles.find(W+'docDefaults/'+W+'rPrDefault/'+W+'rPr'))
def effective(p,r):
 out=base.copy();st=p.find(W+'pPr/'+W+'pStyle');overlay(out,styleprops(st.get(W+'val'))if st is not None else{})
 rp=r.find(W+'rPr');cs=rp.find(W+'rStyle')if rp is not None else None
 if cs is not None:overlay(out,styleprops(cs.get(W+'val')))
 overlay(out,props(rp));return out

def char_states(p,native):
 states=[]
 for ch in p:
  if ch.tag==M+'oMath':
   # Formula objects are independently compared by the mathematics agent.
   states.extend((c,None)for c in h.text(ch)if not c.isspace());continue
  for run in ch.iter(W+'r')if ch.tag!=W+'r'else[ ch ]:
   st=effective(p,run)if native else props(run.find(W+'rPr'))
   # Pandoc's Strong character style conveys supplied **...** emphasis.
   rp=run.find(W+'rPr');cs=rp.find(W+'rStyle')if rp is not None else None
   bold=st.get('b',False)or (not native and cs is not None and cs.get(W+'val')=='Strong')
   for t in run.findall(W+'t'):states.extend((c,bold)for c in(t.text or'')if not c.isspace())
 return states
rep=json.loads((A/'word_sync_build.json').read_text());changed={x['new_flow']for x in rep['journal']if x['kind']in('paragraph','inserted_paragraph')}
bold_diffs=[];checked=0;plain_fonts=Counter();samples=[]
for j in sorted(changed):
 if j not in matched:continue
 p=body[matched[j]];st=p.find(W+'pPr/'+W+'pStyle')
 if st is None or st.get(W+'val')!='body':continue
 expected=char_states(new[j],False);actual=char_states(p,True)
 # Compare ordinary text only; ignore mathematical objects' internal run grouping.
 expected=[x for x in expected if x[1]is not None];actual=[x for x in actual if x[1]is not None]
 if expected!=actual:bold_diffs.append({'flow':j,'paragraph':h.text(p)[:100],'expected_bold_chars':sum(v for c,v in expected),'actual_bold_chars':sum(v for c,v in actual)})
 checked+=1
 for run in p.iter(W+'r'):
  if run.find(W+'t')is None:continue
  state=effective(p,run)
  if state.get('b',False):continue
  fonts=state.get('rFonts',{});plain_fonts[(fonts.get(W+'ascii'),fonts.get(W+'eastAsia'),state.get('sz'))]+=1
 if h.text(p).startswith(('固定项在两个输入相减时精确抵消','由于 x','由于x','F.4 跨保真实验批次','将下文导数界应用','命题D.1')):
  samples.append({'flow':j,'text':h.text(p)[:90],'runs':[{ 'text':(run.find(W+'t').text or'')[:30],'effective':effective(p,run)}for run in p.iter(W+'r')if run.find(W+'t')is not None][:5]})
report={'path':str(path),'new_source_blocks':len(new)-376,'source_match':not missing,'source_missing':missing,'changed_body_paragraphs_checked':checked,'ordinary_text_bold_matches_MD':not bold_diffs,'bold_differences':bold_diffs,'ordinary_font_size_states':[{'ascii':k[0],'eastAsia':k[1],'size_halfpoints':k[2],'runs':v}for k,v in plain_fonts.items()],'samples':samples}
(A/'qa_native_source_format.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items()if k!='samples'},ensure_ascii=False))
