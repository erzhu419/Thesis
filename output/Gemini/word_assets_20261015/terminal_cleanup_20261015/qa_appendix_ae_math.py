from pathlib import Path
from zipfile import ZipFile
from copy import deepcopy
import json
from lxml import etree as E
B=Path('/home/erzhu419/mine_code/Thesis/output/Gemini/word_assets_20261015/terminal_cleanup_20261015')
OLD=B/'修改前_通篇校核版.docx';NEW=B.parents[1]/'重构论文_终审局部修订_20261015.docx'
N={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
W='{'+N['w']+'}';M='{'+N['m']+'}'
def load(path):
 with ZipFile(path) as z:r=E.fromstring(z.read('word/document.xml'))
 body=r.find('w:body',N);starts={}
 for i,p in enumerate(body):
  text=''.join(p.xpath('.//w:t/text()',namespaces=N)).strip()
  if p.xpath('./w:pPr/w:pStyle/@w:val',namespaces=N)==['chapter'] and text.startswith('附录'):
   starts[text[2]]=i
 return {c:list(body)[starts[c]:starts[chr(ord(c)+1)]] for c in 'ABCDE'}
def canon(e):
 c=deepcopy(e);c.tail=None
 return E.tostring(c,method='c14n',exclusive=True,with_comments=False)
def paragraphs(seg):return [p for el in seg for p in el.iter(W+'p')]
def paragraph_locations(seg):
 return [(i,E.ElementTree(el).getelementpath(p)) for i,el in enumerate(seg) for p in el.iter(W+'p')]
def signature(p):
 # Run splitting and formatting do not define paragraph boundaries; text, math, tabs, and breaks do.
 parts=[]
 for e in p.iter():
  if e.tag in (W+'t',M+'t'):parts.append(e.text or '')
  elif e.tag==W+'tab':parts.append('\t')
  elif e.tag==W+'br':parts.append('\n')
 return ''.join(parts)
a=load(OLD);b=load(NEW);out=[]
for c in 'ABCDE':
 pa=paragraphs(a[c]);pb=paragraphs(b[c]);ma=[m for el in a[c] for m in el.iter(M+'oMath')];mb=[m for el in b[c] for m in el.iter(M+'oMath')]
 diffs=[]
 for i,(x,y) in enumerate(zip(pa,pb)):
  if signature(x)!=signature(y):diffs.append({'paragraph':i+1,'old':signature(x)[:180],'new':signature(y)[:180]})
  xm=list(x.iter(M+'oMath'));ym=list(y.iter(M+'oMath'))
  if len(xm)!=len(ym):diffs.append({'paragraph':i+1,'math_count_old':len(xm),'math_count_new':len(ym)})
 mathdiff=[i+1 for i,(x,y) in enumerate(zip(ma,mb)) if canon(x)!=canon(y)]
 out.append({'appendix':c,'paragraphs_old':len(pa),'paragraphs_new':len(pb),'paragraph_boundaries_identical':paragraph_locations(a[c])==paragraph_locations(b[c]),'paragraph_text_identical':not diffs,'oMath_old':len(ma),'oMath_new':len(mb),'inline_oMath_old':sum(m.getparent().tag!=M+'oMathPara' for m in ma),'inline_oMath_new':sum(m.getparent().tag!=M+'oMathPara' for m in mb),'all_native_oMath_subtrees_identical':len(ma)==len(mb) and not mathdiff,'changed_math_ordinals':mathdiff,'paragraph_differences':diffs})
result={'baseline':str(OLD),'final':str(NEW),'scope':'附录A–E全部段落（含表内）和原生oMath子树（含内联）','correct':all(x['paragraph_boundaries_identical'] and x['all_native_oMath_subtrees_identical'] for x in out),'native_oMath_total':sum(x['oMath_old'] for x in out),'paragraphs_total':sum(x['paragraphs_old'] for x in out),'appendices':out}
(B/'qa_appendix_ae_math.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='appendices'},ensure_ascii=False))
