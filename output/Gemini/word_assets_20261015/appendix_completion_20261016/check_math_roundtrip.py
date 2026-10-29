"""One batch of TeX -> OMML, matched to every original appendix math object."""
from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
from collections import Counter,defaultdict
import json,re,copy,unicodedata,pypandoc
A=Path(__file__).resolve().parent
W='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}';M='{http://schemas.openxmlformats.org/officeDocument/2006/math}'
def text(e):return ''.join(n.text or ''for n in e.iter()if n.tag in(W+'t',M+'t'))
def clean(s):
 s=unicodedata.normalize('NFKC',s).replace('−','-').replace('′',"'").replace('ϵ','ε').replace('ϕ','φ').replace('∥','‖')
 return ''.join(c for c in s if not c.isspace()and c not in'\u200b\u2061\u2062')
def local(e):return E.QName(e).localname
def prop(e,name,attribute='val',default=''):
 for p in e:
  if local(p).endswith('Pr'):
   n=p.find(M+name)
   if n is not None:return n.get(M+attribute,default)
 return default
def sequence(values):
 out=[]
 for v in values:
  if not v:continue
  if v[0]=='seq':vs=v[1]
  else:vs=[v]
  for x in vs:
   if out and x[0]=='txt'and out[-1][0]=='txt':out[-1]=('txt',out[-1][1]+x[1])
   else:out.append(x)
 if not out:return()
 if len(out)==1:return out[0]
 return('seq',tuple(out))
def scripts(base,sub,sup):
 if base and base[0]=='scripts':base,oldsub,oldsup=base[1:];sub=sub or oldsub;sup=sup or oldsup
 if sup and sup[0]=='txt'and set(sup[1])=={"'"}:base=sequence([base,sup]);sup=()
 return('scripts',base,sub,sup)if sub or sup else base
def canon(e,layout_matrix=False):
 tag=local(e)
 if e.tag.startswith(W)or tag.endswith('Pr'):return()
 if tag=='t':return('txt',clean(e.text or''))if clean(e.text or'')else()
 child=lambda name:canon(e.find(M+name),layout_matrix)if e.find(M+name)is not None else()
 if tag in('sSub','sSup','sSubSup'):return scripts(child('e'),child('sub'),child('sup'))
 if tag=='sPre':return('prescripts',child('e'),child('sub'),child('sup'))
 if tag=='f':return('frac',prop(e,'type',default='bar'),child('num'),child('den'))
 if tag=='rad':return('root',child('deg'),child('e'))
 if tag in('acc','bar','groupChr'):
  if tag=='bar':mark='underbar'if prop(e,'pos',default='top')=='bot'else'bar'
  else:
   mark=prop(e,'chr',default='̂'if tag=='acc'else'⏞')
   mark={'̂':'hat','ˆ':'hat','^':'hat','̅':'bar','¯':'bar','̃':'tilde','~':'tilde'}.get(mark,mark)
  return('accent',mark,child('e'))
 if tag=='nary':
  operator=('txt',clean(prop(e,'chr',default='∫')))
  return sequence([scripts(operator,child('sub'),child('sup')),child('e')])
 if tag in('limLow','limUpp'):return scripts(child('e'),child('lim')if tag=='limLow'else(),child('lim')if tag=='limUpp'else())
 if tag=='d':
  args=[canon(c,layout_matrix)for c in e if c.tag==M+'e'];sep=prop(e,'sepChr',default='|')
  values=[('txt',clean(prop(e,'begChr',default='(')))]
  for i,x in enumerate(args):
   if i:values.append(('txt',clean(sep)))
   values.append(x)
  values.append(('txt',clean(prop(e,'endChr',default=')'))));return sequence(values)
 if tag=='m':
  rows=[tuple(canon(c,layout_matrix)for c in row if c.tag==M+'e')for row in e if row.tag==M+'mr']
  if layout_matrix:return sequence([c for row in rows for c in row])
  return('matrix',tuple(rows))
 return sequence([canon(c,layout_matrix)for c in e])
ast=json.loads((A/'appendix_AF_raw_ast.json').read_text());blocks=json.loads((A/'word_blocks.json').read_text())
def walk(x):
 if isinstance(x,dict):
  if x.get('t')=='Math':yield x
  else:
   for v in x.values():yield from walk(v)
 elif isinstance(x,list):
  for v in x:yield from walk(v)
entries=[];current=None;within=defaultdict(int)
for block in ast['blocks']:
 marker=next((n['c']for n in block.get('c',[])if isinstance(n,dict)and n.get('t')=='Str'and re.fullmatch(r'APPBLOCK_\d+',n.get('c',''))),None)if block.get('t')=='Para'else None
 if marker:current=int(marker.split('_')[1]);continue
 for math in walk(block):entries.append({'id':len(entries),'block':current,'within':within[current],'ast':math});within[current]+=1
assert len(entries)==994,len(entries)
assert all(within[b['id']]==b['math_count']for b in blocks)
batch={'pandoc-api-version':ast['pandoc-api-version'],'meta':{},'blocks':[]}
for entry in entries:
 batch['blocks'] += [{'t':'Para','c':[{'t':'Str','c':f'RTMATH_{entry["id"]:04d}'}]},{'t':'Para','c':[copy.deepcopy(entry['ast'])]}]
batchfile=A/'roundtrip_math_994.docx'
pypandoc.convert_text(json.dumps(batch),'docx',format='json',outputfile=str(batchfile))
with ZipFile(batchfile)as z:back=E.fromstring(z.read('word/document.xml'))
roundtripped={};current=None
for p in back.find(W+'body'):
 match=re.fullmatch(r'RTMATH_(\d+)',text(p))
 if match:current=int(match[1]);roundtripped[current]=[]
 elif current is not None:roundtripped[current]+=list(p.iter(M+'oMath'))
with ZipFile(Path('/home/erzhu419/mine_code/Thesis/output/Gemini/重构论文_替换库修订_20261015.docx'))as z:source=E.fromstring(z.read('word/document.xml'))
word=list(source.find(W+'body'));differences=[];passed=0;totals=Counter()
for entry in entries:
 meta=blocks[entry['block']];original=list(word[meta['word_index']].iter(M+'oMath'))[entry['within']];returned=roundtripped.get(entry['id'],[])
 tex=entry['ast']['c'][1];labels=meta['equation_labels']
 row=original
 while row is not None and row.tag!=W+'tr':row=row.getparent()
 if row is not None:
  cells=row.findall(W+'tc');match=re.search(r'[（(]([A-F]-\d+[a-z]*)[）)]\s*$',text(cells[-1]))if cells else None
  if match:labels=[match[1]]
 if len(returned)!=1:
  differences.append({'id':entry['id'],'blockID':entry['block'],'labels':labels,'kind':'missing_or_multiple_OMML','returned_count':len(returned),'tex':tex});continue
 returned=returned[0]
 # Aligned equations can round-trip through a layout matrix. Actual mathematical
 # matrices remain structured and are compared cell by cell.
 layout=bool(list(original.iter(M+'eqArr'))and not list(original.iter(M+'m')))
 left=canon(original,layout);right=canon(returned,layout)
 ltxt=clean(text(original));rtxt=clean(text(returned))
 if left==right:passed+=1;continue
 kind='symbol_sequence'if ltxt!=rtxt else'structure_representation'
 totals[kind]+=1
 differences.append({'id':entry['id'],'blockID':entry['block'],'word_index':meta['word_index'],'within':entry['within'],'labels':labels,'kind':kind,'source_text':ltxt,'roundtrip_text':rtxt,'tex':tex,'source_canonical':left,'roundtrip_canonical':right})
report={'math_count':len(entries),'original_block_count':len(blocks),'passed_canonical':passed,'difference_counts':dict(totals),'differences':differences,'one_batch':True}
(A/'math_roundtrip_raw_comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'math_count':len(entries),'passed_canonical':passed,'difference_counts':dict(totals),'missing_or_multiple':[x for x in differences if x['kind']=='missing_or_multiple_OMML'],'first_differences':[{'id':x['id'],'blockID':x['blockID'],'labels':x['labels'],'kind':x['kind'],'source':x.get('source_text','')[:90],'back':x.get('roundtrip_text','')[:90]}for x in differences[:16]]},ensure_ascii=False))
