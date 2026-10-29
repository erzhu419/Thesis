from pathlib import Path
from copy import deepcopy
import json,re,shutil,pypandoc
T=Path(__file__).resolve().parent;R=T.parents[1]
ast=json.loads((T/'appendix_AF_raw_ast.json').read_text());meta=json.loads((T/'word_blocks.json').read_text())
nums=json.loads((T.parent/'terminal_cleanup_20261015/reference_numbers_final.json').read_text());inverse={v:k for k,v in nums.items()}
def plain(x):
 if isinstance(x,list):return ''.join(plain(i)for i in x)
 if not isinstance(x,dict):return ''
 typ=x.get('t');c=x.get('c')
 if typ in ('Str','Code'):return c if isinstance(c,str)else c[-1]
 if typ in ('Space','SoftBreak','LineBreak'):return ' '
 if typ=='Math':return c[1]
 if typ in ('Link','Image'):return plain(c[1])
 if typ=='Span':return plain(c[1])
 if typ=='Header':return plain(c[2])
 return plain(c)
def walk(x,typ):
 if isinstance(x,dict):
  if x.get('t')==typ:yield x
  for v in x.values():yield from walk(v,typ)
 elif isinstance(x,list):
  for v in x:yield from walk(v,typ)
def rows(table):
 c=table['c'];out=list(c[3][1])
 for body in c[4]:out+=body[2]+body[3]
 return out+c[5][1]
def citation_keys(node):
 keys=[]
 for n in walk(node,'Link'):
  m=re.fullmatch(r'#latest_ref_(\d+)',n['c'][2][0])
  if m:
   key=inverse[int(m[1])]
   if key not in keys:keys.append(key)
 return keys
def clean(x):
 if isinstance(x,list):return [clean(v)for v in x]
 if not isinstance(x,dict):return x
 typ=x.get('t');c=x.get('c')
 if typ=='Superscript' and citation_keys(x):return {'t':'RawInline','c':['markdown','['+','.join(citation_keys(x))+']']}
 if typ=='Link':
  m=re.fullmatch(r'#latest_ref_(\d+)',c[2][0])
  if m:return {'t':'RawInline','c':['markdown','['+inverse[int(m[1])]+']']}
  return {'t':'Span','c':[['',[],[]],clean(c[1])]}
 if typ=='Span':return {'t':'Span','c':[['',[],[]],clean(c[1])]}
 return {k:clean(v)for k,v in x.items()}
groups={};current=None
for b in ast['blocks']:
 if b['t']=='Para'and len(b['c'])==1 and b['c'][0].get('t')=='Str'and re.fullmatch(r'APPBLOCK_\d+',b['c'][0]['c']):current=int(b['c'][0]['c'].split('_')[1]);groups[current]=[]
 else:groups[current].append(b)
result=[];eqrecords=[];images=[];chinese_records=[]
for item in meta:
 bs=deepcopy(groups[item['id']]);converted=[]
 if item['equation_labels']:
  assert len(bs)==1 and bs[0]['t']=='Table'
  seen=[]
  for row in rows(bs[0]):
   cells=row[1];label=plain(cells[-1][4]).strip();m=re.fullmatch(r'\(([A-E]-\d+[a-z]?)\)',label)
   assert m,(item['id'],label)
   label=m[1];seen.append(label);expressions=[]
   for cell in cells[:-1]:
    mathnodes=list(walk(cell[4],'Math'))
    assert plain(cell[4]).strip()==''.join(n['c'][1]for n in mathnodes).strip(),('Nonmath display content',label)
    expressions += [n['c'][1] for n in mathnodes]
   assert expressions,(item['id'],label)
   tex=expressions[0]if len(expressions)==1 else '\\begin{aligned}\n'+' \\\\\n'.join(expressions)+'\n\\end{aligned}'
   tex+=' \\tag{'+label+'}'
   converted.append({'t':'Para','c':[{'t':'Math','c':[{'t':'DisplayMath'},tex]}]})
   eqrecords.append({'label':label,'block':item['id'],'expressions':len(expressions)})
  assert seen==item['equation_labels']
 else:
  for b in bs:
   if item['text'].startswith('附录')and re.match(r'附录[A-F] ',item['text']):
    assert b['t']=='Para';b={'t':'Header','c':[1,['',[],[]],b['c']]}
   for im in walk(b,'Image'):
    number=len(images)+1;src=Path(im['c'][2][0]);dst=R/'appendix_figures_20261016'/f'fig_F_{number}.png';dst.parent.mkdir(exist_ok=True);shutil.copy2(src,dst)
    im['c']=[['',[],[]],[],[f'appendix_figures_20261016/fig_F_{number}.png','']];images.append({'figure':f'图F-{number}','source':str(src),'output':str(dst)})
   converted.append(clean(b))
 # Compare every original Chinese character in document order, including formulas.
 original=''.join(re.findall(r'[\u3400-\u9fff]',item['text']))
 new=''.join(re.findall(r'[\u3400-\u9fff]',plain(converted)))
 assert original==new,('Chinese mismatch',item['id'],original[:100],new[:100])
 chinese_records.append({'block':item['id'],'characters':len(original)})
 result+=converted
outast={**ast,'blocks':result};(T/'appendix_AF_final_ast.json').write_text(json.dumps(outast,ensure_ascii=False,indent=2))
md=pypandoc.convert_text(json.dumps(outast,ensure_ascii=False),'markdown-smart+tex_math_dollars+pipe_tables-grid_tables-multiline_tables-simple_tables',format='json',extra_args=['--wrap=none','--markdown-headings=atx'])
(T/'附录A-F.md').write_text(md)
base=(R/'重构论文_替换库修订_20261015.md').read_text();g=base.index('# 附录G 公交跨保真补充实验设置');full=base[:g]+md.rstrip()+'\n\n---\n\n'+base[g:]
assert full[:g]==base[:g]and full.endswith(base[g:])
out=R/'重构论文_附录补全_20261016.md';out.write_text(full)
report={'output':str(out),'word_blocks':len(meta),'display_equations':len(eqrecords),'original_chinese_characters':sum(e['characters']for e in chinese_records),'all_chinese_characters_preserved_per_block':True,'images':images,'equations':eqrecords,'existing_md_prefix_and_G_unchanged':True}
(T/'completion_qa.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in report.items()if k not in ('images','equations')},ensure_ascii=False))
