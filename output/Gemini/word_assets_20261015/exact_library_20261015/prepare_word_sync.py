from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
from difflib import SequenceMatcher
import json,re,unicodedata,pypandoc
A=Path(__file__).resolve().parent;R=A.parents[1]
W='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}';M='{http://schemas.openxmlformats.org/officeDocument/2006/math}';P='{http://schemas.microsoft.com/office/word/2010/wordml}'
def text(e):return ''.join(n.text or ''for n in e.iter()if n.tag in(W+'t',M+'t'))
def norm(s):return ''.join(c for c in unicodedata.normalize('NFKC',s).replace('−','-').replace('′',"'").replace('ℓ','l').replace('ϵ','ε').replace('ς','σ')if not c.isspace()and c!='\u200b')
def read(p):
 with ZipFile(p)as z:return E.fromstring(z.read('word/document.xml'))
def core(r):return[e for e in r.find(W+'body')if norm(text(e))and not text(e).startswith('DISPLAY_EQUATION_')]
nums=json.loads((A.parent/'terminal_cleanup_20261015/reference_numbers_final.json').read_text())
def convert(src,dst):
 s=src.read_text();cites=re.compile(r'\[([A-Za-z]+\d{4}[\w-]*(?:\s*,\s*[A-Za-z]+\d{4}[\w-]*)*)\]')
 s=cites.sub(lambda m:'^\\['+','.join('['+str(nums[k.strip()])+'](#latest_ref_'+str(nums[k.strip()])+')'for k in m[1].split(','))+'\\]^',s)
 s=re.sub(r'^【图[^\n]*】[ \t]*$','',s,flags=re.M)
 s=re.sub(r'\$\$(.*?)\$\$',lambda m:'\n\nDISPLAY_EQUATION_'+re.search(r'\\tag\{([^}]+)\}',m[1])[1]+'\n\n',s,flags=re.S)
 s=re.sub(r'^---\s*$','',s,flags=re.M)
 pypandoc.convert_text(s,'docx',format='markdown+tex_math_dollars-smart',outputfile=str(dst))
convert(A/'修改前.md',A/'word_sync_before_native.docx')
convert(R/'重构论文_替换库修订_20261015.md',A/'word_sync_after_native.docx')
old=core(read(A/'word_sync_before_native.docx'));new=core(read(A/'word_sync_after_native.docx'));word=list(read(A/'修改前.docx').find(W+'body'))
word_norms=[norm(text(e))for e in word]
mapping={};cursor=0
for i,e in enumerate(old):
 wanted=norm(text(e));candidates=[j for j in range(cursor,len(word))if word[j].tag==e.tag and word_norms[j]==wanted]
 assert candidates,('No exact current Word anchor',i,text(e)[:150])
 j=candidates[0];mapping[i]=j;cursor=j+1
groups=[]
for op,a,b,c,d in SequenceMatcher(None,[norm(text(e))for e in old],[norm(text(e))for e in new],autojunk=False).get_opcodes():
 if op=='equal':continue
 groups.append({'kind':op,'old_range':[a,b],'new_range':[c,d],'word_blocks':[mapping[i]for i in range(a,b)],'old':[text(e)[:100]for e in old[a:b]],'new':[text(e)[:100]for e in new[c:d]]})
risks=[]
for g in groups:
 for i in range(*g['old_range']):
  e=word[mapping[i]];ms=list(e.iter(M+'oMath'));marks=[{'tag':E.QName(m).localname,'name':m.get(W+'name'),'id':m.get(W+'id')}for m in e.iter()if m.tag in(W+'bookmarkStart',W+'bookmarkEnd')];fields=[m.text or ''for m in e.iter(W+'instrText')]
  if ms or marks or fields or e.tag==W+'tbl':risks.append({'source_block':i,'word_block':mapping[i],'type':E.QName(e).localname,'paragraph_id':e.get(P+'paraId'),'text':text(e)[:140],'math':len(ms),'bookmarks':marks,'fields':fields})
result={'baseline':str(A/'修改前.docx'),'source_before':str(A/'修改前.md'),'source_after':str(R/'重构论文_替换库修订_20261015.md'),'old_blocks':len(old),'new_blocks':len(new),'mapping':mapping,'groups':groups,'native_risks':risks}
(A/'word_sync_mapping.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'old_blocks':len(old),'new_blocks':len(new),'groups':len(groups),'changed_source_blocks':sum(g['old_range'][1]-g['old_range'][0]for g in groups),'paragraph_count_changes':[(g['old_range'],g['new_range'])for g in groups if g['old_range'][1]-g['old_range'][0]!=g['new_range'][1]-g['new_range'][0]],'fields_in_changed_blocks':[r for r in risks if r['fields']]},ensure_ascii=False))
