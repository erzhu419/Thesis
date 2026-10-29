"""Read-only comparison of the final native Word core with the cleanup source."""
from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
from difflib import SequenceMatcher
import json,re,unicodedata,sys

A=Path(__file__).resolve().parent
final=Path(sys.argv[1]) if len(sys.argv)>1 else Path('/mnt/d/Paper/Thesis/Gemini/terminal_cleanup_20261015/current.docx')
W='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
M='{http://schemas.openxmlformats.org/officeDocument/2006/math}'
P='{http://schemas.microsoft.com/office/word/2010/wordml}'
def read(p):
    with ZipFile(p)as z:return E.fromstring(z.read('word/document.xml'))
def text(e):return ''.join(n.text or ''for n in e.iter()if n.tag in(W+'t',M+'t'))
def norm(s):
    return ''.join(c for c in unicodedata.normalize('NFKC',s).replace('−','-').replace('′',"'").replace('ℓ','l').replace('ϵ','ε').replace('ς','σ') if not c.isspace() and c!='\u200b')
def marker(e):
    return (e.tag,tuple(n.get(P+'paraId')for n in e.iter(W+'p')))
def patch_text(e,start,end,new):
    off=0;placed=False
    for n in e.iter():
        if n.tag not in(W+'t',M+'t'):continue
        old=n.text or '';lo=max(0,start-off);hi=min(len(old),end-off)
        if lo<hi:n.text=old[:lo]+(new if not placed else '')+old[hi:];placed=True
        off+=len(old)
    assert placed
old=list(read(A/'修改前_通篇校核版.docx').find(W+'body'))
mapping=json.loads((A.parent/'language_review_v2/source_word_block_mapping.json').read_text())['mapping']
markers={marker(old[i])for i in mapping.values()}
actual=[(i,text(e))for i,e in enumerate(read(final).find(W+'body'))if marker(e)in markers]
oldnums=json.loads((A.parent/'reference_numbers.json').read_text())
newnums=json.loads((A/'reference_numbers_final.json').read_text())
numbermap={oldnums[k]:v for k,v in newnums.items()}
ren=json.loads((A/'renumbering.json').read_text())
source_native=A/'source_cleanup_native.docx'
final_md=Path(sys.argv[2]) if len(sys.argv)>2 else None
if final_md:
    import pypandoc
    s=final_md.read_text()
    cites=re.compile(r'\[([A-Za-z]+\d{4}[\w-]*(?:\s*,\s*[A-Za-z]+\d{4}[\w-]*)*)\]')
    s=cites.sub(lambda m:'^\\['+','.join('['+str(newnums[k.strip()])+'](#latest_ref_'+str(newnums[k.strip()])+')' for k in m[1].split(','))+'\\]^',s)
    s=re.sub(r'^【图[^\n]*】[ \t]*$','',s,flags=re.M)
    s=re.sub(r'\$\$(.*?)\$\$',lambda m:'\n\nDISPLAY_EQUATION_'+re.search(r'\\tag\{([^}]+)\}',m[1])[1]+'\n\n',s,flags=re.S)
    s=re.sub(r'^---\s*$','',s,flags=re.M)
    source_native=A/'qa_final_md_native.docx'
    pypandoc.convert_text(s,'docx',format='markdown+tex_math_dollars-smart',outputfile=str(source_native))
expected=[]
for e in read(source_native).find(W+'body'):
    if not norm(text(e)) or text(e).startswith('DISPLAY_EQUATION_'):continue
    for h in e.iter(W+'hyperlink'):
        m=re.fullmatch(r'latest_ref_(\d+)',h.get(W+'anchor',''))
        if m and not final_md:
            oldnumber=int(m[1]);assert oldnumber in numbermap
            s=text(h);digit=re.search(r'\d+',s);assert digit
            patch_text(h,digit.start(),digit.end(),str(numbermap[oldnumber]))
    value=text(e)
    if not final_md:
        value=re.sub(r'([图表])6-(\d+)(?!\d)',lambda m:m[1]+ren['figures'if m[1]=='图'else'tables']['6-'+m[2]],value)
        value=re.sub(r'(?<=[（(])6-(\d+)(?=[）)])',lambda m:ren['equations']['6-'+m[1]],value)
        value=re.sub(r'6\.4\.([1-6])(?!\d)',lambda m:'6.4.'+ren['sections'][m[1]],value)
    expected.append(value)
xn=[norm(s)for s in expected];yn=[norm(s)for _,s in actual]
diffs=[]
for op,a,b,c,d in SequenceMatcher(None,xn,yn,autojunk=False).get_opcodes():
    if op=='equal':continue
    diffs.append({'kind':op,'source_range':[a,b],'word_range':[c,d],
                  'source':[s[:400]for s in expected[a:b]],
                  'word':[{'block':i,'text':s[:400]}for i,s in actual[c:d]]})
result={'word':str(final),'source_native':str(source_native),'final_markdown':str(final_md) if final_md else None,
        'source_core_blocks':len(expected),'retained_word_core_blocks':len(actual),
        'correct':not diffs,'differences':diffs,
        'normalization':'NFKC与空白及已核实数学字体等价字符；按最终218引用号、第6章新编号及附录G比较。'}
(A/'independent_source_core_final_qa.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False))
