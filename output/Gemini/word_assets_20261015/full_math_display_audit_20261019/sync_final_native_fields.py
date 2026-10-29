"""Copy only the two approved REF results/codes and automatic TOC page results."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from lxml import etree as E
import json,re
A=Path(__file__).resolve().parent
P=A.parents[1]/'调整完格式的_全公式显示终校_20261019.docx';B=A/'before_final_field_cache_sync.docx'
assert not B.exists();B.write_bytes(P.read_bytes())
W='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
data=json.loads((A/'native_final_field_results.json').read_text(encoding='utf-8-sig'))
refs={i['bookmark'].casefold():i for i in data['ref_updates']}
pages={i['bookmark'].casefold():i['page'] for i in data['toc_pages']}
assert set(refs)=={'eqa_005','eqa_004'} and len(pages)==114
assert {i['new_result'] for i in refs.values()}=={'(A-5)','(A-4)'}
with ZipFile(B) as z:r=E.fromstring(z.read('word/document.xml'))
stack=[];ref_changed=[];toc_changed=[];toc_seen=set()
for n in r.iter():
    if n.tag==W+'fldChar':
        kind=n.get(W+'fldCharType')
        if kind=='begin':stack.append({'instructions':[],'results':[],'inside_result':False})
        elif kind=='separate' and stack:stack[-1]['inside_result']=True
        elif kind=='end' and stack:
            f=stack.pop();code=''.join(t.text or '' for t in f['instructions']);old=''.join(t.text or '' for t in f['results'])
            target=re.search(r'^\s*REF\s+(eqA_00[45])\b',code,re.I)
            page=re.search(r'\bPAGEREF\s+(_Toc[0-9]+)\b',code,re.I)
            if target:
                new=refs[target[1].casefold()]
                assert old==new['old_result'] and code==new['old_code']
                for i,t in enumerate(f['instructions']):t.text=new['new_code'] if i==0 else ''
                for i,t in enumerate(f['results']):t.text=new['new_result'] if i==0 else ''
                ref_changed.append({'bookmark':target[1],'old_result':old,'new_result':new['new_result']})
            elif page:
                key=page[1].casefold();toc_seen.add(key);new=pages[key]
                assert re.fullmatch(r'[0-9IVXLCDMivxlcdm]+',old) and re.fullmatch(r'[0-9IVXLCDMivxlcdm]+',new)
                if old!=new:
                    toc_changed.append({'bookmark':page[1],'old_page':old,'new_page':new})
                    for i,t in enumerate(f['results']):t.text=new if i==0 else ''
    elif n.tag==W+'instrText' and stack and not stack[-1]['inside_result']:stack[-1]['instructions'].append(n)
    elif n.tag==W+'t' and stack and stack[-1]['inside_result']:stack[-1]['results'].append(n)
assert len(ref_changed)==2 and toc_seen==set(pages)
with ZipFile(B) as zin,ZipFile(P,'w',ZIP_DEFLATED) as zout:
    for item in zin.infolist():zout.writestr(item,E.tostring(r,xml_declaration=True,encoding='UTF-8',standalone=True) if item.filename=='word/document.xml' else zin.read(item.filename))
(A/'final_field_cache_sync.json').write_text(json.dumps({'REF_updates':ref_changed,'TOC_page_results_updated':len(toc_changed),'TOC_page_results_checked':len(toc_seen),'TOC_changes':toc_changed,'pages':data['pages'],'native_omath_count':data['math_objects'],'backup':str(B)},ensure_ascii=False,indent=2)+'\n')
print('Updated exactly 2 approved REF fields and',len(toc_changed),'automatic TOC page results; pages',data['pages'])
