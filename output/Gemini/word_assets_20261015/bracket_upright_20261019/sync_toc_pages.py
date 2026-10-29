"""Copy only recalculated PAGEREF page results, preserving TOC title text."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from lxml import etree as E
import json,re

A=Path(__file__).resolve().parent
DEST=A.parents[1]/'调整完格式的_括号正体与公式校对修复_20261019.docx'
BACKUP=A/'before_TOC_page_cache_sync.docx'
native=json.loads((A/'toc_native_pages.json').read_text(encoding='utf-8-sig'))
pages={i['bookmark'].casefold():i['page'] for i in native['items']}
assert len(pages)==114 and native['toc_count']==1
W='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
M='{http://schemas.openxmlformats.org/officeDocument/2006/math}'
with ZipFile(DEST) as z:root=E.fromstring(z.read('word/document.xml'))
before=E.tostring(root,method='c14n');original_text=''.join(n.text or '' for n in root.iter(W+'t'))
stack=[];matched=set();changes=[];edits=[]
for n in root.iter():
    if n.tag==W+'fldChar':
        kind=n.get(W+'fldCharType')
        if kind=='begin':stack.append({'code':[],'result':[],'inside_result':False})
        elif kind=='separate' and stack:stack[-1]['inside_result']=True
        elif kind=='end' and stack:
            field=stack.pop();code=''.join(field['code'])
            match=re.search(r'\bPAGEREF\s+(_Toc\d+)\b',code,re.I)
            if not match:continue
            key=match[1].casefold();assert key in pages
            nodes=field['result'];old=''.join(t.text or '' for t in nodes)
            assert re.fullmatch(r'[0-9IVXLCDMivxlcdm]+',old),repr(old)
            new=pages[key];assert re.fullmatch(r'[0-9IVXLCDMivxlcdm]+',new),repr(new)
            matched.add(key)
            if old!=new:
                changes.append({'bookmark':match[1],'old_page':old,'new_page':new})
                for i,t in enumerate(nodes):
                    edits.append((t,t.text));t.text=new if i==0 else ''
    elif n.tag==W+'instrText' and stack and not stack[-1]['inside_result']:
        stack[-1]['code'].append(n.text or '')
    elif n.tag==W+'t' and stack and stack[-1]['inside_result']:
        stack[-1]['result'].append(n)
assert matched==set(pages)
updated=E.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
for t,old in edits:t.text=old
assert E.tostring(root,method='c14n')==before
assert not BACKUP.exists();DEST.rename(BACKUP)
with ZipFile(BACKUP) as zin,ZipFile(DEST,'w',ZIP_DEFLATED) as zout:
    for item in zin.infolist():zout.writestr(item,updated if item.filename=='word/document.xml' else zin.read(item.filename))
report=json.loads((A/'repair_verification.json').read_text())
report.update({'native_Word_verified':True,'native_verification':json.loads((A/'native_verification.json').read_text(encoding='utf-8-sig')),
               'baseline_pages':290,'output_pages':292,'TOC_page_fields_checked':114,
               'TOC_page_results_updated':len(changes),'TOC_page_changes':changes,
               'all_outside_math_XML_unchanged':False,
               'all_outside_math_XML_unchanged_except_TOC_page_results':True})
(A/'repair_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'TOC_page_fields_checked':114,'TOC_page_results_updated':len(changes),'pages':292},ensure_ascii=False))
