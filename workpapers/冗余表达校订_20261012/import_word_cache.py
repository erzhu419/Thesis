"""Import only Word-native field results, leaving native diagrams untouched."""
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile
import json
import re

from lxml import etree as E

from edit import OUT, HERE, W, body_spans

STAGE = Path('/mnt/c/Users/erzhu/AppData/Local/Temp/thesis_expression_20261012')


def result_nodes(root):
    stack,counts=[],defaultdict(int)
    for n in root.iter():
        if n.tag==W+'fldChar':
            kind=n.get(W+'fldCharType')
            if kind=='begin': stack.append({'code':'','result':False,'nodes':[]})
            elif kind=='separate' and stack: stack[-1]['result']=True
            elif kind=='end' and stack:
                f=stack.pop()
                code=' '.join(f['code'].split())
                if re.match(r'^(SEQ|REF|PAGEREF)\s',code):
                    counts[code]+=1
                    yield (code,counts[code]),f['nodes']
        elif n.tag==W+'instrText' and stack: stack[-1]['code']+=n.text or ''
        elif n.tag==W+'t' and stack and stack[-1]['result']: stack[-1]['nodes'].append(n)


def main():
    with ZipFile(STAGE/'word.docx') as z:
        r=E.fromstring(z.read('word/document.xml'))
        caches={k:''.join(n.text or '' for n in nodes) for k,nodes in result_nodes(r)}
    with ZipFile(OUT) as z:
        raw=z.read('word/document.xml')
        r=E.fromstring(raw)
        body=list(r.find(W+'body'))
        before=[E.tostring(n) for n in body]
        updates=0
        count=0
        for key,nodes in result_nodes(r):
            assert key in caches,('native cache missing',key)
            value=caches[key]
            assert not re.search(r'Error!|错误!|错误！',value),('native field error',key,value)
            count+=1
            if ''.join(n.text or '' for n in nodes)!=value:
                assert nodes,key
                nodes[0].text=value
                for n in nodes[1:]: n.text=''
                updates+=1
        spans=body_spans(raw)
        for i in reversed(range(len(body))):
            if E.tostring(body[i])!=before[i]:
                a,b=spans[i]
                raw=raw[:a]+E.tostring(body[i],encoding='utf-8',with_tail=False)+raw[b:]
        parts=[(info,raw if info.filename=='word/document.xml' else z.read(info.filename)) for info in z.infolist()]
    with ZipFile(OUT,'w') as z:
        for info,data in parts: z.writestr(info,data)
    result={'native_fields_checked':count,'updated_caches':updates}
    (HERE/'Word域更新记录.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__': main()
