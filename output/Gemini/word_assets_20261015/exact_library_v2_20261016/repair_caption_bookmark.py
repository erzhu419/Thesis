"""Fix only tblF_2's original short bookmark after title-prefix insertion."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from lxml import etree as E
import sys,re
import native_helpers as h
W=h.W;M=h.M
path=Path(sys.argv[1]);
with ZipFile(path)as z:parts={n:z.read(n)for n in z.namelist()}
r=E.fromstring(parts['word/document.xml']);start=next(n for n in r.iter(W+'bookmarkStart')if n.get(W+'name')=='tblF_2');mid=start.get(W+'id');p=start.getparent();end=next(n for n in p.iter(W+'bookmarkEnd')if n.get(W+'id')==mid)
before=h.text(p);assert before.startswith('表F-2 原递推'),before
p.remove(end);h.insert_marker(p,len('表F-2 '),end);assert h.text(p)==before
# Correct only this REF's cached result. Word's next refresh now preserves it.
changed=0
for p in r.iter(W+'p'):
 children=list(p);seq=[];active=False;target=False
 for ch in children:
  if any(x.get(W+'fldCharType')=='begin'for x in ch.iter(W+'fldChar')):seq=[];active=True;target=False
  if active:
   seq.append(ch)
   if any(re.search(r'\bREF\s+tblF_2\b',x.text or'')for x in ch.iter(W+'instrText')):target=True
  if active and any(x.get(W+'fldCharType')=='end'for x in ch.iter(W+'fldChar')):
   if target:
    ts=[n for c in seq for n in c.iter(W+'t')];cached=''.join(n.text or''for n in ts)
    assert cached.strip()in('表F-2','表F-2 原递推'),cached
    ts[0].text='表F-2'
    for t in ts[1:]:t.text=''
    changed+=1
   active=False
assert changed==1,changed
parts['word/document.xml']=E.tostring(r,encoding='UTF-8',xml_declaration=True,standalone=True)
with ZipFile(path,'w',ZIP_DEFLATED)as z:
 for n,d in parts.items():z.writestr(n,d)
print(str(path),'tblF_2 range=表F-2 ; cached REF=表F-2; caption text unchanged')
