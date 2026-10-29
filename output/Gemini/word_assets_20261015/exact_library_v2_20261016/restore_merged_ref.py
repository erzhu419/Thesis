"""Retain the original figF_1 REF in its supplied merged paragraph only."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from lxml import etree as E
import ast,re,copy,sys
import native_helpers as h
A=Path(__file__).resolve().parent;R=A.parents[1];W=h.W;M=h.M;text=h.text
names={'fields','split_run','splice'}
mod=ast.parse((A/'build_word_exact_v2.py').read_text())
exec(compile(ast.Module(body=[n for n in mod.body if isinstance(n,ast.FunctionDef)and n.name in names],type_ignores=[]),'<native-ref-helper>','exec'),globals())
path=Path(sys.argv[1])if len(sys.argv)>1 else R/'重构论文_替换库二次修订_20261016.docx'
with ZipFile(A/'修改前.docx')as z:old=E.fromstring(z.read('word/document.xml'))
source=next(p for p in old.find(W+'body')if text(p).startswith('图F-1 比较'))
label,seq,instr=fields(source)[0]
with ZipFile(path)as z:parts={n:z.read(n)for n in z.namelist()}
r=E.fromstring(parts['word/document.xml']);p=next(p for p in r.find(W+'body')if text(p).startswith('如图F-1所示'))
assert not list(p.iter(W+'instrText')),'figF_1 REF already present'
before=text(p);needle=label.strip();at=before.index(needle)
ts=[n for c in seq for n in c.iter(W+'t')];ts[0].text=needle
for t in ts[1:]:t.text=''
splice(p,at,at+len(needle),seq);assert text(p)==before
parts['word/document.xml']=E.tostring(r,encoding='UTF-8',xml_declaration=True,standalone=True)
with ZipFile(path,'w',ZIP_DEFLATED)as z:
 for n,d in parts.items():z.writestr(n,d)
print(str(path),instr.strip(), 'same visible text')
