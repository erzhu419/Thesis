"""Replace only three formula bodies with equivalent explicit-linebreak OMML."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from lxml import etree as E
import json,re,copy,sys
import native_helpers as h
A=Path(__file__).resolve().parent;W=h.W;M=h.M
path=Path(sys.argv[1])if len(sys.argv)>1 else Path('/mnt/d/Paper/Thesis/Gemini/md_sync_latest_20261016/current.docx')
labels={193:'B-25',207:'B-39',376:'E-46'}
with ZipFile(A/'math_three_layout_donor.docx')as z:donor=E.fromstring(z.read('word/document.xml'))
maths={};current=None
for p in donor.find(W+'body'):
 marker=re.fullmatch(r'EXPECT_SLOT_(\d+)',h.text(p))
 if marker:current=int(marker[1]);maths[current]=[]
 elif current is not None:maths[current].extend(copy.deepcopy(m)for m in p.iter(M+'oMath'))
with ZipFile(path)as z:parts={n:z.read(n)for n in z.namelist()}
root=E.fromstring(parts['word/document.xml']);body=root.find(W+'body');modified=[]
for ordinal,label in labels.items():
 candidates=[]
 for table in body.findall(W+'tbl'):
  for row in table.findall(W+'tr'):
   cells=row.findall(W+'tc')
   if cells and h.text(cells[-1]).strip()in('('+label+')','（'+label+'）'):candidates.append((row,cells))
 assert len(candidates)==1,(label,len(candidates))
 row,cells=candidates[0];oldmaths=[m for cell in cells[:-1]for m in cell.iter(M+'oMath')]
 assert len(oldmaths)==1 and len(maths[ordinal])==1,(label,len(oldmaths),len(maths[ordinal]))
 original=oldmaths[0];replacement=copy.deepcopy(maths[ordinal][0]);base=E.Element(W+'rPr')
 font=next((run.find(W+'rPr')for run in original.iter(M+'r')if run.find(W+'rPr')is not None),None)
 if font is not None:
  for child in font:
   if child.tag in h.FONT:base.append(copy.deepcopy(child))
 for run in replacement.iter(M+'r'):h.merge_font(run,base)
 # Pandoc emits TeX alignment separators as literal ampersands. Represent those
 # as native OMML alignment marks, retaining their position without a glyph.
 for t in list(replacement.iter(M+'t')):
  if t.text!='&':continue
  assert any(a.tag==M+'eqArr'for a in t.iterancestors())
  run=t.getparent();rp=run.find(M+'rPr')
  if rp is None:rp=E.Element(M+'rPr');run.insert(0,rp)
  E.SubElement(rp,M+'aln');t.text='\u200b'
 parent=original.getparent();parent.replace(original,replacement)
 modified.append({'label':label,'ordinal':ordinal,'layout_rows':[len(e.findall(M+'e'))for e in replacement.iter(M+'eqArr')],'font_properties':[E.QName(e).localname for e in base]})
parts['word/document.xml']=E.tostring(root,encoding='UTF-8',xml_declaration=True,standalone=True)
with ZipFile(path,'w',ZIP_DEFLATED)as z:
 for name,data in parts.items():z.writestr(name,data)
report={'path':str(path),'modified_formulas':modified,'other_package_parts_unchanged':True,'MD_unchanged':True,'number_cells_and_paragraph_properties_unchanged':True}
(A/'math_three_layout_applied.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False))
