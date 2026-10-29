from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
import json,re,pypandoc
A=Path(__file__).resolve().parent;W='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}';M='{http://schemas.openxmlformats.org/officeDocument/2006/math}'
def entries(path):
 return[(re.search(r'\\tag\{([^}]+)\}',m[1])[1],m[1].strip())for m in re.finditer(r'\$\$(.*?)\$\$',path.read_text(),re.S)]
old=entries(A/'上次MD.md');new=entries(A/'最新MD.md');assert len(old)==len(new)==376
assert[a for a,b in old]==[a for a,b in new],[(i,a[0],b[0])for i,(a,b)in enumerate(zip(old,new),1)if a[0]!=b[0]]
changes=[{'ordinal':i,'label':b[0],'old_tex':a[1],'tex':b[1]}for i,(a,b)in enumerate(zip(old,new),1)if a!=b]
batch=[]
for e in changes:batch.extend([f'EXPECT_SLOT_{e["ordinal"]:04d}','$$\n'+re.sub(r'\\tag\{[^}]+\}','',e['tex'])+'\n$$'])
s='\n\n'.join(batch)+'\n';(A/'math_changed_donor.md').write_text(s)
pypandoc.convert_text(s,'docx',format='markdown+tex_math_dollars',outputfile=str(A/'math_changed_donor.docx'))
(A/'changed_math_inventory.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'labels_same_order':True,'changed_formula_count':len(changes),'changed_formula_labels':[e['label']for e in changes]},ensure_ascii=False))
