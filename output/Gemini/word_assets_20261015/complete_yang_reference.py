"""Complete the user-supplied, publisher-verified bibliography entry 4."""
from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json,re,shutil
from lxml import etree as ET

ROOT=Path('/home/erzhu419/mine_code/Thesis')
A=Path(__file__).resolve().parent
OUT=ROOT/'output/Gemini/重构论文_排版稿_20261015.docx'
BACKUP=A/'参考文献第4条补齐前.docx'
if not BACKUP.exists():shutil.copy2(OUT,BACKUP)
W='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
NS={'w':W[1:-1]}
KEY='yangsj2021dwell'
ENTRY='杨世军, 裴玉龙, 潘恒彦, 等. 城市公交车辆驻站时间特征分析及预测[J]. 吉林大学学报（工学版）, 2021, 51(6): 2031-2039. DOI:10.13229/j.cnki.jdxbgxb20200676.'
URL='https://xuebao.jlu.edu.cn/gxb/CN/10.13229/j.cnki.jdxbgxb20200676'
with ZipFile(BACKUP) as z:
 original=z.read('word/document.xml');tree=ET.fromstring(original)
 p=next(p for p in tree.xpath('.//w:p',namespaces=NS) if p.xpath('./w:bookmarkStart[@w:name="latest_ref_4"]',namespaces=NS))
 ts=p.xpath('.//w:t',namespaces=NS)
 assert ''.join(t.text or '' for t in ts)=='[4]'+KEY
 # Word has split the label/tab and identifier into separate runs.
 t=next(t for t in ts if t.text==KEY);t.text=ENTRY
 updated=ET.tostring(tree,encoding='UTF-8',xml_declaration=True,standalone=True)
 # Only the bibliography text changes in the package; formatting and relations persist.
 with ZipFile(OUT,'w',ZIP_DEFLATED) as out:
  for info in z.infolist():out.writestr(info,updated if info.filename=='word/document.xml' else z.read(info.filename))
with ZipFile(OUT) as z:
 result=ET.fromstring(z.read('word/document.xml'))
 rp=next(p for p in result.xpath('.//w:p',namespaces=NS) if p.xpath('./w:bookmarkStart[@w:name="latest_ref_4"]',namespaces=NS))
 assert ''.join(rp.xpath('.//w:t/text()',namespaces=NS))=='[4]'+ENTRY
 assert len(result.xpath('.//w:hyperlink[@w:anchor="latest_ref_4"]',namespaces=NS))==2
 # Removing entry 4 from both trees must leave identical document XML.
 original_tree=ET.fromstring(original)
 for root in [original_tree,result]:
  para=next(p for p in root.xpath('.//w:p',namespaces=NS) if p.xpath('./w:bookmarkStart[@w:name="latest_ref_4"]',namespaces=NS))
  para.getparent().remove(para)
 assert ET.tostring(original_tree)==ET.tostring(result),'Unrelated document content changed'
rec={'old_id':KEY,'reason':'用户提供完整书目，已与吉林大学学报原刊核对一致','title':'城市公交车辆驻站时间特征分析及预测','bibliography_text':'[4] '+ENTRY,'DOI':'10.13229/j.cnki.jdxbgxb20200676','source_url':URL,'verification_status':'verified','metadata_source':'user and publisher','authors':['杨世军','裴玉龙','潘恒彦','程国柱','张文会'],'year':2021,'volume':'51','issue':'6','pages':'2031-2039'}
(A/'citation_additions_user.json').write_text(json.dumps({KEY:rec},ensure_ascii=False,indent=2))
cm=json.loads((A/'citation_key_map.json').read_text())
for fn in A.glob('citation_additions_*.json'):
 if fn.name.endswith('_unresolved.json'):continue
 d=json.loads(fn.read_text());cm['matched'].update(d.get('matched',d))
cm['matched_count']=len(cm['matched']);cm['unresolved']=[k for k in cm.get('unresolved',[]) if k not in cm['matched']]
(A/'citation_key_map.json').write_text(json.dumps(cm,ensure_ascii=False,indent=2))
numbers=json.loads((A/'reference_numbers.json').read_text());numbers[KEY]=4
(A/'reference_numbers.json').write_text(json.dumps(numbers,ensure_ascii=False,indent=2))
prior=json.loads((A/'citation_additions_chinese_c_unresolved.json').read_text())
prior['unresolved']=[];prior['status']='resolved';prior['resolution']=rec
(A/'citation_additions_chinese_c_unresolved.json').write_text(json.dumps(prior,ensure_ascii=False,indent=2))
build=json.loads((A/'build_result.json').read_text());build['unresolved_citation_keys']=[];build['word_finalized']=False;build['reference_4_completed']=True
(A/'build_result.json').write_text(json.dumps(build,ensure_ascii=False,indent=2))
print(json.dumps({'completed_reference':4,'citation_links_preserved':2,'other_document_content_unchanged':True,'unresolved_citations':cm['unresolved'],'matched_keys':cm['matched_count']},ensure_ascii=False))
