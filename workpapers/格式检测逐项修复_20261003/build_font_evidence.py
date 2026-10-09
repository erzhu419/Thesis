import json
from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
B=Path('workpapers/格式检测逐项修复_20261003')
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
R=E.fromstring(ZipFile(B/'排版完成_修复前.docx').read('word/document.xml'))
P=R.findall('.//w:p',NS); MM=R.findall('.//m:oMath',NS); I={id(m):j+1 for j,m in enumerate(MM)}
ISS=json.loads((B/'issues.json').read_text())
MAP={2:([582],['Sánchez-Martínez'],[]),3:([608],['Hüllermeier'],[]),11:([1363],['ωh','gh'],[315,316]),13:([1405],['exp(-|δ|/τh)','τh-1exp(-|δ|/τh)'],[340,342]),14:([1399,1400],['τh','hf=hb=h','h'],[331,334,335]),16:([1437],['Δh'],[372]),29:([1777],['(τh,whold)'],[538]),34:([2242],['Fréchet'],[]),46:([2824],['Kolmogorov–Smirnov'],[]),99:([4249],['均值±标准差'],[])}
changes=json.loads((B/'font_changes_test.json').read_text())
items=[]
for issue in ISS:
 if issue['issue_id'] not in MAP:continue
 inds,labels,mi=MAP[issue['issue_id']]
 entry={**issue,'verified_visually':True,'visual_evidence':('font_regions.png' if issue['category']=='western_font' else 'math_size_regions.png'),'semantic_targets':labels,'paragraphs':[]}
 for i in inds:
  p=P[i]
  d={'paragraph_index_0based':i,'para_id':p.get('{http://schemas.microsoft.com/office/word/2010/wordml}paraId'),'paragraph_text':''.join(p.xpath('.//w:t/text() | .//m:t/text()',namespaces=NS))}
  if mi:
   d['native_math']=[{'global_omath_index_1based':I[id(m)],'math_text':''.join(m.xpath('.//m:t/text()',namespaces=NS)),'h_math_runs_0based':[j for j,r in enumerate(m.findall('.//m:r',NS)) if r.xpath('string(./m:t)',namespaces=NS) in ['h','ℎ']]} for m in p.findall('.//m:oMath',NS) if I[id(m)] in mi]
  else:
   d['normal_runs']=[x for x in changes['western'] if x['paragraph_index']==i]
  entry['paragraphs'].append(d)
 if mi:
  entry['current_actual']={'inherited_parent_size_points':12,'rendered_main_math_size_points':12,'rendered_subscript_size_points':8.52,'rendered_h_font':'MS Mincho','checker_claimed_size_points':13,'size_claim_corroborated':False}
  entry['fix']={'font':'Cambria Math','font_slots':['ascii','hAnsi','eastAsia','cs'],'hint':'default','explicit_parent_size_points':12,'preserved':['OMML operator tree','math italic and upright semantics','automatic subscript superscript scaling']}
 else:
  entry['current_actual']={'ascii_font':'Times New Roman','hAnsi_font':'Times New Roman','eastAsia_font':'宋体','hint':'eastAsia','rendered_problem_glyph_font':'SimSun'}
  entry['fix']={'font':'Times New Roman','font_slots':['ascii','hAnsi','eastAsia','cs'],'hint':'default','preserved':['Latin accents','en dash','plus-minus symbol','superscripts','bold','italic','run order'],'split_only_mixed_chinese_run':True}
 items.append(entry)
out={'input':'output/排版完成.docx','reference_backup':str(B/'排版完成_修复前.docx'),'index_convention':'paragraph_index is 0-based over document.xml all descendant w:p; native math index is 1-based over all m:oMath','covered_issues':items,'same_class_expansion':{'ordinary_runs':changes['western'],'native_math_h_count':len(changes['native_math_h']),'native_math_h_changes_log':'font_changes_test.json','reason':'h currently selects MS Mincho in the report PDF. Explicit Cambria Math slots resolve the same native mathematical glyph choice throughout. Existing five 10.5pt table h sizes remain unchanged; explicit12 onlysixmarkedbodyparagraphs.'},'implementation':'fix_report_fonts.py: apply_report_fonts(document)','local_copy_verification':{'paragraph_text_and_order':{'pass':True,'paragraphs':5972},'native_math_semantic_structure_except_requested_font_size':{'pass':True,'omath_objects':2038},'all_other_docx_parts_byte_equal':True,'ordinary_run_changes':19,'native_math_h_changes':206,'explicit12_h_runs':20},'pending_final_verification':['Word OMath range actual size on marked expressions','Final Word PDF western glyphs Times New Roman and math h Cambria Math']}
(B/'font_annotation_mapping.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
