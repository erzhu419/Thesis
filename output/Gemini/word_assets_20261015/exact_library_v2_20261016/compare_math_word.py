"""Invoke only after the parent confirms final Word construction is complete."""
from pathlib import Path
import json,re,sys
import prepare_math_verification as h
T=h.T
final=Path(sys.argv[1])
old_md,new_md=h.md_entries(h.OLD),h.md_entries(h.NEW)
source=h.word_rows(h.BASEWORD);actual=h.word_rows(final)
prepared=json.loads((T/'math_expected_inventory.json').read_text());expected={e['ordinal']:e for e in prepared}
obj=lambda x:json.loads(json.dumps(x,ensure_ascii=False))
striptag=lambda t:re.sub(r'\\tag\{[^}]+\}','',t).strip()
diffs=[];passed=0;direct=0;preserved=0;rows=[]
if len(actual)!=376:diffs.append({'kind':'display_row_count','expected':376,'actual':len(actual)})
for i,(old,new,orig,row)in enumerate(zip(old_md,new_md,source,actual),1):
 if row['label']!=new['label']:diffs.append({'ordinal':i,'kind':'tag','expected':new['label'],'actual':row['label']})
 if i in expected:
  entry=expected[i]
  if striptag(entry['tex'])!=striptag(new['tex']):
   diffs.append({'ordinal':i,'label':new['label'],'kind':'prepared_donor_needs_refresh'});continue
  left=entry['expected_content'];direct+=1
 else:
  if old['tex']!=new['tex']:
   diffs.append({'ordinal':i,'label':new['label'],'kind':'changed_formula_has_no_donor'});continue
  left=obj(h.content(orig['maths'],new['tex']));preserved+=1
 right=obj(h.content(row['maths'],new['tex']))
 same=left==right
 if same:passed+=1
 else:diffs.append({'ordinal':i,'label':new['label'],'kind':'math_content','expected_text':h.clean(''.join(h.text(m)for m in orig['maths'])) if i not in expected else None,'actual_text':h.clean(''.join(h.text(m)for m in row['maths'])),'tex':new['tex'],'expected_canonical':left,'actual_canonical':right})
 rows.append({'ordinal':i,'label':new['label'],'actual_oMath_count':len(row['maths']),'canonical_equal':same,'expected_source':'new_MD_donor'if i in expected else'unchanged_baseline_Word'})
report={'final_word':str(final.resolve()),'new_md':str(h.NEW),'display_slots_expected':376,'display_rows_actual':len(actual),'direct_new_MD_donor_comparisons':direct,'unchanged_baseline_comparisons':preserved,'passed_canonical':passed,'all_math_content_matches_MD':passed==376 and not diffs,'multi_expression_rows':[x for x in rows if x['label']in('A-3','C-9')],'all_rows_checked':rows,'differences':diffs}
(T/'qa_word_math_content.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items()if k not in('all_rows_checked','differences')},ensure_ascii=False))
if diffs:print(json.dumps(diffs,ensure_ascii=False)[:8000])
