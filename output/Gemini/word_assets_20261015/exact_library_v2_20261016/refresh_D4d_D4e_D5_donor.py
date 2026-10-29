"""Refresh only three formulas after the confirmed MD repair; leave other donors intact."""
from zipfile import ZipFile
import json,re
from lxml import etree as E
import pypandoc
import prepare_math_verification as h
T=h.T;fresh=h.md_entries(h.NEW);old=h.md_entries(h.OLD)
ordinals=[267,268,269];names=['D-4d','D-4e','D-5'];assert [fresh[i-1]['label']for i in ordinals]==names
entries=[{**fresh[i-1],'old_label':old[i-1]['label'],'changed':fresh[i-1]['tex']!=old[i-1]['tex']}for i in ordinals]
batch=[]
for e in entries:batch.extend([f'EXPECT_SLOT_{e["ordinal"]:04d}','$$\n'+re.sub(r'\\tag\{[^}]+\}','',e['tex']).strip()+'\n$$'])
batch='\n\n'.join(batch)+'\n';(T/'math_expected_D4d_D4e_D5_refresh.md').write_text(batch)
p=T/'math_expected_D4d_D4e_D5_refresh.docx';pypandoc.convert_text(batch,'docx',format='markdown+tex_math_dollars',outputfile=str(p))
with ZipFile(p)as z:r=E.fromstring(z.read('word/document.xml'))
by={};current=None
for node in r.find(h.W+'body'):
 mark=re.fullmatch(r'EXPECT_SLOT_(\d+)',h.text(node))
 if mark:current=int(mark[1]);by[current]=[]
 elif current is not None:by[current].extend(list(node.iter(h.M+'oMath')))
for e in entries:
 ms=by[e['ordinal']];assert ms;e['donor_oMath_count']=len(ms);e['expected_content']=h.content(ms,e['tex'])
p=T/'math_expected_inventory.json';previous=json.loads(p.read_text());merged=[e for e in previous if e['ordinal']not in ordinals]+entries;merged.sort(key=lambda e:e['ordinal']);p.write_text(json.dumps(merged,ensure_ascii=False,indent=2)+'\n')
p=T/'math_verification_preparation.json';report=json.loads(p.read_text());report['pending_slots']=[];report['refreshed_slots']=ordinals;report['expected_donor_slots']=len(merged);report['changed_display_slots']=sum(a['tex']!=b['tex']for a,b in zip(old,fresh));report['tag_differences']=[{'ordinal':i,'old':a['label'],'new':b['label']}for i,(a,b)in enumerate(zip(old,fresh),1)if a['label']!=b['label']];p.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'refreshed':names,'donor_slots':len(merged),'tag_differences':report['tag_differences']},ensure_ascii=False))
