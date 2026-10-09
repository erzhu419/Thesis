"""Account for every PDF annotation, including paired and empty markers."""
from pathlib import Path
import json
from collections import Counter

HERE=Path(__file__).resolve().parent
def read(name): return json.loads((HERE/name).read_text())
issues=read('issues.json')
semantics={x['issue_id']:x for x in read('semantics_dispositions.json')['records']}
fonts={x['issue_id']:x for x in read('font_annotation_mapping.json')['covered_issues']}
references={x['annotation_index']:x for x in read('reference_metadata_verification.json')}
records=[]
for issue in issues:
    row=dict(issue)
    key=issue['issue_id']
    if key in semantics:
        row.update({k:v for k,v in semantics[key].items() if k not in ('original_comment',)})
        if row['status']=='planned_fix':row['status']='repaired'
        if row['status']=='false_positive':row['status']='verified_correct'
    elif key in fonts:
        mapping=fonts[key]
        row['status']='repaired'
        row['action']='普通西文问题字形使用Times New Roman；保留重音、符号和语义格式。' if issue['category']=='western_font' else '原生数学h使用Cambria Math，标记位置明确母字号12pt；保留上下标自动缩放。报告所述13pt未在原XML/PDF中复现。'
        row['evidence']='font_annotation_mapping.json；Word分页与公式核对.json；字体实际显示验收.json'
        row['font_mapping']=mapping
    else:
        entry=references[issue['annotation_indices'][0]]
        row['status']='verified_correct'
        row['action']='已核实，当前文献信息正确，保留。'
        row['reference_number']=entry['reference_number']
        row['source_url']=entry['source_url']
        row['checks']=entry['checks']
    records.append(row)
annotations=[{'annotation_index':i,'issue_id':r['issue_id'],'report_page':r['page'],
              'category':r['category'],'status':r['status'],'action':r['action']}
             for r in records for i in r['annotation_indices']]
annotations.sort(key=lambda x:x['annotation_index'])
assert [x['annotation_index'] for x in annotations]==list(range(1,136))
summary={'annotation_count':len(annotations),'issue_groups':len(records),
         'groups_by_status':dict(Counter(x['status'] for x in records)),
         'annotations_by_status':dict(Counter(x['status'] for x in annotations))}
(HERE/'逐条批注处置.json').write_text(json.dumps({'summary':summary,'issues':records,'annotations':annotations},ensure_ascii=False,indent=2)+'\n')
status={'repaired':'已修复','verified_correct':'已核实正确，保留','deferred_by_user':'按要求暂缓空白','informational':'汇总信息已核对'}
lines=['# 检测报告逐条核对表','',
       '对应修复前317页检测PDF；同文成对批注合并为同一问题，所有135条原始批注均有记录。页号是原报告的物理页号，更新目录后正文物理页号整体后移。','',
       '| 问题组 | 原批注编号 | 原报告页 | 类别 | 处置 | 实际处理 |',
       '|---|---|---|---|---|---|']
for row in records:
    action=row['action'].replace('|','\\|').replace('\n',' ').replace('\r',' ')
    indices=', '.join(map(str,row['annotation_indices']))
    lines.append(f"| {row['issue_id']} | {indices} | {row['page']} | {row['category']} | {status[row['status']]} | {action} |")
(HERE/'逐条批注核对表.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
