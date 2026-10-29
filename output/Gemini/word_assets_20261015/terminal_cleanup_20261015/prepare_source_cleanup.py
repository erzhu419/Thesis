from pathlib import Path
import json,re
A=Path(__file__).resolve().parent;R=A.parents[1]
s=(A/'修改前_重构论文.md').read_text();record=[]
def apply(e,all_matches=False):
 global s
 old,new=e['old'],e['new'];count=s.count(old)
 assert count and (all_matches or count==1),(e.get('id',e.get('location')),count,old[:80])
 s=s.replace(old,new) if all_matches else s.replace(old,new,1)
 record.append(dict(e,count_applied=count if all_matches else 1))
d=json.loads((A/'controlled_deletion_boundaries.json').read_text())
for e in d['continuous_deletions']+d['clause_edits']:
 if e.get('location')=='6.4.6剩余公交项回应编号' and e.get('line',0)<2000:continue
 apply(e)
m=json.loads((A/'minimal_text_edits.json').read_text())
for e in m['edits']:
 e=dict(e)
 if e['id']=='B5-1':e['new']=e['old'].replace('三次训练均值','三次训练均值；另三个共同评价日复评')
 if e['id']=='B11-2':
  line=next(x for x in s.splitlines() if x.startswith('表6-6表明，在策略续训前'))
  e['old']=line;e['new']=line+'自动选择组比无模型基线多耗时约1.98分钟，单状态推理仍为毫秒量级。'
 if not s.count(e['old']):
  record.append(dict(e,count_applied=0,covered_by='authorized controlled-experiment deletion'));continue
 apply(e,e.get('replace_all',False))
for i,cn in enumerate(('一','二','三'),1):apply(dict(id=f'B12-{i}',old=f'#### {i}. ',new=f'#### （{cn}）',reason='统一四级标题编号'))
for e in [
 dict(id='G-reference',old='附录F第五部分',new='附录G'),
 dict(id='G-single-item',old='**（五）独立源经验序列',new='**独立源经验序列'),
 dict(id='controlled-summary-sequence-2',old='第二，**关于公交世界模型样本权重衰减症结的攻克与主实验改进幅度**',new='第一，**关于公交世界模型样本权重衰减症结的攻克与主实验改进幅度**'),
 dict(id='controlled-summary-sequence-3',old='第三，**关于第5章跨保真混合训练与第6章MC-WM世界模型修正的分工演进关系**',new='第二，**关于第5章跨保真混合训练与第6章MC-WM世界模型修正的分工演进关系**'),
 dict(id='rounding-88',old='$88.38$',new='$88.39$'),
]:apply(e,e['id']=='G-reference')
for e in [
 dict(id='controlled-sparse-intro',old='源模型预测＋稀疏与神经残差修正',new='源模型预测＋神经残差修正',reason='删除仅被移除受控SINDy实验使用的稀疏残差描述'),
 dict(id='controlled-intro-1',old='在受控动力学仿真平台与多线路公交跨保真实验中系统检验',new='在多线路公交跨保真实验中系统检验',reason='删除已删除第6章受控实验的前文引线'),
 dict(id='controlled-intro-2',old='在受控动力学平台与设置联合50%、75%源参数标定偏差',new='在设置联合50%、75%源参数标定偏差',reason='删除已删除第6章受控实验的前文引线'),
 dict(id='controlled-intro-3',old='在线模式识别补偿与世界模型自适应选择的基础机理',new='在线模式识别补偿的基础机理',reason='受控机理只保留第4章模式补偿'),
 dict(id='controlled-table-1',old='（受控实验含目标交互）',new='',reason='表1-3第6章删除已移除实验条件'),
 dict(id='controlled-table-2',old='SUMO高保真目标环境（多档源标定偏差跨保真评价）＋受控动力学环境',new='SUMO高保真目标环境（多档源标定偏差跨保真评价）',reason='表1-3第6章删除已移除验证环境'),
]:apply(e)
for e in [
 dict(id='controlled-sindy-context',old='，另一方面在第6章MC-WM中与神经网络残差相结合，共同支撑源预测残差修正与直接目标建模的自适应择优机制',new='',reason='删除第2章只对应已删受控SINDy模型的用途说明'),
 dict(id='controlled-sindy-parallel',old='SINDy稀疏辨识一方面在第4章',new='SINDy稀疏辨识在第4章',reason='删去并列分支后取消一方面'),
 dict(id='controlled-sindy-cite',old='[zhang2026csbapr, zhang2026mcwm]',new='[zhang2026csbapr]',reason='仅保留CS-BAPR用途对应引文'),
]:apply(e)
# Preserve original paragraph boundaries. Source is kept with original numbers for native Word anchoring.
(A/'source_before_renumber.md').write_text(s)
eqs=[tag for tag in re.findall(r'\\tag\{([^}]+)\}',s) if re.fullmatch(r'6-\d+',tag)]
ren=dict(figures={'6-1':'6-1','6-2':'6-2',**{f'6-{i}':f'6-{i-7}' for i in range(10,19)}},tables={'6-1':'6-1',**{f'6-{i}':f'6-{i-3}' for i in range(5,11)}},equations={old:f'6-{i}' for i,old in enumerate(eqs,1)},sections={str(i):str(i-2) for i in range(3,7)})
(A/'renumbering.json').write_text(json.dumps(ren,ensure_ascii=False,indent=2))
for prefix,key in [('图','figures'),('表','tables')]:
 def sub(m):
  old='6-'+m[1];assert old in ren[key],(prefix,old);return prefix+ren[key][old]
 s=re.sub(prefix+r'6-(\d+)(?!\d)',sub,s)
s=re.sub(r'(?<=[（(])6-(\d+)(?=[）)])',lambda m:ren['equations']['6-'+m[1]],s)
s=re.sub(r'\\tag\{(6-\d+)\}',lambda m:r'\tag{'+ren['equations'][m[1]]+'}',s)
s=re.sub(r'6\.4\.([1-6])(?!\d)',lambda m:'6.4.'+ren['sections'][m[1]],s)
# Match the specified Word base's existing K notation in the final MD only.
# The pre-renumber source stays unchanged here because it anchors native Word edits.
a=s.index('表4-2 训练分布内');b=s.index('【图4-3',a)
unit_block=s[a:b].replace('，单位：千','').replace(' / 千','')
unit_block=re.sub(r'\$(-?\d+)\$',r'$\1$K',unit_block)
s=s[:a]+unit_block+s[b:]
(R/'重构论文_终审局部修订_20261015.md').write_text(s)
(A/'source_edit_journal.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
print('New source saved;',len(record),'scoped edit records;',len(eqs),'retained chapter-6 equations')
# The English abstract is absent from the MD, so keep its minimal edit in the Word-only plan.
p=A/'word_only_edits.json';w=json.loads(p.read_text())
for e in m['word_only_edits']:
 if e.get('issue')=='A11' and not any(x['old']==e['old'] for x in w['edits']):w['edits'].append(e)
p.write_text(json.dumps(w,ensure_ascii=False,indent=2))
