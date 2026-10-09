#!/usr/bin/env python3
"""Flatten reviewed template samples to formatting-only OOXML properties."""
from pathlib import Path
import json,copy
P=Path(__file__).with_name('template_analysis')
d=json.loads(P.with_suffix('.json').read_text());rows=[json.loads(x) for x in P.with_name(P.name+'_paragraphs.jsonl').read_text().splitlines()]
roles={'cover_degree':3,'cover_title':6,'abstract_cn_title':82,'abstract_cn_first':84,'abstract_cn_body':85,
       'abstract_en_title':99,'abstract_en_first':101,'abstract_en_body':103,'keywords_cn':95,'keywords_en':112,'toc_title':117,'symbol_title':211,
       'chapter':321,'h2':322,'h3':337,'h4':None,'body':323,'figure_caption':325,'table_caption':613,
       'figure':324,'table':412,'bibliography_title':2324,'bibliography':2325,'toc1':118,'toc2':119,'toc3':121,'equation':409}
result={}
for role,pi in roles.items():
 if pi is None:
  s=d['styles']['4'];pp=copy.deepcopy(s['effective_pPr']);rp=copy.deepcopy(s['effective_rPr']);q=None;rr=None
  location=None;sid='4';evidence='Style definition only; no actual Heading 4 paragraph in template.'
 else:
  q=rows[pi];pp=copy.deepcopy(q['effective_pPr']);rr=q['representative_text_run'];sid=q['style_id'];location=q['location']
  rp=copy.deepcopy(rr['effective'] if rr else d['styles'][sid]['effective_rPr']);evidence='Resolved actual text run and paragraph.' if rr else 'Paragraph sample; no ordinary text run. Equation math retains its separate m:rPr/w:rPr.'
 # Cascade metadata is deliberately absent from the new-style property body.
 pp.pop('pStyle',None);rp.pop('rStyle',None)
 # Resolve theme selectors into explicit faces to avoid source-theme substitution.
 if rr:
  ff=rp.setdefault('rFonts',{})
  for key,field in [('ascii','latin'),('hAnsi','high_ansi'),('eastAsia','east_asia'),('cs','complex_script')]:
   font=rr['display']['fonts'][field]
   if font:ff[key]=font
  for k in ['asciiTheme','hAnsiTheme','eastAsiaTheme','cstheme','csTheme']:ff.pop(k,None)
 result[role]={'location':location,'style_id':sid,'evidence':evidence,'effective_pPr':pp,'effective_rPr':rp,
   'font_size_pt':float(rp['sz']['val'])/2 if 'sz' in rp else None,
   'character_spacing_pt':float(rp.get('spacing',{}).get('val','0'))/20,
   'character_scale_percent':float(rp.get('w',{}).get('val','100')),
   'character_position_pt':float(rp.get('position',{}).get('val','0'))/2,
   'kerning_threshold_pt':float(rp['kern']['val'])/2 if 'kern' in rp else None,
   'effective_display':q['paragraph_display'] if q else None,
   'representative_text':rr['text'] if rr else None}
 if role in ['keywords_cn','keywords_en']:
  label=copy.deepcopy(q['run_samples'][0]['effective']);label.pop('rStyle',None)
  result[role]['mixed_run_rule']={'label_text':'关键词：' if role=='keywords_cn' else 'Keywords:',
      'label_effective_rPr':label,'body_effective_rPr':copy.deepcopy(rp),
      'note':'Label is bold; following terms are regular. Chinese label uses 黑体 14 pt, Chinese terms 宋体 14 pt; English label/terms use TNR 14 pt. Trailing whitespace does not determine visible typography.'}
result['_notes']=[
 'All locations use zero-based body-child / global w:p indexes. pStyle/rStyle identifiers removed from formatting properties; explicit faces replace active theme fonts. Other attributes retain actual OOXML units.',
 'No inherited Normal paragraph spacing is added to styles without basedOn. docDefaults pPr is empty; omitted paragraph line spacing means Word implicit single, subject to section grid/font metrics.',
 'beforeLines/afterLines supersede before/after twips, and hanging/character-based indentation take priority as recorded in effective_display.',
 'Absent run spacing/scale/position mean 0 pt/100%/0 pt respectively; explicit returned auxiliary fields make these defaults clear. kern is a half-point threshold, not character spacing.',
 'Equation effective_rPr describes aff2 paragraph text/numbers. Native math uses mathFont Cambria Math and its own m:rPr/w:rPr; retain mathematical bold/italic/subscript distinctions.',
 'The h4 definition uses 宋体 for Latin as well as East Asian text, unlike h2/h3 TNR Latin. No actual h4 sample exists.',
 'Figure paragraph has no ordinary text run, so effective_rPr comes from the paragraph style plus docDefaults. It does not describe text embedded inside a figure.'
]
P.with_name(P.name+'_values.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(str(P.with_name(P.name+'_values.json')))
