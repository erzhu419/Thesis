#!/usr/bin/env python3
"""Create the reviewed CSU template migration map from template_analysis outputs."""
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as E
from collections import Counter
import json
P=Path(__file__).with_name('template_analysis')
d=json.loads(P.with_suffix('.json').read_text())
rows=[json.loads(x) for x in P.with_name(P.name+'_paragraphs.jsonl').read_text().splitlines()]
W='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
z=ZipFile(d['template']);root=E.fromstring(z.read('word/document.xml'));rawps=root.findall('.//'+W+'p')
def xml(e):return E.tostring(e,encoding='unicode') if e is not None else None
def sample(pi):
 q=rows[pi];p=rawps[pi]
 runs=[(i,r) for i,r in enumerate(p.iter(W+'r')) if r.find(W+'t') is not None]
 ri,r=max(runs,key=lambda ir:len(''.join(t.text or '' for t in ir[1].iter(W+'t')))) if runs else (None,None)
 return {'location':q['location'],'style_id':q['style_id'],'style_name':q['style_name'],'text':q['text'][:80],
         'effective_paragraph':q['paragraph_display'],'dominant_text':q['dominant_text_formats'][:1],
         'base_pPr_xml':xml(p.find(W+'pPr')),'base_real_run_index':ri,
         'base_real_run_rPr_xml':xml(r.find(W+'rPr')) if r is not None else None,
         'paragraph_mark_rPr':q['paragraph_mark_rPr'],'table_context':q['table_context']}
# Reviewed, zero-based global paragraph positions, matched to the live package.
roles={'cover_degree':3,'cover_title_zh':6,'cover_title_zh_continued':7,'cover_title_en':8,
       'abstract_title_zh':82,'abstract_first_zh':84,'abstract_body_zh':85,
       'abstract_title_en':99,'abstract_first_en':101,'abstract_body_en':103,'keywords_cn':95,'keywords_en':112,
       'toc_title':117,'toc_level1':118,'toc_level2':119,'toc_level3':121,
       'symbols_title':211,'chapter_heading':321,'section_heading':322,'subsection_heading':337,
       'body':323,'figure_paragraph':324,'figure_caption':325,'table_caption':613,
       'table_text':412,'bibliography_title':2324,'bibliography_entry':2325,'display_equation':409}
semantics={1:'封面、题名页、原创性与授权声明',2:'中英文摘要、目录、符号说明',3:'第 1 章',4:'第 2 章',5:'第 3 章',6:'第 4 章',7:'第 5 章',8:'第 6 章主体',9:'第 6 章尾段和第 7 章；分界错入第 6 章尾部',10:'参考文献',11:'研究成果',12:'致谢'}
secs=[]
for s in d['sections']:
 secs.append({k:s[k] for k in ['index','body_start','body_end','break_type','page_cm','margins_cm','header_footer_resolved']})
 secs[-1].update(semantic_content=semantics[s['index']],page_number=s['properties'].get('pgNumType'),grid=s['properties'].get('docGrid'))
tables=[]
for t in d['tables']:
 tr=[q for q in rows if q['table_context'] and q['table_context']['table']==t['index']]
 st={q['style_id'] for q in tr}
 kind='cover' if t['index']<2 else 'symbols' if t['index']==2 else 'equation' if st=={'aff2'} else 'data'
 tables.append({'index':t['index'],'body_index':t['body_index'],'kind':kind,'rows':t['rows'],'column_counts':t['column_counts'],
                'style_id':t['style_id'],'paragraph_styles':sorted(st),'width':t['tblPr'].get('tblW'),'layout':t['tblPr'].get('tblLayout')})
aux={}
for part in ['word/header4.xml','word/header13.xml','word/footer4.xml','word/footer7.xml']:
 v=d['auxiliary_parts'][part];r=E.fromstring(z.read(part));p=r.find('.//'+W+'p');q=v['paragraphs'][0]
 aux[part]={'style_id':q['style_id'],'text':v['text'],'fields':v['field_codes'],'effective_paragraph':q['paragraph_display'],
            'dominant_text':q['dominant_text_formats'],'base_p_xml':xml(p)}
summary={'template':d['template'],'index_convention':'All body child and global document.xml w:p indexes are zero-based. Global indexes include all table paragraphs; sample index/text alignment was checked for all 2579 paragraphs.',
 'counts':{'paragraphs':len(rows),'top_level_tables':len(tables),'sections':len(secs),'table_kinds':dict(Counter(t['kind'] for t in tables))},
 'defaults':d['defaults'],'roles':{role:sample(pi) for role,pi in roles.items()},'sections':secs,'tables':tables,'auxiliary_samples':aux,
 'undefined_empirical_formats':{'heading4':{'used_paragraphs':0,'style_only':d['styles']['4']},'footnotes':'No actual footnotes or endnotes; both files contain only separator and continuationSeparator elements.'},
 'formulas':d['formulas'],
 'migration_rules':[
  'Map source semantic elements to reviewed samples; retain source text, actual numbering, captions, references, equations and relationships.',
  'Resolve docDefaults, table style, numbering, paragraph style, character style, direct formatting. Paragraph-mark rPr does not override actual text runs.',
  'Normal body is 12 pt with exact 20 pt line spacing and firstLineChars=200. Abstract bodies explicitly override font size to 14 pt.',
  'Heading 2 is style 20. Style 2 is an unused reference style and does not represent actual bibliography.',
  'aff6 table text and aff7 table captions have no basedOn and no explicit line spacing. docDefaults pPr is empty: implicit single spacing subject to grid/font behavior, not Normal exact 20 pt.',
  'Keep beforeLines/afterLines values: they supersede simultaneous before/after twips. Keep character-based indents; hanging wins over firstLine in bibliography.',
  'Data table appearance derives from direct cell borders: typically top/bottom 1.5 pt and header bottom 0.75 pt; cell nil vertical borders override Table Grid. Preserve source merges and fit column widths to its content.',
  'Keep source OMML editable. Template mixes native OMML with 49 legacy OLE objects; embedded MathType internals cannot be inferred from outer XML.',
  'Figure paragraphs center images and use beforeLines=100. Preserve source image relationships and proportions.',
  'Template headings/captions have literal numbers, not SEQ; all direct paragraph numPr absent. Retain source numbering and refresh source TOC/PAGE fields.',
  'Headers contain literal chapter text, not STYLEREF. Copy formatting and tab structure, substitute source chapter text. Default headers alone are active because evenAndOddHeaders/titlePg are absent.'
 ],
 'template_exceptions':[
  {'location':'body 215 / p 335','issue':'Heading 3 paragraph mark has 黑体 16 pt bold, while actual text remains 宋体 12 pt regular. Copy real run basis from body 217 / p 337.'},
  {'location':'body 86 / p 102','issue':'English abstract paragraph simultaneously has firstLineChars=0 and firstLine=420; character value takes precedence. Use body 87 / p 103 for normal 2-character indent.'},
  {'location':'captions','issue':'21 of 133 figure captions and 10 of 26 table captions carry direct firstLine=480 twips; one additional table-caption variant exists. Reviewed regular samples have zero indent.'},
  {'location':'Heading 3','issue':'44 of 46 share the regular paragraph form; remaining forms include before/after 6 pt and a direct first-line 641 twip indent. These do not define the chapter-wide standard.'},
  {'location':'section 9 / body 1124–1146','issue':'Section starts within chapter 6 summary; chapter 7 starts body 1128. Rebuild sections at source chapter boundaries rather than transplant the raw boundary.'},
  {'location':'word/header15.xml','issue':'Active acknowledgements header incorrectly repeats research-achievements text. Retain source semantic heading, copy only formatting.'},
  {'location':'word/header7.xml','issue':'Inactive even-page header has 第X章 placeholder; do not activate/copy placeholder text.'},
  {'location':'style aff2','issue':'Right tab is 9600 twips, wider than 8306-twip body text area; equation layout also uses borderless 3-column tables. Do not force that tab across all source formulas.'},
  {'location':'named reference style 2','issue':'Unused 楷体 10.5 pt double-spaced reference style conflicts with actual 213 entries, which are Normal with 12 pt / exact 20 pt and 2.5-character hanging indent.'}
 ],
 'limits':'This report establishes XML-defined effective properties and actual representative runs. Font metrics, pagination, table-autofit, document-grid interaction and embedded OLE appearance require the separate Word rendering/visual acceptance task.',
 'spec_reference':d['primary_spec_reference']}
P.with_name(P.name+'_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps({'summary':str(P.with_name(P.name+'_summary.json')),'roles':len(roles),'table_kinds':summary['counts']['table_kinds']},ensure_ascii=False))
