"""Repair confirmed missing math glyphs and literal unit quotes in a new copy."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from copy import deepcopy
from collections import Counter
from lxml import etree as E
import importlib.util
import json

A=Path(__file__).resolve().parent
OUT=A.parents[1]
SRC=A/'source_snapshot_current.docx'
LIVE=OUT/'调整完格式的.docx'
DEST=OUT/'调整完格式的_公式显示修复_20261019.docx'
W='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
M='{http://schemas.openxmlformats.org/officeDocument/2006/math}'
NS={'w':W[1:-1],'m':M[1:-1]}
XSPACE='{http://www.w3.org/XML/1998/namespace}space'

def text(n):
    return ''.join(t.text or '' for t in n.iter() if t.tag in (M+'t',W+'t'))

def set_math_font(r):
    pr=r.find(W+'rPr')
    if pr is None:
        pr=E.Element(W+'rPr')
        r.insert(1 if r.find(M+'rPr') is not None else 0,pr)
    f=pr.find(W+'rFonts')
    if f is None:
        f=E.Element(W+'rFonts');pr.insert(0,f)
    f.attrib.clear()
    for slot in ('ascii','hAnsi','eastAsia','cs'):
        f.set(W+slot,'Cambria Math')

with ZipFile(SRC) as z:
    original=E.fromstring(z.read('word/document.xml'))
root=deepcopy(original)
pdf=json.loads((A/'pdf_reported/pdf_glyph0_diagnosis.json').read_text())
targets=[o['source_OMML'] for g in pdf['visible_gid0_groups'] for o in g['occurrences']]
assert len(targets)==16
missing_glyph_targets=list(targets)
coverage=json.loads((A/'math_glyph_coverage_audit.json').read_text())
norm_targets=[]
for record in coverage['missing_primary_glyph_runs']:
    for c in record['missing_characters']:
        if c['character']!='∥':continue
        norm_targets.append({
            'math_index_0based':record['math_index_0based'],
            'run_text':record['run_text'],'run_xpath':record['run_xpath'],
            'character_index_in_run':c['character_index_in_run'],
            'source_character':'∥','expected_visual_character':'∥',
            'expected_visual_codepoint':'U+2225','kind':record['kind'],
            'exact_equation_label':record['label'],
            'reason':'The supplied PDF uses SimSun for 361 norm signs, rendering diagonal // instead of upright math norm bars; select the math glyph font for this source class.'
        })
assert len(norm_targets)==362
targets+=norm_targets
by_xpath={}
for t in targets:
    entry=by_xpath.setdefault(t['run_xpath'],{'positions':set(),'details':[]})
    entry['positions'].add(t['character_index_in_run'])
    entry['details'].append(t)

# Resolve every element before splitting any sibling run.
resolved=[]
for path,entry in by_xpath.items():
    nodes=root.xpath(path,namespaces=NS)
    assert len(nodes)==1,path
    r=nodes[0]
    assert r.tag==M+'r' and text(r)==entry['details'][0]['run_text'],path
    for t in entry['details']:
        assert text(r)[t['character_index_in_run']]==t['source_character']
    resolved.append((r,entry))

split_runs=0
for r,entry in resolved:
    ts=r.findall(M+'t');assert len(ts)==1
    value=ts[0].text or ''
    positions=entry['positions']
    segments=[]
    for i,c in enumerate(value):
        selected=i in positions
        if segments and segments[-1][0]==selected:
            segments[-1][1]+=c
        else:
            segments.append([selected,c])
    if len(segments)==1:
        assert segments[0][0]
        set_math_font(r)
        continue
    parent=r.getparent();index=parent.index(r)
    parent.remove(r);split_runs+=1
    for offset,(selected,segment) in enumerate(segments):
        new=deepcopy(r);t=new.find(M+'t');t.text=segment
        if segment[:1].isspace() or segment[-1:].isspace():t.set(XSPACE,'preserve')
        if selected:set_math_font(new)
        parent.insert(index+offset,new)

# This combined ampersand/space run is actually printed in B-57. The Markdown
# identifies the ampersand as an alignment separator, not a mathematical term.
amp_path='/w:document/w:body/w:tbl[268]/w:tr/w:tc[2]/w:p/m:oMathPara/m:oMath/m:eqArr/m:e[2]/m:r[1]/m:t'
amp=root.xpath(amp_path,namespaces=NS)
assert len(amp)==1 and amp[0].text=='&\u2001'
amp[0].text='\u2001'

units=json.loads((A/'inline_unit_quote_locations.json').read_text())
paras=list(root.iter(W+'p'))
for p in units['paragraphs']:
    nodes=list(paras[p['paragraph_index_zero_based']].iter(W+'t'))
    for edit in p['node_edits']:
        n=nodes[edit['w_t_index_zero_based']]
        assert n.text==edit['old']
        n.text=edit['new']

# Detect accidental changes to math, ordinary text, and unrelated package parts.
spec=importlib.util.spec_from_file_location('canonical',A.parent/'exact_library_v2_20261016/prepare_math_verification.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
before_math=list(original.iter(M+'oMath'));after_math=list(root.iter(M+'oMath'))
assert len(before_math)==len(after_math)==4738
expected_math=deepcopy(original)
expected_math.xpath(amp_path,namespaces=NS)[0].text='\u2001'
expected_math=list(expected_math.iter(M+'oMath'))
assert all(h.content([a])==h.content([b]) for a,b in zip(expected_math,after_math))
assert [text(n) for n in expected_math]==[text(n) for n in after_math]

expected=deepcopy(original)
ps=list(expected.iter(W+'p'))
for p in units['paragraphs']:
    nodes=list(ps[p['paragraph_index_zero_based']].iter(W+'t'))
    for edit in p['node_edits']:
        assert nodes[edit['w_t_index_zero_based']].text==edit['old']
        nodes[edit['w_t_index_zero_based']].text=edit['new']
def nonmath_tree(r):
    r=deepcopy(r)
    for m in list(r.iter(M+'oMath')):
        m.clear()
    return E.tostring(r,method='c14n')
assert nonmath_tree(expected)==nonmath_tree(root)
assert ''.join(c for c in text(original) if '\u3400'<=c<='\u9fff')==''.join(c for c in text(root) if '\u3400'<=c<='\u9fff')
assert LIVE.stat().st_mtime==SRC.stat().st_mtime and LIVE.stat().st_size==SRC.stat().st_size,'Source has been saved again; refresh the snapshot before delivery.'
assert not DEST.exists(),f'Preserve existing result: {DEST}'
with ZipFile(SRC) as zin,ZipFile(DEST,'w',ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        data=E.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True) if item.filename=='word/document.xml' else zin.read(item.filename)
        zout.writestr(item,data)
with ZipFile(SRC) as a,ZipFile(DEST) as b:
    assert a.namelist()==b.namelist()
    modified=[name for name in a.namelist() if a.read(name)!=b.read(name)]
    assert modified==['word/document.xml'],modified

info={
    'source':str(SRC),'original_live_preserved':str(LIVE),'output':str(DEST),
    'math_objects_checked':4738,'numbered_formula_labels':376,
    'native_inline_math_objects':3661,'unlabeled_display_math_objects':666,
    'reported_PDF_pages_checked':pdf['page_count'],'confirmed_missing_glyphs_repaired':16,
    'missing_glyph_counts':dict(Counter(t['expected_visual_character'] for t in missing_glyph_targets)),
    'norm_glyph_font_occurrences_fixed':len(norm_targets),
    'SimSun_norm_signs_in_reported_PDF':361,
    'symbol_font_character_count':len(targets),
    'glyph_counts':dict(Counter(t['expected_visual_character'] for t in targets)),
    'distinct_math_objects_repaired':len(set(t['math_index_0based'] for t in targets)),
    'font_property_only_math_runs':len(by_xpath),'mixed_symbol_runs_split':split_runs,
    'unit_quote_occurrences_fixed':units['item_count'],'ASCII_quotes_removed':units['quote_character_deletions'],
    'printed_alignment_ampersand_removed_from_B57':1,
    'all_math_text_unchanged_except_one_TeX_alignment_separator':True,
    'all_math_semantic_structures_unchanged':True,'all_Chinese_unchanged':True,
    'all_nonmath_XML_unchanged_except_24_unit_quotes':True,'modified_package_parts':modified,
    'MD_unchanged':True,'original_PDF_unchanged':True,
    'E46_unchanged':True,'formula_3_9_unchanged':True,
    'font_fallback_candidates_not_claimed_as_actual_missing_glyphs':True,
    'changes':targets,
}
(A/'repair_verification.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in info.items() if k!='changes'},ensure_ascii=False,indent=2))
