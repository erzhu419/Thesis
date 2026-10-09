#!/usr/bin/env python3
"""Compare DOCX substantive OOXML in order; no digests and no text-only rebuild.

Projection omits paragraph/run/table appearance, page/section geometry, TOC fields
and page-number fields. It retains text, logical table cells/merges, effective list
numbering, complete REF/SEQ instructions and displayed text, named bookmark
boundaries, links, native OMML operators/arguments, and ordered drawing payloads.
OMML excludes only w:rPr, m:ctrlPr and m:argSz (formatting); m:sty remains.
Running headers/footers are reported separately as template-owned running matter.
"""
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
from collections import Counter
from io import BytesIO
import argparse, json, posixpath, re
from lxml import etree as E
from PIL import Image

NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
      'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
      'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
      'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'}
def q(prefix, local): return '{' + NS[prefix] + '}' + local
def local(e): return E.QName(e).localname
def visible(e): return ''.join(e.xpath('.//w:t/text()|.//m:t/text()', namespaces=NS))
def sval(e, path):
    v = e.find(path, NS)
    return v.get(q('w', 'val')) if v is not None else None

def compact(tokens):
    out = []
    for x in tokens:
        if isinstance(x, str) and out and isinstance(out[-1], str): out[-1] += x
        else: out.append(x)
    return out

def math_projection(e):
    if e.tag in {q('w','rPr'), q('m','ctrlPr'), q('m','argSz')}: return None
    # Word merges adjacent math runs, including punctuation into an adjacent
    # script-font run. Compare each mathematical character and its meaningful
    # style, retaining every operator and argument node around those characters.
    if e.tag == q('m','r'):
        rp=e.find('m:rPr',NS)
        properties={local(v):v.get(q('m','val'),'1') for v in rp} if rp is not None else {}
        chars=[]
        for char in ''.join(e.xpath('./m:t/text()',namespaces=NS)):
            style=dict(properties)
            if not char.isalpha() and not char.isdigit():
                style.pop('scr',None)
                if style.get('sty','i') in {'i','p'}:style.pop('sty',None)
            chars.append(['math_character',char,sorted(style.items())])
        return ['math_run_characters',chars]
    children=[]
    for c in e:
        v=math_projection(c)
        if v is None:continue
        if v[0]=='math_run_characters':children.extend(v[1])
        else:children.append(v)
    attrs = sorted((k, v) for k, v in e.attrib.items() if E.QName(k).localname not in {'rsidR','rsidRPr'})
    return [e.tag, attrs, e.text if e.tag in {q('m','t'),q('w','t')} else None, children]

class Document:
    def __init__(self, path):
        self.path = str(path); self.zip = ZipFile(path)
        self.names = set(self.zip.namelist()); self.assets = []; self.fields = []; self.math = []
        self.bookmarks = []; self.paragraphs = []; self.relationships = {}
        self.styles = self.xml('word/styles.xml')
        self.style_map = {e.get(q('w','styleId')): e for e in self.styles.findall('w:style',NS)}
        self.style_names = {k: sval(v, 'w:name') for k,v in self.style_map.items()}
        self.numbering = self.xml('word/numbering.xml') if 'word/numbering.xml' in self.names else E.Element('numbering')
        self.num_map = {e.get(q('w','numId')):e for e in self.numbering.findall('w:num',NS)}
        self.abstract_map = {e.get(q('w','abstractNumId')):e for e in self.numbering.findall('w:abstractNum',NS)}
        self.projection = {}
        for part in ['word/document.xml','word/footnotes.xml','word/endnotes.xml','word/comments.xml']:
            if part not in self.names: continue
            tree = self.xml(part)
            self.projection[part] = self.project_story(tree, part)
        self.running = {}
        for part in sorted(self.names):
            if re.fullmatch(r'word/(?:header|footer)\d+\.xml', part):
                root = self.xml(part)
                self.running[part] = {'text': visible(root), 'field_instructions': self.extract_fields(root)[0]}
        self.embedded = sorted(n for n in self.names if n.startswith(('word/embeddings/','word/charts/')))
    def xml(self, part): return E.fromstring(self.zip.read(part))
    def rels(self, part):
        if part in self.relationships: return self.relationships[part]
        f = PurePosixPath(part)
        relfile = str(f.parent / '_rels' / (f.name + '.rels'))
        result = {}
        if relfile in self.names:
            for e in self.xml(relfile):
                external = e.get('TargetMode') == 'External'
                target = e.get('Target') if external else posixpath.normpath(str(f.parent / e.get('Target')))
                result[e.get('Id')] = {'target': target, 'type': e.get('Type').rsplit('/',1)[-1], 'external':external}
        self.relationships[part] = result
        return result
    def num_definition(self, p):
        n = p.find('w:pPr/w:numPr',NS)
        style = sval(p, 'w:pPr/w:pStyle'); seen = set()
        while n is None and style in self.style_map and style not in seen:
            seen.add(style); st = self.style_map[style]
            n = st.find('w:pPr/w:numPr',NS); style = sval(st,'w:basedOn')
        if n is None: return None
        numid = sval(n,'w:numId'); level = sval(n,'w:ilvl') or '0'
        if numid in (None,'0'): return None
        num = self.num_map.get(numid)
        if num is None: return {'unresolved_numId':numid,'level':level}
        abstract = self.abstract_map.get(sval(num,'w:abstractNumId'))
        lvl = None if abstract is None else abstract.find(f'w:lvl[@w:ilvl="{level}"]',NS)
        override = num.find(f'w:lvlOverride[@w:ilvl="{level}"]',NS)
        if override is not None and override.find('w:lvl',NS) is not None: lvl = override.find('w:lvl',NS)
        keys = ['start','numFmt','lvlText','lvlRestart','isLgl','suff']
        result = {'level':level, **({k:sval(lvl,'w:'+k) for k in keys} if lvl is not None else {})}
        if override is not None and sval(override,'w:startOverride') is not None:
            result['start'] = sval(override,'w:startOverride')
        return result
    def extract_fields(self, root):
        elements = list(root.iter()); stack = []; records = []
        for i,e in enumerate(elements):
            if e.tag == q('w','fldChar'):
                kind = e.get(q('w','fldCharType'))
                if kind == 'begin': stack.append({'start':i,'code_parts':[],'sep':None})
                elif kind == 'separate' and stack: stack[-1]['sep'] = i
                elif kind == 'end' and stack:
                    item = stack.pop(); item['end'] = i; item['code'] = ''.join(item.pop('code_parts')).strip()
                    item['display'] = ''.join((e.text or '') if e.tag in {q('w','t'),q('m','t')} else '\t' if e.tag == q('w','tab') else '\n' if e.tag in {q('w','br'),q('w','cr')} else '' for e in elements[(item['sep'] or i)+1:i])
                    records.append(item)
            elif e.tag == q('w','instrText') and stack and stack[-1]['sep'] is None:
                stack[-1]['code_parts'].append(e.text or '')
        if stack: raise ValueError('Unclosed complex field in ' + self.path)
        records.sort(key=lambda x:x['start'])
        return records, elements
    def project_story(self, root, part):
        records, elements = self.extract_fields(root); excluded = set(); field_at = {}
        outer_toc = [(r['start'],r['end']) for r in records if r['code'].split(' ',1)[0].upper() == 'TOC']
        for r in records:
            excluded.update(elements[r['start']:r['end']+1])
            kind = r['code'].split(' ',1)[0].upper()
            if any(a <= r['start'] <= b for a,b in outer_toc) or kind in {'PAGE','PAGEREF','NUMPAGES','SECTIONPAGES'}: continue
            index = len(self.fields); self.fields.append({'part':part,'code':r['code'],'display':r['display']})
            field_at[elements[r['start']]] = index
        bookmark_names = {e.get(q('w','id')):e.get(q('w','name')) for e in elements if e.tag == q('w','bookmarkStart')}
        def walk(e):
            if e in field_at: return [['field',field_at[e]]]
            if e in excluded: return []
            tag = e.tag; name = local(e)
            if tag == q('w','p'):
                data = compact([v for c in e if c.tag != q('w','pPr') for v in walk(c)])
                number = self.num_definition(e)
                if number is not None: data.insert(0,['numbering',number])
                if not data: return []
                self.paragraphs.append({'part':part,'style_id':sval(e,'w:pPr/w:pStyle'), 'style_name':self.style_names.get(sval(e,'w:pPr/w:pStyle')), 'text':visible(e),'math_count':len(e.findall('.//m:oMath',NS))})
                return [['paragraph',data]]
            if tag == q('w','tbl'):
                rows = [walk(c)[0] for c in e if c.tag == q('w','tr')]
                return [['table',rows]]
            if tag == q('w','tr'):
                return [['row',[walk(c)[0] for c in e if c.tag == q('w','tc')]]]
            if tag == q('w','tc'):
                meta = {'gridSpan':sval(e,'w:tcPr/w:gridSpan') or '1'}
                merge = e.find('w:tcPr/w:vMerge',NS)
                if merge is not None: meta['vMerge'] = merge.get(q('w','val')) or 'continue'
                return [['cell',meta,[v for c in e if c.tag != q('w','tcPr') for v in walk(c)]]]
            if tag == q('m','oMath'):
                index = len(self.math); self.math.append({'part':part,'structure':math_projection(e)})
                return [['native_OMML',index]]
            if tag in {q('w','bookmarkStart'),q('w','bookmarkEnd')}:
                mark = e.get(q('w','name')) if name == 'bookmarkStart' else bookmark_names.get(e.get(q('w','id')))
                if mark and (mark.startswith('_Toc') or mark == '_GoBack'): return []
                item = [name,mark]; self.bookmarks.append({'part':part,'kind':name,'name':mark})
                return [item]
            if tag in {q('w','drawing'),q('w','pict'),q('w','object')}:
                indexes = []
                for child in e.iter():
                    for attr,val in child.attrib.items():
                        if attr in {q('r','embed'),q('r','link'),q('r','id')}:
                            rel = self.rels(part).get(val)
                            if rel and rel['type'] in {'image','oleObject','package','chart'}:
                                indexes.append(len(self.assets)); self.assets.append({'part':part,**rel})
                descr = [{k:v for k,v in c.attrib.items() if k in {'descr','title'}} for c in e.iter() if c.tag == q('wp','docPr')]
                crop = [sorted(c.attrib.items()) for c in e.iter() if c.tag == q('a','srcRect')]
                return [['drawing',indexes,descr,crop]]
            if tag == q('w','hyperlink'):
                rel = self.rels(part).get(e.get(q('r','id')))
                target = rel['target'] if rel else None
                return [['hyperlink',target,e.get(q('w','anchor')),compact([v for c in e for v in walk(c)])]]
            if tag == q('w','fldSimple'):
                code = e.get(q('w','instr'),'').strip(); kind = code.split(' ',1)[0].upper()
                if kind in {'TOC','PAGE','PAGEREF','NUMPAGES','SECTIONPAGES'}: return []
                index=len(self.fields);self.fields.append({'part':part,'code':code,'display':visible(e)})
                return [['field',index]]
            if tag in {q('w','t'),q('w','delText')}: return [e.text or '']
            if tag == q('w','tab'): return ['\t']
            if tag in {q('w','br'),q('w','cr')}:
                return [] if e.get(q('w','type')) in {'page','column'} else ['\n']
            if tag in {q('w','footnoteReference'),q('w','endnoteReference'),q('w','commentReference')}:
                return [[name,e.get(q('w','id'))]]
            if tag in {q('w','footnote'),q('w','endnote'),q('w','comment')}:
                return [[name,sorted(e.attrib.items()),[v for c in e for v in walk(c)]]]
            if name in {'separator','continuationSeparator','sym','noBreakHyphen','softHyphen'}:
                return [[tag,sorted(e.attrib.items())]]
            if tag in {q('w','rPr'),q('w','pPr'),q('w','sectPr'),q('w','tblPr'),q('w','tblGrid'),q('w','trPr'),q('m','oMathParaPr'),q('w','proofErr'),q('w','lastRenderedPageBreak')}:
                return []
            return compact([v for c in e for v in walk(c)])
        return walk(root)
    def inventory(self):
        tree=self.xml('word/document.xml'); rawstyles=Counter(sval(p,'w:pPr/w:pStyle') or 'DEFAULT' for p in tree.findall('.//w:p',NS))
        headings=[{'level':self.style_names.get(sval(p,'w:pPr/w:pStyle')),'text':visible(p)} for p in tree.findall('.//w:p',NS) if re.fullmatch('heading [1-4]',self.style_names.get(sval(p,'w:pPr/w:pStyle')) or '',re.I)]
        cats=Counter()
        for p in self.paragraphs:
            text=p['text'];style=p['style_name'] or ''
            category='heading' if re.fullmatch('heading [1-4]',style,re.I) else 'figure_caption' if style=='Thesis Figure Caption' else 'table_caption' if re.match(r'^表[\dA-F]+-\d+\s',text) else 'bibliography' if style=='书目1' else 'equation_or_math' if p['math_count'] and not text.strip() else 'body_or_cell'
            p['category']=category;cats[category]+=1
        return {'source':self.path,'main_ooxml_counts':dict(Counter(local(e) for e in tree.iter())), 'paragraph_style_counts':[{'id':k,'name':self.style_names.get(k),'count':v} for k,v in rawstyles.items()], 'substantive_categories':dict(cats),'headings':headings,'fields':self.fields,'field_kind_counts':dict(Counter(f['code'].split(' ',1)[0] for f in self.fields)),'ordered_assets':self.assets,'native_OMML_count':len(self.math),'bookmarks':self.bookmarks,'running_headers_footers':self.running,'embedded_parts':self.embedded,'notes':{p:self.projection[p] for p in self.projection if p!='word/document.xml'},'numbering_note':'Chapter and caption numbers are literal text; equation numbers use SEQ fields. Four English abstract paragraphs use an effective automatic list.'}

def first_difference(a,b,path=''):
    if type(a) is not type(b): return {'at':path,'source':repr(a)[:350],'candidate':repr(b)[:350]}
    if isinstance(a,dict):
        if a.keys()!=b.keys():return {'at':path,'source_keys':list(a),'candidate_keys':list(b)}
        for k in a:
            if (d:=first_difference(a[k],b[k],path+'/'+str(k))) is not None:return d
    elif isinstance(a,list):
        if len(a)!=len(b):return {'at':path,'source_length':len(a),'candidate_length':len(b)}
        for i,(x,y) in enumerate(zip(a,b)):
            if (d:=first_difference(x,y,path+'/'+str(i))) is not None:return d
    elif a!=b:return {'at':path,'source':repr(a)[:350],'candidate':repr(b)[:350]}
    return None

def asset_equal(a,ra,b,rb):
    if ra['type']!=rb['type'] or ra['external']!=rb['external']:return False,'relationship type differs'
    if ra['external']:return ra['target']==rb['target'],'external target'
    xa=a.zip.read(ra['target']);xb=b.zip.read(rb['target'])
    if xa==xb:return True,'identical bytes'
    if ra['type']=='image':
        try:
            ia=Image.open(BytesIO(xa));ib=Image.open(BytesIO(xb))
            if ia.size!=ib.size:return False,'decoded image dimensions differ'
            return ia.convert('RGBA').tobytes()==ib.convert('RGBA').tobytes(),'decoded RGBA pixels'
        except (OSError,ValueError):return False,'image bytes differ and cannot decode'
    return False,'embedded object bytes differ'

def compare(a,b):
    checks={};diffs={}
    def content_fields(fields):
        result=[]
        for f in fields:
            item=dict(f)
            if re.match(r'^REF\b',item['code'],re.I):
                item['code']=re.sub(r'\s+\\\*\s+CHARFORMAT\b','',item['code'],flags=re.I).strip()
            result.append(item)
        return result
    format_switches=[{'index':i,'source':x['code'],'candidate':y['code']} for i,(x,y) in enumerate(zip(a.fields,b.fields)) if x['code']!=y['code'] and content_fields([x])==content_fields([y])]
    for name,left,right in [('ordered_content',a.projection,b.projection),('field_codes_and_displays',content_fields(a.fields),content_fields(b.fields)),('native_OMML_structures',a.math,b.math),('bookmarks_in_order',a.bookmarks,b.bookmarks)]:
        d=first_difference(left,right,name);checks[name]=d is None
        if d is not None:diffs[name]=d
    results=[]
    for i,(ra,rb) in enumerate(zip(a.assets,b.assets)):
        ok,how=asset_equal(a,ra,b,rb);results.append({'index':i,'equal':ok,'comparison':how,'source':ra['target'],'candidate':rb['target']})
    checks['ordered_images_and_embedded_objects']=len(a.assets)==len(b.assets) and all(x['equal'] for x in results)
    checks['embedded_package_parts']=len(a.embedded)==len(b.embedded) and all(a.zip.read(x)==b.zip.read(y) for x,y in zip(a.embedded,b.embedded))
    if not checks['ordered_images_and_embedded_objects']:diffs['assets']={'source_count':len(a.assets),'candidate_count':len(b.assets),'mismatches':[x for x in results if not x['equal']]}
    return {'passed':all(checks.values()),'source':a.path,'candidate':b.path,'checks':checks,'first_differences':diffs,'asset_comparisons':results,'object_counts':{'source':{'fields':len(a.fields),'math':len(a.math),'assets':len(a.assets),'bookmarks':len(a.bookmarks)},'candidate':{'fields':len(b.fields),'math':len(b.math),'assets':len(b.assets),'bookmarks':len(b.bookmarks)}},'allowed_REF_format_switch_changes':format_switches,'math_normalization':'Math runs compared as ordered styled characters within unchanged operator/argument topology. Script/italic selectors on punctuation are normalized because Word merges those glyphs into adjacent runs; letter/digit script and bold/italic, normal-text mode and all operators/arguments remain checked.','excluded_from_content_equivalence':['TOC and page-number fields, including their display','formatting and pagination properties','REF CHARFORMAT switch only; original bookmark arguments and displayed text remain compared','empty formatting-only paragraphs','automatic _Toc and _GoBack bookmarks','headers/footers: recorded separately below'],'running_material_changes':{'source':a.running,'candidate':b.running}}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('source',type=Path);ap.add_argument('candidate',nargs='?',type=Path);ap.add_argument('--inventory-prefix',type=Path);ap.add_argument('--report',type=Path);args=ap.parse_args()
    source=Document(args.source)
    if args.inventory_prefix:
        inv=source.inventory();prefix=args.inventory_prefix;prefix.parent.mkdir(parents=True,exist_ok=True)
        Path(str(prefix)+'.json').write_text(json.dumps(inv,ensure_ascii=False,indent=2)+'\n')
        with Path(str(prefix)+'_paragraphs.jsonl').open('w') as f:
            for i,p in enumerate(source.paragraphs):f.write(json.dumps({'index':i,**p},ensure_ascii=False)+'\n')
    if args.candidate:
        report=compare(source,Document(args.candidate))
        if args.report:args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({k:report[k] for k in ['passed','checks','first_differences','object_counts']},ensure_ascii=False,indent=2))
        raise SystemExit(0 if report['passed'] else 1)
    print(json.dumps({'paragraphs':len(source.paragraphs),'substantive_fields':len(source.fields),'native_OMML':len(source.math),'assets':len(source.assets),'bookmarks':len(source.bookmarks),'embedded_parts':len(source.embedded)},ensure_ascii=False))
if __name__=='__main__':main()
