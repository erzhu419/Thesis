#!/usr/bin/env python3
"""Read DOCX XML without extracting media or modifying the package.
Produces reusable semantic samples, property layers and OOXML-effective values.
Usage: python3 template_analysis.py [template.docx] [output_prefix]
"""
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as E
from collections import Counter, defaultdict
import json,re,sys,copy,posixpath
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'm':'http://schemas.openxmlformats.org/officeDocument/2006/math',
    'a':'http://schemas.openxmlformats.org/drawingml/2006/main',
    'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    'o':'urn:schemas-microsoft-com:office:office'}
W='{'+NS['w']+'}'; R='{'+NS['r']+'}'
def local(x): return x.split('}')[-1]
def val(e,default=None): return e.get(W+'val',default) if e is not None else default
def attrs(e): return {local(k):v for k,v in e.attrib.items()} if e is not None else {}
def xml(e): return E.tostring(e,encoding='unicode') if e is not None else None
def text(e): return ''.join(x.text or '' for x in e.iter() if local(x.tag) in ['t','delText'] and x.tag.startswith((W,'{'+NS['m']+'}')))
def props(e):
    if e is None:return {}
    d={}
    for c in e:
        k=local(c.tag)
        if k in ['rPr','sectPr','pPrChange','rPrChange','tblPrChange','tcPrChange','trPrChange']:continue
        v=attrs(c)
        if len(c): v['children']=[{'tag':local(a.tag),**attrs(a)} for a in c]
        if not v:v={'val':'1'}
        if k in d:
            if not isinstance(d[k],list):d[k]=[d[k]]
            d[k].append(v)
        else:d[k]=v
    return d
TOGGLES={'b','bCs','i','iCs','caps','smallCaps','strike','dstrike','outline','shadow','emboss','imprint','vanish','specVanish'}
def merge(base,add,toggle=False):
    out=copy.deepcopy(base)
    for k,v in add.items():
        if toggle and k in TOGGLES:
            if v.get('val','1') not in ['0','false','off']:
                old=out.get(k,{}).get('val','0') not in ['0','false','off']
                out[k]={'val':'0' if old else '1'}
        elif isinstance(v,dict) and isinstance(out.get(k),dict):out[k].update(copy.deepcopy(v))
        else:out[k]=copy.deepcopy(v)
    return out

def main(template,prefix):
    z=ZipFile(template)
    root=E.fromstring(z.read('word/document.xml'));body=root.find('w:body',NS)
    sr=E.fromstring(z.read('word/styles.xml'));settings=E.fromstring(z.read('word/settings.xml'))
    styles={x.get(W+'styleId'):x for x in sr.findall('w:style',NS)}
    default_style=next((k for k,x in styles.items() if x.get(W+'type')=='paragraph' and x.get(W+'default')=='1'),'Normal')
    default_p=props(sr.find('w:docDefaults/w:pPrDefault/w:pPr',NS));default_r=props(sr.find('w:docDefaults/w:rPrDefault/w:rPr',NS))
    def chain(sid):
        result=[];seen=set()
        while sid in styles and sid not in seen:
            seen.add(sid);result.insert(0,sid);sid=val(styles[sid].find('w:basedOn',NS))
        return result
    def styleprops(sid,kind):
        out={}
        for k in chain(sid):out=merge(out,props(styles[k].find('w:'+kind,NS)),toggle=kind=='rPr')
        return out
    def apply_style(base,sid,kind):
        for k in chain(sid):base=merge(base,props(styles[k].find('w:'+kind,NS)),toggle=kind=='rPr')
        return base
    themes={}
    theme=E.fromstring(z.read('word/theme/theme1.xml')) if 'word/theme/theme1.xml' in z.namelist() else None
    if theme is not None:
        for family in ['majorFont','minorFont']:
            node=theme.find('.//a:fontScheme/a:'+family,NS)
            if node is not None:themes[family]={local(v.tag)+(':'+v.get('script') if v.get('script') else ''):v.get('typeface','') for v in node}
    def describe_r(p):
        f=p.get('rFonts',{});fonts={};source={}
        for role,key in [('latin','ascii'),('east_asia','eastAsia'),('high_ansi','hAnsi'),('complex_script','cs')]:
            tk={'ascii':'asciiTheme','eastAsia':'eastAsiaTheme','hAnsi':'hAnsiTheme','cs':'cstheme'}[key]
            if tk in f:
                fam='majorFont' if f[tk].startswith('major') else 'minorFont';ft=themes.get(fam,{})
                fonts[role]=(ft.get('font:Hans') or ft.get('ea')) if key=='eastAsia' else (ft.get('cs') if key=='cs' else ft.get('latin'))
                source[role]='theme:'+f[tk]
            elif key in f:fonts[role]=f[key];source[role]='explicit'
            else:fonts[role]=None;source[role]='not specified'
        d={'fonts':fonts,'font_sources':source,'size_pt':float(p['sz']['val'])/2 if 'sz' in p else None,
           'bold':p.get('b',{}).get('val','0') not in ['0','false','off'],
           'italic':p.get('i',{}).get('val','0') not in ['0','false','off'],
           'vertical_align':p.get('vertAlign',{}).get('val','baseline'),'color':p.get('color',{}).get('val','unspecified')}
        return d
    def describe_p(p):
        d={'alignment':p.get('jc',{}).get('val','unspecified'),'outline_level':p.get('outlineLvl',{}).get('val','body/unspecified'),'spacing':p.get('spacing',{}),'indent':p.get('ind',{}),'tabs':p.get('tabs',{})}
        sp=d['spacing'];it=d['indent']
        for k in ['before','after','line']:
            if k in sp and (k!='line' or sp.get('lineRule','auto')!='auto'):d[k+'_pt']=float(sp[k])/20
        if sp.get('lineRule','auto')=='auto' and 'line' in sp:d['line_multiple']=float(sp['line'])/240
        for k in ['firstLine','left','right','hanging']:
            if k in it:d[k+'_pt']=float(it[k])/20
        for k in ['firstLineChars','leftChars','rightChars','hangingChars']:
            if k in it:d[k+'_characters']=float(it[k])/100
        for k in ['before','after']:
            if sp.get(k+'Autospacing','0') not in ['0','false','off']:
                d['effective_'+k]={'unit':'automatic'}
            elif k+'Lines' in sp:d['effective_'+k]={'value':float(sp[k+'Lines'])/100,'unit':'lines'}
            elif k in sp:d['effective_'+k]={'value':float(sp[k])/20,'unit':'pt'}
        if 'hangingChars' in it:d['effective_first_line']={'value':-float(it['hangingChars'])/100,'unit':'characters','kind':'hanging'}
        elif 'hanging' in it:d['effective_first_line']={'value':-float(it['hanging'])/20,'unit':'pt','kind':'hanging'}
        elif 'firstLineChars' in it:d['effective_first_line']={'value':float(it['firstLineChars'])/100,'unit':'characters','kind':'firstLine'}
        elif 'firstLine' in it:d['effective_first_line']={'value':float(it['firstLine'])/20,'unit':'pt','kind':'firstLine'}
        d['indent_precedence']='Character units precede twips for the same property; hanging precedes firstLine. Raw values retained above are not all effective.'
        return d
    # Numbering properties and text are retained separately so labels are not mistaken for body text.
    nr=E.fromstring(z.read('word/numbering.xml')) if 'word/numbering.xml' in z.namelist() else E.Element('none')
    nums={x.get(W+'numId'):x for x in nr.findall('w:num',NS)};abnums={x.get(W+'abstractNumId'):x for x in nr.findall('w:abstractNum',NS)}
    def numbering(pp):
        np=pp.get('numPr',{});children={v['tag']:v.get('val') for v in np.get('children',[])}
        ni=children.get('numId');il=children.get('ilvl','0')
        if not ni or ni=='0' or ni not in nums:return None
        an=val(nums[ni].find('w:abstractNumId',NS));ab=abnums.get(an)
        lv=next((v for v in ab.findall('w:lvl',NS) if v.get(W+'ilvl')==il),None) if ab is not None else None
        ov=next((v for v in nums[ni].findall('w:lvlOverride',NS) if v.get(W+'ilvl')==il),None)
        if ov is not None and ov.find('w:lvl',NS) is not None:lv=ov.find('w:lvl',NS)
        if lv is None:return {'numId':ni,'ilvl':il,'unresolved':True}
        return {'numId':ni,'abstractNumId':an,'ilvl':il,'level':props(lv),'pPr':props(lv.find('w:pPr',NS)),'rPr':props(lv.find('w:rPr',NS)),'override':xml(ov)}
    rels={x.get('Id'):x.get('Target') for x in E.fromstring(z.read('word/_rels/document.xml.rels'))}
    # Section properties describe the content ending at that sectPr, not the next content.
    sections=[];start=0;inherited_hf={}
    for bi,node in enumerate(body):
        sec=node if node.tag==W+'sectPr' else node.find('w:pPr/w:sectPr',NS)
        if sec is None:continue
        refs={local(v.tag)+':'+v.get(W+'type'):rels.get(v.get(R+'id')) for v in sec if local(v.tag) in ['headerReference','footerReference']}
        inherited_hf.update(refs)
        seg=list(body)[start:bi+1];texts=[text(t).strip() for t in seg if text(t).strip()]
        pg=attrs(sec.find('w:pgSz',NS));mar=attrs(sec.find('w:pgMar',NS))
        sections.append({'index':len(sections)+1,'body_start':start,'body_end':bi,'first_text':texts[:2],'last_text':texts[-1:],'properties':props(sec),'page_cm':{k:round(float(v)*2.54/1440,3) for k,v in pg.items() if k in ['w','h']},'margins_cm':{k:round(float(v)*2.54/1440,3) for k,v in mar.items()},'header_footer_explicit':refs,'header_footer_resolved':dict(inherited_hf),'break_type':val(sec.find('w:type',NS),'nextPage')})
        start=bi+1
    rows=[];tables=[];table_context={};parent_map={c:p for p in root.iter() for c in p}
    for ti,tbl in enumerate(body.findall('w:tbl',NS)):
        tp=tbl.find('w:tblPr',NS);ts=val(tp.find('w:tblStyle',NS)) if tp is not None else None
        trs=tbl.findall('w:tr',NS)
        cells=[]
        for ri,tr in enumerate(trs):
            for ci,tc in enumerate(tr.findall('w:tc',NS)):
                cp=props(tc.find('w:tcPr',NS));cells.append({'row':ri,'column':ci,'text':text(tc)[:70],'tcPr':cp})
                for pp in tc.findall('.//w:p',NS):table_context[pp]={'table':ti,'row':ri,'column':ci,'style':ts,'tcPr':cp}
        tables.append({'index':ti,'body_index':list(body).index(tbl),'rows':len(trs),'column_counts':[len(t.findall('w:tc',NS)) for t in trs],'text':text(tbl)[:180],
                       'tblPr':props(tp),'grid':[attrs(g) for g in tbl.findall('w:tblGrid/w:gridCol',NS)],'style_id':ts,'style_effective_tblPr':styleprops(ts,'tblPr'),
                       'style_conditional':[xml(v) for k in chain(ts) for v in styles[k].findall('w:tblStylePr',NS)],
                       'row_properties':[props(t.find('w:trPr',NS)) for t in trs],'cell_samples':cells[:6]+cells[-3:]})
    def para_record(p,location,category=None):
        dp=props(p.find('w:pPr',NS));sid=dp.get('pStyle',{}).get('val',default_style);sp=styleprops(sid,'pPr');tc=table_context.get(p)
        effective_p=copy.deepcopy(default_p);effective_r=copy.deepcopy(default_r)
        if tc and tc['style']:
            effective_p=apply_style(effective_p,tc['style'],'pPr');effective_r=apply_style(effective_r,tc['style'],'rPr')
        n=numbering(merge(sp,dp));effective_p=merge(effective_p,n.get('pPr',{}) if n else {})
        effective_p=apply_style(effective_p,sid,'pPr');effective_p=merge(effective_p,dp)
        effective_r=apply_style(effective_r,sid,'rPr')
        rr=[]
        for run in p.iter(W+'r'):
            txt=''.join(t.text or '' for t in run.findall('.//w:t',NS));rp=props(run.find('w:rPr',NS));rs=rp.get('rStyle',{}).get('val')
            er=apply_style(copy.deepcopy(effective_r),rs,'rPr');er=merge(er,rp)
            if txt or run.find('w:footnoteReference',NS) is not None or run.find('w:endnoteReference',NS) is not None:
                rr.append({'text':txt[:90],'chars':len(txt),'character_style':rs,'direct':rp,'effective':er,'display':describe_r(er)})
        dist=Counter();fm={}
        for run in rr:
            key=json.dumps(run['display'],sort_keys=True,ensure_ascii=False);dist[key]+=max(1,run['chars']);fm[key]=run['display']
        dominant=[{'characters':count,**fm[key]} for key,count in dist.most_common(4)]
        return {'location':location,'text':text(p)[:180],'style_id':sid,'style_name':val(styles.get(sid,E.Element('x')).find('w:name',NS)),
                'style_chain':chain(sid),'category':category,'direct_pPr':dp,'paragraph_mark_rPr':props(p.find('w:pPr/w:rPr',NS)),
                'effective_pPr':effective_p,'paragraph_display':describe_p(effective_p),'default_text_run':describe_r(effective_r),
                'dominant_text_formats':dominant,'run_samples':rr[:4]+rr[-1:] if len(rr)>4 else rr,
                'representative_text_run':max(rr,key=lambda r:r['chars']) if rr else None,
                'numbering':n,'table_context':tc,'math_count':len(p.findall('.//m:oMath',NS)),'object_count':len(p.findall('.//w:object',NS)),
                'math_run_samples':[{'text':text(v)[:40],'math_properties':props(v.find('m:rPr',NS)),'text_properties':props(v.find('w:rPr',NS))} for v in p.findall('.//m:r',NS)[:3]],
                'drawing_count':len(p.findall('.//w:drawing',NS)),'fields':[''.join(v.itertext())[:140] for v in p.findall('.//w:instrText',NS)],
                'conditional_table_style_resolved':False if tc and tc['style'] and any(styles[k].find('w:tblStylePr',NS) is not None for k in chain(tc['style'])) else True}
    current='cover';pidx=0
    for bi,node in enumerate(body):
        for pi,p in enumerate(node.iter(W+'p')):
            if parent_map.get(p) is not body and node.tag!=W+'tbl':continue
            t=text(p).strip();sid=val(p.find('w:pPr/w:pStyle',NS),default_style)
            if node.tag==W+'p':
                if t.startswith('摘要：'):current='abstract_zh'
                elif sid=='1' and t.startswith('Mechanical'):current='abstract_en'
                elif sid=='1' and re.sub(r'\s','',t)=='目录':current='toc'
                elif sid=='1' and t=='符号说明':current='symbols'
                elif sid=='1' and re.match(r'第\d+章',t):current='body'
                elif sid=='1' and t=='参考文献':current='references'
                elif sid=='1' and '研究成果' in t:current='achievements'
                elif sid=='1' and re.sub(r'\s','',t)=='致谢':current='acknowledgements'
            if p in table_context:cat='table_text'
            elif sid=='1':cat='chapter_heading' if re.match(r'第\d+章',t) else 'front_back_heading'
            elif sid=='20':cat='heading2'
            elif sid=='3':cat='heading3'
            elif sid=='4':cat='heading4'
            elif sid.startswith('TOC'):cat='toc_'+sid
            elif sid=='aff1':cat='figure_caption'
            elif sid=='aff7':cat='table_caption'
            elif sid=='aff2':cat='display_equation'
            elif sid=='aff5':cat='figure_paragraph'
            elif current=='references' and t:cat='reference_entry'
            elif current.startswith('abstract') and t:cat=current
            elif current=='body' and t:cat='body_text'
            elif current=='cover' and t:cat='cover_declaration'
            else:cat=current if t else 'blank'
            rows.append(para_record(p,{'body_index':bi,'paragraph_index':pidx,'nested_paragraph':pi if node.tag==W+'tbl' else None},cat));pidx+=1
    groups=defaultdict(list)
    for p in rows:groups[p['category']].append(p)
    summary={};samples={}
    for cat,items in groups.items():
        pc=Counter(json.dumps(p['paragraph_display'],sort_keys=True,ensure_ascii=False) for p in items)
        rc=Counter();stylecounts=Counter(p['style_id'] for p in items)
        for p in items:
            for f in p['dominant_text_formats']:
                k=json.dumps({kk:vv for kk,vv in f.items() if kk!='characters'},sort_keys=True,ensure_ascii=False);rc[k]+=f['characters']
        summary[cat]={'paragraphs':len(items),'styles':dict(stylecounts),'paragraph_formats':[{'count':v,**json.loads(k)} for k,v in pc.most_common(5)],'run_formats':[{'characters':v,**json.loads(k)} for k,v in rc.most_common(5)]}
        seen=set();picked=[]
        for p in items:
            sig=json.dumps([p['style_id'],p['paragraph_display'],p['dominant_text_formats'][:1]],sort_keys=True,ensure_ascii=False)
            if sig not in seen:seen.add(sig);picked.append(p)
            if len(picked)>=5:break
        samples[cat]=picked
    auxiliary={}
    for part in z.namelist():
        if re.match(r'word/(header\d+|footer\d+|footnotes|endnotes)\.xml$',part):
            ar=E.fromstring(z.read(part));ps=ar.findall('.//w:p',NS)
            auxiliary[part]={'text':text(ar)[:400],'field_codes':[a.text for a in ar.findall('.//w:instrText',NS)],'paragraphs':[para_record(p,{'part':part,'paragraph_index':i}) for i,p in enumerate(ps[:5])],
                             'note_ids':[attrs(v) for v in ar if local(v.tag) in ['footnote','endnote']]}
    styles_out={k:{'name':val(v.find('w:name',NS)),'type':v.get(W+'type'),'basedOn':val(v.find('w:basedOn',NS)),'next':val(v.find('w:next',NS)),'link':val(v.find('w:link',NS)),'chain':chain(k),'pPr':props(v.find('w:pPr',NS)),'rPr':props(v.find('w:rPr',NS)),'effective_pPr':apply_style(copy.deepcopy(default_p),k,'pPr'),'effective_rPr':apply_style(copy.deepcopy(default_r),k,'rPr')} for k,v in styles.items()}
    mathpr=settings.find('m:mathPr',NS)
    report={'template':str(Path(template).resolve()),'package_bytes':Path(template).stat().st_size,'method':'OOXML only; all indexes zero-based. pPr/rPr direct properties resolved over document defaults and basedOn styles; paragraph mark rPr is not applied to ordinary runs. Style toggle properties honored. Theme fonts resolved for Hans/Latin. Conditional table styles retained but not guessed where active.',
            'defaults':{'paragraph_style':default_style,'pPr':default_p,'rPr':default_r,'theme_fonts':themes},'settings':{k:xml(settings.find('w:'+k,NS)) for k in ['evenAndOddHeaders','mirrorMargins','defaultTabStop','compat','footnotePr','endnotePr']},
            'sections':sections,'styles':styles_out,'category_summary':summary,'representative_samples':samples,'tables':tables,'auxiliary_parts':auxiliary,
            'formulas':{'omml_total':len(root.findall('.//m:oMath',NS)),'display_omml_paragraphs':len(root.findall('.//m:oMathPara',NS)),'legacy_objects':len(root.findall('.//w:object',NS)),'math_settings':xml(mathpr),'math_run_properties_counts':dict(Counter(xml(v) for v in root.findall('.//m:rPr',NS)).most_common(10)),
                        'math_text_properties_counts':[{'count':n,'properties':json.loads(k)} for k,n in Counter(json.dumps(props(v.find('w:rPr',NS)),ensure_ascii=False,sort_keys=True) for v in root.findall('.//m:r',NS)).most_common(12)]},
            'notes':['Missing properties are not invented; application defaults/font fallback still require Word rendering.','Section properties belong to the section ending at their position. Header/footer omissions inherit prior same-type reference; first/even activation also depends on titlePg/evenAndOddHeaders.','A paragraph-mark rPr is not a replacement for the actual run formatting.','Default 20pt exact body spacing can be overridden by per-paragraph line/lineRule and section document-grid settings.','Style IDs are not semantic names: heading 2 uses ID 20, ID 2 is an unused reference style.'],
            'primary_spec_reference':'https://learn.microsoft.com/en-us/dotnet/api/documentformat.openxml.wordprocessing.runstyle?view=openxml-3.0.1'}
    pre=Path(prefix);pre.parent.mkdir(parents=True,exist_ok=True)
    pre.with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    pre.with_name(pre.name+'_paragraphs.jsonl').write_text('\n'.join(json.dumps(p,ensure_ascii=False) for p in rows)+'\n')
    print(json.dumps({'report':str(pre.with_suffix('.json')),'paragraph_records':len(rows),'sections':len(sections),'tables':len(tables),'categories':{k:v['paragraphs'] for k,v in summary.items()}},ensure_ascii=False))
if __name__=='__main__':
    template=Path(sys.argv[1]) if len(sys.argv)>1 else next(Path('.').glob('*gty*.docx'))
    prefix=sys.argv[2] if len(sys.argv)>2 else 'workpapers/模板迁移_20261002/template_analysis'
    main(template,prefix)
