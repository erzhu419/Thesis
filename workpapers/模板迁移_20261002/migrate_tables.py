"""In-place, formatting-only table migration. No file I/O or content rebuilding."""
from collections import Counter
from copy import deepcopy
import re
from lxml import etree as E


def format_tables(docroot, template_docroot, NS):
    """Format existing OOXML tables and return the observed mapping/counts."""
    w = '{' + NS['w'] + '}'
    m = '{' + NS['m'] + '}'
    def child(parent, name, first=False):
        e = parent.find(w + name)
        if e is None:
            e = E.Element(w + name)
            if name == 'rPr' and parent.tag == m+'r':
                parent.insert(1 if len(parent) and parent[0].tag == m+'rPr' else 0,e)
            elif first:parent.insert(0,e)
            else:parent.append(e)
        return e
    def attrs(e):
        return {} if e is None else {E.QName(k).localname:v for k,v in e.attrib.items()}
    def active(e):
        return e is not None and e.get(w+'val') not in (None,'nil','none','0')
    def is_three_line(table):
        borders = table.find('w:tblPr/w:tblBorders',NS)
        if borders is None or not all(active(borders.find(w+k)) for k in ('top','bottom')):
            return False
        if any(active(borders.find(w+k)) for k in ('left','right','insideH','insideV')):
            return False
        rows = table.findall('w:tr',NS)
        for ri,row in enumerate(rows):
            for edge in row.findall('w:tc/w:tcPr/w:tcBorders/*',NS):
                name=E.QName(edge).localname
                allowed=(ri==0 and name in ('top','bottom')) or (ri==1 and name=='top') or (ri==len(rows)-1 and name=='bottom')
                if active(edge) and not allowed:return False
        return True
    def apply_font(rpr, halfpoints, math=False):
        if not math:
            fonts=child(rpr,'rFonts',True)
            for key in ('asciiTheme','hAnsiTheme','eastAsiaTheme','csTheme'):
                fonts.attrib.pop(w+key,None)
            for key,value in [('ascii','Times New Roman'),('hAnsi','Times New Roman'),('cs','Times New Roman'),('eastAsia','宋体')]:
                fonts.set(w+key,value)
        for key in ('sz','szCs'):child(rpr,key).set(w+'val',str(halfpoints))
    def paragraph_format(p, formula, is_header):
        pp=child(p,'pPr',True)
        # Direct minimum spacing overrides a fixed body-style line box.
        sp=child(pp,'spacing')
        if not sp.get(w+'line'):sp.set(w+'line','400' if formula else '300')
        if sp.get(w+'lineRule') not in ('auto','atLeast'):sp.set(w+'lineRule','atLeast')
        if sp.get(w+'before') is None:sp.set(w+'before','100' if formula else '60')
        if sp.get(w+'after') is None:sp.set(w+'after','100' if formula else '60')
        if not formula:
            sp.attrib.clear()
            for key,val in [('line','240'),('lineRule','auto'),('before','0'),('after','0')]:sp.set(w+key,val)
            child(pp,'jc').set(w+'val','center')
        ind=child(pp,'ind');ind.set(w+'firstLine','0');ind.set(w+'firstLineChars','0')
        for key in ('hanging','hangingChars'):ind.attrib.pop(w+key,None)
        # Preserve explicit centering/right alignment (notably equation numbers).
        if pp.find(w+'jc') is None and not formula:child(pp,'jc').set(w+'val','center')
        if not formula and not is_header:
            for e in pp.findall(w+'keepNext'):pp.remove(e)
        apply_font(child(pp,'rPr'),24 if formula else 21)
        for r in p.findall('.//w:r',NS):apply_font(child(r,'rPr',True),24 if formula else 21)
        if not formula:
            for r in p.findall('.//m:r',NS):apply_font(child(r,'rPr'),21,math=True)
            for cp in p.findall('.//m:ctrlPr',NS):apply_font(child(cp,'rPr'),21,math=True)
    # Template body sections agree on this width; covers may use different margins.
    widths=[]
    for sect in template_docroot.findall('.//w:sectPr',NS):
        page=sect.find('w:pgSz',NS);margin=sect.find('w:pgMar',NS)
        if page is not None and margin is not None:
            widths.append(int(page.get(w+'w'))-int(margin.get(w+'left'))-int(margin.get(w+'right'))-int(margin.get(w+'gutter','0')))
    usable=Counter(widths).most_common(1)[0][0]
    template_tables=template_docroot.findall('.//w:tbl',NS)
    samples=[(i,t) for i,t in enumerate(template_tables) if is_three_line(t)]
    # The template's table 26/27 family has consistent 1.5pt outer, 0.75pt header lines.
    chosen=next((i,t) for i,t in samples if t.find('w:tblPr/w:tblBorders/w:top',NS).get(w+'sz')=='12')
    sample_index,sample=chosen
    sample_borders=sample.find('w:tblPr/w:tblBorders',NS)
    header_edge=next(e for e in sample.findall('w:tr/w:tc/w:tcPr/w:tcBorders/w:bottom',NS) if active(e))
    counts=Counter(); old_widths=Counter(); eq_shapes=Counter(); field_count=0
    for table in docroot.findall('.//w:tbl',NS):
        rows=table.findall('w:tr',NS);grid=table.findall('w:tblGrid/w:gridCol',NS)
        codes=''.join(table.xpath('.//w:instrText/text()',namespaces=NS))
        equation=bool(re.search(r'\bSEQ\s+Eq',codes)) and len(grid)==3
        field_count+=len(re.findall(r'SEQ\s+Eq',codes))
        role='equation_layout' if equation else 'data';counts[role]+=1
        if equation:eq_shapes[str((len(rows),len(grid)))]+=1
        pr=child(table,'tblPr',True)
        if not equation and is_three_line(table):
            existing=pr.find(w+'tblBorders');pr.replace(existing,deepcopy(sample_borders))
            for ri,row in enumerate(rows):
                for cell in row.findall('w:tc',NS):
                    cp=child(cell,'tcPr',True);cb=cp.find(w+'tcBorders')
                    if ri==0:
                        cb=child(cp,'tcBorders');edge=cb.find(w+'bottom')
                        if edge is not None:cb.remove(edge)
                        cb.append(deepcopy(header_edge))
                    if cb is not None:
                        for name,where in [('top',0),('bottom',len(rows)-1)]:
                            edge=cb.find(w+name)
                            if ri==where and active(edge):cb.replace(edge,deepcopy(sample_borders.find(w+name)))
            counts['three_line_template_mapping']+=1
        elif not equation:counts['preserved_other_border_structure']+=1
        widths_now=[int(c.get(w+'w')) for c in grid]
        total=sum(widths_now);old_widths[str(total)]+=1
        target=min(total,usable)
        if total>usable:
            scaled=[round(v*target/total) for v in widths_now]
            scaled[-1]+=target-sum(scaled)
            for c,val in zip(grid,scaled):c.set(w+'w',str(val))
            for cw in table.findall('w:tr/w:tc/w:tcPr/w:tcW',NS):
                if cw.get(w+'type','dxa')=='dxa':cw.set(w+'w',str(round(int(cw.get(w+'w'))*target/total)))
            counts['width_reduced']+=1
        tw=child(pr,'tblW');tw.set(w+'type','dxa');tw.set(w+'w',str(target))
        child(pr,'jc').set(w+'val','center')
        for row_index,row in enumerate(rows):
            for h in row.findall('w:trPr/w:trHeight',NS):
                if h.get(w+'hRule')=='exact':h.set(w+'hRule','atLeast');counts['exact_row_heights_relaxed']+=1
            header=row_index==0 and not equation
            for p in row.findall('w:tc/w:p',NS):paragraph_format(p,equation,header)
    return {'usable_width_twips':usable,'template_three_line_sample_zero_based_index':sample_index,
            'template_border_attributes':{E.QName(e).localname:attrs(e) for e in sample_borders},
            'template_header_bottom':attrs(header_edge),'counts':dict(counts),'equation_SEQ_fields_in_tables':field_count,
            'equation_table_shapes':dict(eq_shapes),'original_widths_twips':dict(old_widths),
            'classification':'SEQ Eq field plus three-column layout; inline OMML alone never makes a data table an equation table.',
            'typography':'Data: Song/TNR 10.5pt, native math 10.5pt with original math font and m:sty; formula layouts: normal text/number 12pt, native OMML formatting retained. Existing bold/italic and explicit cell alignment retained.',
            'spacing':'Data tables: template implicit single spacing made explicit, before/after zero. Formula tables: preserve source minimum/automatic spacing, replace exact with atLeast.',
            'borders':'Map only recognized simple three-line data tables to an actual template example; preserve other borders. Equation-table borders and column roles untouched.',
            'width_advice':'Widths constrain the table frame, not the rendered length of an unbreakable OMML expression. Review genuinely overwide equations visually; resize their font only if needed, keeping OMML intact. Scale large drawings by both extents proportionally to body width, not their image pixels.'}
