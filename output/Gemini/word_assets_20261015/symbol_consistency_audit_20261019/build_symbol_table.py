"""Replace only the reviewed notation table and its introductory convention."""
from pathlib import Path
from zipfile import ZipFile
from copy import deepcopy, copy
from lxml import etree as E
import json
import re
import pypandoc

A = Path(__file__).resolve().parent
SOURCE = A / 'source_snapshot.docx'
OUTPUT = A.parents[1] / '调整完格式的_符号表修订_20261019.docx'
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
M = '{http://schemas.openxmlformats.org/officeDocument/2006/math}'
W14 = '{http://schemas.microsoft.com/office/word/2010/wordml}'
plan = json.loads((A / 'symbol_table_plan.json').read_text())
changes = [row for row in plan['rows'] if row['action'] in ('update', 'add')]

def inline_notation(text):
    """Typeset notation already present in the reviewed Chinese definitions."""
    symbols = {
        'd^{svc}(h)': r'd^{svc}(h)', '𝒱_πQ': r'\mathcal V_\pi Q',
        '𝒱_π Q': r'\mathcal V_\pi Q', 'κ′_k': r'\kappa_k^\prime',
        'κ_k': r'\kappa_k', 'κ_ale': r'\kappa_{ale}',
        'τ_IQL': r'\tau_{IQL}', 'τ_e': r'\tau_e', 'τ_h': r'\tau_h',
        'c_pen(s,a)': r'c_{pen}(s,a)', 'c_pen': r'c_{pen}',
        'c_const': r'c_{const}', 'm_grp': r'm_{grp}', 'h_s': r'h_s',
        'ρ_ipm': r'\rho_{ipm}', 'ε_act': r'\varepsilon_{act}',
        'x_0': r'x_0', 'Δd': r'\mathrm{\Delta}d', 'H∘G': r'H\circ G',
        'ρ(z)': r'\rho(z)', 'b(z)': r'b(z)',
        '𝒝': r'\mathcal B', '𝓛': r'\mathcal L', '𝒫': r'\mathcal P',
        '𝒰': r'\mathcal U', '𝒱': r'\mathcal V',
        'ν': r'\nu', 'ξ': r'\xi', 'ζ': r'\zeta', 'τ': r'\tau',
        'ε': r'\varepsilon', 'ρ': r'\rho', 'κ': r'\kappa',
    }
    pattern = re.compile('|'.join(re.escape(s) for s in sorted(symbols, key=len, reverse=True)))
    text = pattern.sub(lambda m: '$' + symbols[m[0]] + '$', text)
    parts = re.split(r'(\$[^$]*\$)', text)
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r'(?<![A-Za-z_])([A-Za-z])(?![A-Za-z_])', r'$\1$', parts[i])
    return ''.join(parts)

donor_md = '\n\n'.join('$$' + row['correct_tex'] + '$$\n\n' +
                        inline_notation(row['correct_definition']) for row in changes)
(A / 'symbol_donor.md').write_text(donor_md, encoding='utf-8')
pypandoc.convert_text(donor_md, 'docx', format='markdown+tex_math_dollars',
                     outputfile=str(A / 'symbol_donor.docx'))
with ZipFile(A / 'symbol_donor.docx') as z:
    donor_root = E.fromstring(z.read('word/document.xml'))
donor_paragraphs = donor_root.find(W + 'body').findall(W + 'p')
assert len(donor_paragraphs) == 2 * len(changes)
donor_by_row = {id(row): donor_paragraphs[2 * i].find('.//' + M + 'oMath')
                for i, row in enumerate(changes)}
definition_by_row = {id(row): donor_paragraphs[2 * i + 1] for i, row in enumerate(changes)}

def symbol_shape(node):
    """Match actual variable structures while allowing roman/italic label restoration."""
    tag = E.QName(node).localname
    if tag.endswith('Pr'):
        return None
    if node.tag == M + 'r':
        props = node.find(M + 'rPr')
        scr = props.find(M + 'scr') if props is not None else None
        sty = props.find(M + 'sty') if props is not None else None
        script = scr.get(M + 'val') if scr is not None else 'roman'
        bold = sty is not None and sty.get(M + 'val') in ('b', 'bi')
        chars = ''.join(t.text or '' for t in node.iter(M + 't'))
        return ('r', tuple((c, script if c.isalpha() else '', bold if c.isalnum() else False)
                           for c in chars if not c.isspace()))
    parts = []
    for child in node:
        if not child.tag.startswith(M):
            continue
        shape = symbol_shape(child)
        if shape is None:
            continue
        if parts and shape[0] == 'r' and parts[-1][0] == 'r':
            parts[-1] = ('r', parts[-1][1] + shape[1])
        else:
            parts.append(shape)
    accent = node.find(M + 'accPr/' + M + 'chr')
    accent = accent.get(M + 'val') if accent is not None else '\u0302' if tag == 'acc' else ''
    return (tag, accent, tuple(parts))

def adopt_actual_styles(math, source_atoms):
    reused = 0
    for node in list(math.iter())[::-1]:
        if node is math or node.tag not in (M + 'r', M + 'sSub', M + 'sSup', M + 'sSubSup', M + 'acc', M + 'bar'):
            continue
        actual = source_atoms.get(symbol_shape(node))
        if actual is not None and node.getparent() is not None:
            node.getparent().replace(node, deepcopy(actual))
            reused += 1
    return reused

def math_font(math):
    for parent in list(math.iter(M + 'r')) + list(math.iter(M + 'ctrlPr')):
        props = parent.find(W + 'rPr')
        if props is None:
            props = E.Element(W + 'rPr')
            parent.insert(1 if parent.find(M + 'rPr') is not None else 0, props)
        for child in list(props):
            props.remove(child)
        fonts = E.SubElement(props, W + 'rFonts')
        for key in ('ascii', 'hAnsi', 'cs', 'eastAsia'):
            fonts.set(W + key, 'Cambria Math')
        E.SubElement(props, W + 'noProof')
        for key in ('sz', 'szCs'):
            E.SubElement(props, W + key).set(W + 'val', '21')

def replace_cell(cell, children):
    paragraph = cell.find(W + 'p')
    assert paragraph is not None
    for child in list(paragraph):
        if child.tag != W + 'pPr':
            paragraph.remove(child)
    for child in children:
        paragraph.append(child)
    for other in cell.findall(W + 'p')[1:]:
        cell.remove(other)

def text_run(template, text):
    run = E.Element(W + 'r')
    props = template.find(W + 'rPr')
    if props is not None:
        run.append(deepcopy(props))
    E.SubElement(run, W + 't').text = text
    return run

with ZipFile(SOURCE) as z:
    root = E.fromstring(z.read('word/document.xml'))
    source_atoms = {}
    body_started = False
    for child in root.find(W + 'body'):
        if child.tag == W + 'p' and ''.join(child.itertext()) == '第1章 绪论':
            body_started = True
        if body_started:
            for math in child.iter(M + 'oMath'):
                for node in math.iter():
                    if node.tag in (M + 'r', M + 'sSub', M + 'sSup', M + 'sSubSup', M + 'acc', M + 'bar'):
                        source_atoms.setdefault(symbol_shape(node), node)
    assert source_atoms
    table = list(root.iter(W + 'tbl'))[2]
    original_rows = list(table.findall(W + 'tr'))
    assert len(original_rows) == 99
    for row in original_rows[1:]:
        table.remove(row)
    for item in plan['rows']:
        if item['action'] == 'delete':
            continue
        row = deepcopy(original_rows[item['source_row']]) if item['source_row'] else deepcopy(original_rows[1])
        if item['action'] == 'add':
            for node in row.iter():
                for key in (W14 + 'paraId', W14 + 'textId'):
                    node.attrib.pop(key, None)
        if item['action'] in ('update', 'add'):
            cells = row.findall(W + 'tc')
            math = deepcopy(donor_by_row[id(item)])
            adopt_actual_styles(math, source_atoms)
            math_font(math)
            replace_cell(cells[0], [math])
            template = next(cells[1].iter(W + 'r'))
            content = []
            for child in definition_by_row[id(item)]:
                if child.tag == W + 'pPr':
                    continue
                child = deepcopy(child)
                for embedded in child.iter(M + 'oMath'):
                    adopt_actual_styles(embedded, source_atoms)
                    math_font(embedded)
                if child.tag == W + 'r':
                    props = child.find(W + 'rPr')
                    if props is not None:
                        child.remove(props)
                    if template.find(W + 'rPr') is not None:
                        child.insert(0, deepcopy(template.find(W + 'rPr')))
                content.append(child)
            replace_cell(cells[1], content)
        table.append(row)
    intro = next(p for p in root.iter(W + 'p') if ''.join(p.itertext()).startswith('数学记号按完整的'))
    template = next(intro.iter(W + 'r'))
    for child in list(intro):
        if child.tag != W + 'pPr':
            intro.remove(child)
    intro.append(text_run(template, plan['intro']))
    with ZipFile(OUTPUT, 'w') as out:
        for item in z.infolist():
            out.writestr(copy(item), E.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
                         if item.filename == 'word/document.xml' else z.read(item.filename))
print(OUTPUT)
print('notation rows:', len(table.findall(W + 'tr')) - 1)
