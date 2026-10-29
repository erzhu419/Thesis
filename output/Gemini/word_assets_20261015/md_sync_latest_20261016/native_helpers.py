from lxml import etree as E
from difflib import SequenceMatcher
from zipfile import ZipFile
import copy, re, unicodedata
W="{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
M="{http://schemas.openxmlformats.org/officeDocument/2006/math}"
P="{http://schemas.microsoft.com/office/word/2010/wordml}"
FONT={W+x for x in ("rFonts","sz","szCs","lang","kern")}
EMPH={W+x for x in ("b","bCs","i","iCs")}
def text(e):
    return ''.join((n.text or '' for n in e.iter() if n.tag in (W + 't', M + 't')))

def norm(s):
    return ''.join((c for c in unicodedata.normalize('NFKC', s).replace('−', '-').replace('′', "'").replace('ℓ', 'l').replace('ϵ', 'ε').replace('ς', 'σ') if not c.isspace() and c != '\u200b'))

def read(p):
    with ZipFile(p) as z:
        return E.fromstring(z.read('word/document.xml'))

def core(r):
    return [e for e in r.find(W + 'body') if norm(text(e)) and (not text(e).startswith('DISPLAY_EQUATION_'))]

def mathkey(e):
    if e.tag in (W + 'rPr', M + 'ctrlPr', M + 'argPr'):
        return None
    content = (e.text or '').replace('−', '-').replace('′', "'").replace('ϵ', 'ε').replace('ς', 'σ')
    children = tuple((k for ch in e if (k := mathkey(ch)) is not None))
    return (e.tag, tuple(sorted(e.attrib.items())), content, children)

def signature(e):
    if e.tag == W + 'pPr' or e.tag in (W + 'tcPr', W + 'tblPr', W + 'tblGrid'):
        return None
    attrs = tuple(sorted(((k, v) for (k, v) in e.attrib.items() if k not in (P + 'paraId', P + 'textId'))))
    return (e.tag, attrs, e.text or '', tuple((k for ch in e if (k := signature(ch)) is not None)))

def category(e):
    if e.tag == W + 'tbl':
        return 'table'
    st = e.find(W + 'pPr/' + W + 'pStyle')
    v = st.get(W + 'val', '') if st is not None else ''
    if v.lower().startswith('heading'):
        return v
    m = re.match('([图表])\\d+-\\d+\\s', text(e))
    return m[1] if m else 'p'

def base_rpr(p):
    candidates = [r.find(W + 'rPr') for r in p.iter(W + 'r') if r.find(W + 't') is not None and (r.find(W + 't').text or '').strip()]
    # The first ordinary text run establishes the paragraph's body font.
    # A later accented Latin character may have a character-specific fallback;
    # applying that fallback to all new Chinese text changes its font.
    source = candidates[0] if candidates else None
    result = E.Element(W + 'rPr')
    if source is not None:
        for x in source:
            if x.tag in FONT:
                result.append(copy.deepcopy(x))
    for tag in (W + 'sz', W + 'szCs'):
        if result.find(tag) is None:
            x = p.find(W + 'pPr/' + W + 'rPr/' + tag)
            if x is not None:
                result.append(copy.deepcopy(x))
    return result

def merge_font(run, base):
    rp = run.find(W + 'rPr')
    if rp is None:
        rp = E.Element(W + 'rPr')
        run.insert(0, rp)
    for x in base:
        previous = rp.find(x.tag)
        if previous is not None:
            rp.remove(previous)
        rp.append(copy.deepcopy(x))

def reuse_emphasis(original, donor):
    oo = list(original.iter(M + 'r'))
    dd = list(donor.iter(M + 'r'))
    assert len(oo) == len(dd)
    for (o, d) in zip(oo, dd):
        rp = o.find(W + 'rPr')
        dp = d.find(W + 'rPr')
        if rp is None:
            rp = E.Element(W + 'rPr')
            o.insert(0, rp)
        for x in list(rp):
            if x.tag in EMPH:
                rp.remove(x)
        if dp is not None:
            for x in dp:
                if x.tag in EMPH:
                    rp.append(copy.deepcopy(x))

def locate_offsets(p):
    offsets = []
    position = 0
    for n in p.iter():
        if n.tag in (W + 'bookmarkStart', W + 'bookmarkEnd'):
            offsets.append((position, copy.deepcopy(n)))
        if n.tag in (W + 't', M + 't'):
            position += len(n.text or '')
    return offsets

def map_offset(pos, oldtext, newtext):
    if pos == 0:
        return 0
    if pos == len(oldtext):
        return len(newtext)
    for (mode, a, b, c, d) in SequenceMatcher(None, oldtext, newtext, autojunk=False).get_opcodes():
        if a <= pos <= b:
            if mode == 'equal':
                return c + pos - a
            if pos == a:
                return c
            if pos == b:
                return d
            return c + round((pos - a) * (d - c) / (b - a))
    raise AssertionError(pos)

def insert_marker(p, pos, marker):
    offset = 0
    for (k, ch) in enumerate(p):
        if ch.tag == W + 'pPr':
            continue
        size = len(text(ch))
        if pos == offset:
            p.insert(k, marker)
            return
        if offset < pos < offset + size:
            assert ch.tag == W + 'r' and len(ch.findall(W + 't')) == 1, ('Bookmark within mathematical/link object', pos, text(p))
            t = ch.find(W + 't')
            left = pos - offset
            right = copy.deepcopy(ch)
            t.text = (t.text or '')[:left]
            right.find(W + 't').text = (right.find(W + 't').text or '')[left:]
            p.insert(k + 1, marker)
            p.insert(k + 2, right)
            return
        offset += size
    if pos == offset:
        p.append(marker)
        return
    raise AssertionError((pos, offset, text(p)))

def patch_plain(p, donor):
    oldtext = text(p)
    newtext = text(donor)
    assert not list(donor.iter(M + 'oMath'))
    label = re.match('(?:图|表)\\d+-\\d+', oldtext)
    for (mode, a, b, c, d) in reversed(SequenceMatcher(None, oldtext, newtext, autojunk=False).get_opcodes()):
        if mode == 'equal':
            continue
        if label:
            assert a >= label.end()
        nodes = [n for n in p.iter(W + 't')]
        off = 0
        placed = False
        for n in nodes:
            value = n.text or ''
            lo = max(0, a - off)
            hi = min(len(value), b - off)
            if lo < hi:
                n.text = value[:lo] + (newtext[c:d] if not placed else '') + value[hi:]
                placed = True
            elif a == b and (not placed) and (off <= a <= off + len(value)):
                k = a - off
                n.text = value[:k] + newtext[c:d] + value[k:]
                placed = True
            off += len(value)
        assert placed
    assert norm(text(p)) == norm(newtext)
