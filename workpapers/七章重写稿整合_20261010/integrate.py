"""Insert the supplied chapter prose into the existing formatted thesis."""

from collections import Counter, defaultdict
from copy import deepcopy
from difflib import SequenceMatcher
from pathlib import Path
from zipfile import ZipFile
from xml.parsers import expat
import json
import re
import shutil

from lxml import etree as ET
import pypandoc


ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent
BASE = ROOT / "output/排版完成_逐节校订_第2轮_20261010.docx"
OUT = ROOT / "output/排版完成_七章重写稿整合_20261010.docx"
NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}
W, M, R = ("{" + NS[k] + "}" for k in ("w", "m", "r"))
P_REL = "http://schemas.openxmlformats.org/package/2006/relationships"


def text(node):
    return "".join(node.xpath(".//w:t/text()", namespaces=NS))


def style(node):
    value = node.find("w:pPr/w:pStyle", NS)
    return value.get(W + "val", "") if value is not None else ""


def body_spans(data):
    """Return byte ranges of direct body children, keeping untouched XML exact."""
    parser = expat.ParserCreate()
    depth = 0
    body_depth = None
    start = None
    spans = []

    def on_start(name, attrs):
        nonlocal depth, body_depth, start
        depth += 1
        if name == "w:body":
            body_depth = depth
        elif body_depth is not None and depth == body_depth + 1:
            start = parser.CurrentByteIndex

    def on_end(name):
        nonlocal depth, body_depth
        if body_depth is not None and depth == body_depth + 1:
            index = parser.CurrentByteIndex
            end = data.index(b">", index) + 1 if data[index:index + 2] == b"</" else index
            spans.append((start, end))
        if name == "w:body":
            body_depth = None
        depth -= 1

    parser.StartElementHandler = on_start
    parser.EndElementHandler = on_end
    parser.Parse(data, True)
    return spans


def inline_text(inlines):
    result = []
    for item in inlines:
        kind, value = item["t"], item.get("c")
        if kind == "Str":
            result.append(value)
        elif kind in ("Space", "SoftBreak", "LineBreak"):
            result.append(" ")
        elif kind in ("Strong", "Emph", "Superscript", "Subscript", "Strikeout", "SmallCaps"):
            result.append(inline_text(value))
        elif kind in ("Link", "Span", "Image"):
            result.append(inline_text(value[1]))
        elif kind == "Code":
            result.append(value[1])
        elif kind == "Math":
            result.append("<MATH>")
        else:
            raise ValueError((kind, value))
    return "".join(result)


def ast_key(block, table_caption):
    kind, value = block["t"], block.get("c")
    if kind == "Header":
        return "H:" + inline_text(value[2])
    if kind == "Table":
        return "T:" + table_caption
    assert kind == "Para", kind
    if len(value) == 1 and value[0]["t"] == "Math" and value[0]["c"][0]["t"] == "DisplayMath":
        return "E:" + re.search(r"\\tag\{([1-7]-\d+)\}", value[0]["c"][1]).group(1)
    match = re.match(r"([图表][1-7]-\d+)\s", inline_text(value))
    return "C:" + match.group(1) if match else None


def math_semantics(node):
    """Ignore run splitting and typography, but retain mathematical operators."""
    local = ET.QName(node).localname
    if local == "r":
        scripts = tuple(node.xpath("./m:rPr/m:scr/@m:val", namespaces=NS))
        return ("text", "".join(node.xpath(".//m:t/text()", namespaces=NS)), scripts)
    if local.endswith("Pr") or node.tag.startswith(W):
        return None
    children = []
    for child in node:
        value = math_semantics(child)
        if value is None:
            continue
        if value[0] == "text" and children and children[-1][0] == "text" and children[-1][2] == value[2]:
            children[-1] = ("text", children[-1][1] + value[1], value[2])
        else:
            children.append(value)
    return (local, tuple(children))


def serialize(node):
    return ET.tostring(node, encoding="utf-8", with_tail=False)


def normalized_prose(node):
    copied = deepcopy(node)
    for item in copied.xpath(".//m:oMath", namespaces=NS):
        item.getparent().replace(item, ET.Element("MATH"))
    parts = []

    def walk(item):
        if item.tag == W + "t":
            parts.append(item.text or "")
        elif item.tag == "MATH":
            parts.append("<MATH>")
        else:
            for child in item:
                walk(child)

    walk(copied)
    return "".join(parts)


def field_runs(instruction, display, run_properties):
    result = []
    for kind in ("begin", "instruction", "separate", "result", "end"):
        run = ET.Element(W + "r")
        if run_properties is not None:
            run.append(deepcopy(run_properties))
        if kind == "instruction":
            item = ET.SubElement(run, W + "instrText")
            item.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
            item.text = " " + instruction + " "
        elif kind == "result":
            ET.SubElement(run, W + "t").text = display
        else:
            ET.SubElement(run, W + "fldChar").set(W + "fldCharType", kind)
        result.append(run)
    return result


def make_run(value, properties=None):
    run = ET.Element(W + "r")
    if properties is not None:
        run.append(deepcopy(properties))
    node = ET.SubElement(run, W + "t")
    node.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    node.text = value
    return run


def main():
    WORK.mkdir(exist_ok=True)
    backup = WORK / "整合前_第2轮成品.docx"
    if not backup.exists():
        shutil.copy2(BASE, backup)
    with ZipFile(backup) as source:
        original = source.read("word/document.xml")
        root = ET.fromstring(original)
        body = root.find("w:body", NS)
        spans = body_spans(original)
        assert len(spans) == len(body)
        raw = [original[a:b] for a, b in spans]
        first = next(i for i, x in enumerate(body) if style(x) == "chapter" and text(x).startswith("第1章"))
        last = next(i for i, x in enumerate(body) if style(x) == "bibliographytitle")

        anchors = {}
        figures = {}
        sections = {}
        old_by_section = defaultdict(list)
        section_ending = []
        body_markers = defaultdict(list)
        caption = ""
        section = ""
        ref_targets = {}
        added_bookmarks = []
        next_bookmark = max(int(v) for v in root.xpath(".//w:bookmarkStart/@w:id", namespaces=NS)) + 1
        for index in range(first, last):
            node = body[index]
            st, value = style(node), text(node)
            key = None
            if st in ("chapter", "h2", "h3"):
                key = "H:" + value
                section = key
                marks = node.findall("w:bookmarkStart", NS)
                if marks:
                    target = marks[0].get(W + "name")
                    if st != "chapter":
                        ref_targets["第" + value.split()[0] + "节"] = target
            elif st in ("figurecaption", "tablecaption"):
                label = re.match(r"[图表][1-7]-\d+", value).group()
                key = "C:" + label
                mark = node.find("w:bookmarkStart", NS)
                if mark is None:
                    node = deepcopy(node)
                    mark = ET.SubElement(node, W + "bookmarkStart")
                    mark.set(W + "id", str(next_bookmark))
                    mark.set(W + "name", "tbl" + label[1:].replace("-", "_"))
                    ET.SubElement(node, W + "bookmarkEnd").set(W + "id", str(next_bookmark))
                    added_bookmarks.append(str(next_bookmark))
                    next_bookmark += 1
                ref_targets[label] = mark.get(W + "name")
                if st == "figurecaption":
                    figure_index = index - 1
                    while style(body[figure_index]) != "figure":
                        assert not text(body[figure_index]).strip(), label
                        figure_index -= 1
                    figures[label] = raw[figure_index]
                else:
                    caption = label
            elif node.tag == W + "tbl":
                marks = node.xpath('.//w:bookmarkStart[starts-with(@w:name,"eq")]', namespaces=NS)
                if marks:
                    label = re.search(r"\(([1-7]-\d+)\)\s*$", value).group(1)
                    key = "E:" + label
                    ref_targets["(" + label + ")"] = marks[0].get(W + "name")
                else:
                    key = "T:" + caption
            if key:
                assert key not in anchors, key
                anchors[key] = (index, node, raw[index])
            elif node.find("w:pPr/w:sectPr", NS) is not None:
                section_ending.append((index, raw[index]))
            elif st == "body":
                old_by_section[section].append((index, node))
            elif node.tag in (W + "bookmarkStart", W + "bookmarkEnd"):
                body_markers[section].append((index, node))
            else:
                assert st == "figure" or not value.strip(), (index, st, value[:80])

        # Recreate citations against the preserved bibliography bookmarks.
        bibliography = {}
        for node in body:
            if style(node) != "bibliography":
                continue
            match = re.match(r"\[(\d+)\]", text(node))
            marks = node.xpath('./w:bookmarkStart[starts-with(@w:name,"ref-")]', namespaces=NS)
            if match and marks:
                bibliography[match.group(1)] = marks[0].get(W + "name")
        for node in body:
            if style(node) == "chapter" and text(node).startswith("附录"):
                match = re.match(r"附录[A-Z]", text(node))
                marks = node.findall("w:bookmarkStart", NS)
                if match and marks:
                    ref_targets[match.group()] = marks[0].get(W + "name")

        relation_root = ET.fromstring(source.read("word/_rels/document.xml.rels"))
        url_rel = {x.get("Target"): x.get("Id") for x in relation_root if x.get("TargetMode") == "External"}
        old_math = {}
        for node in body[first:last]:
            if node.tag == W + "p" and style(node) == "body":
                for math in node.findall(".//m:oMath", NS):
                    old_math.setdefault(math_semantics(math), math)

        records = []
        full_ast = None
        donor_differences = []
        for chapter in range(1, 8):
            path = next((ROOT / "markdown").glob(f"第{chapter}章*润色重写稿_20261010.md"))
            ast = json.loads(pypandoc.convert_text(path.read_text(), "json", format="markdown+tex_math_dollars-auto_identifiers-smart"))
            if full_ast is None:
                full_ast = {"pandoc-api-version": ast["pandoc-api-version"], "meta": {}, "blocks": []}
            current_section, current_caption = "", ""
            chapter_plain = []
            for block in ast["blocks"]:
                key = ast_key(block, current_caption)
                if key and key.startswith("H:"):
                    current_section = key
                if key and key.startswith("C:表"):
                    current_caption = key[2:]
                generated = deepcopy(block)
                if key and key.startswith("E:"):
                    generated = {"t": "Para", "c": [{"t": "Str", "c": key}]}
                if block["t"] == "Para" and not (key and key.startswith("E:")):
                    chapter_plain.append(re.sub(r"\s+", "", inline_text(block["c"]).replace("<MATH>", "")))
                full_ast["blocks"].append(generated)
                records.append({"chapter": chapter, "section": current_section, "key": key, "ast": block})
            # The companion DOCX is corroboration, not authority over the Markdown.
            with ZipFile(path.with_suffix(".docx")) as donor:
                donor_body = ET.fromstring(donor.read("word/document.xml")).find("w:body", NS)
                donor_plain = [re.sub(r"\s+", "", text(x)) for x in donor_body if x.tag == W + "p" and style(x) not in ("Heading1", "Heading2", "Heading3") and text(x).strip()]
                if donor_plain != chapter_plain:
                    donor_differences.append({"chapter": chapter, "markdown_paragraphs": len(chapter_plain), "companion_paragraphs": len(donor_plain)})

        converted = WORK / "Markdown原生公式转换_中间件.docx"
        warnings = pypandoc.convert_text(json.dumps(full_ast, ensure_ascii=False), "docx", format="json", outputfile=str(converted))
        with ZipFile(converted) as generated_zip:
            generated_body = ET.fromstring(generated_zip.read("word/document.xml")).find("w:body", NS)
            generated_nodes = [x for x in generated_body if x.tag != W + "sectPr"]
            generated_rel = ET.fromstring(generated_zip.read("word/_rels/document.xml.rels"))
            generated_urls = {x.get("Id"): x.get("Target") for x in generated_rel if x.get("TargetMode") == "External"}
        assert len(generated_nodes) == len(records), (len(generated_nodes), len(records))
        for record, node in zip(records, generated_nodes):
            record["generated"] = node
        assert {x["key"] for x in records if x["key"]} == set(anchors)

        new_by_section = defaultdict(list)
        for record in records:
            if record["key"] is None:
                new_by_section[record["section"]].append(record)
        metadata = defaultdict(list)
        old_to_new = {}
        paragraph_template = {}
        for section, new_items in new_by_section.items():
            old_items = old_by_section[section]
            metadata_items = old_items
            if not old_items:
                chapter = new_items[0]["chapter"]
                old_items = next(items for name, items in old_by_section.items() if name.startswith(f"H:{chapter}.") and items)
            old_values = [normalized_prose(x) for _, x in old_items]
            new_values = [normalized_prose(x["generated"]) for x in new_items]
            for j, value in enumerate(new_values):
                nearest = max(range(len(old_items)), key=lambda k: SequenceMatcher(None, old_values[k], value, autojunk=False).ratio())
                paragraph_template[id(new_items[j])] = old_items[nearest][1]
            for old_index, old_node in metadata_items:
                special = [deepcopy(x) for x in old_node if x.tag in (W + "bookmarkStart", W + "bookmarkEnd", W + "commentRangeStart", W + "commentRangeEnd")]
                special += [deepcopy(x) for x in old_node.findall("w:r", NS) if x.find("w:commentReference", NS) is not None]
                value = normalized_prose(old_node)
                nearest = max(range(len(new_values)), key=lambda k: SequenceMatcher(None, value, new_values[k], autojunk=False).ratio())
                if old_node.xpath('./w:bookmarkStart[@w:name="从仿真标定到数据驱动的模型选择"]', namespaces=NS):
                    nearest = 0
                old_to_new[old_index] = new_items[nearest]
                if special:
                    metadata[id(new_items[nearest])].extend(special)
        for section, markers in body_markers.items():
            candidates = [index for index, _ in old_by_section[section]]
            for index, marker in markers:
                if marker.tag == W + "bookmarkStart":
                    adjacent = next(i for i in candidates if i > index)
                else:
                    adjacent = max(i for i in candidates if i < index)
                metadata[id(old_to_new[adjacent])].append(deepcopy(marker))

        replacement_counts = Counter()
        ref_pattern = re.compile(r"\[\d+(?:[,，\-–]\d+)*\]|[图表][1-7]-\d+|\([1-7]-\d+\)|第[1-7](?:\.\d+){1,2}节|附录[A-Z]")

        def richen(paragraph):
            for link in paragraph.findall(".//w:hyperlink", NS):
                rid = link.get(R + "id")
                if rid:
                    url = generated_urls[rid]
                    assert url in url_rel, url
                    link.set(R + "id", url_rel[url])
            for math in list(paragraph.findall(".//m:oMath", NS)):
                old = old_math.get(math_semantics(math))
                if old is not None:
                    math.getparent().replace(math, deepcopy(old))
                    replacement_counts["reused_inline_math"] += 1
                else:
                    replacement_counts["converted_inline_math"] += 1
            for run in list(paragraph.findall(".//w:r", NS)):
                if run.getparent().tag == W + "hyperlink":
                    continue
                texts = run.findall("w:t", NS)
                if len(texts) != 1 or any(x.tag not in (W + "rPr", W + "t") for x in run):
                    continue
                value = texts[0].text or ""
                matches = list(ref_pattern.finditer(value))
                if not matches:
                    continue
                properties = run.find("w:rPr", NS)
                nodes, offset = [], 0
                for match in matches:
                    label = match.group()
                    if match.start() > offset:
                        nodes.append(make_run(value[offset:match.start()], properties))
                    if label.startswith("[") and all(n in bibliography for n in re.findall(r"\d+", label)):
                        cite_properties = deepcopy(properties) if properties is not None else ET.Element(W + "rPr")
                        vertical = cite_properties.find("w:vertAlign", NS)
                        if vertical is None:
                            vertical = ET.SubElement(cite_properties, W + "vertAlign")
                        vertical.set(W + "val", "superscript")
                        for fragment in re.findall(r"\d+|[^\d]+", label):
                            if fragment.isdigit():
                                assert fragment in bibliography, fragment
                                link = ET.Element(W + "hyperlink")
                                link.set(W + "anchor", bibliography[fragment])
                                link.append(make_run(fragment, cite_properties))
                                nodes.append(link)
                                replacement_counts["citation_links"] += 1
                            else:
                                nodes.append(make_run(fragment, cite_properties))
                    elif label in ref_targets:
                        if label.startswith("第") or label.startswith("附录"):
                            link = ET.Element(W + "hyperlink")
                            link.set(W + "anchor", ref_targets[label])
                            link.append(make_run(label, properties))
                            nodes.append(link)
                        else:
                            nodes.extend(field_runs(f'REF {ref_targets[label]} \\h \\* CHARFORMAT', label, properties))
                        replacement_counts["cross_references"] += 1
                    else:
                        nodes.append(make_run(label, properties))
                    offset = match.end()
                if offset < len(value):
                    nodes.append(make_run(value[offset:], properties))
                parent, position = run.getparent(), run.getparent().index(run)
                parent.remove(run)
                for child in nodes:
                    parent.insert(position, child)
                    position += 1
            return paragraph

        def paragraph(source_node, template):
            result = ET.Element(W + "p", nsmap=body.nsmap)
            properties = template.find("w:pPr", NS)
            if properties is not None:
                result.append(deepcopy(properties))
            for child in source_node:
                if child.tag != W + "pPr" and child.tag not in (W + "bookmarkStart", W + "bookmarkEnd"):
                    result.append(deepcopy(child))
            template_run = next((r for r in template.findall("w:r", NS) if r.find("w:t", NS) is not None and (r.find("w:t", NS).text or "").strip()), None)
            template_properties = template_run.find("w:rPr", NS) if template_run is not None else None
            if template_properties is not None:
                typography = ("rFonts", "sz", "szCs", "color", "spacing", "kern", "w", "lang")
                for run in result.findall("w:r", NS):
                    properties = run.find("w:rPr", NS)
                    if properties is None:
                        properties = ET.Element(W + "rPr")
                        run.insert(0, properties)
                    for name in typography:
                        old_value = template_properties.find(W + name)
                        if old_value is not None:
                            for existing in properties.findall(W + name):
                                properties.remove(existing)
                            properties.append(deepcopy(old_value))
            richen(result)
            return result

        output_blocks = []
        chapter_nodes = defaultdict(list)
        used_anchors = Counter()
        for record in records:
            key, generated, chapter = record["key"], record["generated"], record["chapter"]
            if key:
                used_anchors[key] += 1
                index, old, old_raw = anchors[key]
                if key.startswith(("H:", "E:")):
                    node, data = deepcopy(old), old_raw
                    if key.startswith("E:"):
                        replacement_counts["original_numbered_equations"] += 1
                elif key.startswith("C:"):
                    label = key[2:]
                    if label.startswith("图"):
                        output_blocks.append(figures[label])
                        chapter_nodes[chapter].append(("image", figures[label]))
                    node = deepcopy(old)
                    for child in list(node):
                        if child.tag != W + "pPr":
                            node.remove(child)
                    for mark in old.findall("w:bookmarkStart", NS):
                        node.append(deepcopy(mark))
                    node.append(make_run(label))
                    for mark in old.findall("w:bookmarkEnd", NS):
                        node.append(deepcopy(mark))
                    caption_rest = deepcopy(generated)
                    caption_text = caption_rest.find(".//w:t", NS)
                    assert caption_text.text.startswith(label)
                    caption_text.text = caption_text.text[len(label):]
                    formatted_rest = paragraph(caption_rest, old)
                    for child in formatted_rest:
                        if child.tag != W + "pPr":
                            node.append(deepcopy(child))
                    assert normalized_prose(node) == inline_text(record["ast"]["c"]), label
                    data = serialize(node)
                    replacement_counts["captions"] += 1
                else:
                    node = deepcopy(old)
                    cells = node.findall(".//w:tc", NS)
                    new_cells = generated.findall(".//w:tc", NS)
                    assert len(cells) == len(new_cells), key
                    for cell, new_cell in zip(cells, new_cells):
                        old_paras = cell.findall("w:p", NS)
                        new_paras = new_cell.findall("w:p", NS)
                        assert old_paras
                        for child in list(cell):
                            if child.tag != W + "tcPr":
                                cell.remove(child)
                        for j, new_para in enumerate(new_paras):
                            cell.append(paragraph(new_para, old_paras[min(j, len(old_paras) - 1)]))
                    assert normalized_prose(node) == normalized_prose(generated), key
                    data = serialize(node)
                    replacement_counts["formatted_data_tables"] += 1
            else:
                node = paragraph(generated, paragraph_template[id(record)])
                marks = metadata[id(record)]
                leading = [x for x in marks if x.tag in (W + "bookmarkStart", W + "commentRangeStart")]
                trailing = [x for x in marks if x.tag not in (W + "bookmarkStart", W + "commentRangeStart")]
                for mark in reversed(leading):
                    node.insert(1 if node.find("w:pPr", NS) is not None else 0, mark)
                for mark in trailing:
                    node.append(mark)
                assert normalized_prose(node) == normalized_prose(generated), (chapter, text(node)[:80])
                data = serialize(node)
                replacement_counts["markdown_prose_paragraphs"] += 1
            output_blocks.append(data)
            chapter_nodes[chapter].append((key or "prose", data))
            if record is records[-1] or record["chapter"] != records[records.index(record) + 1]["chapter"]:
                data = section_ending[chapter - 1][1]
                output_blocks.append(data)
                chapter_nodes[chapter].append(("section", data))

        assert all(x == 1 for x in used_anchors.values())
        updated = original[:spans[first][0]] + b"".join(output_blocks) + original[spans[last][0]:]
        updated_root = ET.fromstring(updated)
        updated_body = updated_root.find("w:body", NS)
        names = updated_root.xpath(".//w:bookmarkStart/@w:name", namespaces=NS)
        assert not [k for k, v in Counter(names).items() if v > 1]
        ids = updated_root.xpath(".//w:bookmarkStart/@w:id", namespaces=NS)
        expected_ids = Counter(root.xpath(".//w:bookmarkStart/@w:id", namespaces=NS) + added_bookmarks)
        assert Counter(ids) == expected_ids, {
            "lost": [(x.get(W + "id"), x.get(W + "name")) for x in root.findall(".//w:bookmarkStart", NS) if x.get(W + "id") in expected_ids - Counter(ids)],
            "extra": dict(Counter(ids) - expected_ids),
        }
        assert Counter(updated_root.xpath(".//w:bookmarkEnd/@w:id", namespaces=NS)) == Counter(root.xpath(".//w:bookmarkEnd/@w:id", namespaces=NS) + added_bookmarks)
        assert Counter(updated_root.xpath(".//w:commentReference/@w:id", namespaces=NS)) == Counter(root.xpath(".//w:commentReference/@w:id", namespaces=NS))
        new_spans = body_spans(updated)
        new_last = next(i for i, x in enumerate(updated_body) if style(x) == "bibliographytitle")
        assert original[:spans[first][0]] == updated[:new_spans[first][0]]
        assert original[spans[last][0]:] == updated[new_spans[new_last][0]:]
        original_objects = root.xpath(".//w:drawing|.//w:object|.//w:pict", namespaces=NS)
        updated_objects = updated_root.xpath(".//w:drawing|.//w:object|.//w:pict", namespaces=NS)
        assert sorted(serialize(x) for x in original_objects) == sorted(serialize(x) for x in updated_objects)
        assert [serialize(x) for x in root.findall(".//w:sectPr", NS)] == [serialize(x) for x in updated_root.findall(".//w:sectPr", NS)]
        with ZipFile(OUT, "w") as target:
            for info in source.infolist():
                target.writestr(deepcopy(info), updated if info.filename == "word/document.xml" else source.read(info.filename))
        with ZipFile(OUT) as target:
            assert all(source.read(n) == target.read(n) for n in source.namelist() if n != "word/document.xml")

    summary = {
        "baseline": str(BASE), "output": str(OUT),
        "counts": dict(replacement_counts), "body_blocks": len(updated_body),
        "chapters": {str(k): len(v) for k, v in chapter_nodes.items()},
        "markdown_prose_and_cells_exact": True,
        "protected_front_and_bibliography_appendices_byte_exact": True,
        "all_other_zip_parts_unchanged": True,
        "original_images_sections_bookmarks_preserved": True,
        "missing_original_caption_bookmarks_added": added_bookmarks,
        "companion_docx_differences_markdown_takes_precedence": donor_differences,
    }
    (WORK / "整合核对.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
