"""Check changed Word content and locate its rendered tables and figures."""

import json
from pathlib import Path
from zipfile import ZipFile

import fitz
from lxml import etree as E

HERE = Path(__file__).resolve().parent
NS = {"w":"http://schemas.openxmlformats.org/wordprocessingml/2006/main",
      "m":"http://schemas.openxmlformats.org/officeDocument/2006/math",
      "a":"http://schemas.openxmlformats.org/drawingml/2006/main"}


def text(n):
    return "".join(n.xpath(".//w:t/text() | .//m:t/text()",namespaces=NS))


def math(n):
    return [tuple(m.xpath(".//m:t/text()",namespaces=NS))
            for m in n.xpath(".//m:oMath",namespaces=NS)]


def check():
    with ZipFile(HERE/"排版完成_修改前.docx") as old, ZipFile(HERE/"排版完成_结果整合.docx") as new:
        before=E.fromstring(old.read("word/document.xml"))
        after=E.fromstring(new.read("word/document.xml"))
        replaced={r["index"] for r in json.loads((HERE/"word_changes.json").read_text())}
        blocks=list(before.find("w:body",NS))
        expected=[m for i,b in enumerate(blocks) if i not in replaced for m in math(b)]
        assert math(after)==expected, "An untouched native equation changed."
        names=after.xpath(".//w:bookmarkStart/@w:name",namespaces=NS)
        assert len(names)==len(set(names)), "Duplicate bookmark names."
        original={name for name in before.xpath(".//w:bookmarkStart/@w:name",namespaces=NS)
                  if not name.startswith("_Toc")}
        assert original.issubset(names), "An existing reference anchor disappeared."
        links=set(after.xpath(".//w:hyperlink/@w:anchor",namespaces=NS))
        assert links.issubset(names), "A generated contents link has no target."
        assert len(after.xpath(".//w:sectPr",namespaces=NS))==16
        for name in ("word/styles.xml","word/numbering.xml"):
            assert old.read(name)==new.read(name), "Original styles or numbering changed."
        labels={"tbl3_10":"表3-10","fig3_13":"图3-13","tbl4_10":"表4-10",
                "fig4_11":"图4-11","tbl4_11":"表4-11","tbl5_13":"表5-13",
                "tbl6_16":"表6-16","tbl6_17":"表6-17","fig6_18":"图6-18"}
        new_tables={}
        for name,label in labels.items():
            bookmark=after.xpath('.//w:bookmarkStart[@w:name="'+name+'"]',namespaces=NS)[0]
            p=bookmark.getparent()
            assert text(p).startswith(label)
            if name.startswith("tbl"):
                table=p.getnext()
                assert E.QName(table).localname=="tbl"
                new_tables[label]=[[text(c) for c in r.findall("w:tc",NS)]
                                   for r in table.findall("w:tr",NS)]
        assert new_tables["表6-16"][2][2]=="2632.69"
        assert new_tables["表6-16"][5][2]=="2573.64"
        assert new_tables["表6-17"][2][2:]==["309.11","2681.68"]
        assert new_tables["表6-17"][5][2:]==["314.58","2614.78"]
        report=dict(sections=16,native_equations=len(expected),
            original_reference_anchors_preserved=len(original),new_tables=len(new_tables),
            images=len(after.xpath(".//a:blip",namespaces=NS)))
    pdf=fitz.open(HERE/"结果整合_预览.pdf")
    pages=[p.get_text() for p in pdf]
    assert not any("错误!" in t or "Error!" in t for t in pages)
    report["pages"]=len(pdf)
    report["locations"]={label:[i+1 for i,t in enumerate(pages) if label in t] for label in labels.values()}
    targets=set()
    for page,text_value in enumerate(pages):
        if any(heading in text_value for heading in (
                "3.4.9", "4.4.7", "5.2.7", "5.4.4", "6.4.4")):
            targets.add(page+1)
        if any(label in text_value for label in labels.values()):targets.add(page+1)
    for page in sorted(targets):
        if page<22:continue
        pdf[page-1].get_pixmap(matrix=fitz.Matrix(1.4,1.4)).save(str(HERE/"pages"/f"final_{page}.png"))
    (HERE/"verification.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(report,ensure_ascii=False))


if __name__=="__main__":check()
