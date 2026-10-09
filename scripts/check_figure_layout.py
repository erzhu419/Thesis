"""Check inserted figure placement and render the affected Word-exported pages."""

from pathlib import Path
from zipfile import ZipFile
import json
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET

import fitz
from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
PDF = Path(sys.argv[1])
DOCX = OUTPUT / "博士论文_正文与数学附录_阶段稿.docx"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"


def main():
    build = json.loads((OUTPUT / "word_build.json").read_text())
    # Missing media or lost OMML would require rebuilding the Word document.
    with ZipFile(DOCX) as archive:
        tree = ET.fromstring(archive.read("word/document.xml"))
        equations = len(tree.findall(f".//{{{M}}}oMath"))
        media = [n for n in archive.namelist() if n.startswith("word/media/")]
        references = [n for n in tree.findall(f".//{{{W}}}bookmarkStart") if n.get(f"{{{W}}}name", "").startswith("ref-")]
        assert equations == build["native_equations"], equations
        assert len(media) == build["embedded_media"], len(media)
        assert len(references) == build["cited_references"], len(references)
        for extent in tree.findall(f".//{{{WP}}}extent"):
            assert int(extent.get("cx")) / 12700 < 416
    pdf = fitz.open(PDF)
    source = (OUTPUT / "博士论文_正文与数学附录_阶段稿.md").read_text()
    caption_texts = {re.sub(r"\s+", "", line) for line in source.splitlines() if re.match(r"^图[1-7A-F]-\d+ ", line)}
    paragraphs = source.split("\n\n")
    notes = {}
    for index, paragraph in enumerate(paragraphs[:-1]):
        caption = re.sub(r"\s+", "", paragraph)
        if caption in caption_texts and not paragraphs[index + 1].startswith("#"):
            notes[caption] = re.sub(r"\s+", "", paragraphs[index + 1])
    previews = OUTPUT / "figure_page_previews"
    previews.mkdir(exist_ok=True)
    figures, boundary_errors = [], []
    # A clipped image or separated caption would require a layout adjustment.
    for page in pdf:
        for word in page.get_text("words"):
            x0, y0, x1, y1, text, *_ = word
            if 71 < y0 < 770 and (x0 < 87 or x1 > 520):
                boundary_errors.append([page.number + 1, text, [x0, y0, x1, y1]])
        images = page.get_image_info()
        if not images:
            continue
        captions = []
        for block in page.get_text("dict")["blocks"]:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                text = "".join(span["text"] for span in line["spans"])
                if re.sub(r"\s+", "", text) in caption_texts:
                    captions.append((text, line["bbox"]))
        assert len(images) == len(captions), (page.number + 1, len(images), captions)
        for image, (caption, caption_box) in zip(sorted(images, key=lambda r: r["bbox"][1]), sorted(captions, key=lambda r: r[1][1])):
            x0, y0, x1, y1 = image["bbox"]
            assert 88 < x0 < x1 < 507 and 70 < y0 < y1 < 772, (page.number+1, image["bbox"])
            assert y1 <= caption_box[1] + 3, (page.number+1, caption)
            key = re.sub(r"\s+", "", caption)
            if key in notes:
                # Word emits mathematical italic characters for editable inline equations.
                note_source = re.sub(r"\[([^\]]+)\]\(#[^)]+\)", r"\1", notes[key])
                note = unicodedata.normalize("NFKC", note_source.replace("$", ""))
                page_text = re.sub(r"\s+", "", unicodedata.normalize("NFKC", page.get_text()))
                assert note in page_text, (page.number+1, "figure note separated", caption)
            figures.append({"caption": caption, "pdf_page": page.number + 1})
        page.get_pixmap(matrix=fitz.Matrix(1.45, 1.45)).save(previews / f"page_{page.number+1:03}.png")
    assert len(figures) == build["embedded_media"], figures
    assert not boundary_errors, boundary_errors[:20]
    sheet = Image.new("RGB", (1600, ((len(figures)+3)//4)*610), "#EEEEEE")
    draw = ImageDraw.Draw(sheet)
    for i, figure in enumerate(figures):
        page_number = figure["pdf_page"]
        image = Image.open(previews / f"page_{page_number:03}.png").convert("RGB")
        image = ImageOps.contain(image, (390, 575))
        x, y = (i % 4)*400, (i // 4)*610
        sheet.paste(image, (x+(400-image.width)//2, y+25))
        draw.text((x+8, y+5), f"PDF page {page_number}", fill="black")
    sheet.save(OUTPUT / "figure_pages_contact_sheet.jpg", quality=92)
    report = {"pdf_pages": len(pdf), "native_equations": equations, "embedded_media": len(media),
              "reference_bookmarks": len(references), "figures": figures,
              "image_caption_same_page": True, "figure_notes_same_page": True, "body_boundary_errors": boundary_errors,
              "visual_review_complete": False}
    (OUTPUT / "figure_layout_check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
