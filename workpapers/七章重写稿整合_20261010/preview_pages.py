"""Make body-only layout sheets and selected full-size page previews."""

from collections import Counter
from pathlib import Path
import json

import fitz
from PIL import Image, ImageDraw

from integrate import ROOT, WORK


def main():
    pdf = fitz.open(ROOT / "output/排版完成_七章重写稿整合_20261010.pdf")
    folder = WORK / "版面预览"
    folder.mkdir(exist_ok=True)
    first = 18
    last = next(i for i in range(213, 235) if "参考文献" in [line.strip() for line in pdf[i].get_text().splitlines()])
    fonts = Counter()
    out_of_page = []
    blank_pages = []
    for index in range(first, last):
        page = pdf[index]
        blocks = page.get_text("dict")["blocks"]
        body_lines = 0
        for block in blocks:
            for line in block.get("lines", []):
                if 75 < line["bbox"][1] < page.rect.height - 75:
                    body_lines += 1
                for span in line["spans"]:
                    fonts[span["font"]] += 1
                    x0, y0, x1, y1 = span["bbox"]
                    if x0 < -1 or y0 < -1 or x1 > page.rect.width + 1 or y1 > page.rect.height + 1:
                        out_of_page.append({"page": index + 1, "text": span["text"], "bbox": span["bbox"]})
        if body_lines == 0 and not page.get_images():
            blank_pages.append(index + 1)
    for start in range(first, last, 12):
        sheet = Image.new("RGB", (1600, 1760), "#d5d5d5")
        draw = ImageDraw.Draw(sheet)
        for cell, index in enumerate(range(start, min(start + 12, last))):
            pixmap = pdf[index].get_pixmap(matrix=fitz.Matrix(400 / pdf[index].rect.width, 400 / pdf[index].rect.width))
            preview = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            x, y = (cell % 4) * 400, (cell // 4) * 586
            sheet.paste(preview, (x, y + 20))
            draw.text((x + 8, y + 3), str(index + 1), fill="black")
        sheet.save(folder / f"body_{start + 1:03d}_{min(start + 12, last):03d}.jpg", quality=88)
    selected = [19, 38, 39, 68, 69, 105, 106, 133, 134, 167, 168, 212, 213, last]
    for index in range(first, last):
        lines = pdf[index].get_text().splitlines()
        if any(label in "".join(lines) for label in ("表3-1", "表6-16", "图3-1", "图4-1", "图5-2", "图6-1")):
            selected.append(index + 1)
    selected = sorted(set(selected))
    for page_number in selected:
        pixmap = pdf[page_number - 1].get_pixmap(matrix=fitz.Matrix(1.7, 1.7))
        pixmap.save(folder / f"page_{page_number:03d}.png")
    result = {"pdf_pages": len(pdf), "body_first_page": first + 1, "body_last_page": last, "selected_full_size_pages": selected, "out_of_page_text": out_of_page, "blank_body_pages": blank_pages, "rendered_fonts": dict(fonts)}
    (WORK / "版面扫描.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
