from pathlib import Path

import fitz
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "6.4预览"
OUTPUT.mkdir(exist_ok=True)
document = fitz.open(ROOT / "6.4预览.pdf")
for index, page in enumerate(document):
    page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6), alpha=False).save(OUTPUT / f"page_{179 + index}.png")
    text = page.get_text()
    lines = [line for line in text.splitlines() if line.strip()]
    print(179 + index, "BEGIN", " | ".join(lines[2:5]), "END", " | ".join(lines[-4:]))
for index in range(0, len(document), 2):
    first = Image.open(OUTPUT / f"page_{179 + index}.png").convert("RGB")
    second = Image.open(OUTPUT / f"page_{180 + index}.png").convert("RGB") if index + 1 < len(document) else Image.new("RGB", first.size, "white")
    pair = Image.new("RGB", (first.width + second.width + 12, max(first.height, second.height)), "#cccccc")
    pair.paste(first, (0, 0))
    pair.paste(second, (first.width + 12, 0))
    pair.save(OUTPUT / f"pair_{179 + index}_{180 + index}.png")
