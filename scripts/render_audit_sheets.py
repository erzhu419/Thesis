"""Render every PDF page at readable scale, four per numbered review sheet."""

import argparse
from pathlib import Path
import fitz
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(args.pdf)
    width, height = 596, 866
    for start in range(0, len(doc), 4):
        sheet = Image.new("RGB", (width*2, height*2), "#d8d8d8")
        draw = ImageDraw.Draw(sheet)
        for j in range(4):
            if start+j >= len(doc):
                break
            page = doc[start+j]
            pix = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
            x, y = (j % 2)*width, (j // 2)*height
            draw.text((x+8, y+5), f"PDF PAGE {start+j+1}", fill="black")
            sheet.paste(Image.frombytes("RGB", (pix.width, pix.height), pix.samples), (x, y+22))
        sheet.save(args.output / f"{start+1:03d}-{min(start+4,len(doc)):03d}.png")
    print(f"Rendered {len(doc)} pages to {(len(doc)+3)//4} sheets")


if __name__ == "__main__":
    main()
