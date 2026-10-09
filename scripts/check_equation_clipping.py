"""Find Word math glyphs painted outside their display-equation cell."""

import json
from pathlib import Path
import re
import subprocess
import sys

import fitz


def inspect_pages(path, start, stop):
    findings = []
    with fitz.open(path) as pdf:
        for index in range(start, min(stop, len(pdf))):
            page = pdf[index]
            # The Word builder reserves 34 pt at both sides for equation numbers.
            left, right = 89.85 + 34, page.rect.width - 89.85 - 34
            cells = {tuple(d["scissor"]) for d in page.get_drawings(extended=True)
                     if d["type"] == "clip" and abs(d["scissor"].x0-left) < .2
                     and abs(d["scissor"].x1-right) < .2}
            if not cells:
                continue
            spans = [s for s in page.get_texttrace() if "Math" in s["font"]]
            for bounds in sorted(cells):
                cell = fitz.Rect(bounds)
                clipped = []
                for span in spans:
                    for code, _, origin, box in span["chars"]:
                        if (cell.y0 <= origin[1] <= cell.y1
                                and (box[0] < cell.x0-.5 or box[2] > cell.x1+.5)
                                and not chr(code).isspace()):
                            clipped.append(chr(code))
                if clipped:
                    number_box = fitz.Rect(right, cell.y0, page.rect.width-89, cell.y1)
                    numbers = re.findall(r"\([2-6A-E]-\d+\)", page.get_textbox(number_box))
                    findings.append(dict(page=index+1,
                                         equation=numbers[-1] if numbers else "unknown",
                                         cell=[round(v, 2) for v in bounds],
                                         clipped="".join(clipped)))
    return findings


if __name__ == "__main__":
    path = sys.argv[1]
    if len(sys.argv) == 4:
        print(json.dumps(inspect_pages(path, int(sys.argv[2]), int(sys.argv[3]))))
    else:
        with fitz.open(path) as pdf:
            pages = len(pdf)
        findings = []
        # This installed PyMuPDF texttrace build crashes after many pages in one process.
        for start in range(0, pages, 16):
            result = subprocess.check_output(
                [sys.executable, __file__, path, str(start), str(start+16)], text=True)
            findings.extend(json.loads(result))
        report = dict(pdf=str(Path(path).resolve()), pages=pages, clipping=findings)
        target = Path(__file__).resolve().parents[1] / "output/equation_clipping_check.json"
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        sys.exit(bool(findings))
