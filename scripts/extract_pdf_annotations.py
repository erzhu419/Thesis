"""Extract saved PDF comments together with their marked text and page context."""

import argparse
import json
from pathlib import Path

import fitz


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    records = []
    with fitz.open(args.pdf) as doc:
        for page in doc:
            for annotation in page.annots() or []:
                vertices = annotation.vertices or []
                selected = [page.get_textbox(fitz.Quad(vertices[i:i + 4]).rect)
                            for i in range(0, len(vertices), 4)]
                records.append(dict(index=len(records) + 1, page=page.number + 1,
                                    xref=annotation.xref, type=annotation.type[1],
                                    selected_text=" ".join(selected),
                                    comment=annotation.info.get("content", ""),
                                    page_text=page.get_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    print(f"Extracted {len(records)} annotations to {args.output}")


if __name__ == "__main__":
    main()
