"""Compile the new manuscript and verify the main reported result anchors."""
from pathlib import Path
import json
import re
import subprocess

HERE = Path(__file__).resolve().parent
PAPER = HERE/'paper'


def main():
    source = (PAPER/'main.tex').read_text()
    abstract = source.split('\\begin{abstract}')[1].split('\\end{abstract}')[0]
    words = len(abstract.split())
    if not 150 <= words <= 250:
        raise ValueError('T-ITS abstract must contain150--250 words')
    process = subprocess.run(['latexmk', '-pdf', '-interaction=nonstopmode', '-halt-on-error',
                              'main.tex'], cwd=PAPER, capture_output=True, text=True)
    (PAPER/'build_stdout.log').write_text(process.stdout+process.stderr)
    if process.returncode:
        raise RuntimeError('Inspect the saved manuscript build log')
    log = (PAPER/'main.log').read_text(errors='replace')
    if re.search(r'undefined references|Citation .* undefined|Reference .* undefined', log):
        raise RuntimeError('Manuscript has unresolved references')
    subprocess.run(['pdftotext', '-layout', str(PAPER/'main.pdf'), str(PAPER/'main.txt')], check=True)
    text = (PAPER/'main.txt').read_text()
    anchors = {value:value in text for value in ('5.946','0.269','7.199','0.325','4.656','8.359')}
    if not all(anchors.values()):
        raise RuntimeError('A main reported result is absent from the compiled draft')
    info = subprocess.check_output(['pdfinfo', str(PAPER/'main.pdf')], text=True)
    pages = int(re.search(r'Pages:\s+(\d+)', info).group(1))
    result = dict(status='draft_compiled', pages=pages, abstract_words=words,
        main_result_anchors=anchors,
        overfull_boxes=re.findall(r'Overfull \\[hv]box[^\n]*', log),
        new_simulation_seconds=0, training_updates=0, old_results_modified=False,
        submission_status='not_submitted; author details, supplementary evidence and release materials pending')
    with (HERE/'build_receipt.json').open('w') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
