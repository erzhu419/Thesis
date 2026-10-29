"""Locate and render the pages affected by the wording review."""
from pathlib import Path
import json
import shutil

import fitz
from PIL import Image, ImageDraw

from edit import ROOT, HERE

STAGE=Path('/mnt/c/Users/erzhu/AppData/Local/Temp/thesis_expression_20261012/final_preview.pdf')
OUT=ROOT/'output/排版完成_冗余表达与用词校订_20261012.pdf'


def main():
    shutil.copy2(STAGE,OUT)
    doc=fitz.open(OUT)
    compact=[''.join(p.get_text().split()) for p in doc]
    targets=['表3-1RE-SAC','表3-11','4.3.2输出结构及其响应性质',
             '符号导数约束的解释依赖','数据内优势可记',
             '三类网络分别','原递推目标及样本权重为',
             '表6-8公交','表6-9','D.1基本条件','交通管制通告、活动安排']
    locations={s:[i+1 for i,t in enumerate(compact) if s in t] for s in targets}
    pages=[4,8,15,74,92,99,106,108,145,184,185,186,192,208,209,234,235]
    appendix=locations['D.1基本条件']
    if appendix: pages.append(appendix[-1])
    for n in pages:
        doc[n-1].get_pixmap(matrix=fitz.Matrix(1.45,1.45),alpha=False).save(HERE/f'页面_{n}.png')
    for group in range((len(pages)+5)//6):
        subset=pages[group*6:(group+1)*6]
        board=Image.new('RGB',(1200,1150),'#e7e7e7')
        draw=ImageDraw.Draw(board)
        for j,n in enumerate(subset):
            pic=Image.open(HERE/f'页面_{n}.png')
            pic.thumbnail((380,540))
            x,y=(j%3)*400+10,(j//3)*570+25
            board.paste(pic,(x,y))
            draw.text((x,y-17),f'Page {n}',fill='black')
        board.save(HERE/f'重点页_{group+1}.png')
    errors=[i+1 for i,t in enumerate(compact) if any(s in t for s in ['Error!','错误!','错误！未定义书签','错误！找不到引用源'])]
    assert not errors,errors
    page4=compact[3]
    assert '(填写阿拉伯数字)' not in page4
    (HERE/'页面检查记录.json').write_text(json.dumps({'pages':len(doc),'field_errors':errors,'rendered_pages':pages,'locations':locations},ensure_ascii=False,indent=2))
    print(json.dumps({'pages':len(doc),'rendered_pages':pages,'locations':locations},ensure_ascii=False))


if __name__=='__main__':main()
