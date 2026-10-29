"""Render the edited regions, keeping the visual audit explicitly scoped."""
import json
from pathlib import Path

import fitz
from PIL import Image, ImageDraw

from revise import HERE, OUT


def main():
    document = fitz.open(OUT.with_suffix('.pdf'))
    text = [p.get_text() for p in document]
    compact = [''.join(s.split()) for s in text]
    appendix_start = next(i for i,s in enumerate(compact) if i>220 and '附录A' in s)
    main_patterns = ['确定性动作映射、参考状态、符号参考',
                     '实际软符号配置在策略损失中',
                     '非空合法残差闭集', '有限时域公交任务的对应条件分析']
    body_pages = [next(i for i,s in enumerate(compact) if pattern in s) for pattern in main_patterns]
    focus_patterns = ['(D-4a)','(D-4b)','(D-4c)','(D-4d)','(D-4e)',
                      '(E-23a)','(E-23b)','(E-23c)','(E-27)',
                      '高斯密度，类别先验相等','原精确权重为','第6.4.5节的车头时距奖励补充试验']
    focus_pages = set(body_pages+list(range(3,10)))
    focus_pages.update(i for i,s in enumerate(compact[:19]) if 'C.8持续证据下的后验阈值' in s)
    for pattern in focus_patterns:
        matches = [i for i,s in enumerate(compact) if pattern in s and i>=appendix_start]
        assert matches, pattern
        focus_pages.update(matches)
    for index in sorted(focus_pages):
        document[index].get_pixmap(matrix=fitz.Matrix(1.4,1.4)).save(str(HERE/f'页_{index+1:03d}.png'))
    pages = list(range(3,19))+sorted(set(body_pages))+list(range(appendix_start,len(document)))
    pages = list(dict.fromkeys(pages))
    for start in range(0,len(pages),12):
        sheet = Image.new('RGB',(4*380,3*565),'#eeeeee')
        draw = ImageDraw.Draw(sheet)
        for slot,index in enumerate(pages[start:start+12]):
            pix = document[index].get_pixmap(matrix=fitz.Matrix(.62,.62),alpha=False)
            page = Image.frombytes('RGB',(pix.width,pix.height),pix.samples)
            page.thumbnail((370,535))
            x,y=(slot%4)*380,(slot//4)*565
            draw.text((x+12,y+4),f'PDF {index+1}',fill='black')
            sheet.paste(page,(x+(380-page.width)//2,y+25))
        sheet.save(HERE/f'缩略复查_{start//12+1}.png')
    result={'pages':len(document),'appendix_pdf_pages':[appendix_start+1,len(document)],
            'main_body_adaptation_pdf_pages':[p+1 for p in body_pages],
            'full_size_previews':[p+1 for p in sorted(focus_pages)],
            'contact_sheet_pages':[p+1 for p in pages]}
    (HERE/'预览页索引.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
