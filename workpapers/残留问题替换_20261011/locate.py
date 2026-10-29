"""Locate user replacements, tables, and appendix references in the saved Word."""
from pathlib import Path
from zipfile import ZipFile
import json
import sys

from lxml import etree as E

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'workpapers/全文编号符号简称复核_20261011'))
from scan import W, M, NS, content

BASE = HERE / '修改前_用户已完成1至13项.docx'


def main():
    with ZipFile(BASE) as z:
        d = E.fromstring(z.read('word/document.xml'))
    body = d.find(W+'body')
    needles = ['本文借鉴这些方法', '利用近期奖励与价值分歧', '本文采用自建',
               '表2-6', '表2-7', '表2-8', '六类条件下，RE-SAC和SAC',
               '本章在BAPR前期', '式(4-5)列出', '4.3.2',
               '为考察不同函数结构', '图4-1展示', '（2）针对客流', '（2）提出',
               '为将含记忆', 'B.6', 'B.7', '最终公交软符号', 'D.2',
               '式(D-43)中的符号', '设φ', '设ϕ', '于是正数', '一步备份差',
               '若只在状态子集', '环境空间', '本节比较特定误差',
               '附录F', 'F.2', '人均广义时间', 'eq6_norm']
    hit = set()
    for i, p in enumerate(body):
        text = content(p)
        if any(x in text for x in needles) or 'eq6_norm' in E.tostring(p).decode():
            hit.update(range(max(0,i-1),min(len(body),i+3)))
    lines = []
    for i in sorted(hit):
        p = body[i]
        st = p.find(W+'pPr/'+W+'pStyle')
        style = st.get(W+'val') if st is not None else E.QName(p).localname
        text = content(p)
        if p.tag == W+'tbl':
            text = '\n'.join(' | '.join(content(c) for c in row.findall(W+'tc')) for row in p.findall(W+'tr'))
        lines.append(f'[{i} {style}] {text}')
    (HERE / '定位上下文.txt').write_text('\n\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
