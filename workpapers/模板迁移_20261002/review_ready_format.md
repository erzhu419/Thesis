实际格式核验：未通过。对象：`review_ready.docx`。本报告由本次文件的 XML 实际计算生成。

| 分类 | 段落 | 文字运行 | 判定 | 差异 |
|---|---:|---:|---|---|
| cover_title | 1 | 2 | 通过 | — |
| cover_degree | 1 | 1 | 通过 | — |
| body | 1593 | 13368 | 未通过 | paragraph.tabs: 1 |
| abstract_cn_title | 1 | 1 | 通过 | — |
| abstract_cn_first | 1 | 1 | 通过 | — |
| abstract_cn_body | 6 | 127 | 通过 | — |
| keywords_cn | 1 | 2 | 通过 | — |
| abstract_en_title | 1 | 1 | 通过 | — |
| abstract_en_first | 1 | 1 | 通过 | — |
| abstract_en_body | 6 | 9 | 通过 | — |
| keywords_en | 1 | 2 | 通过 | — |
| toc_title | 1 | 1 | 通过 | — |
| toc1 | 17 | 76 | 通过 | — |
| toc2 | 89 | 270 | 通过 | — |
| toc3 | 218 | 661 | 通过 | — |
| symbol_title | 1 | 1 | 通过 | — |
| table | 1759 | 2119 | 通过 | — |
| chapter | 13 | 61 | 通过 | — |
| h2 | 89 | 181 | 通过 | — |
| h3 | 218 | 443 | 通过 | — |
| table_caption | 57 | 254 | 通过 | — |
| figure | 63 | 0 | 通过 | — |
| figure_caption | 63 | 290 | 通过 | — |
| equation | 1149 | 1149 | 通过 | — |
| h4 | 53 | 53 | 通过 | — |
| bibliography_title | 1 | 1 | 通过 | — |
| bibliography | 152 | 654 | 通过 | — |

推断与冲突：
- Accepted: 18 tall inline-math paragraphs use atLeast 20 pt instead of exact 20 pt.
- Accepted: preserve the source two-paragraph cover order; title spacing 100 pt above/30 pt below with atLeast 28 pt, degree atLeast 20 pt.
- Accepted: Heading 4 has no actual template sample; use its style definition plus exact 20 pt.
- Accepted: long chapter headers retain 10.5 pt font and full text, with two exact-12-pt lines instead of template single exact-20-pt line.
- Migration layout controls: keepNext/keepLines for headings/captions; figure auto line and snapToGrid=0; source numbered-list indents and cell alignments retained.
- Source bold/italic emphasis is accepted only where actual weighted character counts match the corresponding source paragraph; superscript/subscript remain semantic. Keywords instead follow the template mixed-font rule: bold label, regular terms, zero first-line indent.
- Only font selectors active for the actual run text are compared. The unused EastAsian selector on ASCII equation numbers differs (Song vs TNR), while visible number text uses TNR 12 pt.
- Equation table spacing/alignment follow source layout because template formula examples are mixed and its 9600-twip tab is overwide; typography is checked separately.

未定义／未验收：
- No actual Heading 4, footnote or endnote text sample in the template.
- This is XML format acceptance; it does not claim that Word pagination or every rendered page passed visual inspection.
