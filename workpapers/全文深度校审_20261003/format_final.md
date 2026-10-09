实际格式核验：未通过。对象：`排版完成.docx`。本报告由本次文件的 XML 实际计算生成。

| 分类 | 段落 | 文字运行 | 判定 | 差异 |
|---|---:|---:|---|---|
| cover_title | 1 | 2 | 通过 | — |
| cover_degree | 1 | 1 | 通过 | — |
| body | 1628 | 13776 | 通过 | — |
| abstract_cn_title | 1 | 1 | 通过 | — |
| abstract_cn_first | 1 | 1 | 通过 | — |
| abstract_cn_body | 6 | 132 | 通过 | — |
| keywords_cn | 1 | 2 | 通过 | — |
| abstract_en_title | 1 | 1 | 通过 | — |
| abstract_en_first | 1 | 1 | 通过 | — |
| abstract_en_body | 6 | 8 | 通过 | — |
| keywords_en | 1 | 2 | 通过 | — |
| toc_title | 1 | 1 | 通过 | — |
| toc1 | 17 | 76 | 通过 | — |
| toc2 | 89 | 270 | 通过 | — |
| toc3 | 223 | 680 | 通过 | — |
| symbol_title | 1 | 1 | 通过 | — |
| table | 2025 | 2453 | 通过 | — |
| chapter | 13 | 61 | 通过 | — |
| h2 | 89 | 181 | 通过 | — |
| h3 | 223 | 453 | 通过 | — |
| table_caption | 63 | 280 | 通过 | — |
| figure | 66 | 0 | 通过 | — |
| figure_caption | 66 | 303 | 通过 | — |
| equation | 1149 | 1149 | 通过 | — |
| h4 | 53 | 53 | 通过 | — |
| bibliography_title | 1 | 1 | 通过 | — |
| bibliography | 154 | 663 | 通过 | — |

推断与冲突：
- Accepted: 18 tall inline-math paragraphs use atLeast 20 pt instead of exact 20 pt.
- Accepted: preserve the source two-paragraph cover order; title spacing 100 pt above/30 pt below with atLeast 28 pt, degree atLeast 20 pt.
- Accepted: Heading 4 has no actual template sample; use its style definition plus exact 20 pt.
- Accepted: chapter 3/4 headers retain 10.5 pt and full text, with two exact-12-pt lines; other headers use exact 20 pt. Inherited center tab 4153 is cleared, right tab 8306 retained, and bottom-border distance is 3 pt to prevent the rule crossing text.
- Migration layout controls: keepNext/keepLines for headings/captions; figure auto line and snapToGrid=0; source numbered-list indents and cell alignments retained.
- The empty paragraph carrying the TOC closing field marker may use exact 1 pt as a layout-only marker; it has no source prose and is identified by the actual TOC field boundary.
- The single symbols-table introductory paragraph beginning 数学记号按完整的字母、上下标及字体识别 may use 6 pt after-spacing to separate its final line from the table top border. Ordinary body paragraphs retain template after-spacing 0.
- Source bold/italic emphasis is checked at matching character positions in the corresponding source paragraph; superscript/subscript remain semantic. Keywords instead follow the template mixed-font rule: bold label, regular terms, zero first-line indent. REF results use the referencing paragraph baseline, so prior inherited heading bold is not treated as intended emphasis.
- Only font selectors active for the actual run text are compared. The unused EastAsian selector on ASCII equation numbers differs (Song vs TNR), while visible number text uses TNR 12 pt.
- Equation table spacing/alignment follow source layout because template formula examples are mixed and its 9600-twip tab is overwide; typography is checked separately.

未定义／未验收：
- No actual Heading 4, footnote or endnote text sample in the template.
- This is XML format acceptance; it does not claim that Word pagination or every rendered page passed visual inspection.
