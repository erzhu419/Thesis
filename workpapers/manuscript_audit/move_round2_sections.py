"""Move existing evidence without changing values or stable equation labels."""
from pathlib import Path
import re

root = Path(__file__).resolve().parents[2] / "manuscript"
c5 = next((root / "chapters").glob("05_*.md"))
c6 = next((root / "chapters").glob("06_*.md"))
s5, s6 = c5.read_text(), c6.read_text()
start = s6.index("### 6.7.5 相同停站")
end = s6.index("### 6.7.7 多训练")
block = s6[start:end]
s6 = s6[:start] + s6[end:]
s5 = s5.replace("## 5.7 讨论与局限", "## 5.7 训练方案、执行保障与运营规则的作用\n\n" + block + "## 5.8 讨论与局限")
s5 = s5.replace("### 5.7.", "### 5.8.").replace("## 5.8 本章小结", "## 5.9 本章小结")
c5.write_text(s5)
c6.write_text(s6)
mapping = {
    "6.7.5": "5.7.1", "6.7.6": "5.7.2", "6.7.7": "6.7.5", "6.7.8": "6.7.6",
    "图6-12": "图5-11", "图6-13": "图5-12", "图6-14": "图6-12",
    "表6-8": "表5-7", "表6-9": "表6-8", "表6-10": "表6-9",
}
pattern = re.compile("|".join(re.escape(k) + r"(?!\d)" for k in mapping))
for folder in ("chapters", "appendices", "frontmatter"):
    for path in (root / folder).glob("*.md"):
        text = path.read_text()
        text = pattern.sub(lambda m: mapping[m.group()], text)
        text = text.replace("第5.7节集中讨论", "第5.8节集中讨论")
        path.write_text(text)
