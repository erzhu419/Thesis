"""One-time scoped consistency edits for the 2026-09-29 thesis audit."""

import re
from pathlib import Path

root = Path(__file__).resolve().parents[2] / "manuscript"
for pattern, first, last in (
    ("chapters/04_*.md", "### 4.3.2", "### 4.3.3"),
    ("appendices/D_*.md", "## D.5", "## D.6"),
):
    path = next(root.glob(pattern))
    text = path.read_text()
    start, end = text.index(first), text.index(last)
    section = text[start:end]
    def replace_math(match):
        math = match[0].replace("L_h", r"L_{\mathrm{out}}").replace("B_h", r"B_{\mathrm{out}}")
        math = math.replace("Dh", r"D f_{\mathrm{out}}")
        return re.sub(r"(?<![\\A-Za-z])h(?![A-Za-z])", lambda _: r"{f_{\mathrm{out}}}", math)
    section = re.sub(r"\$\$.*?\$\$|(?<!\$)\$(?!\$).*?(?<!\$)\$(?!\$)", replace_math, section, flags=re.S)
    section = section.replace("；这些映射记号与带方向标记的车头时距$h^{\\mathrm f},h^{\\mathrm b}$区分", "")
    path.write_text(text[:start] + section + text[end:])

path = next(root.glob("chapters/01_*.md"))
text = path.read_text().replace("单调价值混合网络QMIX", "单调价值混合网络（QMIX）")
text = text.replace("仿真优化方法（Simulation Optimization，SimOpt）", "仿真参数优化方法SimOpt")
path.write_text(text)
