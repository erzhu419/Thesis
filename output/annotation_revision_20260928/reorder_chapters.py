"""One-off chapter and cross-reference relocation for the annotated revision."""

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
CHAPTERS = ROOT / "manuscript/chapters"
old5 = CHAPTERS / "05_仿真模型失配下的世界模型适应方法.md"
old6 = CHAPTERS / "06_有限交互条件下公交在途控制的跨保真迁移.md"
wm = old5.read_text()
bus = old6.read_text()
wm_main, wm_end = wm.split("## 5.7 讨论与适用范围", 1)
bus_main, bus_rest = bus.split("## 6.7 数据预算、模型失配与执行保障的补充实验", 1)
bus_extra, bus_end = bus_rest.split("## 6.8 讨论与局限", 1)


def section(chapter, section):
    if chapter == 5:
        return 6, section if section < 7 else section + 1
    return (5, section) if section < 7 else ((6, 7) if section == 7 else (5, section - 1))


def transform(text):
    def heading(m):
        c, s = section(int(m[2]), int(m[3]))
        return f"{m[1]}{c}.{s}"

    def reference(m):
        c, s = section(int(m[1]), int(m[2]))
        return f"第{c}.{s}"

    def figure_table(m):
        kind, c, n = m[1], int(m[2]), int(m[3])
        if c == 5:
            return f"{kind}6-{n}"
        cutoff = 10 if kind == "图" else 6
        if n <= cutoff:
            return f"{kind}5-{n}"
        return f"{kind}6-{n - (2 if kind == '图' else 1)}"

    text = re.sub(r"^(#{2,3} )[56]\.\d+", lambda m: m[0], text, flags=re.M)
    text = re.sub(r"^(#{2,3} )([56])\.(\d+)", heading, text, flags=re.M)
    text = re.sub(r"第([56])\.(\d+)", reference, text)
    text = re.sub(r"第([56])章", lambda m: f"第{11-int(m[1])}章", text)
    return re.sub(r"([图表])([56])-(\d+)", figure_table, text)


new5 = CHAPTERS / "05_有限交互条件下公交在途控制的跨保真迁移.md"
new6 = CHAPTERS / "06_训练与使用环境失配下的世界模型适应方法.md"
new5.write_text(transform(bus_main + "## 6.8 讨论与局限" + bus_end))
new6.write_text(transform(wm_main + "## 6.7 数据预算、模型失配与执行保障的补充实验" + bus_extra
                          + "## 5.7 讨论与适用范围" + wm_end))
for folder in ("chapters", "appendices"):
    for path in (ROOT / "manuscript" / folder).glob("*.md"):
        if path not in (old5, old6, new5, new6):
            path.write_text(transform(path.read_text()))
old5.unlink()
old6.unlink()
print(new5.name)
print(new6.name)
