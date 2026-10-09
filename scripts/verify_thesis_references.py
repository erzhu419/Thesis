"""Fetch small primary-source records and adopt verified arXiv versions."""

import argparse
import json
import time
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

from lxml import html

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / "manuscript/references.json"
DEST = ROOT / "workpapers/manuscript_audit/reference_verification.json"
ARXIV = {
    "wang2021asynchronous": "2105.00376", "wang2023robust": "2111.01946",
    "zhang2026single": "2508.20784", "haarnoja2018soft": "1801.01290",
    "kumar2020cql": "2006.04779", "kostrikov2022iql": "2110.06169",
    "ball2023efficient": "2302.02948", "zhou2025wsrl": "2412.07762",
    "eysenbach2021off": "2006.13916", "niu2022h2o": "2206.13464",
    "niu2025h2oplus": "2309.12716v2", "peng2018": "1710.06537",
    "chua2018pets": "1805.12114", "janner2019mbpo": "1906.08253",
    "hafner2020dreamer": "1912.01603", "brunton2016sindy": "1509.03580",
    "lanier2026redraw": "2504.02252", "ross2012agnostic": "1203.1007",
    "achiam2017cpo": "1705.10528", "chen2021redq": "2101.05982",
    "Fujimoto2018TD3": "1802.09477", "Osband2016BootstrappedDQN": "1602.04621",
    "Depeweg2018Decomposition": "1710.07283", "Bellemare2017Distributional": "1707.06887",
    "Geist2019TheoryRegularized": "1901.11275", "Asadi2018Lipschitz": "1804.07193",
    "madsen2020nau": "2001.05016",
}


def normalized(title):
    return "".join(c for c in title.casefold() if c.isalnum())


def get(url):
    with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0 (thesis-reference-review)"}), timeout=18) as response:
        return response.read()


def arxiv_record(ref, number):
    url = "https://arxiv.org/abs/" + number
    page = html.fromstring(get(url).decode("utf-8"))
    def meta(name):
        return page.xpath('//meta[@name=$name]/@content', name="citation_"+name)
    title = meta("title")[0]
    authors = []
    for author in meta("author"):
        if "," in author:
            family, given = author.split(",", 1)
            authors.append(dict(family=family.strip(), given=given.strip()))
        else:
            authors.append(dict(literal=author))
    abstract = page.xpath('//blockquote[contains(@class,"abstract")]')[0].text_content()
    date = meta("online_date" if "v" in number else "date")[0]
    entry = dict(id=ref["id"], type="article", title=title, author=authors,
                 issued={"date-parts": [[int(date[:4])]]},
                 URL=url, archive="arXiv", number=number, language="en",
                 accessed={"date-parts": [[2026, 9, 29]]})
    return dict(id=ref["id"], source=url, checked="2026-09-29", entry=entry,
                title_match=normalized(title)==normalized(ref["title"]),
                abstract=" ".join(abstract.strip().removeprefix("Abstract:").split()))


def fetch():
    records = json.loads(DEST.read_text()) if DEST.exists() else {}
    for ref in json.loads(LIBRARY.read_text()):
        key = ref["id"]
        if key in records and "error" not in records[key]:
            continue
        number = ARXIV.get(key)
        if not number and "arxiv.org/abs/" in ref.get("URL", ""):
            number = ref["URL"].split("/abs/")[1]
        try:
            if number:
                record = arxiv_record(ref, number)
            elif ref.get("DOI"):
                url = "https://api.crossref.org/works/" + quote(ref["DOI"], safe="/")
                data = json.loads(get(url))["message"]
                record = dict(id=key, source=url, checked="2026-09-29", metadata=data,
                              title_match=normalized(data["title"][0])==normalized(ref["title"]),
                              abstract=data.get("abstract", ""))
            else:
                record = dict(id=key, source=ref.get("URL", ""), needs_source_review=True)
            records[key] = record
            print(key, "TITLE_MATCH" if record.get("title_match") else "REVIEW", flush=True)
        except Exception as error:
            records[key] = dict(id=key, error=str(error))
            print(key, "FETCH_ERROR", str(error), flush=True)
        DEST.write_text(json.dumps(records, ensure_ascii=False, indent=2)+"\n")
        time.sleep(0.5)


def apply_arxiv():
    records = json.loads(DEST.read_text())
    entries = json.loads(LIBRARY.read_text())
    changed = []
    for i, old in enumerate(entries):
        record = records.get(old["id"], {})
        if (record.get("title_match") or record.get("same_work_verified")) and record.get("entry", {}).get("archive") == "arXiv":
            entries[i] = record["entry"]
            if old != entries[i]:
                changed.append(old["id"])
    LIBRARY.write_text(json.dumps(entries, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(dict(updated=changed), ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["fetch", "apply-arxiv"])
    args = parser.parse_args()
    fetch() if args.action == "fetch" else apply_arxiv()
