"""Import selected local references and check DOI metadata for the thesis."""

from pathlib import Path
from urllib.parse import quote, unquote, urlsplit
from urllib.request import Request, urlopen
import json
import re

import pypandoc


ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ROOT.parent
SELECTIONS = {
    "TransitDuet/paper/references.bib": [
        "daganzo2009headway", "xuan2011dynamic", "wang2020real",
        "wang2021asynchronous", "wang2023robust", "wang2023multiobjective",
        "rodriguez2023cooperative", "yu2024hierarchical", "li2026graphaware",
        "alesiani2018reinforcement", "zhang2026single", "resac2026",
    ],
    "sumo-rl/H2Oplus/paper/references.bib": [
        "delgado2012holding", "andres2017predictive", "haarnoja2018soft",
        "kumar2020cql", "kostrikov2022iql", "ball2023efficient",
        "zhou2025wsrl", "eysenbach2021off", "niu2022h2o", "niu2025h2oplus",
        "sutton1999between", "lopez2018microscopic",
    ],
    "MC-WM/paper/references.bib": [
        "peng2018", "chua2018pets", "janner2019mbpo", "hafner2020dreamer",
        "brunton2016sindy", "lanier2026redraw", "todorov2012mujoco",
        "ross2012agnostic", "achiam2017cpo", "chen2021redq",
    ],
    "RE-SAC/paper/ensemble_refs_jmlr.bib": [
        "Fujimoto2018TD3", "Osband2016BootstrappedDQN",
        "Depeweg2018Decomposition", "Bellemare2017Distributional",
        "Geist2019TheoryRegularized", "Iyengar2005RobustDP",
        "Asadi2018Lipschitz",
    ],
    "Thesis/manuscript/references_extra.bib": [
        "adams2007bocd", "madsen2020nau", "arjovsky2019irm",
    ],
}


def normalized_title(title):
    return "".join(c for c in re.sub(r"<[^>]+>", "", title).casefold() if c.isalnum())


def main():
    references, audit = [], []
    for source, keys in SELECTIONS.items():
        entries = json.loads(pypandoc.convert_file(str(PROJECTS / source), "csljson", format="biblatex"))
        by_id = {entry["id"]: entry for entry in entries}
        for key in keys:
            entry = by_id[key]
            entry.pop("abstract", None)
            record = {"id": key, "source": source, "title": entry["title"]}
            if doi := entry.get("DOI"):
                url = "https://api.crossref.org/works/" + quote(doi, safe="/")
                request = Request(url, headers={"User-Agent": "ThesisReferenceCheck/1.0"})
                with urlopen(request, timeout=30) as response:
                    work = json.load(response)["message"]
                record.update({
                    "verification_url": url,
                    "registered_title": work["title"][0],
                    "title_match": normalized_title(entry["title"]) == normalized_title(work["title"][0]),
                    "registered_authors": work.get("author", []),
                    "registered_container": work.get("container-title", []),
                    "registered_year": work.get("published-print", work["issued"])["date-parts"][0][0],
                    "registered_volume": work.get("volume"),
                    "registered_issue": work.get("issue"),
                    "registered_page": work.get("page"),
                })
                if "abstract" in work:
                    record["registered_abstract"] = work["abstract"]
                if not record["title_match"]:
                    print("TITLE DIFFERENCE:", key, record["registered_title"], flush=True)
            else:
                record["verification_url"] = entry.get("URL", "")
                record["verification_basis"] = "Primary proceedings or author preprint page; see reference_sources.md"
            # The publisher now assigns the early-view bus paper to volume 8(2).
            if key == "zhang2026single":
                entry.update({"volume": "8", "issue": "2", "page": "tdag005"})
            if key == "zhou2025wsrl":
                entry.pop("volume", None)
                entry.pop("page", None)
            if key == "kumar2020cql":
                entry["URL"] = "https://proceedings.neurips.cc/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html"
                record["verification_url"] = entry["URL"]
            if key == "Fujimoto2018TD3":
                entry["author"][1] = {"family": "van Hoof", "given": "Herke"}
            link = urlsplit(entry.get("URL", ""))
            if link.netloc in ("doi.org", "dx.doi.org") and unquote(link.path).lstrip("/").casefold() == entry.get("DOI", "").casefold():
                entry.pop("URL")
            entry["language"] = "en"
            references.append(entry)
            audit.append(record)
            print(key, flush=True)
    destination = ROOT / "manuscript"
    additions_file = destination / "reference_additions.json"
    if additions_file.exists():
        additions = json.loads(additions_file.read_text(encoding="utf-8"))
        references.extend(additions)
        selected = {entry["id"] for entry in additions}
        for name in ("reference_preprints.json", "reference_transport.json", "reference_chinese.json"):
            audit.extend(record for record in json.loads((destination / name).read_text())
                         if record["id"] in selected)
    assert len({r["id"] for r in references}) == len(references)
    (destination / "references.json").write_text(
        json.dumps(references, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (destination / "reference_metadata.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("Imported references:", len(references))


if __name__ == "__main__":
    main()
