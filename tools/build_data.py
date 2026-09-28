"""Build prototype/data/wines.json from the scraped shop feed.

Inputs (from the scrape): wine_1.json (collections/wine/products.json),
membership.json (collection handle -> product handles), colcounts.txt
(every filter collection linked from the shop, with its product count),
norm_report.json (from normalise.py) and the raw/ and norm/ image folders.

Outputs: prototype/data/wines.json and prototype/img/{now,studio}/*.jpg.

Usage: python3 build_data.py <scrape_dir> <prototype_dir>
"""
import html
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

TYPES = {"red-wine": "Red", "white-wine": "White", "sparkling-wine": "Sparkling",
         "rose": "Rosé", "vermouth": "Vermouth"}
COUNTRIES = {"italy": "Italy", "spain": "Spain", "france": "France", "portugal": "Portugal",
             "new-zealand": "New Zealand", "south-africa": "South Africa"}
REGIONS = {"alella": "Alella", "alto-adige": "Alto Adige", "bordeaux": "Bordeaux",
           "burgundy": "Burgundy", "champagne": "Champagne", "languedoc": "Languedoc",
           "loire-valley": "Loire", "marlborough": "Marlborough", "medoc": "Médoc",
           "piedmonte": "Piedmont", "rhone": "Rhône", "rias-baixas": "Rías Baixas",
           "ribeira-sacra": "Ribeira Sacra", "ribera-del-duero-1": "Ribera del Duero",
           "rioja": "Rioja", "sancerre": "Sancerre", "tuscany": "Tuscany",
           "val-do-salnes": "Val do Salnés", "valtellina-superiore": "Valtellina",
           "veneto": "Veneto", "muscadet": "Muscadet"}
GRAPES = {"albarino": "Albariño", "albillo": "Albillo", "brancello": "Brancellao",
          "cabernet-sauvignon": "Cabernet Sauvignon", "caino-tinto": "Caíño Tinto",
          "carignan": "Carignan", "chardonnay": "Chardonnay", "corvina": "Corvina",
          "esadeiro": "Espadeiro", "garnacha": "Garnacha", "godello": "Godello",
          "grenache": "Grenache", "macabeu": "Macabeu", "malbec": "Malbec",
          "mataro": "Mataró", "mencia": "Mencía", "merlot": "Merlot",
          "monastrell": "Monastrell", "nebbiolo": "Nebbiolo", "pansa-blanca": "Pansa Blanca",
          "parellada": "Parellada", "pinot-noir": "Pinot Noir", "sangiovese": "Sangiovese",
          "sauvignon-blanc": "Sauvignon Blanc", "souson": "Sousón", "syrah": "Syrah",
          "tempranillo": "Tempranillo", "xarel-lo": "Xarel·lo"}
STYLES = {"organic": "Organic", "biodynamic": "Biodynamic", "natural": "Natural",
          "vegan": "Vegan", "practicing-organic": "Practising organic"}

YEAR = re.compile(r"(?<!\d)(19[5-9]\d|20[0-2]\d)(?!\d)")


def text(fragment):
    return html.unescape(re.sub(r"<[^>]+>", "", fragment)).replace("﻿", "").strip()


def paragraphs(body):
    return [t for t in (text(p) for p in re.findall(r"<p[^>]*>(.*?)</p>", body or "", re.S)) if t]


def parse(p, member_of):
    body = p["body_html"] or ""
    paras = paragraphs(body)
    header = paras[0] if paras else ""
    title = text(p["title"])
    rest = paras[1:]
    pairing = next((x for x in rest if x.lower().startswith(("pairs", "pair "))), "")
    quote = ""
    m = re.search(r"<em>(.*?)</em>(?:.*?<strong>(.*?)</strong>)?", body, re.S)
    if m:
        quote = text(m.group(1)).strip('"“” ')
        by = text(m.group(2) or "")
        quote = {"text": quote, "by": by} if quote else ""
    desc = [x for x in rest if x != pairing and (not quote or quote["text"][:30] not in x)
            and not x.lower().startswith(("click here", "for multiorders"))]
    cols = member_of.get(p["handle"], set())
    title_years = YEAR.findall(title)
    header_years = YEAR.findall(header)
    vintage = (title_years or header_years or [None])[0]
    if not vintage and re.search(r"\bNV\b", header + " " + title):
        vintage = "NV"
    v = p["variants"][0]
    img = p["images"][0] if p["images"] else None
    return {
        "handle": p["handle"],
        "title": title,
        "price": float(v["price"]),
        "available": any(x["available"] for x in p["variants"]),
        "type": next((TYPES[c] for c in TYPES if c in cols), None),
        "country": next((COUNTRIES[c] for c in COUNTRIES if c in cols), None),
        "region": next((REGIONS[c] for c in REGIONS if c in cols), None),
        "grapes": [GRAPES[c] for c in GRAPES if c in cols],
        "styles": [STYLES[c] for c in STYLES if c in cols],
        "magnum": "magnum" in cols or "MAGNUM" in title.upper() or "JEROBOAM" in title.upper(),
        "vintage": vintage,
        "header": header,
        "description": desc,
        "pairing": re.sub(r"^pairs? (well )?with\s*", "", pairing, flags=re.I).rstrip(". "),
        "quote": quote,
        "published": p["published_at"][:10],
        "image": {
            "file": img["src"].split("/")[-1].split("?")[0] if img else None,
            "w": img["width"] if img else None,
            "h": img["height"] if img else None,
            "count": len(p["images"]),
        },
    }


def find_issues(wines):
    by_file = defaultdict(list)
    for w in wines:
        if w["image"]["file"]:
            by_file[w["image"]["file"]].append(w)
    titles = Counter(w["title"].lower() for w in wines)
    for w in wines:
        issues = []
        f, iw, ih = w["image"]["file"], w["image"]["w"], w["image"]["h"]
        others = [o for o in by_file.get(f, []) if o is not w]
        if others:
            issues.append({"kind": "image", "level": "high",
                           "text": "Same image as " + ", ".join(o["title"] for o in others[:3])
                           + (" and more" if len(others) > 3 else "")})
        file_years = set(YEAR.findall(f or ""))
        if w["vintage"] and w["vintage"] != "NV" and file_years and w["vintage"] not in file_years:
            issues.append({"kind": "image", "level": "high",
                           "text": f"Image file says {', '.join(sorted(file_years))}, listing says {w['vintage']}"})
        if iw and max(iw, ih) < 600:
            issues.append({"kind": "image", "level": "medium", "text": f"Low resolution ({iw}×{ih} px)"})
        if iw and ih and abs(iw / ih - 1) > 0.05:
            issues.append({"kind": "image", "level": "low", "text": f"Not square ({iw}×{ih} px), crops differently in the grid"})
        if f and f.startswith("https___"):
            issues.append({"kind": "image", "level": "low", "text": "Image saved from another website"})
        url_years = set(YEAR.findall(w["handle"]))
        if w["vintage"] and w["vintage"] != "NV" and url_years and w["vintage"] not in url_years:
            issues.append({"kind": "data", "level": "medium",
                           "text": f"Web address says {', '.join(sorted(url_years))}, listing says {w['vintage']}"})
        if "copy" in w["handle"].split("-"):
            issues.append({"kind": "data", "level": "low",
                           "text": "Web address still says “copy”, from a duplicated listing"})
        if not w["vintage"]:
            issues.append({"kind": "data", "level": "medium", "text": "No vintage in title or description"})
        if re.search(r"\b20\d{3}\b", w["header"]):
            issues.append({"kind": "data", "level": "medium", "text": "Typo in vintage: " + re.search(r"\b20\d{3}\b", w["header"]).group(0)})
        if w["header"].lower().startswith("for multiorders"):
            issues.append({"kind": "data", "level": "medium", "text": "Description opens with “For multiorders please contact us here.”"})
        if not w["type"]:
            issues.append({"kind": "filter", "level": "medium", "text": "Not in any wine-type collection, so the Red / White filters miss it"})
        if not w["grapes"]:
            issues.append({"kind": "filter", "level": "low", "text": "Not in any grape collection"})
        if titles[w["title"].lower()] > 1:
            issues.append({"kind": "data", "level": "medium", "text": "Two listings with this exact title"})
        w["issues"] = issues


def main(scrape, proto):
    scrape, proto = Path(scrape), Path(proto)
    products = json.loads((scrape / "wine_1.json").read_text())["products"]
    membership = json.loads((scrape / "membership.json").read_text())
    member_of = defaultdict(set)
    for col, handles in membership.items():
        for h in handles:
            member_of[h].add(col)
    report = json.loads((scrape / "norm_report.json").read_text())

    wines = [parse(p, member_of) for p in products]
    find_issues(wines)

    now_dir, studio_dir = proto / "img" / "now", proto / "img" / "studio"
    now_dir.mkdir(parents=True, exist_ok=True)
    studio_dir.mkdir(parents=True, exist_ok=True)
    for w in wines:
        h = w["handle"]
        # "Now": the source as the shop shows it, on white, at most 640 px.
        src = Image.open(scrape / "raw" / h).convert("RGBA")
        flat = Image.new("RGBA", src.size, (255, 255, 255, 255))
        flat.alpha_composite(src)
        flat = flat.convert("RGB")
        flat.thumbnail((640, 640))
        flat.save(now_dir / f"{h}.jpg", quality=80, optimize=True, progressive=True)
        r = report.get(h, {})
        w["studio"] = r.get("status") == "normalised"
        w["studio_note"] = r.get("source")
        if w["studio"]:
            s = Image.open(scrape / "norm" / f"{h}.jpg")
            s.thumbnail((640, 800))
            s.save(studio_dir / f"{h}.jpg", quality=80, optimize=True, progressive=True)
        else:
            w["issues"].append({"kind": "image", "level": "medium",
                                "text": "Photo on a busy background. Needs a reshoot or an AI studio shot"})

    counts = [l.split() for l in (scrape / "colcounts.txt").read_text().splitlines() if l.strip()]
    empty = sorted(c for c, n in counts if n == "0")
    shop = {
        "scraped": "2026-09-28",
        "wines": len(wines),
        "products": len(membership.get("all", [])),
        "filter_links": len(counts),
        "empty_filter_links": empty,
        "sold_out": sum(1 for w in wines if not w["available"]),
    }
    out = {"shop": shop, "wines": wines}
    (proto / "data").mkdir(exist_ok=True)
    (proto / "data" / "wines.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    flagged = sum(1 for w in wines if w["issues"])
    print(f"{len(wines)} wines, {flagged} with issues, {sum(w['studio'] for w in wines)} studio shots")
    print(Counter(i["text"].split(" (")[0].split(":")[0][:40] for w in wines for i in w["issues"]).most_common(20))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
