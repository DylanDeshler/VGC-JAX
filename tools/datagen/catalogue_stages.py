#!/usr/bin/env python3
"""Regenerate the `stage` column of docs/catalogue.csv from data/export/stages/{v0,v1,v2}.json (05 §2).

Rows are matched on (kind, id). MOVE / ABILITY / ITEM rows get the first stage whose list contains
them; CONDITION rows keep `dep` (implemented when the first dependent lands); ENGINE_DELTA / SYSTEM
rows are untouched. Every other column, the row order and the CSV dialect are preserved.

Usage: uv run python tools/datagen/catalogue_stages.py --stages data/export/stages --catalogue docs/catalogue.csv
"""
from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import os
import sys

STAGES = ("v0", "v1", "v2")
KIND_KEY = {"MOVE": "moves", "ABILITY": "abilities", "ITEM": "items"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", required=True)
    ap.add_argument("--catalogue", required=True)
    a = ap.parse_args()

    first: dict[tuple[str, str], str] = {}
    for st in STAGES:
        with open(os.path.join(a.stages, f"{st}.json")) as f:
            doc = json.load(f)
        for kind, key in KIND_KEY.items():
            for eid in doc[key]:
                first.setdefault((kind, eid), st)
        for c in doc["conditions"]:
            first.setdefault(("CONDITION", c), "dep")

    with open(a.catalogue, newline="") as f:
        text = f.read()
    rows = list(csv.DictReader(io.StringIO(text, newline="")))
    fields = list(rows[0].keys())
    before = collections.Counter((r["kind"], r["stage"]) for r in rows)
    seen = set()
    for r in rows:
        k = (r["kind"], r["id"])
        if r["kind"] in KIND_KEY or r["kind"] == "CONDITION":
            if k not in first:
                sys.exit(f"catalogue row {k} is not in any stage list")
            r["stage"] = first[k]
            seen.add(k)
    missing = sorted(set(first) - seen)
    if missing:
        sys.exit(f"stage entities missing from the catalogue: {missing[:20]} ({len(missing)})")

    out = io.StringIO(newline="")
    w = csv.DictWriter(out, fieldnames=fields, lineterminator="\r\n" if "\r\n" in text else "\n")
    w.writeheader()
    w.writerows(rows)
    with open(a.catalogue, "w", newline="") as f:
        f.write(out.getvalue())
    after = collections.Counter((r["kind"], r["stage"]) for r in rows)
    for k in sorted(set(before) | set(after)):
        if before[k] != after[k]:
            print(f"{k[0]:<10} {k[1]:<4} {before[k]:>4} -> {after[k]:>4}")


if __name__ == "__main__":
    main()
