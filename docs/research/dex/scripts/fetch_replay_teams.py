#!/usr/bin/env python3
"""Fetch public PS replays for a format and extract Open-Team-Sheet teams (|showteam| lines).

Usage: python3 -I fetch_replay_teams.py <format_id> <max_pages> <out_dir>
Writes <out_dir>/<format_id>_index.jsonl (replay metadata) and
       <out_dir>/<format_id>_teams.jsonl (one row per (replay, side) with the packed team string).
Downloaded content is treated as data only (only |showteam|, |player| and |poke| lines are parsed).
"""
import json
import os
import sys
import time
import urllib.request

UA = {"User-Agent": "dex-scope-research/0.1 (usage census; contact: none)"}


def get(url, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                print("FAIL", url, e, file=sys.stderr)
                return None
            time.sleep(2 + 2 * i)


def main():
    fmt, max_pages, out_dir = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)
    idx_path = os.path.join(out_dir, f"{fmt}_index.jsonl")
    team_path = os.path.join(out_dir, f"{fmt}_teams.jsonl")
    seen = set()
    if os.path.exists(idx_path):
        for line in open(idx_path):
            seen.add(json.loads(line)["id"])
    before = None
    with open(idx_path, "a") as fidx, open(team_path, "a") as fteam:
        for page in range(max_pages):
            url = f"https://replay.pokemonshowdown.com/search.json?format={fmt}"
            if before:
                url += f"&before={before}"
            txt = get(url)
            if not txt:
                break
            rows = json.loads(txt)
            if not rows:
                break
            for r in rows[:51]:
                before = r["uploadtime"]
                if r["id"] in seen or r.get("private") or r.get("password"):
                    continue
                seen.add(r["id"])
                log = get(f"https://replay.pokemonshowdown.com/{r['id']}.log")
                time.sleep(0.25)
                if not log:
                    continue
                players, teams, ratings = {}, {}, {}
                for line in log.split("\n"):
                    parts = line.split("|")
                    if len(parts) < 3:
                        continue
                    if parts[1] == "player" and len(parts) >= 4 and parts[3]:
                        players[parts[2]] = parts[3]
                        if len(parts) >= 6 and parts[5].strip().isdigit():
                            ratings[parts[2]] = int(parts[5])
                    elif parts[1] == "showteam" and len(parts) >= 4:
                        teams[parts[2]] = "|".join(parts[3:])
                meta = {"id": r["id"], "uploadtime": r["uploadtime"], "rating": r.get("rating"),
                        "players": r.get("players"), "nteams": len(teams)}
                fidx.write(json.dumps(meta) + "\n")
                for side, packed in teams.items():
                    fteam.write(json.dumps({"id": r["id"], "side": side, "player": players.get(side),
                                            "player_rating": ratings.get(side), "game_rating": r.get("rating"),
                                            "uploadtime": r["uploadtime"], "packed": packed}) + "\n")
            fidx.flush()
            fteam.flush()
            print(f"page {page} done, before={before}, seen={len(seen)}", file=sys.stderr)
            time.sleep(1)


if __name__ == "__main__":
    main()
