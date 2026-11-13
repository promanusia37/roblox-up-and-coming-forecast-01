#!/usr/bin/env python3
"""Predict Roblox's Up-and-Coming top 10 (make.fun roblox-up-and-coming-forecast-01).

Model: an empirical rank-transition kernel learned from the chart history in
/input. For pairs of past reads we record, per time-gap bucket (0-14h, 14-38h,
38-90h), how often a game at rank r is later found at each place 1..10 (or
outside the top 10). The forecast for the scored window mixes the three
buckets with weights matching the window's hour distribution (11/72, 24/72,
37/72), applies each game's kernel row from its rank in the last read, tilts
slightly toward games whose player count grew over the last ~24 hours, and
normalises every place to 0.85, leaving 0.15 unnamed for games we did not
name. Rows are Laplace-smoothed toward the bucket's global place
distribution; sparse rows fall back to that distribution.

Backtests on the kit's practice history put this well ahead of the
persistence baseline. Standard library only, no network, deterministic.
"""

import json
import math
import os
from datetime import datetime, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

BUCKETS = ((0.0, 14.0), (14.0, 38.0), (38.0, 90.0))
WEIGHTS = (11.0 / 72.0, 24.0 / 72.0, 37.0 / 72.0)
RESERVE = 0.15          # unnamed share left at every place
MIN_P = 0.003           # drop rows below this before renormalising
SMOOTH = 6.0            # Laplace pseudo-count toward the global distribution
MIN_ROW = 8             # rows with fewer observations fall back to global
BETA = 0.3              # player-count growth tilt
MAX_RANK = 41           # ranks above 40 are pooled into row 41 ("absent")


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def load_reads():
    task_path = os.path.join(INPUT_DIR, "task.json")
    with open(task_path, "r", encoding="utf-8") as fh:
        task = json.load(fh)
    reads = []
    with open(os.path.join(INPUT_DIR, "history.ndjson"), "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            snap = json.loads(line)
            if snap.get("ok") is False:
                continue  # a failed read carries no chart
            rows = list(snap.get("up_and_coming") or [])
            rows.sort(key=lambda g: g["rank"])
            reads.append({
                "at": parse_time(snap["at"]),
                "chart": [int(g["universe_id"]) for g in rows],
                "playing": {int(g["universe_id"]): g.get("playing") for g in rows},
            })
    reads.sort(key=lambda r: r["at"])
    return task, reads


def build_kernel(reads):
    """K[bucket][rank] -> list of 12 shares: index 0 unused, 1..10 places, 11 = outside top 10."""
    counts = [{r: [0.0] * 12 for r in range(1, MAX_RANK + 1)} for _ in BUCKETS]
    rank_maps = [{g: i + 1 for i, g in enumerate(r["chart"])} for r in reads]
    for i, a in enumerate(reads):
        for j in range(i + 1, len(reads)):
            gap = (reads[j]["at"] - a["at"]).total_seconds() / 3600.0
            bucket = None
            for bi, (lo, hi) in enumerate(BUCKETS):
                if lo <= gap < hi:
                    bucket = bi
                    break
            if bucket is None:
                continue
            later = rank_maps[j]
            for game, rank in rank_maps[i].items():
                dest = later.get(game, 99)
                counts[bucket][min(rank, MAX_RANK)][dest if dest <= 10 else 11] += 1.0
    kernel = []
    for bi in range(len(BUCKETS)):
        glob = [0.0] * 12
        for r in range(1, MAX_RANK + 1):
            for k in range(12):
                glob[k] += counts[bi][r][k]
        total = sum(glob) or 1.0
        glob = [x / total for x in glob]
        rows = {}
        for r in range(1, MAX_RANK + 1):
            row_counts = counts[bi][r]
            n = sum(row_counts)
            if n < MIN_ROW:
                rows[r] = glob[:]
            else:
                rows[r] = [(row_counts[k] + SMOOTH * glob[k]) / (n + SMOOTH) for k in range(12)]
        kernel.append(rows)
    return kernel


def growth_tilts(reads):
    """exp(BETA * clipped log player-count growth over the last ~24h), per game."""
    if len(reads) < 2:
        return {}
    last = reads[-1]
    ref = None
    for cand in reversed(reads[:-1]):
        if (last["at"] - cand["at"]).total_seconds() >= 20 * 3600:
            ref = cand
            break
    if ref is None:
        ref = reads[0]
    tilts = {}
    for game, playing in last["playing"].items():
        before = ref["playing"].get(game)
        if playing and before:
            growth = max(-1.0, min(1.5, math.log(playing / before)))
            tilts[game] = math.exp(BETA * growth)
    return tilts


def main():
    task, reads = load_reads()
    places = int(task.get("places", 10))
    kernel = build_kernel(reads)
    tilts = growth_tilts(reads)

    last = reads[-1]
    rank_last = {g: i + 1 for i, g in enumerate(last["chart"])}
    candidates = {g for g, r in rank_last.items() if r <= 40}
    for snap in reads[-6:]:
        for g in snap["chart"][:20]:
            candidates.add(g)

    raw = {k: {} for k in range(1, places + 1)}
    for game in candidates:
        rank = min(rank_last.get(game, MAX_RANK), MAX_RANK)
        tilt = tilts.get(game, 1.0)
        for bi, weight in enumerate(WEIGHTS):
            row = kernel[bi][rank]
            for k in range(1, places + 1):
                raw[k][game] = raw[k].get(game, 0.0) + weight * row[k] * tilt

    probabilities = []
    pin, weighted = {}, {}
    for k in range(1, places + 1):
        col = raw[k]
        total = sum(col.values())
        if total <= 0:
            continue
        col = {g: p / total * (1.0 - RESERVE) for g, p in col.items()}
        col = {g: p for g, p in col.items() if p >= MIN_P}
        col = dict(sorted(col.items(), key=lambda kv: (-kv[1], kv[0]))[:100])
        tot2 = sum(col.values())
        if tot2 <= 0:
            continue
        for g, p in col.items():
            p = round(p / tot2 * (1.0 - RESERVE), 6)
            if p <= 0:
                continue
            probabilities.append({"universe_id": g, "place": k, "p": p})
            pin[g] = pin.get(g, 0.0) + p
            weighted[g] = weighted.get(g, 0.0) + p * k

    ranked = sorted(pin, key=lambda g: (-pin[g], weighted[g] / pin[g], g))
    top10 = ranked[:places]
    for g in last["chart"]:
        if len(top10) >= places:
            break
        if g not in top10:
            top10.append(g)

    probabilities.sort(key=lambda r: (r["place"], -r["p"], r["universe_id"]))
    out_path = os.path.join(OUTPUT_DIR, task.get("output", "forecast.json"))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"probabilities": probabilities, "top10": top10}) + "\n")


if __name__ == "__main__":
    main()
