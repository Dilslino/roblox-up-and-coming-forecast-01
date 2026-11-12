"""Roblox Up-and-Coming top-10 forecaster - V1 pure wide persistence kernel.

Reads /input/history.ndjson (+ task.json), writes /output/forecast.json.
Stdlib only, no network, no clock, no randomness: a pure function of /input.
Design: wide persistence kernel (span 11, named 0.65 per place), 28-day
expiry scaling from details.created, Top Trending games discounted x0.9
(measured mean-reversion), top10 by expected place.
"""
import json
import os
from datetime import datetime, timedelta


def _parse_time(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def _load_history(input_dir):
    path = os.path.join(input_dir, "history.ndjson")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _close_chart(last_read):
    games = last_read.get("up_and_coming") or []
    games = [g for g in games if not g.get("sponsored")]
    games = [
        g for g in games
        if isinstance(g.get("universe_id"), int) and isinstance(g.get("rank"), int)
    ]
    games.sort(key=lambda g: g["rank"])
    unique = []
    seen_ids = set()
    for g in games:
        if g["universe_id"] in seen_ids:
            continue
        seen_ids.add(g["universe_id"])
        unique.append(g)
    return unique


def _expiry_fractions(games, details, window_from, window_hours):
    """Fraction of window hours before each game's 28-day expiry (1.0 if unknown)."""
    frac = {}
    if not isinstance(details, dict):
        return frac
    for g in games:
        uid = g["universe_id"]
        d = details.get(str(uid)) or details.get(uid)
        if not isinstance(d, dict):
            continue
        created = _parse_time(d.get("created"))
        if created is None or window_from is None:
            continue
        expiry = created + timedelta(days=28)
        hours_left = (expiry - window_from).total_seconds() / 3600.0
        frac[uid] = max(0.0, min(1.0, hours_left / float(window_hours)))
    return frac


def build_forecast(input_dir):
    with open(os.path.join(input_dir, "task.json"), "r", encoding="utf-8") as f:
        task = json.load(f)
    places = int(task.get("places", 10))
    output_name = task.get("output", "forecast.json")
    window = task.get("window") or {}
    window_from = _parse_time(window.get("from"))
    window_until = _parse_time(window.get("until"))
    window_hours = 72
    if window_from is not None and window_until is not None:
        window_hours = max(
            1, int(round((window_until - window_from).total_seconds() / 3600.0))
        )

    history = _load_history(input_dir)
    if not history:
        raise ValueError("empty history")
    chart = _close_chart(history[-1])
    if not chart:
        raise ValueError("empty close chart")
    n_chart = len(chart)
    rank_of = {g["universe_id"]: g["rank"] for g in chart}
    details = history[-1].get("details")
    expiry = _expiry_fractions(chart, details, window_from, window_hours)
    trend_entries = history[-1].get("top_trending") or []
    if isinstance(trend_entries, dict):
        trend_entries = list(trend_entries.values())
    trending_ids = set()
    for entry in trend_entries:
        if isinstance(entry, dict) and isinstance(entry.get("universe_id"), int):
            trending_ids.add(entry["universe_id"])

    span = 11
    named_total = 0.65
    cap = 0.25

    probabilities = []
    place_mass = {}
    for k in range(1, places + 1):
        weights = {}
        for g in chart:
            r = g["rank"]
            if abs(r - k) <= span:
                w = 1.0 / (1.0 + abs(r - k))
                if g["universe_id"] in trending_ids:
                    w *= 0.9
                weights[g["universe_id"]] = w
        if k >= 8:
            for g in chart:
                r = g["rank"]
                if 21 <= r <= 30 and g["universe_id"] not in weights:
                    w = 1.0 / (1.0 + abs(r - k) + 5.0)
                    if g["universe_id"] in trending_ids:
                        w *= 0.9
                    weights[g["universe_id"]] = w
        scaled = {}
        for uid, w in weights.items():
            scaled[uid] = w * expiry.get(uid, 1.0)
        total = sum(scaled.values())
        if total <= 0:
            scaled = dict(weights)
            total = sum(scaled.values())
        rows = {}
        for uid, w in scaled.items():
            rows[uid] = min(named_total * w / total, cap)
        s = sum(rows.values())
        if 0 < s < named_total:
            for uid in rows:
                rows[uid] = rows[uid] * named_total / s
        place_mass[k] = rows
        for uid in sorted(rows):
            probabilities.append({"universe_id": uid, "place": k, "p": rows[uid]})

    expected = {}
    for uid in rank_of:
        exp_place = 0.0
        for k in range(1, places + 1):
            rows = place_mass[k]
            if uid in rows:
                p = rows[uid]
            else:
                unnamed = (1.0 - sum(rows.values())) / max(1, n_chart - len(rows))
                p = unnamed
            exp_place += k * p
        expected[uid] = exp_place
    top10 = [uid for uid, _ in sorted(expected.items(), key=lambda kv: (kv[1], kv[0]))[:places]]
    if len(top10) < places:
        # Edge case: close chart holds fewer than `places` games. Pad with
        # games seen on earlier reads (most recent first) so top10 stays
        # exactly `places` distinct ids. Never triggers on a normal chart
        # (90+ games), but keeps the output valid on degenerate input.
        seen = set(top10)
        for read in reversed(history[:-1]):
            for g in _close_chart(read):
                uid = g["universe_id"]
                if uid not in seen:
                    seen.add(uid)
                    top10.append(uid)
                    if len(top10) >= places:
                        break
            if len(top10) >= places:
                break
    top10 = top10[:places]

    probabilities.sort(key=lambda r: (r["place"], r["universe_id"]))
    return {"probabilities": probabilities, "top10": top10}, output_name


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    forecast, output_name = build_forecast(input_dir)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, output_name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(forecast, sort_keys=True, separators=(",", ":")))
        f.write("\n")


if __name__ == "__main__":
    main()
