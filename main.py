"""Roblox Up-and-Coming top-10 forecast.

Wide persistence kernel over the close chart: each place's named mass is
spread over nearby close ranks with a 1 / (1 + distance) kernel, named share
0.65, a small discount for games on Top Trending, and 28-day expiry scaling
from each game's creation time. Deterministic, standard library only.
"""

import json
import os
from datetime import datetime, timedelta, timezone


NAMED_TOTAL = 0.65
MAX_P = 0.25
SPAN = 11
TRENDING_FACTOR = 0.9
EXPIRY_DAYS = 28
BENCH_RANKS = range(21, 31)


def parse_ts(value):
    """Parse an ISO 8601 timestamp (or epoch number) to an aware datetime."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def to_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_history(input_dir):
    """Read history.ndjson, oldest first, skipping unusable lines."""
    path = os.path.join(input_dir, "history.ndjson")
    reads = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if not isinstance(record, dict):
                continue
            if record.get("ok") is False:
                continue
            moment = parse_ts(record.get("at"))
            if moment is None:
                continue
            reads.append((moment, record))
    reads.sort(key=lambda item: item[0])
    return reads


def is_sponsored(game):
    for key in ("sponsored", "is_sponsored", "sponsored_content", "ad"):
        if game.get(key):
            return True
    return False


def chart_games(record):
    """Deduplicated close-chart games, sorted by rank then universe id."""
    games = []
    seen = set()
    for game in record.get("up_and_coming") or []:
        if not isinstance(game, dict):
            continue
        uid = to_int(game.get("universe_id"))
        if uid is None or uid in seen:
            continue
        if is_sponsored(game):
            continue
        rank = to_int(game.get("rank"))
        if rank is None:
            continue
        seen.add(uid)
        games.append({"universe_id": uid, "rank": rank,
                      "playing": game.get("playing")})
    games.sort(key=lambda item: (item["rank"], item["universe_id"]))
    return games


def cap_and_normalize(weights):
    """Scale weights to NAMED_TOTAL with each share capped at MAX_P.

    Water-filling: entries that would exceed the cap are pinned at the cap
    and the remaining mass is shared among the rest in proportion to weight.
    Returns a list of (universe_id, p) sorted by universe id.
    """
    result = {}
    remaining = list(weights)
    mass_left = NAMED_TOTAL
    while remaining and mass_left > 0.0:
        weight_sum = sum(weight for _, weight in remaining)
        if weight_sum <= 0.0:
            break
        share = mass_left / weight_sum
        capped = [(uid, w) for uid, w in remaining if w * share >= MAX_P]
        if not capped:
            for uid, weight in remaining:
                result[uid] = weight * share
            break
        for uid, _weight in capped:
            result[uid] = MAX_P
            mass_left -= MAX_P
        remaining = [(uid, w) for uid, w in remaining if w * share < MAX_P]
    return sorted(result.items())


def build_forecast(task, reads):
    places = to_int(task.get("places")) or 10
    window = task.get("window") or {}
    close_at, close_record = reads[-1]
    win_from = parse_ts(window.get("from")) or close_at
    win_until = parse_ts(window.get("until")) or (
        win_from + timedelta(hours=72))
    window_seconds = max((win_until - win_from).total_seconds(), 1.0)

    games = chart_games(close_record)
    close_rank = {game["universe_id"]: game["rank"] for game in games}
    close_playing = {}
    for game in games:
        playing = game.get("playing")
        close_playing[game["universe_id"]] = (
            playing if isinstance(playing, (int, float)) else None)

    trending = set()
    for game in close_record.get("top_trending") or []:
        if isinstance(game, dict):
            uid = to_int(game.get("universe_id"))
            if uid is not None:
                trending.add(uid)

    details_by_id = {}
    details = close_record.get("details")
    if isinstance(details, dict):
        for key, info in details.items():
            uid = to_int(key)
            if uid is not None:
                details_by_id[uid] = info

    def expiry_fraction(uid):
        info = details_by_id.get(uid)
        if not isinstance(info, dict):
            return 1.0
        created = parse_ts(info.get("created"))
        if created is None:
            return 1.0
        expiry = created + timedelta(days=EXPIRY_DAYS)
        if expiry <= win_from:
            return 0.0
        if expiry >= win_until:
            return 1.0
        return (expiry - win_from).total_seconds() / window_seconds

    probabilities = []
    expected_place = {}
    for place in range(1, places + 1):
        weights = []
        for game in games:
            uid = game["universe_id"]
            rank = game["rank"]
            distance = abs(rank - place)
            in_span = distance <= SPAN
            in_bench = place >= 8 and rank in BENCH_RANKS
            if not in_span and not in_bench:
                continue
            weight = 1.0 / (1.0 + distance)
            if uid in trending:
                weight *= TRENDING_FACTOR
            weight *= expiry_fraction(uid)
            if weight <= 0.0:
                continue
            weights.append((uid, weight))
        if not weights:
            continue
        for uid, prob in cap_and_normalize(weights):
            if prob <= 0.0:
                continue
            probabilities.append(
                {"universe_id": uid, "place": place, "p": prob})
            expected_place[uid] = expected_place.get(uid, 0.0) + place * prob

    probabilities.sort(key=lambda row: (row["place"], row["universe_id"]))

    # Rank by conditional expected place: raw expected place divided by
    # the game's total named mass. Dividing matters because a bench game
    # named at only a few places with tiny probabilities would otherwise
    # look like it belongs at the top. Tie-break by close rank, then by
    # higher close playing count.
    mass = {}
    for row in probabilities:
        mass[row["universe_id"]] = mass.get(row["universe_id"], 0.0) + row["p"]
    ranked = sorted(
        expected_place,
        key=lambda uid: (
            expected_place[uid] / mass.get(uid, 1.0),
            close_rank.get(uid, 10 ** 9),
            -(close_playing.get(uid) or 0),
        ),
    )
    top10 = list(ranked[:10])
    if len(top10) < 10:
        chosen = set(top10)
        for _moment, record in reversed(reads[:-1]):
            for game in chart_games(record):
                uid = game["universe_id"]
                if uid in chosen:
                    continue
                chosen.add(uid)
                top10.append(uid)
                if len(top10) >= 10:
                    break
            if len(top10) >= 10:
                break
    return {"probabilities": probabilities, "top10": top10[:10]}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    with open(os.path.join(input_dir, "task.json"), "r", encoding="utf-8") as handle:
        task = json.load(handle)
    reads = load_history(input_dir)
    if not reads:
        raise SystemExit("no usable reads in history.ndjson")
    forecast = build_forecast(task, reads)
    os.makedirs(output_dir, exist_ok=True)
    out_name = task.get("output") or "forecast.json"
    out_path = os.path.join(output_dir, out_name)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(forecast, handle, separators=(",", ":"))
        handle.write("\n")


if __name__ == "__main__":
    main()
