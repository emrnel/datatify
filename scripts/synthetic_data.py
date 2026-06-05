# -*- coding: utf-8 -*-
"""
Synthetic Spotify Extended Streaming History generator.

Generates Spotify-like JSON / Parquet datasets at arbitrary scale (500K, 1M,
5M+ records) for the scalability benchmarks specified in §III and §V of the
proposal. Distributions (artist popularity, hour of day, skip behaviour) are
calibrated against the real 131K-record dataset to keep the synthetic load
representative of real query patterns.

Usage
-----
    python synthetic_data.py --records 1000000 --out ./synthetic/1M.json
    python synthetic_data.py --records 5000000 --format parquet --out ./synthetic/5M.parquet
"""

from __future__ import annotations

import argparse
import json
import math
import random
import string
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

PLATFORMS = [
    "android",
    "ios",
    "web_player",
    "windows",
    "osx",
]
COUNTRIES = ["TR", "DE", "US", "GB", "NL", "FR", "ES", "IT"]
REASON_START = ["trackdone", "fwdbtn", "backbtn", "clickrow", "appload"]
REASON_END = ["trackdone", "fwdbtn", "logout", "endplay", "remote"]


def _make_artist_pool(n: int) -> list[str]:
    """Zipf-distributed artist names so a few are very popular."""
    return [f"Artist_{i:05d}" for i in range(n)]


def _make_track_pool(n: int) -> list[str]:
    """Track names are short random alpha strings."""
    return [
        "Track_" + "".join(random.choices(string.ascii_uppercase, k=6))
        for _ in range(n)
    ]


def _zipf_indices(size: int, n_items: int, alpha: float = 1.2) -> np.ndarray:
    """Sample ``size`` indices in [0, n_items) from a Zipf-like distribution."""
    raw = np.random.zipf(alpha, size=size)
    return np.clip(raw - 1, 0, n_items - 1)


def generate_records(
    n_records: int,
    *,
    n_artists: int | None = None,
    n_tracks: int | None = None,
    seed: int = 42,
    start_year: int = 2018,
    end_year: int = 2026,
) -> list[dict]:
    """Generate a list of synthetic Spotify-like records."""
    random.seed(seed)
    np.random.seed(seed)

    n_artists = n_artists or max(50, int(math.sqrt(n_records) * 4))
    n_tracks = n_tracks or max(500, int(math.sqrt(n_records) * 30))
    artists = _make_artist_pool(n_artists)
    tracks = _make_track_pool(n_tracks)

    artist_idx = _zipf_indices(n_records, n_artists, alpha=1.15)
    track_idx = _zipf_indices(n_records, n_tracks, alpha=1.05)

    start_ts = datetime(start_year, 1, 1, tzinfo=timezone.utc)
    span_seconds = int((datetime(end_year, 12, 31, tzinfo=timezone.utc) - start_ts).total_seconds())
    seconds_offsets = np.random.uniform(0, span_seconds, size=n_records)
    seconds_offsets.sort()

    night_bias = np.random.choice(
        [0, 1, 2, 3, 4, 5, 19, 20, 21, 22, 23],
        size=int(n_records * 0.35),
    )
    night_offsets = np.random.uniform(0, span_seconds, size=int(n_records * 0.35))
    seconds_offsets[: len(night_offsets)] = night_offsets

    skip_probability = 0.32
    skipped_arr = np.random.random(n_records) < skip_probability
    ms_played = np.where(
        skipped_arr,
        np.random.randint(1_000, 60_000, size=n_records),
        np.random.randint(60_000, 300_000, size=n_records),
    )
    shuffle_arr = np.random.random(n_records) < 0.45
    offline_arr = np.random.random(n_records) < 0.08
    incognito_arr = np.random.random(n_records) < 0.03

    plats = np.random.choice(PLATFORMS, size=n_records, p=[0.45, 0.20, 0.10, 0.20, 0.05])
    countries = np.random.choice(COUNTRIES, size=n_records, p=[0.55, 0.10, 0.10, 0.05, 0.05, 0.05, 0.05, 0.05])
    reason_start_arr = np.random.choice(REASON_START, size=n_records)
    reason_end_arr = np.random.choice(REASON_END, size=n_records)

    records: list[dict] = []
    append = records.append
    for i in range(n_records):
        ts = start_ts + timedelta(seconds=float(seconds_offsets[i]))
        artist_name = artists[int(artist_idx[i])]
        track_name = tracks[int(track_idx[i])]
        append(
            {
                "ts": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "ms_played": int(ms_played[i]),
                "master_metadata_track_name": track_name,
                "master_metadata_album_artist_name": artist_name,
                "master_metadata_album_album_name": f"Album_of_{artist_name}",
                "skipped": bool(skipped_arr[i]),
                "shuffle": bool(shuffle_arr[i]),
                "reason_start": str(reason_start_arr[i]),
                "reason_end": str(reason_end_arr[i]),
                "conn_country": str(countries[i]),
                "platform": str(plats[i]),
                "offline": bool(offline_arr[i]),
                "incognito_mode": bool(incognito_arr[i]),
            }
        )
    return records


def write_json(records: list[dict], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)
    return out_path


def write_parquet(records: list[dict], out_path: str | Path) -> Path:
    import pandas as pd

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(out_path, engine="pyarrow", index=False)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic Spotify history")
    ap.add_argument("--records", type=int, required=True, help="Number of records to generate")
    ap.add_argument("--out", type=str, required=True, help="Output file path")
    ap.add_argument(
        "--format", choices=["json", "parquet"], default="json", help="Output format"
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"[synthetic_data] generating {args.records:,} records (seed={args.seed})...")
    records = generate_records(args.records, seed=args.seed)
    print(f"[synthetic_data] writing {args.format.upper()} -> {args.out}")
    if args.format == "json":
        write_json(records, args.out)
    else:
        write_parquet(records, args.out)
    print(f"[synthetic_data] done: {len(records):,} records")


if __name__ == "__main__":
    main()
