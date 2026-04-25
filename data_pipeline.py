# -*- coding: utf-8 -*-
"""
Data Processing Pipeline (Methods §B of proposal).

Pandas-based ingestion → validation → Parquet partitioning → aggregation.
Provides a single-node DataFrame implementation that mirrors the metric
definitions from §C of the proposal and serves as the baseline against
which the PySpark pipeline (``spark_pipeline.py``) is benchmarked.

Pipeline stages
---------------
    1. ``load_json_files(paths)``        Read raw Spotify exports → DataFrame
    2. ``validate_schema(df)``           Drop / fill records lacking required cols
    3. ``write_parquet(df, root)``       Year/month partitioned Parquet write
    4. ``read_parquet(root)``            Round-trip read for analytics
    5. ``compute_metrics_pandas(df)``    GroupBy aggregations → metrics dict
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLS = [
    "ts",
    "ms_played",
    "master_metadata_track_name",
    "master_metadata_album_artist_name",
]

OPTIONAL_COLS = [
    "master_metadata_album_album_name",
    "skipped",
    "shuffle",
    "reason_start",
    "reason_end",
    "conn_country",
    "platform",
    "offline",
    "incognito_mode",
]

SESSION_GAP_MINUTES = 30


def load_json_files(paths: list[str | Path]) -> pd.DataFrame:
    """Read one or more Spotify Extended Streaming History JSON exports."""
    frames: list[pd.DataFrame] = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, list):
            continue
        frames.append(pd.DataFrame(payload))
    if not frames:
        return pd.DataFrame(columns=REQUIRED_COLS + OPTIONAL_COLS)
    return pd.concat(frames, ignore_index=True)


def load_records(records: list[dict]) -> pd.DataFrame:
    """Build a DataFrame from already-parsed records (used by FastAPI)."""
    if not records:
        return pd.DataFrame(columns=REQUIRED_COLS + OPTIONAL_COLS)
    return pd.DataFrame(records)


def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Filter unusable rows and coerce dtypes."""
    if df.empty:
        return df

    for col in REQUIRED_COLS + OPTIONAL_COLS:
        if col not in df.columns:
            df[col] = pd.NA

    df = df.dropna(subset=["ts", "master_metadata_track_name"]).copy()

    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).copy()

    df["ms_played"] = pd.to_numeric(df["ms_played"], errors="coerce").fillna(0).astype(np.int64)
    for col in ("skipped", "shuffle", "offline", "incognito_mode"):
        df[col] = df[col].fillna(False).astype(bool)

    for col in (
        "master_metadata_album_artist_name",
        "master_metadata_album_album_name",
        "reason_start",
        "reason_end",
        "conn_country",
        "platform",
    ):
        df[col] = df[col].fillna("Unknown").astype(str)

    df["year"] = df["ts"].dt.year.astype(np.int32)
    df["month"] = df["ts"].dt.month.astype(np.int32)
    df["hour_utc"] = df["ts"].dt.hour.astype(np.int32)
    df["weekday"] = df["ts"].dt.weekday.astype(np.int32)

    return df


def write_parquet(df: pd.DataFrame, root: str | Path) -> Path:
    """Write a year/month partitioned Parquet dataset."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return root
    df.to_parquet(
        root,
        engine="pyarrow",
        partition_cols=["year", "month"],
        index=False,
    )
    return root


def read_parquet(root: str | Path) -> pd.DataFrame:
    """Read back a partitioned Parquet dataset as a single DataFrame."""
    return pd.read_parquet(root, engine="pyarrow")


# ───────────────────────── Metric computation ──────────────────────────────

TZ_OFFSETS = {
    "TR": 3, "DE": 1, "FR": 1, "NL": 1, "GB": 0, "US": -5,
    "CA": -5, "AT": 1, "CZ": 1, "IT": 1, "ES": 1, "SE": 1,
    "NO": 1, "DK": 1, "FI": 2, "PL": 1, "BE": 1, "CH": 1,
    "PT": 0, "GR": 2, "RO": 2, "BG": 2, "JP": 9, "KR": 9,
    "AU": 10, "NZ": 12, "BR": -3, "MX": -6, "AR": -3, "IN": 5,
}


def _platform_category(raw: str) -> str:
    p = (raw or "").split("(")[0].strip().lower()
    if any(k in p for k in ("android", "ios", "iphone")):
        return "Mobile"
    if any(k in p for k in ("web", "chrome", "firefox")):
        return "Web"
    if any(k in p for k in ("desktop", "windows", "mac")):
        return "Desktop"
    return "Other"


def _detect_sessions(df: pd.DataFrame) -> tuple[int, float]:
    """Return (session_count, avg_session_hours) using a 30-min gap heuristic."""
    if df.empty:
        return 0, 0.0
    sorted_df = df.sort_values("ts")
    ts = sorted_df["ts"].to_numpy()
    ms = sorted_df["ms_played"].to_numpy()
    gap = np.timedelta64(SESSION_GAP_MINUTES * 60, "s")
    new_session = np.zeros(len(ts), dtype=bool)
    new_session[0] = True
    new_session[1:] = (ts[1:] - ts[:-1]) > gap
    session_id = np.cumsum(new_session) - 1
    session_ms = pd.Series(ms).groupby(session_id).sum().to_numpy()
    n_sessions = int(session_ms.size)
    avg_h = float(session_ms.mean()) / (1000 * 3600) if n_sessions else 0.0
    return n_sessions, avg_h


def _shannon_entropy(counts: pd.Series) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    return float(-(p * np.log2(p.where(p > 0, 1))).sum())


def compute_metrics_pandas(df: pd.DataFrame) -> dict:
    """Pandas reference implementation of the §C metrics.

    Returns a metrics dict mirroring the structure produced by ``analyzer.analyze``.
    Used by ``benchmark.py`` to measure Pandas vs. PySpark performance and
    referenced from the FastAPI app as the production data path.
    """
    df = validate_schema(df)
    if df.empty:
        return {"error": "no valid records"}

    df = df[df["master_metadata_track_name"].astype(bool)].copy()
    total_plays = int(len(df))
    if total_plays == 0:
        return {"error": "no playable tracks"}

    total_ms = int(df["ms_played"].sum())
    total_hours = total_ms / (1000 * 3600)

    # ── Aggregations ──
    by_artist = df.groupby("master_metadata_album_artist_name").agg(
        ms=("ms_played", "sum"),
        plays=("ms_played", "size"),
        skips=("skipped", "sum"),
    )
    by_song = df.groupby(
        ["master_metadata_track_name", "master_metadata_album_artist_name"]
    ).agg(ms=("ms_played", "sum"), plays=("ms_played", "size"))
    by_year = df.groupby("year").agg(ms=("ms_played", "sum"), plays=("ms_played", "size"))
    by_hour = df.groupby("hour_utc").agg(ms=("ms_played", "sum"))
    by_weekday = df.groupby("weekday").agg(ms=("ms_played", "sum"))
    by_country = df.groupby("conn_country").agg(ms=("ms_played", "sum"))

    df["platform_cat"] = df["platform"].map(_platform_category)
    by_platform = df.groupby("platform_cat").agg(ms=("ms_played", "sum"))

    # ── Behavioral metrics ──
    skipped_count = int(df["skipped"].sum())
    completed_count = int((df["reason_end"].str.lower() == "endplay").sum())
    shuffle_count = int(df["shuffle"].sum())
    offline_ms = int(df.loc[df["offline"], "ms_played"].sum())
    incognito_count = int(df["incognito_mode"].sum())
    early_skip_count = int(((df["skipped"]) & (df["ms_played"] < 30_000)).sum())
    skipped_ms_sum = int(df.loc[df["skipped"], "ms_played"].sum())
    focus_play_count = int(((~df["skipped"]) & (df["ms_played"] >= 4 * 60_000)).sum())
    album_play_count = int(
        df["master_metadata_album_album_name"].astype(str).str.strip().ne("").sum()
    )

    calendar_first_ts = df["ts"].min()
    calendar_last_ts = df["ts"].max()
    calendar_days = max(1, (calendar_last_ts - calendar_first_ts).days)

    # ── Sessions ──
    n_sessions, avg_session_h = _detect_sessions(df)

    # ── Habit loop (consecutive identical (track, artist) pair repetitions) ──
    sorted_df = df.sort_values("ts")
    pair_keys = list(
        zip(
            sorted_df["master_metadata_track_name"].to_list(),
            sorted_df["master_metadata_album_artist_name"].to_list(),
        )
    )
    pair_counts: dict[tuple, int] = defaultdict(int)
    for i in range(len(pair_keys) - 1):
        pair_counts[(pair_keys[i], pair_keys[i + 1])] += 1
    total_pairs = max(0, len(pair_keys) - 1)
    repeated_pairs = sum(1 for c in pair_counts.values() if c > 1)
    habit_loop_score = round(100 * repeated_pairs / total_pairs, 1) if total_pairs else 0.0

    # ── Monthly novelty ──
    ts_naive = df["ts"].dt.tz_convert(None)
    df_naive = df.assign(_ts_naive=ts_naive)
    first_seen = (
        df_naive.sort_values("_ts_naive")
        .groupby(["master_metadata_track_name", "master_metadata_album_artist_name"])["_ts_naive"]
        .min()
        .dt.to_period("M")
    )
    new_per_month = first_seen.value_counts().sort_index()
    plays_per_month = ts_naive.dt.to_period("M").value_counts().sort_index()
    novelty = (100 * new_per_month / plays_per_month).fillna(0)
    avg_novelty = float(novelty.mean()) if not novelty.empty else 0.0

    # ── Yearly growth ──
    years_sorted = sorted(by_year.index.tolist())
    yearly_hours = [by_year.loc[y, "ms"] / (1000 * 3600) for y in years_sorted]
    yearly_growth: dict[str, float] = {}
    for i in range(1, len(years_sorted)):
        prev_h = yearly_hours[i - 1]
        if prev_h > 0:
            yearly_growth[str(years_sorted[i])] = round(
                100 * (yearly_hours[i] - prev_h) / prev_h, 1
            )

    unique_tracks = int(by_song.shape[0])
    unique_artists = int(by_artist.shape[0])
    avg_listen_sec = (total_ms / 1000) / total_plays
    avg_track_duration_sec = 210
    avg_listening_ratio = min(1.0, avg_listen_sec / avg_track_duration_sec)
    skip_latency_ratio = (
        (skipped_ms_sum / 1000) / (avg_track_duration_sec * skipped_count)
        if skipped_count else 0
    )

    # ── Timezone-corrected circadian profile ──
    dominant_country = (
        by_country.idxmax().iloc[0] if not by_country.empty else "TR"
    )
    tz_offset = TZ_OFFSETS.get(str(dominant_country), 3)
    night_utc_hours = [(h + 24 - tz_offset) % 24 for h in range(6)]
    night_ms = int(by_hour.reindex(night_utc_hours, fill_value=0)["ms"].sum())
    mobile_ms = int(by_platform.loc["Mobile", "ms"]) if "Mobile" in by_platform.index else 0

    circadian_local: dict[str, float] = {}
    for utc_h in range(24):
        local_h = (utc_h + tz_offset) % 24
        ms_val = int(by_hour.loc[utc_h, "ms"]) if utc_h in by_hour.index else 0
        circadian_local[f"{local_h:02d}"] = round(ms_val / (1000 * 3600), 1)
    circadian_sorted = {f"{h:02d}": circadian_local[f"{h:02d}"] for h in range(24)}

    # ── Top lists ──
    top_artists_df = by_artist.sort_values("ms", ascending=False).head(20)
    top_songs_df = by_song.sort_values("ms", ascending=False).head(20)

    metrics: dict = {
        "bizim_rapor": {
            "toplam_kayit": total_plays,
            "toplam_saat": round(total_hours, 2),
            "toplam_gun_esdeger": round(total_hours / 24, 1),
            "tam_dinlenen": completed_count,
            "atlanan": skipped_count,
            "tam_dinlenme_orani_pct": round(100 * completed_count / total_plays, 1),
            "oturum_sayisi": n_sessions,
            "ortalama_oturum_saat": round(avg_session_h, 2),
            "shuffle_orani_pct": round(100 * shuffle_count / total_plays, 1),
            "cevrimdisi_saat": round(offline_ms / (1000 * 3600), 1),
            "cevrimdisi_orani_pct": round(100 * offline_ms / total_ms, 1) if total_ms else 0,
            "gizli_oturum_kayit": incognito_count,
            "takvim_gun_sayisi": calendar_days,
            "ilk_yil": int(calendar_first_ts.year),
            "son_yil": int(calendar_last_ts.year),
        },
        "metrikler": {
            "impatience_score_pct": round(100 * skipped_count / total_plays, 1),
            "completion_rate_pct": round(100 * completed_count / total_plays, 1),
            "avg_listen_sec": round(avg_listen_sec, 1),
            "avg_listening_ratio": round(avg_listening_ratio, 3),
            "exploration_score": round(100 * unique_tracks / total_plays, 1),
            "artist_diversity_entropy": round(_shannon_entropy(by_artist["plays"]), 2),
            "skip_latency_ratio": round(skip_latency_ratio, 3),
            "early_skip_rate_pct": round(100 * early_skip_count / skipped_count, 1) if skipped_count else 0,
            "listening_intensity_h_per_day": round(total_hours / calendar_days, 2),
            "yearly_growth_pct": yearly_growth,
            "circadian_hours": circadian_sorted,
            "timezone": f"UTC+{tz_offset}" if tz_offset >= 0 else f"UTC{tz_offset}",
            "night_listening_ratio_pct": round(100 * night_ms / total_ms, 1) if total_ms else 0,
            "mobile_usage_ratio_pct": round(100 * mobile_ms / total_ms, 1) if total_ms else 0,
            "album_listening_ratio_pct": round(100 * album_play_count / total_plays, 1),
            "focus_session_score_pct": round(100 * focus_play_count / total_plays, 1),
            "music_novelty_rate_pct": round(avg_novelty, 1),
            "artist_loyalty_score_pct": round(
                100 * top_artists_df.head(10)["plays"].sum() / total_plays, 1
            ),
            "song_replay_density_pct": round(
                100 * (top_songs_df["plays"].iloc[0] if not top_songs_df.empty else 0) / total_plays, 1
            ),
            "listening_fragmentation_index": round(skipped_count / n_sessions, 1) if n_sessions else 0,
            "habit_loop_score_pct": habit_loop_score,
        },
        "top_sanatcilar": [
            {
                "sira": i + 1,
                "sanatci": str(idx),
                "saat": round(row["ms"] / (1000 * 3600), 1),
                "dinleme": int(row["plays"]),
                "skip_pct": round(100 * row["skips"] / row["plays"], 1) if row["plays"] else 0,
            }
            for i, (idx, row) in enumerate(top_artists_df.iterrows())
        ],
        "top_sarkilar": [
            {
                "sira": i + 1,
                "sarki": str(idx[0]),
                "sanatci": str(idx[1]),
                "saat": round(row["ms"] / (1000 * 3600), 1),
            }
            for i, (idx, row) in enumerate(top_songs_df.iterrows())
        ],
        "yillara_gore": {
            str(int(y)): {"kayit": int(by_year.loc[y, "plays"]), "saat": round(by_year.loc[y, "ms"] / (1000 * 3600), 1)}
            for y in years_sorted
        },
        "haftanin_gunu": {
            ["Pzt", "Sal", "Çar", "Per", "Cum", "Cmt", "Paz"][w]: round(
                by_weekday.loc[w, "ms"] / (1000 * 3600), 1
            )
            if w in by_weekday.index else 0
            for w in range(7)
        },
    }
    return metrics
