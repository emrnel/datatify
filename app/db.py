# -*- coding: utf-8 -*-
"""Benchmark DB: schema, queries, and metric-vector bridge.

All public functions are pure DB operations; no FastAPI types here.
"""
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .constants import METRIC_KEYS, METRIC_LABELS

_BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = os.environ.get("DB_PATH", str(_BASE_DIR / "benchmark.db"))


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    with get_db() as conn:
        cols = ", ".join(f"{k} REAL DEFAULT 0" for k in METRIC_KEYS)
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS submissions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                {cols}
            )
        """)
        conn.commit()


def extract_metric_vector(analysis_result: dict) -> dict:
    """Bridge analyze() output to the 15-key benchmark metric shape."""
    M = analysis_result["metrikler"]
    R = analysis_result["bizim_rapor"]
    return {
        "impatience_score_pct":          M["impatience_score_pct"],
        "completion_rate_pct":           M["completion_rate_pct"],
        "exploration_score":             M["exploration_score"],
        "artist_diversity_entropy":      M["artist_diversity_entropy"],
        "early_skip_rate_pct":           M["early_skip_rate_pct"],
        "listening_intensity_h_per_day": M["listening_intensity_h_per_day"],
        "night_listening_ratio_pct":     M["night_listening_ratio_pct"],
        "mobile_usage_ratio_pct":        M["mobile_usage_ratio_pct"],
        "focus_session_score_pct":       M["focus_session_score_pct"],
        "music_novelty_rate_pct":        M["music_novelty_rate_pct"],
        "artist_loyalty_score_pct":      M["artist_loyalty_score_pct"],
        "habit_loop_score_pct":          M["habit_loop_score_pct"],
        "listening_fragmentation_index": M["listening_fragmentation_index"],
        "total_hours":                   R["toplam_saat"],
        "shuffle_pct":                   R["shuffle_orani_pct"],
    }


def _generate_labels(percentiles: dict) -> dict:
    labels = {}
    for k, pct in percentiles.items():
        _, suffix = METRIC_LABELS.get(k, (k, ""))
        labels[k] = f"%{pct:.0f} {suffix}"
    return labels


def compute_percentiles(user_values: dict) -> dict:
    with get_db() as conn:
        count = conn.execute("SELECT COUNT(*) FROM submissions").fetchone()[0]
        if count < 2:
            p = {k: 50.0 for k in METRIC_KEYS}
            return {"total_users": count, "percentiles": p, "labels": _generate_labels(p)}
        percentiles = {}
        for k in METRIC_KEYS:
            rows = conn.execute(f"SELECT {k} FROM submissions ORDER BY {k}").fetchall()
            all_vals = [r[0] for r in rows]
            below = sum(1 for v in all_vals if v < user_values[k])
            percentiles[k] = round(100 * below / len(all_vals), 1)
    return {"total_users": count, "percentiles": percentiles, "labels": _generate_labels(percentiles)}


def submit(values: dict) -> dict:
    """Insert a metric submission and return percentile rankings."""
    with get_db() as conn:
        cols = ", ".join(METRIC_KEYS)
        ph = ", ".join(["?"] * len(METRIC_KEYS))
        conn.execute(
            f"INSERT INTO submissions ({cols}) VALUES ({ph})",
            [values[k] for k in METRIC_KEYS],
        )
        conn.commit()
    return compute_percentiles(values)


def stats() -> dict:
    with get_db() as conn:
        count = conn.execute("SELECT COUNT(*) FROM submissions").fetchone()[0]
        if count == 0:
            return {"total_users": 0, "averages": {}}
        avgs = {}
        for k in METRIC_KEYS:
            row = conn.execute(f"SELECT AVG({k}) as avg_val FROM submissions").fetchone()
            avgs[k] = round(row["avg_val"], 2) if row["avg_val"] is not None else 0
    return {"total_users": count, "averages": avgs}


def all_users() -> list[dict]:
    with get_db() as conn:
        return [dict(r) for r in conn.execute(
            f"SELECT {', '.join(METRIC_KEYS)} FROM submissions"
        ).fetchall()]
