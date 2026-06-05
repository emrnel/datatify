# -*- coding: utf-8 -*-
"""Single source of truth for shared constants across all Datatify modules."""
from __future__ import annotations

TZ_OFFSETS: dict[str, int] = {
    "TR": 3, "DE": 1, "FR": 1, "NL": 1, "GB": 0, "US": -5,
    "CA": -5, "AT": 1, "CZ": 1, "IT": 1, "ES": 1, "SE": 1,
    "NO": 1, "DK": 1, "FI": 2, "PL": 1, "BE": 1, "CH": 1,
    "PT": 0, "GR": 2, "RO": 2, "BG": 2, "JP": 9, "KR": 9,
    "AU": 10, "NZ": 12, "BR": -3, "MX": -6, "AR": -3, "IN": 5,
}

SESSION_GAP_MINUTES: int = 30

METRIC_KEYS: list[str] = [
    "impatience_score_pct",
    "completion_rate_pct",
    "exploration_score",
    "artist_diversity_entropy",
    "early_skip_rate_pct",
    "listening_intensity_h_per_day",
    "night_listening_ratio_pct",
    "mobile_usage_ratio_pct",
    "focus_session_score_pct",
    "music_novelty_rate_pct",
    "artist_loyalty_score_pct",
    "habit_loop_score_pct",
    "listening_fragmentation_index",
    "total_hours",
    "shuffle_pct",
]

REQUIRED_COLS: list[str] = [
    "ts",
    "ms_played",
    "master_metadata_track_name",
    "master_metadata_album_artist_name",
]

OPTIONAL_COLS: list[str] = [
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

METRIC_LABELS: dict[str, tuple[str, str]] = {
    "impatience_score_pct":          ("Sabırsızlık",  "kullanıcıdan daha sabırsız"),
    "completion_rate_pct":           ("Tamamlama",    "kullanıcıdan daha fazla şarkı bitiriyor"),
    "exploration_score":             ("Keşif",        "kullanıcıdan daha fazla keşfediyor"),
    "artist_diversity_entropy":      ("Çeşitlilik",   "kullanıcıdan daha eklektik"),
    "early_skip_rate_pct":           ("Erken Atlama", "kullanıcıdan daha hızlı atlıyor"),
    "listening_intensity_h_per_day": ("Yoğunluk",    "kullanıcıdan daha yoğun dinliyor"),
    "night_listening_ratio_pct":     ("Gece Kuşu",   "kullanıcıdan daha çok gece dinliyor"),
    "mobile_usage_ratio_pct":        ("Mobil",        "kullanıcıdan daha çok mobil kullanıyor"),
    "focus_session_score_pct":       ("Odak",         "kullanıcıdan daha odaklı"),
    "music_novelty_rate_pct":        ("Yenilik",      "kullanıcıdan daha çok yeni parça keşfediyor"),
    "artist_loyalty_score_pct":      ("Sadakat",      "kullanıcıdan daha sadık"),
    "habit_loop_score_pct":          ("Alışkanlık",   "kullanıcıdan daha alışkanlık odaklı"),
    "listening_fragmentation_index": ("Parçalılık",   "kullanıcıdan daha parçalı dinliyor"),
    "total_hours":                   ("Toplam Süre",  "kullanıcıdan daha çok dinlemiş"),
    "shuffle_pct":                   ("Shuffle",      "kullanıcıdan daha çok shuffle kullanıyor"),
}
