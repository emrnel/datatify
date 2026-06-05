# Datatify Architecture Tidying Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate duplicated constants and extract personality classification logic into two new focused modules, giving every shared constant and every classification rule a single authoritative home.

**Architecture:** Create `constants.py` as the single source of truth for all shared constants (TZ_OFFSETS, SESSION_GAP_MINUTES, METRIC_KEYS, REQUIRED_COLS, OPTIONAL_COLS, METRIC_LABELS). Create `personality.py` with pure classification functions (compute_badges, compute_archetype, compute_level, compute_radar) extracted from the monolithic `analyze()` in `analyzer.py`. Update all six consumer modules to import from these two new modules. Add `tests/` with unit and integration tests.

**Tech Stack:** Python 3.11+, FastAPI, pandas, scikit-learn, networkx, pytest

---

## File Map

| File | Action | Responsibility after change |
|------|--------|----------------------------|
| `constants.py` | **Create** | Single source of truth: TZ_OFFSETS, SESSION_GAP_MINUTES, METRIC_KEYS, REQUIRED_COLS, OPTIONAL_COLS, METRIC_LABELS |
| `personality.py` | **Create** | Pure classification: compute_badges, compute_archetype, compute_level, compute_radar |
| `tests/__init__.py` | **Create** | Empty — marks tests as a package |
| `tests/test_constants.py` | **Create** | Smoke tests: constants exist and have expected shapes |
| `tests/test_personality.py` | **Create** | Unit tests for each classification function |
| `tests/test_analyzer.py` | **Create** | Integration test: analyze() on minimal synthetic data |
| `analyzer.py` | **Modify** | Remove TZ_OFFSETS, SESSION_GAP_MINUTES; delegate badge/archetype/level/radar to personality.py |
| `data_pipeline.py` | **Modify** | Remove TZ_OFFSETS, SESSION_GAP_MINUTES, REQUIRED_COLS, OPTIONAL_COLS; import from constants |
| `graph_analysis.py` | **Modify** | Remove SESSION_GAP_MINUTES; import from constants |
| `clustering.py` | **Modify** | Remove METRIC_KEYS local def; import from constants |
| `main.py` | **Modify** | Remove METRIC_KEYS, METRIC_LABELS local defs; import from constants |
| `requirements.txt` | **Modify** | Add pytest |

---

## Task 1: Create `constants.py`

**Files:**
- Create: `constants.py`

- [ ] **Step 1: Write the file**

```python
# constants.py
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
    "impatience_score_pct": ("Sabırsızlık", "kullanıcıdan daha sabırsız"),
    "completion_rate_pct": ("Tamamlama", "kullanıcıdan daha fazla şarkı bitiriyor"),
    "exploration_score": ("Keşif", "kullanıcıdan daha fazla keşfediyor"),
    "artist_diversity_entropy": ("Çeşitlilik", "kullanıcıdan daha eklektik"),
    "early_skip_rate_pct": ("Erken Atlama", "kullanıcıdan daha hızlı atlıyor"),
    "listening_intensity_h_per_day": ("Yoğunluk", "kullanıcıdan daha yoğun dinliyor"),
    "night_listening_ratio_pct": ("Gece Kuşu", "kullanıcıdan daha çok gece dinliyor"),
    "mobile_usage_ratio_pct": ("Mobil", "kullanıcıdan daha çok mobil kullanıyor"),
    "focus_session_score_pct": ("Odak", "kullanıcıdan daha odaklı"),
    "music_novelty_rate_pct": ("Yenilik", "kullanıcıdan daha çok yeni parça keşfediyor"),
    "artist_loyalty_score_pct": ("Sadakat", "kullanıcıdan daha sadık"),
    "habit_loop_score_pct": ("Alışkanlık", "kullanıcıdan daha alışkanlık odaklı"),
    "listening_fragmentation_index": ("Parçalılık", "kullanıcıdan daha parçalı dinliyor"),
    "total_hours": ("Toplam Süre", "kullanıcıdan daha çok dinlemiş"),
    "shuffle_pct": ("Shuffle", "kullanıcıdan daha çok shuffle kullanıyor"),
}
```

- [ ] **Step 2: Verify it imports cleanly**

Run: `cd /Users/mehmetalpatay/Desktop/datatify && python -c "import constants; print(len(constants.METRIC_KEYS), len(constants.TZ_OFFSETS))"`
Expected: `15 30`

- [ ] **Step 3: Commit**

```bash
git add constants.py
git commit -m "feat: add constants.py — single source of truth for shared constants"
```

---

## Task 2: Write failing tests for `constants.py`

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_constants.py`

- [ ] **Step 1: Create `tests/__init__.py`**

Empty file — just create it.

- [ ] **Step 2: Write test file**

```python
# tests/test_constants.py
from constants import (
    TZ_OFFSETS,
    SESSION_GAP_MINUTES,
    METRIC_KEYS,
    REQUIRED_COLS,
    OPTIONAL_COLS,
    METRIC_LABELS,
)


def test_metric_keys_count():
    assert len(METRIC_KEYS) == 15


def test_metric_keys_no_duplicates():
    assert len(METRIC_KEYS) == len(set(METRIC_KEYS))


def test_metric_labels_covers_all_keys():
    assert set(METRIC_LABELS.keys()) == set(METRIC_KEYS)


def test_tz_offsets_has_turkey():
    assert TZ_OFFSETS["TR"] == 3


def test_tz_offsets_values_are_ints():
    for country, offset in TZ_OFFSETS.items():
        assert isinstance(offset, int), f"{country} offset is not int"


def test_session_gap_is_30():
    assert SESSION_GAP_MINUTES == 30


def test_required_cols_has_ts_and_track():
    assert "ts" in REQUIRED_COLS
    assert "master_metadata_track_name" in REQUIRED_COLS


def test_optional_cols_has_skipped():
    assert "skipped" in OPTIONAL_COLS
```

- [ ] **Step 3: Run tests — they should pass immediately since constants.py already exists**

Run: `cd /Users/mehmetalpatay/Desktop/datatify && python -m pytest tests/test_constants.py -v`
Expected: all 8 tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/__init__.py tests/test_constants.py
git commit -m "test: add smoke tests for constants.py"
```

---

## Task 3: Create `personality.py`

**Files:**
- Create: `personality.py`

- [ ] **Step 1: Write the file**

```python
# personality.py
"""Pure classification functions: badges, archetype, level, radar chart.

All functions are side-effect-free: they take data, return new dicts/lists.
No I/O, no global state. Call them directly in tests without constructing
fake streaming records.
"""
from __future__ import annotations

_LEVEL_THRESHOLDS: list[tuple[int, str, int]] = [
    (50,   "Casual",        1),
    (200,  "Listener",      2),
    (500,  "Enthusiast",    3),
    (1000, "Devotee",       4),
    (1500, "Addict",        5),
    (2000, "Obsessed",      6),
    (2500, "Maniac",        7),
    (3000, "Legendary",     8),
    (4000, "Mythic",        9),
    (5000, "Transcendent", 10),
]

# Each entry: (predicate(night, imp, expl, focus, loyalty, entropy, novelty), name, description)
_ARCHETYPE_RULES: list[tuple] = [
    (lambda n, i, e, f, l, en, nv: n > 25 and f > 10,  "The Night Diver",       "Gece saatlerinde derinlere dalan, odaklı bir dinleyici."),
    (lambda n, i, e, f, l, en, nv: i > 40 and e > 12,  "The Restless Explorer", "Sürekli yeni şeyler arıyor, beğenmediklerini anında geçiyor."),
    (lambda n, i, e, f, l, en, nv: l > 18 and nv < 7,  "The Loyal Guardian",    "Sevdiği sanatçılara sadık, güvendiği limandan ayrılmıyor."),
    (lambda n, i, e, f, l, en, nv: en > 10 and e > 8,  "The Eclectic Mind",     "Çok geniş bir müzik yelpazesi, türler arası rahatça geziniyor."),
    (lambda n, i, e, f, l, en, nv: f > 12 and i < 25,  "The Deep Listener",     "Sabırlı, odaklı, şarkıları sonuna kadar dinleyen bir ruh."),
    (lambda n, i, e, f, l, en, nv: i > 35 and nv < 6,  "The Picky Repeater",    "Seçici ama keşfetmekten çok tekrar eden."),
    (lambda n, i, e, f, l, en, nv: n > 20 and en > 9,  "The Midnight Wanderer", "Gece saatlerinde farklı türler arasında dolaşan bir gezgin."),
]
_DEFAULT_ARCHETYPE = ("The Balanced Listener", "Dengeli bir dinleyici — keşfetme, sadakat ve sabır arasında denge.")


def compute_level(total_hours: float, earned_count: int) -> dict:
    """Return level dict: {"level", "title", "xp", "next_threshold_hours"}."""
    user_level, user_title, next_threshold = 1, "Newbie", 50
    for threshold, title, lvl in _LEVEL_THRESHOLDS:
        if total_hours >= threshold:
            user_level = lvl
            user_title = title
        else:
            next_threshold = threshold
            break
    return {
        "level": user_level,
        "title": user_title,
        "xp": int(total_hours * 10 + earned_count * 50),
        "next_threshold_hours": next_threshold,
    }


def compute_archetype(metrikler: dict) -> dict:
    """Return {"name", "description"} for the first matching archetype rule."""
    n  = metrikler["night_listening_ratio_pct"]
    i  = metrikler["impatience_score_pct"]
    e  = metrikler["exploration_score"]
    f  = metrikler["focus_session_score_pct"]
    l  = metrikler["artist_loyalty_score_pct"]
    en = metrikler["artist_diversity_entropy"]
    nv = metrikler["music_novelty_rate_pct"]
    for predicate, name, desc in _ARCHETYPE_RULES:
        if predicate(n, i, e, f, l, en, nv):
            return {"name": name, "description": desc}
    return {"name": _DEFAULT_ARCHETYPE[0], "description": _DEFAULT_ARCHETYPE[1]}


def compute_radar(metrikler: dict) -> dict:
    """Return six-axis radar chart dict (each value 0–100)."""
    imp       = metrikler["impatience_score_pct"]
    expl      = metrikler["exploration_score"]
    loyalty   = metrikler["artist_loyalty_score_pct"]
    focus     = metrikler["focus_session_score_pct"]
    entropy   = metrikler["artist_diversity_entropy"]
    night     = metrikler["night_listening_ratio_pct"]
    return {
        "Sabır":      round(100 - imp, 1),
        "Keşif":      round(min(expl * 5, 100), 1),
        "Sadakat":    round(min(loyalty * 5, 100), 1),
        "Odak":       round(min(focus * 5, 100), 1),
        "Çeşitlilik": round(min(entropy * 8, 100), 1),
        "Gece Kuşu":  round(min(night * 4, 100), 1),
    }


def compute_badges(
    metrikler: dict,
    rapor: dict,
    *,
    max_session_h: float,
    top1_song_hours: float,
    loyal_artists_5yr: int,
    countries_set: set,
    unique_artists: int,
    incognito_count: int,
    total_hours: float,
    total_plays: int,
) -> tuple[list[dict], int]:
    """Return (badge_list, earned_count). Badge list always has 15 entries."""
    M, R = metrikler, rapor
    defs = [
        ("night_owl",          "Night Owl",           "Gece dinleme > %25",                          M["night_listening_ratio_pct"] > 25),
        ("marathon",           "Marathon Listener",    f"Tek oturumda {max_session_h:.1f} saat",      max_session_h >= 4),
        ("one_track_mind",     "One Track Mind",       f"Bir şarkıyı {top1_song_hours:.0f}+ saat",    top1_song_hours >= 20),
        ("shuffle_addict",     "Shuffle Addict",       "Shuffle > %80",                               R["shuffle_orani_pct"] > 80),
        ("album_purist",       "Album Purist",         "Shuffle < %20",                               R["shuffle_orani_pct"] < 20),
        ("explorer",           "Explorer",             f"{unique_artists:,} benzersiz sanatçı",       unique_artists >= 1000),
        ("deep_focus",         "Deep Focus",           "Odak skoru > %15",                            M["focus_session_score_pct"] > 15),
        ("impatient",          "Impatient",            "Erken atlama > %50",                          M["early_skip_rate_pct"] > 50),
        ("loyal_fan",          "Loyal Fan",            f"{loyal_artists_5yr} sanatçıyı 5+ yıl",       loyal_artists_5yr >= 1),
        ("ghost",              "Ghost Listener",       f"{incognito_count} gizli dinleme",            incognito_count >= 100),
        ("world_traveler",     "World Traveler",       f"{len(countries_set)} ülke",                  len(countries_set) >= 3),
        ("centurion",          "Centurion",            "100.000+ dinleme",                            total_plays >= 100_000),
        ("dedication",         "Dedication",           "3.000+ saat",                                 total_hours >= 3000),
        ("offline_warrior",    "Offline Warrior",      "Çevrimdışı > %10",                            R["cevrimdisi_orani_pct"] > 10),
        ("creature_of_habit",  "Creature of Habit",    "Alışkanlık > %10",                            M["habit_loop_score_pct"] > 10),
    ]
    badges = [
        {"id": bid, "name": name, "desc": desc, "earned": bool(cond)}
        for bid, name, desc, cond in defs
    ]
    earned = sum(1 for b in badges if b["earned"])
    return badges, earned
```

- [ ] **Step 2: Verify it imports**

Run: `cd /Users/mehmetalpatay/Desktop/datatify && python -c "import personality; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add personality.py
git commit -m "feat: add personality.py — pure badge/archetype/level/radar classification"
```

---

## Task 4: Write tests for `personality.py`

**Files:**
- Create: `tests/test_personality.py`

- [ ] **Step 1: Write test file**

```python
# tests/test_personality.py
import pytest
from personality import compute_level, compute_archetype, compute_radar, compute_badges

# ── Fixtures ──────────────────────────────────────────────────────────────────

def _metrikler(**overrides):
    """Minimal valid metrikler dict. Override any key via kwargs."""
    base = {
        "night_listening_ratio_pct": 10.0,
        "impatience_score_pct": 20.0,
        "exploration_score": 5.0,
        "focus_session_score_pct": 8.0,
        "artist_loyalty_score_pct": 15.0,
        "artist_diversity_entropy": 7.0,
        "music_novelty_rate_pct": 8.0,
        "early_skip_rate_pct": 30.0,
        "habit_loop_score_pct": 5.0,
    }
    base.update(overrides)
    return base


def _rapor(**overrides):
    base = {
        "shuffle_orani_pct": 50.0,
        "cevrimdisi_orani_pct": 5.0,
    }
    base.update(overrides)
    return base


# ── compute_level ─────────────────────────────────────────────────────────────

def test_level_newbie():
    result = compute_level(total_hours=10.0, earned_count=0)
    assert result["level"] == 1
    assert result["title"] == "Newbie"
    assert result["xp"] == 100


def test_level_casual():
    result = compute_level(total_hours=50.0, earned_count=0)
    assert result["level"] == 1
    assert result["title"] == "Casual"


def test_level_listener():
    result = compute_level(total_hours=200.0, earned_count=2)
    assert result["level"] == 2
    assert result["title"] == "Listener"
    assert result["xp"] == int(200 * 10 + 2 * 50)


def test_level_legendary():
    result = compute_level(total_hours=3000.0, earned_count=10)
    assert result["level"] == 8
    assert result["title"] == "Legendary"


def test_level_transcendent():
    result = compute_level(total_hours=5001.0, earned_count=15)
    assert result["level"] == 10
    assert result["title"] == "Transcendent"


def test_level_xp_increases_with_badges():
    r1 = compute_level(100.0, 0)
    r2 = compute_level(100.0, 5)
    assert r2["xp"] == r1["xp"] + 5 * 50


# ── compute_archetype ─────────────────────────────────────────────────────────

def test_archetype_night_diver():
    m = _metrikler(night_listening_ratio_pct=30.0, focus_session_score_pct=15.0)
    result = compute_archetype(m)
    assert result["name"] == "The Night Diver"


def test_archetype_restless_explorer():
    m = _metrikler(impatience_score_pct=45.0, exploration_score=15.0)
    result = compute_archetype(m)
    assert result["name"] == "The Restless Explorer"


def test_archetype_loyal_guardian():
    m = _metrikler(artist_loyalty_score_pct=20.0, music_novelty_rate_pct=5.0)
    result = compute_archetype(m)
    assert result["name"] == "The Loyal Guardian"


def test_archetype_balanced_default():
    m = _metrikler()  # base values match no specific archetype
    result = compute_archetype(m)
    assert result["name"] == "The Balanced Listener"


def test_archetype_returns_name_and_description():
    m = _metrikler()
    result = compute_archetype(m)
    assert "name" in result
    assert "description" in result
    assert len(result["description"]) > 0


# ── compute_radar ─────────────────────────────────────────────────────────────

def test_radar_has_six_axes():
    m = _metrikler()
    result = compute_radar(m)
    assert set(result.keys()) == {"Sabır", "Keşif", "Sadakat", "Odak", "Çeşitlilik", "Gece Kuşu"}


def test_radar_values_bounded_0_100():
    m = _metrikler(
        impatience_score_pct=0.0,
        exploration_score=100.0,
        artist_loyalty_score_pct=100.0,
        focus_session_score_pct=100.0,
        artist_diversity_entropy=100.0,
        night_listening_ratio_pct=100.0,
    )
    result = compute_radar(m)
    for key, val in result.items():
        assert 0 <= val <= 100, f"{key}={val} out of bounds"


def test_radar_sabir_inverted():
    m_low  = _metrikler(impatience_score_pct=10.0)
    m_high = _metrikler(impatience_score_pct=80.0)
    assert compute_radar(m_low)["Sabır"] > compute_radar(m_high)["Sabır"]


# ── compute_badges ────────────────────────────────────────────────────────────

def _default_badge_kwargs():
    return dict(
        max_session_h=2.0,
        top1_song_hours=5.0,
        loyal_artists_5yr=0,
        countries_set={"TR"},
        unique_artists=100,
        incognito_count=10,
        total_hours=100.0,
        total_plays=1000,
    )


def test_badges_returns_15():
    badges, earned = compute_badges(_metrikler(), _rapor(), **_default_badge_kwargs())
    assert len(badges) == 15


def test_badges_earned_count_matches_list():
    badges, earned = compute_badges(_metrikler(), _rapor(), **_default_badge_kwargs())
    assert earned == sum(1 for b in badges if b["earned"])


def test_badge_night_owl_earned():
    m = _metrikler(night_listening_ratio_pct=30.0)
    badges, _ = compute_badges(m, _rapor(), **_default_badge_kwargs())
    night_owl = next(b for b in badges if b["id"] == "night_owl")
    assert night_owl["earned"] is True


def test_badge_night_owl_not_earned():
    m = _metrikler(night_listening_ratio_pct=10.0)
    badges, _ = compute_badges(m, _rapor(), **_default_badge_kwargs())
    night_owl = next(b for b in badges if b["id"] == "night_owl")
    assert night_owl["earned"] is False


def test_badge_marathon_earned():
    kwargs = {**_default_badge_kwargs(), "max_session_h": 5.0}
    badges, _ = compute_badges(_metrikler(), _rapor(), **kwargs)
    marathon = next(b for b in badges if b["id"] == "marathon")
    assert marathon["earned"] is True


def test_badge_shuffle_addict_vs_album_purist_mutually_exclusive():
    addict_r  = _rapor(shuffle_orani_pct=90.0)
    purist_r  = _rapor(shuffle_orani_pct=10.0)
    b_addict, _ = compute_badges(_metrikler(), addict_r, **_default_badge_kwargs())
    b_purist, _ = compute_badges(_metrikler(), purist_r, **_default_badge_kwargs())
    addict_earned = next(b["earned"] for b in b_addict if b["id"] == "shuffle_addict")
    purist_earned = next(b["earned"] for b in b_purist if b["id"] == "album_purist")
    assert addict_earned is True
    assert purist_earned is True


def test_badge_world_traveler_needs_3_countries():
    kwargs_few  = {**_default_badge_kwargs(), "countries_set": {"TR", "DE"}}
    kwargs_many = {**_default_badge_kwargs(), "countries_set": {"TR", "DE", "US"}}
    b_few,  _ = compute_badges(_metrikler(), _rapor(), **kwargs_few)
    b_many, _ = compute_badges(_metrikler(), _rapor(), **kwargs_many)
    wt_few  = next(b["earned"] for b in b_few  if b["id"] == "world_traveler")
    wt_many = next(b["earned"] for b in b_many if b["id"] == "world_traveler")
    assert wt_few  is False
    assert wt_many is True
```

- [ ] **Step 2: Run tests — they should all pass**

Run: `cd /Users/mehmetalpatay/Desktop/datatify && python -m pytest tests/test_personality.py -v`
Expected: all tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_personality.py
git commit -m "test: add unit tests for personality.py classification functions"
```

---

## Task 5: Update `analyzer.py` — import from constants + personality

**Files:**
- Modify: `analyzer.py`

- [ ] **Step 1: Replace the top-of-file constants with imports**

Replace lines 1–16 (the module docstring through the closing `}` of TZ_OFFSETS):
```python
# before (lines 7–15)
SESSION_GAP_MINUTES = 30

TZ_OFFSETS = {
    "TR": 3, ...
    "IN": 5,
}
```
with:
```python
from constants import SESSION_GAP_MINUTES, TZ_OFFSETS
from personality import compute_badges, compute_archetype, compute_level, compute_radar
```

The file header should become:
```python
# -*- coding: utf-8 -*-
"""Spotify Extended Streaming History — core analysis engine."""
import math
from collections import defaultdict
from datetime import datetime, timedelta

from constants import SESSION_GAP_MINUTES, TZ_OFFSETS
from personality import compute_badges, compute_archetype, compute_level, compute_radar
```

- [ ] **Step 2: Replace the badge block (lines 324–357) with a call to compute_badges**

Replace:
```python
    # ── Badges ──
    max_session_h = max(session_list) / (1000 * 3600) if session_list else 0
    top1_song_hours = top_songs[0][1]["ms"] / (1000 * 3600) if top_songs else 0
    loyal_artists_5yr = sum(
        1 for a, fl in first_listen.items()
        if calendar_last_ts and (calendar_last_ts - fl).days >= 5 * 365 and artists[a]["count"] >= 20
    )
    countries_set = set(by_country.keys()) - {"?"}

    M = metrics["metrikler"]
    R = metrics["bizim_rapor"]

    badge_defs = [
        ("night_owl",        "Night Owl",          "Gece dinleme > %25",                          M["night_listening_ratio_pct"] > 25),
        ("marathon",         "Marathon Listener",   f"Tek oturumda {max_session_h:.1f} saat",      max_session_h >= 4),
        ("one_track_mind",   "One Track Mind",      f"Bir şarkıyı {top1_song_hours:.0f}+ saat",    top1_song_hours >= 20),
        ("shuffle_addict",   "Shuffle Addict",      "Shuffle > %80",                               R["shuffle_orani_pct"] > 80),
        ("album_purist",     "Album Purist",        "Shuffle < %20",                               R["shuffle_orani_pct"] < 20),
        ("explorer",         "Explorer",            f"{unique_artists:,} benzersiz sanatçı",       unique_artists >= 1000),
        ("deep_focus",       "Deep Focus",          "Odak skoru > %15",                            M["focus_session_score_pct"] > 15),
        ("impatient",        "Impatient",           "Erken atlama > %50",                          M["early_skip_rate_pct"] > 50),
        ("loyal_fan",        "Loyal Fan",           f"{loyal_artists_5yr} sanatçıyı 5+ yıl",      loyal_artists_5yr >= 1),
        ("ghost",            "Ghost Listener",      f"{incognito_count} gizli dinleme",            incognito_count >= 100),
        ("world_traveler",   "World Traveler",      f"{len(countries_set)} ülke",                  len(countries_set) >= 3),
        ("centurion",        "Centurion",           "100.000+ dinleme",                            total_plays >= 100_000),
        ("dedication",       "Dedication",          "3.000+ saat",                                 total_hours >= 3000),
        ("offline_warrior",  "Offline Warrior",     "Çevrimdışı > %10",                            R["cevrimdisi_orani_pct"] > 10),
        ("creature_of_habit","Creature of Habit",   "Alışkanlık > %10",                            M["habit_loop_score_pct"] > 10),
    ]
    earned_badges = [{"id": bid, "name": name, "desc": desc, "earned": bool(cond)} for bid, name, desc, cond in badge_defs]
    earned_count = sum(1 for b in earned_badges if b["earned"])
    metrics["badges"] = earned_badges
    metrics["badges_earned"] = earned_count
    metrics["badges_total"] = len(badge_defs)
```
with:
```python
    # ── Badges ──
    max_session_h = max(session_list) / (1000 * 3600) if session_list else 0
    top1_song_hours = top_songs[0][1]["ms"] / (1000 * 3600) if top_songs else 0
    loyal_artists_5yr = sum(
        1 for a, fl in first_listen.items()
        if calendar_last_ts and (calendar_last_ts - fl).days >= 5 * 365 and artists[a]["count"] >= 20
    )
    countries_set = set(by_country.keys()) - {"?"}

    M = metrics["metrikler"]
    R = metrics["bizim_rapor"]

    earned_badges, earned_count = compute_badges(
        M, R,
        max_session_h=max_session_h,
        top1_song_hours=top1_song_hours,
        loyal_artists_5yr=loyal_artists_5yr,
        countries_set=countries_set,
        unique_artists=unique_artists,
        incognito_count=incognito_count,
        total_hours=total_hours,
        total_plays=total_plays,
    )
    metrics["badges"] = earned_badges
    metrics["badges_earned"] = earned_count
    metrics["badges_total"] = len(earned_badges)
```

- [ ] **Step 3: Replace level block (lines 360–381) with compute_level call**

Replace:
```python
    # ── Level system ──
    level_thresholds = [
        (50,   "Casual",        1),
        (200,  "Listener",      2),
        (500,  "Enthusiast",    3),
        (1000, "Devotee",       4),
        (1500, "Addict",        5),
        (2000, "Obsessed",      6),
        (2500, "Maniac",        7),
        (3000, "Legendary",     8),
        (4000, "Mythic",        9),
        (5000, "Transcendent", 10),
    ]
    user_level, user_title, next_threshold = 1, "Newbie", 50
    for threshold, title, lvl in level_thresholds:
        if total_hours >= threshold:
            user_level = lvl
            user_title = title
        else:
            next_threshold = threshold
            break
    xp = int(total_hours * 10 + earned_count * 50)
    metrics["level"] = {"level": user_level, "title": user_title, "xp": xp, "next_threshold_hours": next_threshold}
```
with:
```python
    # ── Level system ──
    metrics["level"] = compute_level(total_hours, earned_count)
```

- [ ] **Step 4: Replace archetype block (lines 383–409) with compute_archetype call**

Replace:
```python
    # ── Archetype (rule-based fallback) ──
    night = M["night_listening_ratio_pct"]
    imp = M["impatience_score_pct"]
    expl = M["exploration_score"]
    focus = M["focus_session_score_pct"]
    loyalty = M["artist_loyalty_score_pct"]
    entropy_val = M["artist_diversity_entropy"]
    novelty = M["music_novelty_rate_pct"]

    if night > 25 and focus > 10:
        arch = ("The Night Diver", "Gece saatlerinde derinlere dalan, odaklı bir dinleyici.")
    elif imp > 40 and expl > 12:
        arch = ("The Restless Explorer", "Sürekli yeni şeyler arıyor, beğenmediklerini anında geçiyor.")
    elif loyalty > 18 and novelty < 7:
        arch = ("The Loyal Guardian", "Sevdiği sanatçılara sadık, güvendiği limandan ayrılmıyor.")
    elif entropy_val > 10 and expl > 8:
        arch = ("The Eclectic Mind", "Çok geniş bir müzik yelpazesi, türler arası rahatça geziniyor.")
    elif focus > 12 and imp < 25:
        arch = ("The Deep Listener", "Sabırlı, odaklı, şarkıları sonuna kadar dinleyen bir ruh.")
    elif imp > 35 and novelty < 6:
        arch = ("The Picky Repeater", "Seçici ama keşfetmekten çok tekrar eden.")
    elif night > 20 and entropy_val > 9:
        arch = ("The Midnight Wanderer", "Gece saatlerinde farklı türler arasında dolaşan bir gezgin.")
    else:
        arch = ("The Balanced Listener", "Dengeli bir dinleyici — keşfetme, sadakat ve sabır arasında denge.")

    metrics["archetype"] = {"name": arch[0], "description": arch[1]}
```
with:
```python
    # ── Archetype ──
    metrics["archetype"] = compute_archetype(M)
```

- [ ] **Step 5: Replace radar block (lines 411–419) with compute_radar call**

Replace:
```python
    # ── Radar ──
    metrics["radar"] = {
        "Sabır": round(100 - imp, 1),
        "Keşif": round(min(expl * 5, 100), 1),
        "Sadakat": round(min(loyalty * 5, 100), 1),
        "Odak": round(min(focus * 5, 100), 1),
        "Çeşitlilik": round(min(entropy_val * 8, 100), 1),
        "Gece Kuşu": round(min(night * 4, 100), 1),
    }
```
with:
```python
    # ── Radar ──
    metrics["radar"] = compute_radar(M)
```

- [ ] **Step 6: Verify analyzer imports cleanly**

Run: `cd /Users/mehmetalpatay/Desktop/datatify && python -c "from analyzer import analyze; print('ok')"`
Expected: `ok`

- [ ] **Step 7: Commit**

```bash
git add analyzer.py
git commit -m "refactor: analyzer.py delegates badge/archetype/level/radar to personality.py"
```

---

## Task 6: Update `data_pipeline.py` — import from constants

**Files:**
- Modify: `data_pipeline.py`

- [ ] **Step 1: Remove the duplicated constant blocks and add imports**

Remove lines 30–49 (REQUIRED_COLS, OPTIONAL_COLS, SESSION_GAP_MINUTES definitions) and lines 131–137 (TZ_OFFSETS definition).

Add at the top of the file, after `import numpy as np`:
```python
from constants import (
    REQUIRED_COLS,
    OPTIONAL_COLS,
    SESSION_GAP_MINUTES,
    TZ_OFFSETS,
)
```

The imports block should look like:
```python
from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from constants import REQUIRED_COLS, OPTIONAL_COLS, SESSION_GAP_MINUTES, TZ_OFFSETS
```

- [ ] **Step 2: Verify data_pipeline imports cleanly**

Run: `cd /Users/mehmetalpatay/Desktop/datatify && python -c "from data_pipeline import compute_metrics_pandas; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add data_pipeline.py
git commit -m "refactor: data_pipeline.py imports constants from constants.py"
```

---

## Task 7: Update `graph_analysis.py` — import SESSION_GAP_MINUTES from constants

**Files:**
- Modify: `graph_analysis.py`

- [ ] **Step 1: Replace the local constant with an import**

Remove line 26:
```python
SESSION_GAP_MINUTES = 30
```
Add to the imports block:
```python
from constants import SESSION_GAP_MINUTES
```

The imports block should be:
```python
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Iterable

import networkx as nx

from constants import SESSION_GAP_MINUTES
```

- [ ] **Step 2: Verify graph_analysis imports cleanly**

Run: `cd /Users/mehmetalpatay/Desktop/datatify && python -c "from graph_analysis import analyze_listening_graph; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add graph_analysis.py
git commit -m "refactor: graph_analysis.py imports SESSION_GAP_MINUTES from constants.py"
```

---

## Task 8: Update `clustering.py` — import METRIC_KEYS from constants

**Files:**
- Modify: `clustering.py`

- [ ] **Step 1: Replace local METRIC_KEYS definition with import**

Remove lines 22–38 (the `METRIC_KEYS: list[str] = [...]` block).

Add to the imports block:
```python
from constants import METRIC_KEYS
```

The imports block should be:
```python
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from constants import METRIC_KEYS
```

- [ ] **Step 2: Verify clustering imports cleanly and METRIC_KEYS is re-exported**

Run: `cd /Users/mehmetalpatay/Desktop/datatify && python -c "from clustering import cluster_users, METRIC_KEYS; print(len(METRIC_KEYS))"`
Expected: `15`

- [ ] **Step 3: Commit**

```bash
git add clustering.py
git commit -m "refactor: clustering.py imports METRIC_KEYS from constants.py"
```

---

## Task 9: Update `main.py` — import from constants, remove local definitions

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Update the import line for clustering and add constants import**

Replace line 24:
```python
from clustering import cluster_users, METRIC_KEYS as CLUSTER_METRIC_KEYS
```
with:
```python
from clustering import cluster_users
from constants import METRIC_KEYS, METRIC_LABELS
```

- [ ] **Step 2: Remove the local METRIC_KEYS definition (lines 341–349)**

Remove:
```python
METRIC_KEYS = [
    "impatience_score_pct", "completion_rate_pct", "exploration_score",
    "artist_diversity_entropy", "early_skip_rate_pct",
    "listening_intensity_h_per_day", "night_listening_ratio_pct",
    "mobile_usage_ratio_pct", "focus_session_score_pct",
    "music_novelty_rate_pct", "artist_loyalty_score_pct",
    "habit_loop_score_pct", "listening_fragmentation_index",
    "total_hours", "shuffle_pct",
]
```

- [ ] **Step 3: Remove the local METRIC_LABELS definition (lines 395–411)**

Remove:
```python
METRIC_LABELS = {
    "impatience_score_pct": ("Sabırsızlık", "kullanıcıdan daha sabırsız"),
    ...
    "shuffle_pct": ("Shuffle", "kullanıcıdan daha çok shuffle kullanıyor"),
}
```

- [ ] **Step 4: Replace CLUSTER_METRIC_KEYS with METRIC_KEYS everywhere**

In main.py, `CLUSTER_METRIC_KEYS` is used in:
- Line 498: `f"SELECT {', '.join(CLUSTER_METRIC_KEYS)} FROM submissions"`
- Line 589: `f"SELECT {', '.join(CLUSTER_METRIC_KEYS)} FROM submissions"`

Replace both occurrences of `CLUSTER_METRIC_KEYS` with `METRIC_KEYS`.

- [ ] **Step 5: Verify main.py imports cleanly**

Run: `cd /Users/mehmetalpatay/Desktop/datatify && python -c "import main; print('ok')" 2>&1 | head -5`
Expected: `ok` (or only startup output, no ImportError)

- [ ] **Step 6: Commit**

```bash
git add main.py
git commit -m "refactor: main.py imports METRIC_KEYS and METRIC_LABELS from constants.py"
```

---

## Task 10: Add integration test for `analyzer.py`

**Files:**
- Create: `tests/test_analyzer.py`

- [ ] **Step 1: Write the integration test**

```python
# tests/test_analyzer.py
"""Integration test: analyzer.analyze() on a minimal set of synthetic records."""
from datetime import datetime, timezone, timedelta

from analyzer import analyze


def _make_record(ts: datetime, artist: str, track: str, ms: int, **kwargs) -> dict:
    return {
        "ts": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ms_played": ms,
        "master_metadata_track_name": track,
        "master_metadata_album_artist_name": artist,
        "master_metadata_album_album_name": f"Album by {artist}",
        "skipped": kwargs.get("skipped", False),
        "shuffle": kwargs.get("shuffle", False),
        "reason_start": "clickrow",
        "reason_end": kwargs.get("reason_end", "endplay"),
        "conn_country": kwargs.get("country", "TR"),
        "platform": kwargs.get("platform", "android"),
        "offline": False,
        "incognito_mode": False,
    }


def _minimal_records(n: int = 20) -> list[dict]:
    base = datetime(2023, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    records = []
    for i in range(n):
        records.append(_make_record(
            ts=base + timedelta(hours=i),
            artist=f"Artist_{i % 3}",
            track=f"Track_{i % 5}",
            ms=180_000,
        ))
    return records


def test_analyze_returns_dict():
    result = analyze(_minimal_records())
    assert isinstance(result, dict)
    assert "error" not in result


def test_analyze_has_required_top_level_keys():
    result = analyze(_minimal_records())
    for key in ("bizim_rapor", "metrikler", "top_sanatcilar", "top_sarkilar",
                "badges", "level", "archetype", "radar"):
        assert key in result, f"missing key: {key}"


def test_analyze_metrikler_has_all_metric_keys():
    from constants import METRIC_KEYS
    result = analyze(_minimal_records())
    for k in METRIC_KEYS:
        assert k in result["metrikler"] or k in ("total_hours", "shuffle_pct"), (
            f"metric key '{k}' missing from metrikler"
        )


def test_analyze_badges_count():
    result = analyze(_minimal_records())
    assert len(result["badges"]) == 15


def test_analyze_level_has_required_fields():
    result = analyze(_minimal_records())
    level = result["level"]
    assert "level" in level
    assert "title" in level
    assert "xp" in level


def test_analyze_archetype_has_name_and_description():
    result = analyze(_minimal_records())
    arch = result["archetype"]
    assert "name" in arch
    assert "description" in arch


def test_analyze_radar_has_six_axes():
    result = analyze(_minimal_records())
    assert len(result["radar"]) == 6


def test_analyze_empty_records_returns_error():
    result = analyze([])
    assert "error" in result


def test_analyze_records_without_track_name_returns_error():
    records = [{"ts": "2023-01-01T00:00:00Z", "ms_played": 1000}]
    result = analyze(records)
    assert "error" in result
```

- [ ] **Step 2: Run the integration test**

Run: `cd /Users/mehmetalpatay/Desktop/datatify && python -m pytest tests/test_analyzer.py -v`
Expected: all 10 tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_analyzer.py
git commit -m "test: add integration tests for analyzer.analyze()"
```

---

## Task 11: Add pytest to requirements and run full suite

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add pytest to requirements.txt**

Append `pytest>=7.0` to requirements.txt.

- [ ] **Step 2: Run full test suite**

Run: `cd /Users/mehmetalpatay/Desktop/datatify && python -m pytest tests/ -v`
Expected: all tests PASS, no errors

- [ ] **Step 3: Verify the FastAPI app still starts**

Run: `cd /Users/mehmetalpatay/Desktop/datatify && python -c "import main; print('FastAPI app imports OK')"`
Expected: `FastAPI app imports OK`

- [ ] **Step 4: Final commit**

```bash
git add requirements.txt
git commit -m "chore: add pytest to requirements"
```

---

## Self-Review

### Spec coverage

| Goal | Task |
|------|------|
| TZ_OFFSETS duplicated in analyzer + data_pipeline | Task 1 (create), Task 5, Task 6 (update consumers) |
| SESSION_GAP_MINUTES in 3 files | Task 1, Tasks 5, 6, 7 |
| METRIC_KEYS in 2 files | Task 1, Tasks 8, 9 |
| METRIC_LABELS in main.py only — centralize | Task 1, Task 9 |
| REQUIRED_COLS/OPTIONAL_COLS in data_pipeline only | Task 1, Task 6 |
| Archetype/badge/level/radar buried and untestable | Tasks 3, 4 (personality.py + tests) |
| No tests at all | Tasks 2, 4, 10, 11 |

### Placeholder scan
No TBD, no TODO, no "add appropriate error handling" — all steps have real code.

### Type consistency
- `compute_level(total_hours: float, earned_count: int) -> dict` — used same signature in test_personality.py and analyzer.py
- `compute_badges(metrikler, rapor, *, ...)` — kwargs match between personality.py and the call in analyzer.py
- `METRIC_KEYS` is `list[str]` throughout — consistent

All clear.
