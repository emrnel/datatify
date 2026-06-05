# -*- coding: utf-8 -*-
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

# Each row: (predicate(night, imp, expl, focus, loyalty, entropy, novelty), name, description)
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
    imp     = metrikler["impatience_score_pct"]
    expl    = metrikler["exploration_score"]
    loyalty = metrikler["artist_loyalty_score_pct"]
    focus   = metrikler["focus_session_score_pct"]
    entropy = metrikler["artist_diversity_entropy"]
    night   = metrikler["night_listening_ratio_pct"]
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
