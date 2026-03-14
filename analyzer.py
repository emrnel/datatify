# -*- coding: utf-8 -*-
"""Spotify Extended Streaming History — core analysis engine."""
import math
from collections import defaultdict
from datetime import datetime, timedelta

SESSION_GAP_MINUTES = 30

TZ_OFFSETS = {
    "TR": 3, "DE": 1, "FR": 1, "NL": 1, "GB": 0, "US": -5,
    "CA": -5, "AT": 1, "CZ": 1, "IT": 1, "ES": 1, "SE": 1,
    "NO": 1, "DK": 1, "FI": 2, "PL": 1, "BE": 1, "CH": 1,
    "PT": 0, "GR": 2, "RO": 2, "BG": 2, "JP": 9, "KR": 9,
    "AU": 10, "NZ": 12, "BR": -3, "MX": -6, "AR": -3, "IN": 5,
}


def _entropy(count_dict: dict) -> float:
    total = sum(count_dict.values())
    if total <= 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in count_dict.values() if c > 0)


def _parse_ts(r: dict):
    try:
        return datetime.fromisoformat(r["ts"].replace("Z", "+00:00"))
    except Exception:
        return None


def _compute_sessions(tracks: list[dict]):
    sorted_tracks = sorted(
        [r for r in tracks if _parse_ts(r) is not None],
        key=lambda r: r["ts"],
    )
    if not sorted_tracks:
        return [], 0, 0.0
    sessions = []
    current_end = None
    current_ms = 0
    gap_ms = SESSION_GAP_MINUTES * 60 * 1000
    for r in sorted_tracks:
        ts = _parse_ts(r)
        ms = r.get("ms_played") or 0
        start_ts = ts - timedelta(milliseconds=ms)
        if current_end is None:
            current_end = ts
            current_ms = ms
        else:
            gap = (start_ts - current_end).total_seconds() * 1000
            if gap > gap_ms:
                sessions.append(current_ms)
                current_end = ts
                current_ms = ms
            else:
                current_end = ts
                current_ms += ms
    if current_ms:
        sessions.append(current_ms)
    total_ms = sum(sessions)
    avg_h = (total_ms / len(sessions)) / (1000 * 3600) if sessions else 0
    return sessions, len(sessions), avg_h


def analyze(records: list[dict]) -> dict:
    """Run full analysis on a list of streaming history records.
    Returns a metrics dict ready for JSON serialisation / dashboard rendering.
    """
    tracks = [x for x in records if x.get("master_metadata_track_name")]

    total_ms = sum(x.get("ms_played", 0) or 0 for x in tracks)
    total_hours = total_ms / (1000 * 3600)
    total_days = total_hours / 24

    artists = defaultdict(lambda: {"ms": 0, "count": 0, "skipped": 0})
    songs = defaultdict(lambda: {"ms": 0, "count": 0, "artist": None})
    albums = defaultdict(lambda: {"ms": 0, "count": 0})
    by_year = defaultdict(lambda: {"ms": 0, "count": 0})
    by_month = defaultdict(lambda: {"ms": 0, "count": 0})
    by_hour = defaultdict(lambda: {"ms": 0, "count": 0})
    by_weekday = defaultdict(lambda: {"ms": 0, "count": 0})
    by_platform = defaultdict(lambda: {"ms": 0, "count": 0})
    by_country = defaultdict(lambda: {"ms": 0, "count": 0})
    reason_start_counts = defaultdict(int)
    reason_end_counts = defaultdict(int)

    skipped_count = 0
    completed_count = 0
    shuffle_count = 0
    offline_ms = 0
    incognito_count = 0
    first_listen: dict[str, datetime] = {}
    first_listen_track: dict[tuple, datetime] = {}
    early_skip_count = 0
    skipped_ms_sum = 0
    focus_play_count = 0
    album_play_count = 0

    for r in tracks:
        ms = r.get("ms_played") or 0
        artist = r.get("master_metadata_album_artist_name") or "Bilinmeyen"
        track = r.get("master_metadata_track_name") or "Bilinmeyen"
        album = r.get("master_metadata_album_album_name") or "Bilinmeyen"

        artists[artist]["ms"] += ms
        artists[artist]["count"] += 1
        if r.get("skipped"):
            artists[artist]["skipped"] += 1

        key = (track, artist)
        songs[key]["ms"] += ms
        songs[key]["count"] += 1
        songs[key]["artist"] = artist

        albums[(album, artist)]["ms"] += ms
        albums[(album, artist)]["count"] += 1

        ts = _parse_ts(r)
        if ts:
            by_year[ts.year]["ms"] += ms
            by_year[ts.year]["count"] += 1
            by_month[(ts.year, ts.month)]["ms"] += ms
            by_month[(ts.year, ts.month)]["count"] += 1
            by_hour[ts.hour]["ms"] += ms
            by_hour[ts.hour]["count"] += 1
            by_weekday[ts.weekday()]["ms"] += ms
            by_weekday[ts.weekday()]["count"] += 1
            if artist not in first_listen or ts < first_listen[artist]:
                first_listen[artist] = ts
            if key not in first_listen_track or ts < first_listen_track[key]:
                first_listen_track[key] = ts

        if r.get("skipped"):
            skipped_ms_sum += ms
            if ms < 30_000:
                early_skip_count += 1
        elif ms >= 4 * 60 * 1000:
            focus_play_count += 1
        if album and album.strip() and album != "Bilinmeyen":
            album_play_count += 1

        plat_raw = (r.get("platform") or "unknown").strip()
        plat = plat_raw.split("(")[0].strip().lower()
        if any(k in plat for k in ("android", "ios", "iphone")):
            plat_cat = "Mobil"
        elif any(k in plat for k in ("web", "chrome", "firefox")):
            plat_cat = "Web"
        elif any(k in plat for k in ("desktop", "windows", "mac")):
            plat_cat = "Masaüstü"
        else:
            plat_cat = plat[:20] if len(plat) > 20 else (plat or "Diğer")
        by_platform[plat_cat]["ms"] += ms
        by_platform[plat_cat]["count"] += 1

        country = r.get("conn_country") or "?"
        by_country[country]["ms"] += ms
        by_country[country]["count"] += 1

        reason_start_counts[r.get("reason_start") or "unknown"] += 1
        reason_end_counts[r.get("reason_end") or "unknown"] += 1

        if r.get("shuffle"):
            shuffle_count += 1
        if r.get("offline"):
            offline_ms += ms
        if r.get("incognito_mode"):
            incognito_count += 1

        if r.get("skipped"):
            skipped_count += 1
        if (r.get("reason_end") or "").lower() == "endplay":
            completed_count += 1

    total_plays = len(tracks)
    if total_plays == 0:
        return {"error": "Hiç müzik kaydı bulunamadı."}

    top_artists = sorted(artists.items(), key=lambda x: -x[1]["ms"])[:20]
    top_songs = sorted(songs.items(), key=lambda x: -x[1]["ms"])[:20]

    # Sessions
    session_list, num_sessions, avg_session_h = _compute_sessions(tracks)

    # Calendar span
    sorted_by_ts = sorted([r for r in tracks if _parse_ts(r)], key=lambda r: r["ts"])
    calendar_first_ts = _parse_ts(sorted_by_ts[0]) if sorted_by_ts else None
    calendar_last_ts = _parse_ts(sorted_by_ts[-1]) if sorted_by_ts else None
    calendar_days = (calendar_last_ts - calendar_first_ts).days if (calendar_first_ts and calendar_last_ts) else 1

    # Monthly novelty
    new_tracks_by_month: dict[tuple, int] = defaultdict(int)
    for (_track, _artist), t in first_listen_track.items():
        new_tracks_by_month[(t.year, t.month)] += 1
    novelty_by_month = {}
    for (y, m_), play_info in by_month.items():
        new_in = new_tracks_by_month.get((y, m_), 0)
        novelty_by_month[f"{y}-{m_:02d}"] = round(100 * new_in / play_info["count"], 1) if play_info["count"] else 0
    avg_novelty = sum(novelty_by_month.values()) / len(novelty_by_month) if novelty_by_month else 0

    # Habit loop
    pair_counts: dict[tuple, int] = defaultdict(int)
    for i in range(len(sorted_by_ts) - 1):
        r1, r2 = sorted_by_ts[i], sorted_by_ts[i + 1]
        k1 = (r1.get("master_metadata_track_name"), r1.get("master_metadata_album_artist_name"))
        k2 = (r2.get("master_metadata_track_name"), r2.get("master_metadata_album_artist_name"))
        if k1[0] and k2[0]:
            pair_counts[(k1, k2)] += 1
    total_pairs = len(sorted_by_ts) - 1
    repeated_pairs = sum(1 for c in pair_counts.values() if c > 1)
    habit_loop_score = round(100 * repeated_pairs / total_pairs, 1) if total_pairs else 0

    # Yearly growth
    years_sorted = sorted(by_year.keys())
    yearly_hours = [by_year[y]["ms"] / (1000 * 3600) for y in years_sorted]
    yearly_growth: dict[str, float] = {}
    for i in range(1, len(years_sorted)):
        prev_h = yearly_hours[i - 1]
        curr_h = yearly_hours[i]
        if prev_h > 0:
            yearly_growth[str(years_sorted[i])] = round(100 * (curr_h - prev_h) / prev_h, 1)

    # Derived values
    unique_tracks = len(songs)
    unique_artists = len(artists)
    artist_counts = {a: d["count"] for a, d in artists.items()}
    top10_artist_plays = sum(d["count"] for _, d in top_artists)
    top1_song_plays = top_songs[0][1]["count"] if top_songs else 0
    avg_listen_sec = (total_ms / 1000) / total_plays
    avg_track_duration_sec = 210
    avg_listening_ratio = min(1.0, avg_listen_sec / avg_track_duration_sec)
    skip_latency_ratio = (
        (skipped_ms_sum / 1000) / (avg_track_duration_sec * skipped_count)
        if skipped_count and avg_track_duration_sec else 0
    )

    # Timezone detection
    dominant_country = max(by_country.items(), key=lambda x: x[1]["ms"])[0] if by_country else "TR"
    tz_offset = TZ_OFFSETS.get(dominant_country, 3)
    night_utc_hours = [(h + 24 - tz_offset) % 24 for h in range(6)]
    night_ms = sum(by_hour[h]["ms"] for h in night_utc_hours)
    mobile_ms = by_platform.get("Mobil", {}).get("ms", 0)

    circadian_local = {}
    for utc_h in range(24):
        local_h = (utc_h + tz_offset) % 24
        circadian_local[local_h] = round(by_hour[utc_h]["ms"] / (1000 * 3600), 1)
    circadian_local_sorted = {f"{h:02d}": circadian_local[h] for h in range(24)}

    # Date range for display
    first_year = calendar_first_ts.year if calendar_first_ts else "?"
    last_year = calendar_last_ts.year if calendar_last_ts else "?"

    metrics: dict = {
        "bizim_rapor": {
            "toplam_kayit": total_plays,
            "toplam_saat": round(total_hours, 2),
            "toplam_gun_esdeger": round(total_days, 1),
            "tam_dinlenen": completed_count,
            "atlanan": skipped_count,
            "tam_dinlenme_orani_pct": round(100 * completed_count / total_plays, 1),
            "oturum_sayisi": num_sessions,
            "ortalama_oturum_saat": round(avg_session_h, 2),
            "shuffle_orani_pct": round(100 * shuffle_count / total_plays, 1),
            "cevrimdisi_saat": round(offline_ms / (1000 * 3600), 1),
            "cevrimdisi_orani_pct": round(100 * offline_ms / total_ms, 1) if total_ms else 0,
            "gizli_oturum_kayit": incognito_count,
            "takvim_gun_sayisi": calendar_days,
            "ilk_yil": first_year,
            "son_yil": last_year,
        },
        "metrikler": {
            "impatience_score_pct": round(100 * skipped_count / total_plays, 1),
            "completion_rate_pct": round(100 * completed_count / total_plays, 1),
            "avg_listen_sec": round(avg_listen_sec, 1),
            "avg_listening_ratio": round(avg_listening_ratio, 3),
            "exploration_score": round(100 * unique_tracks / total_plays, 1),
            "artist_diversity_entropy": round(_entropy(artist_counts), 2),
            "skip_latency_ratio": round(skip_latency_ratio, 3),
            "early_skip_rate_pct": round(100 * early_skip_count / skipped_count, 1) if skipped_count else 0,
            "listening_intensity_h_per_day": round(total_hours / calendar_days, 2) if calendar_days else 0,
            "yearly_growth_pct": yearly_growth,
            "circadian_hours": circadian_local_sorted,
            "timezone": f"UTC+{tz_offset}" if tz_offset >= 0 else f"UTC{tz_offset}",
            "night_listening_ratio_pct": round(100 * night_ms / total_ms, 1) if total_ms else 0,
            "mobile_usage_ratio_pct": round(100 * mobile_ms / total_ms, 1) if total_ms else 0,
            "album_listening_ratio_pct": round(100 * album_play_count / total_plays, 1),
            "focus_session_score_pct": round(100 * focus_play_count / total_plays, 1),
            "music_novelty_rate_pct": round(avg_novelty, 1),
            "artist_loyalty_score_pct": round(100 * top10_artist_plays / total_plays, 1),
            "song_replay_density_pct": round(100 * top1_song_plays / total_plays, 1),
            "listening_fragmentation_index": round(skipped_count / num_sessions, 1) if num_sessions else 0,
            "habit_loop_score_pct": habit_loop_score,
        },
        "top_sanatcilar": [
            {
                "sira": i,
                "sanatci": a,
                "saat": round(d["ms"] / (1000 * 3600), 1),
                "dinleme": d["count"],
                "skip_pct": round(100 * d["skipped"] / d["count"], 1) if d["count"] else 0,
            }
            for i, (a, d) in enumerate(top_artists, 1)
        ],
        "top_sarkilar": [
            {
                "sira": i,
                "sarki": t,
                "sanatci": a,
                "saat": round(d["ms"] / (1000 * 3600), 1),
            }
            for i, ((t, a), d) in enumerate(top_songs, 1)
        ],
        "yillara_gore": {
            str(y): {"kayit": by_year[y]["count"], "saat": round(by_year[y]["ms"] / (1000 * 3600), 1)}
            for y in years_sorted
        },
        "haftanin_gunu": {
            ["Pzt", "Sal", "Çar", "Per", "Cum", "Cmt", "Paz"][w]: round(by_weekday[w]["ms"] / (1000 * 3600), 1)
            for w in range(7)
        },
    }

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

    # ── Radar ──
    metrics["radar"] = {
        "Sabır": round(100 - imp, 1),
        "Keşif": round(min(expl * 5, 100), 1),
        "Sadakat": round(min(loyalty * 5, 100), 1),
        "Odak": round(min(focus * 5, 100), 1),
        "Çeşitlilik": round(min(entropy_val * 8, 100), 1),
        "Gece Kuşu": round(min(night * 4, 100), 1),
    }

    return metrics
