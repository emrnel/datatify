from app.personality import compute_level, compute_archetype, compute_radar, compute_badges


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _metrikler(**overrides):
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
    base = {"shuffle_orani_pct": 50.0, "cevrimdisi_orani_pct": 5.0}
    base.update(overrides)
    return base


def _badge_kwargs(**overrides):
    base = dict(
        max_session_h=2.0,
        top1_song_hours=5.0,
        loyal_artists_5yr=0,
        countries_set={"TR"},
        unique_artists=100,
        incognito_count=10,
        total_hours=100.0,
        total_plays=1000,
    )
    base.update(overrides)
    return base


# ── compute_level ─────────────────────────────────────────────────────────────

def test_level_newbie():
    r = compute_level(10.0, 0)
    assert r["level"] == 1
    assert r["title"] == "Newbie"
    assert r["xp"] == 100


def test_level_casual():
    r = compute_level(50.0, 0)
    assert r["title"] == "Casual"


def test_level_listener():
    r = compute_level(200.0, 2)
    assert r["level"] == 2
    assert r["xp"] == int(200 * 10 + 2 * 50)


def test_level_legendary():
    r = compute_level(3000.0, 10)
    assert r["title"] == "Legendary"


def test_level_transcendent():
    r = compute_level(5001.0, 15)
    assert r["level"] == 10
    assert r["title"] == "Transcendent"


def test_level_xp_scales_with_badges():
    r1 = compute_level(100.0, 0)
    r2 = compute_level(100.0, 5)
    assert r2["xp"] == r1["xp"] + 5 * 50


# ── compute_archetype ─────────────────────────────────────────────────────────

def test_archetype_night_diver():
    m = _metrikler(night_listening_ratio_pct=30.0, focus_session_score_pct=15.0)
    assert compute_archetype(m)["name"] == "The Night Diver"


def test_archetype_restless_explorer():
    m = _metrikler(impatience_score_pct=45.0, exploration_score=15.0)
    assert compute_archetype(m)["name"] == "The Restless Explorer"


def test_archetype_loyal_guardian():
    m = _metrikler(artist_loyalty_score_pct=20.0, music_novelty_rate_pct=5.0)
    assert compute_archetype(m)["name"] == "The Loyal Guardian"


def test_archetype_balanced_default():
    assert compute_archetype(_metrikler())["name"] == "The Balanced Listener"


def test_archetype_has_name_and_description():
    r = compute_archetype(_metrikler())
    assert "name" in r and "description" in r
    assert len(r["description"]) > 0


# ── compute_radar ─────────────────────────────────────────────────────────────

def test_radar_has_six_axes():
    axes = set(compute_radar(_metrikler()).keys())
    assert axes == {"Sabır", "Keşif", "Sadakat", "Odak", "Çeşitlilik", "Gece Kuşu"}


def test_radar_values_bounded():
    m = _metrikler(
        impatience_score_pct=0.0, exploration_score=100.0,
        artist_loyalty_score_pct=100.0, focus_session_score_pct=100.0,
        artist_diversity_entropy=100.0, night_listening_ratio_pct=100.0,
    )
    for k, v in compute_radar(m).items():
        assert 0 <= v <= 100, f"{k}={v} out of range"


def test_radar_sabir_inverted():
    low  = compute_radar(_metrikler(impatience_score_pct=10.0))["Sabır"]
    high = compute_radar(_metrikler(impatience_score_pct=80.0))["Sabır"]
    assert low > high


# ── compute_badges ────────────────────────────────────────────────────────────

def test_badges_returns_15():
    badges, earned = compute_badges(_metrikler(), _rapor(), **_badge_kwargs())
    assert len(badges) == 15


def test_badges_earned_count_matches_list():
    badges, earned = compute_badges(_metrikler(), _rapor(), **_badge_kwargs())
    assert earned == sum(1 for b in badges if b["earned"])


def test_badge_night_owl_earned():
    m = _metrikler(night_listening_ratio_pct=30.0)
    badges, _ = compute_badges(m, _rapor(), **_badge_kwargs())
    assert next(b["earned"] for b in badges if b["id"] == "night_owl") is True


def test_badge_night_owl_not_earned():
    badges, _ = compute_badges(_metrikler(night_listening_ratio_pct=10.0), _rapor(), **_badge_kwargs())
    assert next(b["earned"] for b in badges if b["id"] == "night_owl") is False


def test_badge_marathon_earned():
    badges, _ = compute_badges(_metrikler(), _rapor(), **_badge_kwargs(max_session_h=5.0))
    assert next(b["earned"] for b in badges if b["id"] == "marathon") is True


def test_badge_world_traveler_threshold():
    few  = compute_badges(_metrikler(), _rapor(), **_badge_kwargs(countries_set={"TR", "DE"}))[0]
    many = compute_badges(_metrikler(), _rapor(), **_badge_kwargs(countries_set={"TR", "DE", "US"}))[0]
    assert next(b["earned"] for b in few  if b["id"] == "world_traveler") is False
    assert next(b["earned"] for b in many if b["id"] == "world_traveler") is True


def test_badge_shuffle_addict_and_album_purist():
    b_addict, _ = compute_badges(_metrikler(), _rapor(shuffle_orani_pct=90.0), **_badge_kwargs())
    b_purist, _ = compute_badges(_metrikler(), _rapor(shuffle_orani_pct=10.0), **_badge_kwargs())
    assert next(b["earned"] for b in b_addict if b["id"] == "shuffle_addict") is True
    assert next(b["earned"] for b in b_purist if b["id"] == "album_purist") is True
