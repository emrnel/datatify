"""Integration test: analyzer.analyze() on minimal synthetic records."""
from datetime import datetime, timezone, timedelta

from analyzer import analyze


def _make_record(ts, artist, track, ms, **kwargs):
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


def _minimal_records(n=20):
    base = datetime(2023, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    return [
        _make_record(base + timedelta(hours=i), f"Artist_{i % 3}", f"Track_{i % 5}", 180_000)
        for i in range(n)
    ]


def test_analyze_returns_dict():
    assert isinstance(analyze(_minimal_records()), dict)


def test_analyze_no_error_key():
    assert "error" not in analyze(_minimal_records())


def test_analyze_has_required_top_level_keys():
    r = analyze(_minimal_records())
    for key in ("bizim_rapor", "metrikler", "top_sanatcilar", "top_sarkilar",
                "badges", "level", "archetype", "radar"):
        assert key in r, f"missing key: {key}"


def test_analyze_badges_count():
    assert len(analyze(_minimal_records())["badges"]) == 15


def test_analyze_level_has_required_fields():
    level = analyze(_minimal_records())["level"]
    assert {"level", "title", "xp", "next_threshold_hours"} <= set(level.keys())


def test_analyze_archetype_has_name_and_description():
    arch = analyze(_minimal_records())["archetype"]
    assert "name" in arch and "description" in arch


def test_analyze_radar_has_six_axes():
    assert len(analyze(_minimal_records())["radar"]) == 6


def test_analyze_empty_records_returns_error():
    assert "error" in analyze([])


def test_analyze_records_without_track_name_returns_error():
    assert "error" in analyze([{"ts": "2023-01-01T00:00:00Z", "ms_played": 1000}])
