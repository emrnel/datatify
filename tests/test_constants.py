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
