# -*- coding: utf-8 -*-
"""Unit tests for app.db — benchmark DB and metric-vector bridge."""
import pytest
import app.db as db_mod
from app.constants import METRIC_KEYS

_METRIKLER_KEYS = [k for k in METRIC_KEYS if k not in ("total_hours", "shuffle_pct")]


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "test.db"))
    db_mod.init_db()


def _make_analysis(toplam_saat: float = 10.0, shuffle_orani: float = 25.0) -> dict:
    return {
        "metrikler": {k: 0.0 for k in _METRIKLER_KEYS},
        "bizim_rapor": {"toplam_saat": toplam_saat, "shuffle_orani_pct": shuffle_orani},
    }


def test_extract_metric_vector_has_all_keys():
    vec = db_mod.extract_metric_vector(_make_analysis())
    assert set(vec.keys()) == set(METRIC_KEYS)


def test_extract_metric_vector_pulls_from_bizim_rapor():
    vec = db_mod.extract_metric_vector(_make_analysis(toplam_saat=42.0, shuffle_orani=75.0))
    assert vec["total_hours"] == 42.0
    assert vec["shuffle_pct"] == 75.0


def test_stats_empty_db():
    result = db_mod.stats()
    assert result["total_users"] == 0
    assert result["averages"] == {}


def test_submit_increments_count():
    db_mod.submit({k: 1.0 for k in METRIC_KEYS})
    assert db_mod.stats()["total_users"] == 1


def test_all_users_empty_then_one():
    assert db_mod.all_users() == []
    db_mod.submit({k: 5.0 for k in METRIC_KEYS})
    rows = db_mod.all_users()
    assert len(rows) == 1
    assert set(rows[0].keys()) == set(METRIC_KEYS)


def test_percentiles_single_user_defaults_to_50():
    values = {k: 10.0 for k in METRIC_KEYS}
    db_mod.submit(values)
    result = db_mod.compute_percentiles(values)
    assert result["total_users"] == 1
    for pct in result["percentiles"].values():
        assert pct == 50.0


def test_percentiles_two_users():
    low = {k: 10.0 for k in METRIC_KEYS}
    high = {k: 90.0 for k in METRIC_KEYS}
    db_mod.submit(low)
    db_mod.submit(high)
    result_high = db_mod.compute_percentiles(high)
    # high user: 1 of 2 rows below → 50 %
    assert result_high["percentiles"]["impatience_score_pct"] == 50.0
    result_low = db_mod.compute_percentiles(low)
    # low user: 0 of 2 rows below → 0 %
    assert result_low["percentiles"]["impatience_score_pct"] == 0.0
