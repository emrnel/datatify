# -*- coding: utf-8 -*-
"""Unit tests for app.clustering."""
import pytest
from app.clustering import label_cluster, cluster_users, _validate_rows
from app.constants import METRIC_KEYS


def _row(**kwargs) -> dict:
    base = {k: 0.0 for k in METRIC_KEYS}
    base.update(kwargs)
    return base


# ── _validate_rows ────────────────────────────────────────────────────────────

def test_validate_rows_passes_complete_row():
    _validate_rows([_row()])  # must not raise


def test_validate_rows_empty_list_passes():
    _validate_rows([])  # must not raise


def test_validate_rows_raises_on_missing_key():
    bad = {k: 0.0 for k in METRIC_KEYS if k != "total_hours"}
    with pytest.raises(ValueError, match="total_hours"):
        _validate_rows([bad])


# ── label_cluster ─────────────────────────────────────────────────────────────

def test_label_night_explorers():
    assert label_cluster(_row(night_listening_ratio_pct=30, exploration_score=15)) == "Night Explorers"


def test_label_impatient_skippers():
    assert label_cluster(_row(impatience_score_pct=40, exploration_score=13)) == "Impatient Skippers"


def test_label_loyal_repeaters():
    assert label_cluster(_row(artist_loyalty_score_pct=20, music_novelty_rate_pct=5)) == "Loyal Repeaters"


def test_label_deep_listeners():
    assert label_cluster(_row(focus_session_score_pct=15, impatience_score_pct=10)) == "Deep Listeners"


def test_label_default():
    assert label_cluster(_row()) == "Balanced Listeners"


# ── cluster_users ─────────────────────────────────────────────────────────────

def test_cluster_users_insufficient_data():
    result = cluster_users([_row() for _ in range(3)])
    assert result["status"] == "insufficient_data"
    assert result["current_samples"] == 3


def test_cluster_users_ok():
    rows = [_row(impatience_score_pct=float(i * 5), exploration_score=float(i * 3), total_hours=float(i))
            for i in range(20)]
    result = cluster_users(rows)
    assert result["status"] == "ok"
    assert result["n_users"] == 20
    assert len(result["clusters"]) >= 2


def test_cluster_users_with_user_vector():
    rows = [_row(impatience_score_pct=float(i * 5)) for i in range(20)]
    result = cluster_users(rows, user_vector=_row(impatience_score_pct=25.0))
    assert result["user_cluster"] is not None
    assert "label" in result["user_cluster"]
    assert "cluster_id" in result["user_cluster"]


def test_cluster_users_raises_on_bad_rows():
    bad = [{k: 0.0 for k in METRIC_KEYS if k != "shuffle_pct"} for _ in range(10)]
    with pytest.raises(ValueError, match="shuffle_pct"):
        cluster_users(bad)
