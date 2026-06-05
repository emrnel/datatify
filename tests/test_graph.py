# -*- coding: utf-8 -*-
"""Unit tests for app.graph_analysis."""
import networkx as nx
import pytest
from app.graph_analysis import (
    _parse_artist_timeline,
    build_artist_transition_graph,
    graph_summary,
    analyze_listening_graph,
)


def _rec(ts: str, artist: str | None = "Artist A") -> dict:
    return {
        "ts": ts,
        "master_metadata_album_artist_name": artist,
        "master_metadata_track_name": "Track",
        "ms_played": 180000,
    }


# ── _parse_artist_timeline ────────────────────────────────────────────────────

def test_parse_timeline_counts_kept():
    _, diag = _parse_artist_timeline([_rec("2023-01-01T10:00:00Z"), _rec("2023-01-01T10:05:00Z")])
    assert diag == {"kept": 2, "dropped_no_artist": 0, "dropped_bad_ts": 0}


def test_parse_timeline_drops_missing_artist():
    _, diag = _parse_artist_timeline([_rec("2023-01-01T10:00:00Z", artist=None)])
    assert diag["dropped_no_artist"] == 1
    assert diag["kept"] == 0


def test_parse_timeline_drops_bad_ts():
    _, diag = _parse_artist_timeline([{"ts": "not-a-date", "master_metadata_album_artist_name": "A"}])
    assert diag["dropped_bad_ts"] == 1
    assert diag["kept"] == 0


def test_parse_timeline_empty():
    _, diag = _parse_artist_timeline([])
    assert diag == {"kept": 0, "dropped_no_artist": 0, "dropped_bad_ts": 0}


# ── build_artist_transition_graph ─────────────────────────────────────────────

def test_graph_returns_tuple_with_diagnostics():
    G, diag = build_artist_transition_graph([_rec("2023-01-01T10:00:00Z")])
    assert isinstance(G, nx.DiGraph)
    assert "kept" in diag


def test_graph_empty_input():
    G, diag = build_artist_transition_graph([])
    assert G.number_of_nodes() == 0
    assert diag["kept"] == 0


def test_graph_builds_edge_within_session():
    records = [_rec("2023-01-01T10:00:00Z", "Artist A"), _rec("2023-01-01T10:05:00Z", "Artist B")]
    G, _ = build_artist_transition_graph(records)
    assert G.has_edge("Artist A", "Artist B")


def test_graph_no_edge_across_session_gap():
    records = [_rec("2023-01-01T10:00:00Z", "Artist A"), _rec("2023-01-01T11:00:00Z", "Artist B")]
    G, _ = build_artist_transition_graph(records)
    assert not G.has_edge("Artist A", "Artist B")


def test_graph_self_loop_same_artist():
    records = [_rec("2023-01-01T10:00:00Z", "Artist A"), _rec("2023-01-01T10:03:00Z", "Artist A")]
    G, _ = build_artist_transition_graph(records)
    assert G.has_edge("Artist A", "Artist A")


# ── graph_summary ─────────────────────────────────────────────────────────────

def test_graph_summary_empty():
    s = graph_summary(nx.DiGraph())
    assert s["nodes"] == 0
    assert s["edges"] == 0


def test_graph_summary_non_empty():
    G = nx.DiGraph()
    G.add_edge("A", "B", weight=2)
    s = graph_summary(G)
    assert s["nodes"] == 2
    assert s["edges"] == 1


# ── analyze_listening_graph ───────────────────────────────────────────────────

def test_analyze_includes_parse_diagnostics():
    records = [_rec("2023-01-01T10:00:00Z"), _rec("2023-01-01T10:05:00Z")]
    result = analyze_listening_graph(records)
    assert "parse_diagnostics" in result["summary"]
    assert result["summary"]["parse_diagnostics"]["kept"] == 2


def test_analyze_empty_records():
    result = analyze_listening_graph([])
    assert result["summary"]["nodes"] == 0
    assert result["visualization"] == {"nodes": [], "edges": []}
