# -*- coding: utf-8 -*-
"""
Listening Transition Graph Analysis (Methods §D of proposal).

Builds a directed graph of artist transitions from a user's streaming history
and applies three classical algorithms:

    * PageRank                — central artists in the listening flow
    * Label Propagation       — co-listened artist communities
    * Connected Components    — isolated listening "islands"

Nodes  : artists (vertices)
Edges  : directed A -> B when track of artist B is played immediately
         after a track of artist A *within the same session* (30-min gap).
Weight : number of such consecutive occurrences.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Iterable

import networkx as nx

SESSION_GAP_MINUTES = 30


def _parse_ts(record: dict) -> datetime | None:
    try:
        return datetime.fromisoformat(record["ts"].replace("Z", "+00:00"))
    except Exception:
        return None


def build_artist_transition_graph(records: Iterable[dict]) -> nx.DiGraph:
    """Build a weighted directed graph of artist→artist transitions.

    Two consecutive plays form an edge only if the gap between them is
    less than ``SESSION_GAP_MINUTES`` (i.e. they belong to the same session).
    Self-loops (same artist played twice) are intentionally kept to detect
    repetitive single-artist sessions.
    """
    parsed: list[tuple[datetime, str]] = []
    for r in records:
        artist = r.get("master_metadata_album_artist_name")
        ts = _parse_ts(r)
        if artist and ts:
            parsed.append((ts, artist))

    parsed.sort(key=lambda x: x[0])

    edge_weights: dict[tuple[str, str], int] = defaultdict(int)
    gap = timedelta(minutes=SESSION_GAP_MINUTES)
    for i in range(len(parsed) - 1):
        t1, a1 = parsed[i]
        t2, a2 = parsed[i + 1]
        if t2 - t1 <= gap:
            edge_weights[(a1, a2)] += 1

    G = nx.DiGraph()
    for (src, dst), w in edge_weights.items():
        G.add_edge(src, dst, weight=w)
    return G


def compute_pagerank(G: nx.DiGraph, top_k: int = 20) -> list[dict]:
    """PageRank over the weighted transition graph."""
    if G.number_of_nodes() == 0:
        return []
    pr = nx.pagerank(G, alpha=0.85, weight="weight", max_iter=200, tol=1e-7)
    ranked = sorted(pr.items(), key=lambda x: -x[1])[:top_k]
    return [
        {"rank": i + 1, "artist": a, "score": round(s, 6)}
        for i, (a, s) in enumerate(ranked)
    ]


def detect_communities(G: nx.DiGraph, top_k: int = 10) -> list[dict]:
    """Label-Propagation community detection on the undirected projection."""
    if G.number_of_nodes() == 0:
        return []
    UG = G.to_undirected()
    if UG.number_of_edges() == 0:
        return []
    try:
        communities_iter = nx.algorithms.community.label_propagation_communities(UG)
        communities = [sorted(c) for c in communities_iter]
    except Exception:
        return []

    weighted_degree = dict(UG.degree(weight="weight"))
    enriched = []
    for idx, members in enumerate(communities):
        score = sum(weighted_degree.get(m, 0) for m in members)
        top_members = sorted(
            members, key=lambda m: -weighted_degree.get(m, 0)
        )[:8]
        enriched.append(
            {
                "id": idx,
                "size": len(members),
                "weight": int(score),
                "top_artists": top_members,
            }
        )
    enriched.sort(key=lambda c: -c["weight"])
    return enriched[:top_k]


def connected_components_summary(G: nx.DiGraph, top_k: int = 5) -> dict:
    """Weakly connected components summary."""
    if G.number_of_nodes() == 0:
        return {"count": 0, "largest_size": 0, "components": []}
    comps = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
    return {
        "count": len(comps),
        "largest_size": len(comps[0]) if comps else 0,
        "components": [
            {"size": len(c), "sample_artists": sorted(c)[:5]}
            for c in comps[:top_k]
        ],
    }


def graph_summary(G: nx.DiGraph) -> dict:
    """Lightweight graph stats for the dashboard."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    if n == 0 or m == 0:
        return {
            "nodes": n,
            "edges": m,
            "density": 0.0,
            "avg_in_degree": 0.0,
            "avg_out_degree": 0.0,
            "self_loops": 0,
            "reciprocity": 0.0,
        }
    return {
        "nodes": n,
        "edges": m,
        "density": round(nx.density(G), 6),
        "avg_in_degree": round(sum(d for _, d in G.in_degree()) / n, 3),
        "avg_out_degree": round(sum(d for _, d in G.out_degree()) / n, 3),
        "self_loops": nx.number_of_selfloops(G),
        "reciprocity": round(nx.reciprocity(G) or 0.0, 3),
    }


def analyze_listening_graph(
    records: list[dict],
    *,
    pagerank_top_k: int = 20,
    community_top_k: int = 10,
    viz_top_n_nodes: int = 60,
) -> dict:
    """Full pipeline used by the FastAPI endpoint and dashboard."""
    G = build_artist_transition_graph(records)

    pagerank = compute_pagerank(G, top_k=pagerank_top_k)
    communities = detect_communities(G, top_k=community_top_k)
    components = connected_components_summary(G, top_k=5)
    summary = graph_summary(G)

    viz = _build_visualization_subgraph(
        G, pagerank, communities, top_n=viz_top_n_nodes
    )

    return {
        "summary": summary,
        "pagerank": pagerank,
        "communities": communities,
        "components": components,
        "visualization": viz,
    }


def _build_visualization_subgraph(
    G: nx.DiGraph,
    pagerank: list[dict],
    communities: list[dict],
    top_n: int = 60,
) -> dict:
    """Compact node/edge payload that the frontend renders as a force graph."""
    if not pagerank:
        return {"nodes": [], "edges": []}

    keep = {p["artist"] for p in pagerank[:top_n]}

    artist_to_community: dict[str, int] = {}
    for c in communities:
        for a in c["top_artists"]:
            artist_to_community.setdefault(a, c["id"])

    pr_lookup = {p["artist"]: p["score"] for p in pagerank}

    nodes = [
        {
            "id": a,
            "score": pr_lookup.get(a, 0.0),
            "community": artist_to_community.get(a, -1),
        }
        for a in keep
    ]

    edges = []
    for u, v, data in G.edges(data=True):
        if u in keep and v in keep and u != v:
            edges.append(
                {"source": u, "target": v, "weight": int(data.get("weight", 1))}
            )

    edges.sort(key=lambda e: -e["weight"])
    edges = edges[: max(80, top_n * 2)]

    return {"nodes": nodes, "edges": edges}
