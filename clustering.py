# -*- coding: utf-8 -*-
"""
User Clustering (Methods §E of proposal).

K-Means clustering of anonymous users based on their 15-dimensional
behavioral metric vectors. Optimal k is selected via:

    * Elbow method   (within-cluster sum of squares vs k)
    * Silhouette score (cohesion + separation)

Cluster centroids are inspected post-hoc and assigned data-driven labels
such as "Night Explorers", "Loyal Repeaters", "Impatient Skippers".
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

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


def _vectors_from_rows(rows: list[dict]) -> np.ndarray:
    return np.array(
        [[float(r.get(k, 0) or 0) for k in METRIC_KEYS] for r in rows],
        dtype=np.float64,
    )


def find_optimal_k(
    X_scaled: np.ndarray, k_min: int = 2, k_max: int = 8
) -> dict:
    """Elbow + silhouette evaluation.

    Returns best_k chosen by maximum silhouette score, plus per-k diagnostics
    so the dashboard can render an elbow chart.
    """
    n_samples = X_scaled.shape[0]
    k_max = min(k_max, max(2, n_samples - 1))
    if n_samples < 4 or k_max < 2:
        return {
            "best_k": min(2, max(1, n_samples)),
            "diagnostics": [],
            "reason": "insufficient_samples",
        }

    diagnostics: list[dict] = []
    best_k, best_score = 2, -1.0
    for k in range(max(2, k_min), k_max + 1):
        km = KMeans(
            n_clusters=k, n_init=10, init="k-means++", random_state=42
        )
        labels = km.fit_predict(X_scaled)
        if len(set(labels)) < 2:
            continue
        sil = float(silhouette_score(X_scaled, labels))
        diagnostics.append(
            {
                "k": k,
                "inertia": float(km.inertia_),
                "silhouette": round(sil, 4),
            }
        )
        if sil > best_score:
            best_score = sil
            best_k = k
    return {
        "best_k": best_k,
        "best_silhouette": round(best_score, 4),
        "diagnostics": diagnostics,
    }


def label_cluster(centroid: dict[str, float]) -> str:
    """Heuristic naming for a cluster based on its centroid profile."""
    night = centroid.get("night_listening_ratio_pct", 0)
    impatience = centroid.get("impatience_score_pct", 0)
    exploration = centroid.get("exploration_score", 0)
    loyalty = centroid.get("artist_loyalty_score_pct", 0)
    focus = centroid.get("focus_session_score_pct", 0)
    novelty = centroid.get("music_novelty_rate_pct", 0)
    intensity = centroid.get("listening_intensity_h_per_day", 0)

    if night > 25 and exploration > 10:
        return "Night Explorers"
    if impatience > 35 and exploration > 12:
        return "Impatient Skippers"
    if loyalty > 18 and novelty < 7:
        return "Loyal Repeaters"
    if focus > 12 and impatience < 25:
        return "Deep Listeners"
    if intensity > 3 and exploration > 8:
        return "Heavy Discoverers"
    if night > 20:
        return "Late Night Listeners"
    if loyalty > 25:
        return "Devoted Fans"
    return "Balanced Listeners"


def cluster_users(
    rows: list[dict],
    *,
    user_vector: dict | None = None,
    k_min: int = 2,
    k_max: int = 8,
) -> dict:
    """Cluster all submitted users; optionally place a single new user.

    Args:
        rows:        list of dicts (one per submission), each containing the
                     keys listed in ``METRIC_KEYS``.
        user_vector: optional dict for the current user (so we can return
                     which cluster they belong to without re-fitting).
    """
    if len(rows) < 4:
        return {
            "status": "insufficient_data",
            "min_samples_required": 4,
            "current_samples": len(rows),
            "user_cluster": None,
        }

    X = _vectors_from_rows(rows)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k_info = find_optimal_k(X_scaled, k_min=k_min, k_max=k_max)
    best_k = k_info["best_k"]

    km = KMeans(
        n_clusters=best_k, n_init=20, init="k-means++", random_state=42
    )
    labels = km.fit_predict(X_scaled)

    centroids_scaled = km.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)

    cluster_summaries: list[dict] = []
    for cid in range(best_k):
        members_mask = labels == cid
        size = int(members_mask.sum())
        centroid_dict = {
            METRIC_KEYS[i]: round(float(centroids_original[cid, i]), 3)
            for i in range(len(METRIC_KEYS))
        }
        cluster_summaries.append(
            {
                "id": cid,
                "size": size,
                "label": label_cluster(centroid_dict),
                "centroid": centroid_dict,
            }
        )

    user_cluster: dict | None = None
    if user_vector is not None:
        u = np.array(
            [[float(user_vector.get(k, 0) or 0) for k in METRIC_KEYS]],
            dtype=np.float64,
        )
        u_scaled = scaler.transform(u)
        cid = int(km.predict(u_scaled)[0])
        info = next(c for c in cluster_summaries if c["id"] == cid)
        user_cluster = {
            "cluster_id": cid,
            "label": info["label"],
            "size": info["size"],
            "share_pct": round(100 * info["size"] / len(rows), 1),
        }

    return {
        "status": "ok",
        "n_users": len(rows),
        "k": best_k,
        "silhouette": k_info.get("best_silhouette", 0.0),
        "diagnostics": k_info.get("diagnostics", []),
        "clusters": cluster_summaries,
        "user_cluster": user_cluster,
    }
