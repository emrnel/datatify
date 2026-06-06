# Datatify — Project Context

> Load this file at session start to skip codebase exploration.
> Deployment: https://web-production-65055.up.railway.app/

---

## What It Is

FastAPI web app that analyzes a user's **Spotify Extended Streaming History** (JSON export). User uploads one or more JSON files → receives an interactive HTML dashboard with listening stats, personality archetype, badges, level, radar chart, artist graph, clustering, and optional Gemini AI character analysis.

**UI language:** Turkish (display labels, metric names, archetype descriptions). Code and identifiers are English.

---

## Tech Stack

| Layer | Tech |
|-------|------|
| Web framework | FastAPI + Uvicorn |
| Data processing | pandas, numpy, pyarrow |
| Graph analysis | NetworkX |
| Clustering | scikit-learn (K-Means) |
| Distributed pipeline | PySpark (benchmark only) |
| AI analysis | Google Gemini (gemini-2.5-flash, fallback chain) |
| Database | SQLite (`benchmark.db` — stores anonymized metric submissions) |
| Deployment | Railway (Nixpacks builder) |
| Python | 3.12+ |

---

## Directory Layout

```
datatify/
├── app/                    ← main Python package
│   ├── __init__.py
│   ├── main.py             ← FastAPI routes only (thin orchestration, ~185 lines)
│   ├── analyzer.py         ← core analysis engine (pure Python, no Pandas)
│   ├── constants.py        ← single source of truth for all shared constants
│   ├── personality.py      ← pure classification functions (badges, archetype, level, radar)
│   ├── db.py               ← benchmark DB (SQLite), metric-vector bridge, percentile math
│   ├── gemini.py           ← Gemini AI client, RPM throttle, retry/backoff engine, prompt
│   ├── clustering.py       ← K-Means user clustering via scikit-learn
│   ├── data_pipeline.py    ← Pandas reference pipeline (mirrors analyzer.py logic)
│   ├── graph_analysis.py   ← artist transition graph (NetworkX)
│   └── spark_pipeline.py   ← PySpark distributed pipeline (benchmark only)
├── scripts/
│   ├── benchmark.py        ← scalability benchmark (Python vs Pandas vs PySpark)
│   ├── synthetic_data.py   ← synthetic Spotify record generator
│   └── plot_benchmark.py   ← renders benchmark_results/ PNGs
├── templates/
│   ├── index.html          ← landing page (file upload form)
│   └── dashboard.html      ← results dashboard (reads SPOTIFY_DATA_PLACEHOLDER JSON)
├── tests/
│   ├── test_constants.py   ← 8 smoke tests for constants
│   ├── test_personality.py ← 21 unit tests for pure classification functions
│   └── test_analyzer.py    ← 9 integration tests for analyze()
├── benchmark_results/      ← CSV/JSON/PNG output from benchmark runs
├── Procfile                ← web: uvicorn app.main:app --host 0.0.0.0 --port $PORT
├── railway.json            ← Railway deployment config (same start command)
└── requirements.txt
```

---

## Module Responsibilities

### `app/constants.py` — Single Source of Truth
All shared constants. Import from here, never redefine locally.

```python
TZ_OFFSETS: dict[str, int]          # country code → UTC offset hours (30 countries)
SESSION_GAP_MINUTES: int = 30       # gap that breaks a listening session
AVG_TRACK_DURATION_SEC: int = 210   # assumed average track length for ratio metrics
METRIC_KEYS: list[str]              # 15 benchmark metric field names
REQUIRED_COLS: list[str]            # 4 columns a record must have
OPTIONAL_COLS: list[str]            # 9 enrichment columns
METRIC_LABELS: dict[str, tuple]     # (Turkish label, Turkish comparison suffix) per metric
```

### `app/analyzer.py` — Core Analysis Engine
`analyze(records: list[dict]) -> dict`

Takes raw Spotify JSON records, returns the full metrics dict used by the dashboard. Pure Python (no Pandas). Key sub-steps:
- Per-track/artist/album/time aggregations
- Session detection (30-min gap heuristic via `_compute_sessions`)
- Timezone correction (dominant country → UTC offset)
- Monthly novelty, habit loop, yearly growth
- Delegates badges/level/archetype/radar to `personality.py`

Private helpers (testable in isolation):
```python
_categorize_platform(raw: str) -> str                # "android (os=10)" → "Mobil"
_compute_habit_loop(sorted_tracks) -> float          # % consecutive (track,artist) pairs that repeat
_compute_sessions(tracks) -> (list, int, float)
_aggregate_records(tracks) -> dict                   # single pass → all per-play accumulators
```

`_aggregate_records` owns the full for-loop (artists, songs, by_year, by_month, by_hour, by_weekday, by_platform, by_country, skip/focus/shuffle counters, first_listen maps). `analyze()` calls it and unpacks the returned dict. This makes aggregation logic independently testable without running the full pipeline.

Output shape:
```python
{
  "bizim_rapor":   {...},   # summary stats (total hours, sessions, etc.)
  "metrikler":     {...},   # 15+ computed metrics
  "top_sanatcilar":[...],   # top 20 artists
  "top_sarkilar":  [...],   # top 20 songs
  "yillara_gore":  {...},   # by-year breakdown
  "haftanin_gunu": {...},   # by-weekday hours
  "badges":        [...],   # 15-item list, each {id, name, desc, earned: bool}
  "badges_earned": int,
  "badges_total":  int,
  "level":         {...},   # {level, title, xp, next_threshold_hours}
  "archetype":     {...},   # {name, description}
  "radar":         {...},   # 6 axes: Sabır, Keşif, Sadakat, Odak, Çeşitlilik, Gece Kuşu
}
# main.py adds: "graph", "clustering", "gemini_analysis" (optional)
```

### `app/personality.py` — Pure Classification
Side-effect-free functions, no I/O, fully unit-tested.

```python
compute_level(total_hours, earned_count) -> dict
compute_archetype(metrikler) -> dict       # 7 rules + default "The Balanced Listener"
compute_radar(metrikler) -> dict           # 6-axis 0–100 values
compute_badges(metrikler, rapor, *, ...) -> tuple[list[dict], int]  # 15 badges
```

Level thresholds: Newbie(0) → Casual(50h) → Listener(200h) → Enthusiast(500h) → Devotee(1000h) → Addict(1500h) → Obsessed(2000h) → Maniac(2500h) → Legendary(3000h) → Mythic(4000h) → Transcendent(5000h)

### `app/clustering.py` — K-Means User Clustering
```python
cluster_users(rows, *, user_vector=None, k_min=2, k_max=8) -> dict
find_optimal_k(X_scaled) -> dict    # elbow + silhouette
label_cluster(centroid) -> str      # heuristic names: "Night Explorers", "Loyal Repeaters", etc.
_validate_rows(rows) -> None        # raises ValueError if first row is missing a METRIC_KEY
```
Clusters anonymous benchmark submissions. Returns which cluster the current user falls into.

`_validate_rows` is called at the top of `cluster_users` — a missing METRIC_KEY raises `ValueError` with a clear message at the seam rather than a cryptic numpy error deep in the call stack.

### `app/graph_analysis.py` — Artist Transition Graph
```python
build_artist_transition_graph(records) -> tuple[nx.DiGraph, dict]  # (G, parse_diagnostics)
analyze_listening_graph(records, ...) -> dict   # full pipeline
compute_pagerank(G) -> list[dict]
detect_communities(G) -> list[dict]             # label propagation
connected_components_summary(G) -> dict
_parse_artist_timeline(records) -> tuple[list[tuple[datetime, str]], dict]
```
Edges: artist A → artist B when B is played within 30 min of A. Used for force-directed graph in dashboard.

`_parse_artist_timeline` does the filtering pass and returns `(parsed_pairs, diagnostics)` where `diagnostics = {kept, dropped_no_artist, dropped_bad_ts}`. `build_artist_transition_graph` calls it and returns `(G, diagnostics)`. `analyze_listening_graph` merges `parse_diagnostics` into the `summary` key of its return dict — so every graph response exposes how many records were dropped and why.

### `app/data_pipeline.py` — Pandas Reference Pipeline
Mirrors `analyzer.py` logic using Pandas groupBy. Used in benchmarks, not in the live request path. Key export: `compute_metrics_pandas(df)`.

### `app/spark_pipeline.py` — PySpark Distributed Pipeline
Mirrors `data_pipeline.py` for distributed execution. Used only in `scripts/benchmark.py`. Includes Spark MLlib K-Means.

### `app/db.py` — Benchmark DB + Metric-Vector Bridge
All SQLite access lives here. No FastAPI types.

```python
init_db()                                          # create submissions table if absent
extract_metric_vector(analysis_result: dict) -> dict  # bridge analyze() output → 15-key shape
submit(values: dict) -> dict                       # insert row, return percentiles
compute_percentiles(user_values: dict) -> dict
stats() -> dict                                    # aggregate averages
all_users() -> list[dict]                          # all submissions for clustering
```

`DB_PATH` defaults to `BASE_DIR / "benchmark.db"`, overridable via `DB_PATH` env var.

`extract_metric_vector` is the single source of truth for the `metrikler`/`bizim_rapor` → METRIC_KEYS mapping. Two keys come from `bizim_rapor` (not `metrikler`): `total_hours` ← `toplam_saat`, `shuffle_pct` ← `shuffle_orani_pct`.

### `app/gemini.py` — AI Character Analysis
Self-contained Gemini subsystem. No FastAPI dependency.

```python
analyze_character(metrics: dict) -> dict | None   # main entry point
```

Internal components (not called directly):
- `_get_client()` — lazy singleton init, returns `None` if no API key
- `_throttle_rpm(deadline)` — sliding-window rate limiter (4 RPM)
- `_classify_error(exc)` → `(code, retry_after, kind)` — HTTP error inspector
- `_build_prompt(metrics)` → `str` — constructs Turkish character analysis prompt
- `_try_once(client, model, prompt, attempt_no)` — single timed attempt with `ThreadPoolExecutor`

Constants: `GEMINI_TIMEOUT=25s`, `GEMINI_MAX_ATTEMPTS=2`, `GEMINI_BUDGET=60s`, `GEMINI_RPM_LIMIT=4`.  
Models tried in order: `gemini-2.5-flash` → `gemini-2.0-flash` → `gemini-flash-latest`.

### `app/main.py` — FastAPI Routes (thin orchestration, ~185 lines)
- `BASE_DIR` = project root (`Path(__file__).parent.parent`)
- `TEMPLATES` = `BASE_DIR / "templates"`
- Routes call domain modules and return; no business logic inline
- `render_dashboard(metrics, template_path) -> str` — serialises metrics to JSON, injects into dashboard template, raises HTTP 500 on serialisation failure. The `/analyze` route calls this instead of doing the string-replace inline.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Landing page (index.html) |
| POST | `/analyze` | Upload Spotify JSONs → full dashboard HTML |
| POST | `/api/submit` | Submit anonymized metrics to benchmark DB |
| POST | `/api/percentiles` | Get percentile ranking vs. benchmark pool |
| GET | `/api/stats` | Aggregate stats from benchmark DB |
| POST | `/api/graph` | Run only graph analysis on uploaded JSONs |
| GET | `/api/cluster` | Cluster all benchmark DB users |
| GET | `/api/health` | `{"status": "ok"}` |

---

## Spotify Record Shape

A single streaming record (from Extended Streaming History export):

```python
{
    "ts":                                  "2023-06-01T12:00:00Z",  # REQUIRED
    "ms_played":                           180000,                   # REQUIRED
    "master_metadata_track_name":          "Song Name",              # REQUIRED
    "master_metadata_album_artist_name":   "Artist Name",            # REQUIRED
    "master_metadata_album_album_name":    "Album Name",
    "skipped":                             False,
    "shuffle":                             False,
    "reason_start":                        "clickrow",
    "reason_end":                          "endplay",
    "conn_country":                        "TR",
    "platform":                            "android",
    "offline":                             False,
    "incognito_mode":                      False,
}
```

---

## The 15 Benchmark Metrics (`METRIC_KEYS`)

```
impatience_score_pct          # % of plays that were skipped
completion_rate_pct           # % of plays ending with "endplay"
exploration_score             # 100 * unique_tracks / total_plays
artist_diversity_entropy      # Shannon entropy of artist play counts
early_skip_rate_pct           # % of skips where ms_played < 30s
listening_intensity_h_per_day # total_hours / calendar_days
night_listening_ratio_pct     # % of ms in local 00–06
mobile_usage_ratio_pct        # % of ms on mobile platform
focus_session_score_pct       # % of non-skipped plays >= 4 min
music_novelty_rate_pct        # avg % new tracks per month
artist_loyalty_score_pct      # % plays by top-10 artists
habit_loop_score_pct          # % of consecutive (track,artist) pairs that repeat
listening_fragmentation_index # skipped_count / num_sessions
total_hours                   # total listening hours
shuffle_pct                   # % of plays in shuffle mode
```

---

## Gemini Integration

File: `app/gemini.py` — `analyze_character(metrics)`

- Models tried in order: `gemini-2.5-flash` → `gemini-2.0-flash` → `gemini-flash-latest`
- Per-call timeout: 25s; max attempts per model: 2; total budget: 60s
- RPM cap: 4/min (sliding window, free tier safe)
- Returns Turkish character analysis: `{title, summary, traits, insights, prediction}`
- Skip via env: `SKIP_GEMINI=1`
- API key via env: `GEMINI_API_KEY`

---

## Benchmark SQLite DB

File: `benchmark.db` (project root, gitignored)  
Table: `submissions` — one row per user analysis, stores all 15 `METRIC_KEYS` as REAL columns.  
Used for: percentile ranking, global clustering.

---

## Tests

Run with: `.venv/bin/python -m pytest tests/ -v`  
70 tests, ~1.7s. No network, no HTTP server.

| File | Count | What |
|------|-------|------|
| `test_constants.py` | 8 | smoke tests on constant shapes/values |
| `test_personality.py` | 21 | unit tests for all 4 pure functions |
| `test_analyzer.py` | 9 | integration: `analyze()` on synthetic records |
| `test_db.py` | 7 | `extract_metric_vector`, `submit`, `compute_percentiles`, `stats`, `all_users` — uses in-memory SQLite via `tmp_path` + `monkeypatch` |
| `test_clustering.py` | 12 | `_validate_rows`, `label_cluster` boundary tests, `cluster_users` edge cases |
| `test_graph.py` | 13 | `_parse_artist_timeline` diagnostics, `build_artist_transition_graph` edge/gap/self-loop, `analyze_listening_graph` parse_diagnostics in summary |

---

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `GEMINI_API_KEY` | `""` | Gemini API key; empty = skip AI analysis |
| `SKIP_GEMINI` | `""` | Set to `1`/`true`/`yes` to skip Gemini entirely |
| `DB_PATH` | `benchmark.db` | Override SQLite path |
| `PORT` | `8000` | Uvicorn port |

---

## Key Architecture Decisions

- **`app/constants.py`** is the single source of truth — never redefine `SESSION_GAP_MINUTES`, `TZ_OFFSETS`, `METRIC_KEYS`, etc. in other modules.
- **`app/personality.py`** contains only pure functions — no side effects, no DB, no I/O. Testable in isolation.
- **`app/constants.py`** is the single source of truth — never redefine `SESSION_GAP_MINUTES`, `AVG_TRACK_DURATION_SEC`, `TZ_OFFSETS`, `METRIC_KEYS`, etc. in other modules.
- **`app/db.py`** owns all SQLite access and the metric-vector bridge — `extract_metric_vector` is the only place that maps `analyze()` output keys to METRIC_KEYS. Two keys diverge from the `metrikler` namespace: `total_hours` ← `bizim_rapor.toplam_saat`, `shuffle_pct` ← `bizim_rapor.shuffle_orani_pct`.
- **`app/gemini.py`** is fully self-contained — no FastAPI imports. The entire retry/throttle/fallback system lives here and is testable without the web layer.
- **`app/main.py` is thin** — routes parse input, call domain modules, return output. No business logic, no SQL, no AI calls inline. `render_dashboard()` is the only place that serialises metrics to HTML.
- **`analyzer.py` vs `data_pipeline.py`**: both compute the same metrics. `analyzer.py` is the live request path (pure Python, fast for typical uploads). `data_pipeline.py` is the Pandas reference used only in benchmarks.
- **`_aggregate_records()` in `analyzer.py`** — the single-pass accumulator. All per-play bucketing lives here. `analyze()` calls it and unpacks the result; nothing else should replicate that loop.
- **`build_artist_transition_graph()` returns `(G, diagnostics)`** — callers must unpack the tuple. Only `analyze_listening_graph()` calls it. Diagnostics are merged into `summary["parse_diagnostics"]` in every graph response.
- **`_validate_rows()` in `clustering.py`** — always the first call inside `cluster_users()`. If METRIC_KEYS changes, the error surfaces here, not inside numpy.
- **`BASE_DIR`** in `app/main.py` is `Path(__file__).parent.parent` (project root) — templates and DB live at project root, not inside `app/`.

---

## Scripts

| Script | Run from | Purpose |
|--------|----------|---------|
| `scripts/benchmark.py` | project root | `python scripts/benchmark.py [--no-spark] [--scales N ...]` |
| `scripts/synthetic_data.py` | project root | `python scripts/synthetic_data.py --records 1000000 --out ./data.json` |
| `scripts/plot_benchmark.py` | project root | `python scripts/plot_benchmark.py` → writes PNGs to `benchmark_results/` |

Scripts add project root + scripts dir to `sys.path` automatically.
