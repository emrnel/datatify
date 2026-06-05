# Datatify

**Personal digital data analysis for Spotify listening behavior.**

Datatify is a FastAPI web app that analyzes Spotify Extended Streaming History exports and turns them into an interactive listening dashboard. Users upload one or more Spotify JSON files and receive deterministic behavioral metrics, a personality archetype, badges, level progress, radar profile, artist transition graph, user clustering, community percentiles, and optional Gemini-powered character analysis.

The dashboard UI is in Turkish. Code, module names, and API identifiers are in English.

**Live demo:** <https://web-production-65055.up.railway.app/>

---

## Features

### Spotify Listening Analysis

- Upload one or more `Streaming_History_Audio_*.json` files from Spotify Extended Streaming History.
- Computes summary statistics such as total hours, sessions, top artists, top songs, yearly listening, weekday distribution, platform mix, and shuffle usage.
- Uses timezone correction based on the dominant listening country.
- Detects listening sessions with a 30-minute gap heuristic.

### Behavioral Metrics

Datatify computes a 15-metric benchmark vector:

- Impatience score
- Completion rate
- Exploration score
- Artist diversity entropy
- Early skip rate
- Listening intensity per day
- Night listening ratio
- Mobile usage ratio
- Focus session score
- Music novelty rate
- Artist loyalty score
- Habit loop score
- Listening fragmentation index
- Total listening hours
- Shuffle percentage

### Personality Layer

- 15 achievement badges.
- 10+ level thresholds from Newbie to Transcendent.
- Rule-based listener archetypes such as night listeners, loyal repeaters, explorers, and balanced listeners.
- 6-axis radar profile for patience, exploration, loyalty, focus, diversity, and night listening.

### Graph Analysis

- Builds a directed artist transition graph from consecutive plays.
- Adds an edge from artist A to artist B when B follows A within 30 minutes.
- Computes PageRank for central artists in the listening flow.
- Detects artist communities with NetworkX label propagation.
- Returns parse diagnostics so skipped graph records are visible.
- Renders an interactive D3 force-directed graph in the dashboard.

### User Clustering and Community Benchmarking

- Stores anonymized metric submissions in SQLite.
- Computes percentile rankings against the benchmark pool.
- Clusters users with scikit-learn K-Means.
- Selects k with elbow and silhouette scoring.
- Labels clusters with heuristic listener-group names.

### Gemini Character Analysis

- Optional Turkish AI narrative generated with Google Gemini.
- Uses `gemini-2.5-flash` first, then falls back to `gemini-2.0-flash` and `gemini-flash-latest`.
- Includes timeout, retry, backoff, total-budget, and RPM-throttle handling.
- Dashboard still renders when Gemini is disabled, unavailable, or times out.

### Scalability and Benchmarks

- Live request path uses a pure-Python analyzer optimized for typical uploads.
- Pandas and PySpark reference pipelines mirror the metric logic for benchmarks.
- Synthetic Spotify-like data generator supports larger-scale testing.
- Benchmark scripts compare Python, Pandas, and PySpark execution.

---

## Architecture

```text
Spotify JSON upload
    -> FastAPI route in app/main.py
    -> app/analyzer.py pure-Python metrics
    -> app/personality.py badges, level, archetype, radar
    -> app/graph_analysis.py NetworkX transition graph
    -> app/db.py SQLite percentiles and metric-vector bridge
    -> app/clustering.py K-Means benchmark clustering
    -> app/gemini.py optional Gemini character analysis
    -> templates/dashboard.html interactive dashboard
```

`app/main.py` is intentionally thin: routes parse inputs, call domain modules, and return HTML or JSON. Business logic lives in dedicated modules.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web framework | FastAPI, Uvicorn |
| Data processing | Pure Python, pandas, NumPy, pyarrow |
| Graph analysis | NetworkX |
| Clustering | scikit-learn K-Means |
| Distributed benchmark pipeline | PySpark |
| AI analysis | Google Gemini via `google-genai` |
| Frontend | HTML, CSS, JavaScript, Chart.js, D3.js |
| Database | SQLite |
| Deployment | Railway, Nixpacks |
| Python | 3.12+ |

---

## Project Structure

```text
datatify/
|-- app/
|   |-- main.py             # FastAPI routes and dashboard rendering
|   |-- analyzer.py         # Core pure-Python listening analysis
|   |-- constants.py        # Shared constants and metric keys
|   |-- personality.py      # Badges, level, archetype, radar profile
|   |-- db.py               # SQLite access, percentiles, metric-vector bridge
|   |-- gemini.py           # Gemini client, prompt, retry/throttle/fallback logic
|   |-- clustering.py       # scikit-learn K-Means user clustering
|   |-- data_pipeline.py    # Pandas reference pipeline for benchmarks
|   |-- graph_analysis.py   # NetworkX artist transition graph
|   `-- spark_pipeline.py   # PySpark benchmark pipeline
|-- scripts/
|   |-- benchmark.py        # Scalability benchmark runner
|   |-- synthetic_data.py   # Synthetic Spotify record generator
|   `-- plot_benchmark.py   # Benchmark plot renderer
|-- templates/
|   |-- index.html          # Upload page
|   `-- dashboard.html      # Interactive dashboard
|-- tests/                  # Unit and integration tests
|-- docs/
|   |-- CONTEXT.md          # Maintainer context
|   `-- datatify-ieee-report-final.tex
|-- benchmark_results/      # CSV, JSON, and PNG benchmark outputs
|-- benchmark.db            # Local SQLite benchmark DB
|-- Procfile                # Railway process command
|-- railway.json            # Railway deployment config
|-- requirements.txt
`-- README.md
```

---

## Setup

### Requirements

- Python 3.12+
- Java 8+ or newer if you want to run PySpark benchmarks
- Google Gemini API key if you want AI character analysis

### Install

```bash
git clone https://github.com/emrnel/datatify
cd datatify

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Configure Environment

Gemini is optional. If no key is configured, Datatify skips AI analysis and still returns the dashboard.

```bash
export GEMINI_API_KEY="your-api-key"
export SKIP_GEMINI=""
export DB_PATH="benchmark.db"
```

On Windows PowerShell:

```powershell
$env:GEMINI_API_KEY = "your-api-key"
$env:SKIP_GEMINI = ""
$env:DB_PATH = "benchmark.db"
```

### Run Locally

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000>.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | empty | Gemini API key. Empty means AI analysis is skipped. |
| `SKIP_GEMINI` | empty | Set to `1`, `true`, or `yes` to disable Gemini explicitly. |
| `DB_PATH` | `benchmark.db` | SQLite database path. |
| `PORT` | `8000` | Server port used by local runs and Railway. |

---

## How to Get Spotify Data

1. Go to Spotify account privacy settings: <https://www.spotify.com/account/privacy/>
2. Request **Extended Streaming History**.
3. Wait for Spotify to email the export ZIP.
4. Extract the ZIP.
5. Upload the `Streaming_History_Audio_*.json` files to Datatify.

Datatify expects records with fields such as `ts`, `ms_played`, `master_metadata_track_name`, and `master_metadata_album_artist_name`. Optional enrichment fields like `platform`, `conn_country`, `skipped`, `shuffle`, and `offline` improve the analysis when present.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Upload page |
| `POST` | `/analyze` | Upload Spotify JSON files and receive the full dashboard HTML |
| `POST` | `/api/submit` | Submit anonymized metric vector to the benchmark DB |
| `POST` | `/api/percentiles` | Compute percentile ranking against stored submissions |
| `GET` | `/api/stats` | Return aggregate benchmark statistics |
| `POST` | `/api/graph` | Run only artist transition graph analysis on uploaded JSON files |
| `GET` | `/api/cluster` | Cluster all benchmark DB users |
| `GET` | `/api/health` | Health check |

---

## Tests

Run the test suite:

```bash
python -m pytest tests/ -v
```

The tests cover constants, personality classification, core analysis, database operations, clustering edge cases, and graph parsing/analysis.

---

## Benchmarks

Run scalability benchmarks:

```bash
python scripts/benchmark.py
```

Skip Spark if Java or PySpark setup is unavailable:

```bash
python scripts/benchmark.py --no-spark
```

Generate benchmark plots:

```bash
python scripts/plot_benchmark.py
```

Generate synthetic data:

```bash
python scripts/synthetic_data.py --records 1000000 --out ./synthetic_spotify.json
```

Benchmark outputs are written to `benchmark_results/`.

---

## Deployment

The project is currently deployed on Railway.

`Procfile` and `railway.json` both use:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Railway uses the Nixpacks builder.

---

## Development Notes

- `app/constants.py` is the single source of truth for shared constants such as metric keys, timezone offsets, required Spotify columns, and metric labels.
- `app/personality.py` is side-effect free and should stay easy to unit test.
- `app/db.py` owns SQLite access and the mapping from full analysis output to the 15-key benchmark vector.
- `app/gemini.py` is self-contained and has no FastAPI dependency.
- `app/analyzer.py` is the live analysis path; `app/data_pipeline.py` and `app/spark_pipeline.py` are benchmark/reference implementations.
- `build_artist_transition_graph()` returns both the graph and parse diagnostics; graph callers should preserve those diagnostics in API output.

---

## Team

CSE 458 - Big Data Analytics Project

- Soner Gunes
- Mehmet Alp Atay
- Emre Ilhan Senel
- H. Muhammet Cengelci

---

## License

MIT
