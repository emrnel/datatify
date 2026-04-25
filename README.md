# Datatify

**Personal Digital Data Analysis for Behavioral Insights**

Datatify is a Big Data analytics platform that processes Spotify Extended Streaming History exports to extract behavioral insights using deterministic metric formulas, graph algorithms, clustering, and LLM-generated narratives. Built as a semester project for CSE 458 - Big Data Analytics.

**Live Demo:** [https://web-production-65055.up.railway.app](https://web-production-65055.up.railway.app)

---

## Features

### Behavioral Metrics (20+ Deterministic Formulas)
- **Attention & Patience:** Impatience score, completion rate, early skip rate, skip latency ratio
- **Diversity & Exploration:** Artist diversity (Shannon entropy), exploration score, music novelty rate
- **Habit & Routine:** Circadian listening profile (timezone-corrected), night listening ratio, habit loop score, listening intensity
- **Loyalty & Focus:** Artist loyalty score, song replay density, focus session score, listening fragmentation index
- **Platform & Context:** Mobile usage ratio, offline ratio, shuffle ratio

### Graph Analysis (NetworkX + Spark GraphFrames)
- Artist transition graph construction (directed, weighted by consecutive-play frequency)
- **PageRank** to identify central artists in listening flow (distinct from raw play count)
- **Label Propagation** community detection for discovering co-listened artist clusters
- **Weakly Connected Components** for isolated listening islands
- D3.js force-directed graph visualization on the dashboard (draggable nodes, community-colored)

### User Clustering (scikit-learn + Spark MLlib)
- K-Means on 15-dimensional behavioral metric vectors
- Optimal k via elbow method + silhouette score
- Rule-based archetype labels (e.g. "Night Explorers", "Loyal Repeaters", "Impatient Skippers")
- Both single-node (scikit-learn) and distributed (Spark MLlib) implementations

### Gamification
- **15 Achievement Badges** with specific unlock conditions (Night Owl, Marathon Listener, Centurion, Shuffle Addict, etc.)
- **10-Level Progression** based on total listening hours (Newcomer to Legendary)
- **8 Listener Archetypes** via rule-based decision tree (The Night Diver, The Comfort Curator, etc.)
- **6-Axis Radar Chart** profile (Diversity, Loyalty, Focus, Exploration, Patience, Routine)

### AI-Powered Narrative (Gemini 2.0 Flash)
- Structured JSON metric summary sent to the Gemini API
- Returns: creative title, psychological character analysis, personality traits, data-supported insights
- 20-second timeout with graceful fallback — dashboard always renders

### Community Benchmarking
- Anonymous metric submission to SQLite database
- Percentile ranking against other users
- Aggregate community statistics

### Scalability & Benchmarking
- **Three processing engines:** Pure Python, Pandas, PySpark
- **Synthetic data generator** (Zipf-distributed, 130K–5M records)
- **Benchmark harness** comparing wall-clock time across engines
- Sub-linear PySpark scaling demonstrated up to 2M records

---

## Architecture

```
JSON Upload → Schema Validation → Parquet (year/month partitioned)
    ↓
Pure Python / Pandas / PySpark metric computation
    ↓
┌─────────────────┬──────────────────┬─────────────────┐
│  Graph Analysis  │  User Clustering │  Gamification   │
│  (NetworkX /     │  (scikit-learn / │  (Badges, Levels│
│   GraphFrames)   │   Spark MLlib)   │   Archetypes)   │
└─────────────────┴──────────────────┴─────────────────┘
    ↓
Gemini 2.0 Flash narrative generation
    ↓
FastAPI → Interactive Dashboard (Chart.js, D3.js, SVG gauges)
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Backend | FastAPI + Uvicorn |
| Single-Node Processing | Pure Python (`analyzer.py`) + Pandas/NumPy (`data_pipeline.py`) |
| Distributed Processing | PySpark (`spark_pipeline.py`) |
| Graph Analysis | NetworkX (single-node) + Spark GraphFrames (distributed) |
| Clustering | scikit-learn K-Means (single-node) + Spark MLlib K-Means (distributed) |
| AI Narrative | Google Gemini 2.0 Flash |
| Frontend | HTML/CSS/JS, Chart.js, D3.js, SVG |
| Database | SQLite (benchmark data) |
| Storage | Apache Parquet (year/month partitioned) |
| Deployment | Railway (current), GCP Cloud Run + Dataproc (target) |

---

## Setup

### Requirements
- Python 3.9+
- Java 8+ (for PySpark — optional, only needed for distributed benchmarks)
- Google Gemini API key ([get one here](https://aistudio.google.com))

### Installation

```bash
git clone https://github.com/emrnel/datatify
cd datatify

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt

export GEMINI_API_KEY="your-api-key-here"   # Windows: set GEMINI_API_KEY=your-api-key-here

python main.py
```

Open `http://localhost:8000` in your browser.

### Environment Variables

| Variable | Description | Required |
|---|---|---|
| `GEMINI_API_KEY` | Google Gemini API key | Yes (for AI narrative) |
| `SKIP_GEMINI` | Set to `1` to skip AI analysis | No |
| `DB_PATH` | SQLite database path | No |
| `PORT` | Server port (default: 8000) | No |

---

## How to Get Your Spotify Data

1. Go to [Spotify Account Privacy](https://www.spotify.com/account/privacy/)
2. Request **"Extended Streaming History"** under the data download section
3. Wait ~30 days for the email with a ZIP file
4. Upload the `Streaming_History_Audio_*.json` files to Datatify

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Upload landing page |
| `POST` | `/analyze` | Upload JSON files and get full behavioral analysis |
| `POST` | `/api/graph` | Artist transition graph + PageRank + communities |
| `GET` | `/api/cluster` | K-Means clustering over benchmark database |
| `POST` | `/api/submit` | Submit anonymous metrics |
| `POST` | `/api/percentiles` | Get percentile rankings |
| `GET` | `/api/stats` | Community aggregate statistics |
| `GET` | `/api/health` | Health check |

---

## Project Structure

```
datatify/
├── main.py                  # FastAPI application + all routes
├── analyzer.py              # Pure Python behavioral metric engine (20+ formulas)
├── data_pipeline.py         # Pandas-based ingestion, validation, Parquet, metrics
├── spark_pipeline.py        # PySpark mirror: metrics + GraphFrames + MLlib K-Means
├── graph_analysis.py        # NetworkX artist transition graph analysis
├── clustering.py            # scikit-learn K-Means with elbow + silhouette
├── synthetic_data.py        # Zipf-distributed synthetic data generator
├── benchmark.py             # Scalability benchmark harness (3 engines)
├── plot_benchmark.py        # Scalability plot generator
├── templates/
│   ├── index.html           # Upload page
│   └── dashboard.html       # Interactive analysis dashboard
├── midterm_report.tex       # IEEE-format midterm report (TikZ/pgfplots figures)
├── requirements.txt
└── benchmark_results/
    ├── benchmark_results.json
    └── benchmark_results.csv
```

---

## Running Benchmarks

```bash
# Generate synthetic data and benchmark all three engines
python benchmark.py

# Generate scalability plots
python plot_benchmark.py
```

Results are written to `benchmark_results/`. PySpark benchmarks require Java 8+ installed.

---

## Deployment

Currently deployed on Railway. For GCP deployment (Phase 3 target):

1. **Cloud Run** — containerized FastAPI app
2. **Dataproc** — managed Spark cluster for distributed benchmarks
3. **GCS** — Parquet storage
4. **Cloud SQL** — PostgreSQL for benchmark data

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
