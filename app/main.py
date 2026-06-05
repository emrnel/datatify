# -*- coding: utf-8 -*-
"""Datatify — Spotify Listening DNA web app (FastAPI)."""
import json
import time
import traceback
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .analyzer import analyze
from .graph_analysis import analyze_listening_graph
from .clustering import cluster_users
from .constants import METRIC_KEYS
from .db import init_db, extract_metric_vector, compute_percentiles, submit, stats, all_users
from .gemini import analyze_character

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES = BASE_DIR / "templates"
STATIC = BASE_DIR / "static"

app = FastAPI(title="Datatify — Listening DNA", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

init_db()


class MetricsSubmission(BaseModel):
    impatience_score_pct: float = 0
    completion_rate_pct: float = 0
    exploration_score: float = 0
    artist_diversity_entropy: float = 0
    early_skip_rate_pct: float = 0
    listening_intensity_h_per_day: float = 0
    night_listening_ratio_pct: float = 0
    mobile_usage_ratio_pct: float = 0
    focus_session_score_pct: float = 0
    music_novelty_rate_pct: float = 0
    artist_loyalty_score_pct: float = 0
    habit_loop_score_pct: float = 0
    listening_fragmentation_index: float = 0
    total_hours: float = 0
    shuffle_pct: float = 0


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def landing():
    return FileResponse(str(TEMPLATES / "index.html"), media_type="text/html")


@app.post("/analyze", response_class=HTMLResponse)
async def analyze_files(files: list[UploadFile] = File(...)):
    """Accept one or more Spotify JSON files, run analysis + Gemini, return dashboard."""
    t_start = time.time()
    all_records: list[dict] = []
    file_names = []
    for f in files:
        try:
            raw = await f.read()
            data = json.loads(raw.decode("utf-8"))
            if isinstance(data, list):
                all_records.extend(data)
                file_names.append(f"{f.filename} ({len(data)} records)")
        except Exception as e:
            print(f"[UPLOAD] Failed to parse {f.filename}: {e}")
            continue

    print(f"[UPLOAD] {len(files)} files -> {len(all_records)} total records")
    for fn in file_names:
        print(f"  · {fn}")

    if not all_records:
        raise HTTPException(status_code=400, detail="Geçerli Spotify JSON dosyası bulunamadı.")

    print("[ANALYZE] Starting analysis...")
    t1 = time.time()
    metrics = analyze(all_records)
    print(f"[ANALYZE] Done in {time.time()-t1:.1f}s")

    if "error" in metrics:
        raise HTTPException(status_code=400, detail=metrics["error"])

    print("[GRAPH] Building artist transition graph...")
    t2 = time.time()
    try:
        metrics["graph"] = analyze_listening_graph(all_records)
        print(f"[GRAPH] Done in {time.time()-t2:.1f}s "
              f"(nodes={metrics['graph']['summary']['nodes']}, "
              f"edges={metrics['graph']['summary']['edges']}, "
              f"communities={len(metrics['graph']['communities'])})")
    except Exception as e:
        print(f"[GRAPH] FAILED: {e}")
        traceback.print_exc()
        metrics["graph"] = {
            "summary": {"nodes": 0, "edges": 0},
            "pagerank": [], "communities": [], "components": {},
            "visualization": {"nodes": [], "edges": []},
            "error": str(e),
        }

    print("[CLUSTER] Computing user clustering vs benchmark pool...")
    try:
        user_vec = extract_metric_vector(metrics)
        metrics["clustering"] = cluster_users(all_users(), user_vector=user_vec)
        print(f"[CLUSTER] {metrics['clustering'].get('status')}, "
              f"k={metrics['clustering'].get('k')}, "
              f"silhouette={metrics['clustering'].get('silhouette')}")
    except Exception as e:
        print(f"[CLUSTER] FAILED: {e}")
        traceback.print_exc()
        metrics["clustering"] = {"status": "error", "error": str(e)}

    ai = analyze_character(metrics)
    if ai:
        metrics["gemini_analysis"] = ai

    template = (TEMPLATES / "dashboard.html").read_text(encoding="utf-8")
    html = template.replace("SPOTIFY_DATA_PLACEHOLDER", json.dumps(metrics, ensure_ascii=False))
    print(f"[DONE] Total request time: {time.time()-t_start:.1f}s")
    return HTMLResponse(content=html)


# ─── Benchmark API ───────────────────────────────────────────────────────────

@app.post("/api/submit")
def submit_metrics(data: MetricsSubmission):
    values = {k: getattr(data, k) for k in METRIC_KEYS}
    return submit(values)


@app.post("/api/percentiles")
def get_percentiles(data: MetricsSubmission):
    values = {k: getattr(data, k) for k in METRIC_KEYS}
    return compute_percentiles(values)


@app.get("/api/stats")
def get_stats():
    return stats()


@app.post("/api/graph")
async def api_graph(files: list[UploadFile] = File(...)):
    """Run only the listening transition graph analysis on uploaded JSONs."""
    all_records: list[dict] = []
    for f in files:
        try:
            data = json.loads((await f.read()).decode("utf-8"))
            if isinstance(data, list):
                all_records.extend(data)
        except Exception:
            continue
    if not all_records:
        raise HTTPException(status_code=400, detail="Geçerli Spotify JSON bulunamadı.")
    return analyze_listening_graph(all_records)


@app.get("/api/cluster")
def api_cluster():
    return cluster_users(all_users())


@app.get("/api/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
