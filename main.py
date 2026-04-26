# -*- coding: utf-8 -*-
"""Datatify — Spotify Listening DNA web app (FastAPI + Gemini)."""
import os
import json
import random
import re
import sqlite3
import threading
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from contextlib import contextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from analyzer import analyze
from graph_analysis import analyze_listening_graph
from clustering import cluster_users, METRIC_KEYS as CLUSTER_METRIC_KEYS

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES = BASE_DIR / "templates"
STATIC = BASE_DIR / "static"
DB_PATH = os.environ.get("DB_PATH", str(BASE_DIR / "benchmark.db"))
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
SKIP_GEMINI = os.environ.get("SKIP_GEMINI", "").strip().lower() in ("1", "true", "yes")

app = FastAPI(title="Datatify — Listening DNA", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

# ─── Gemini ──────────────────────────────────────────────────────────────────

_gemini_model = None

# In-process sliding-window RPM throttle. Free tier is 5 RPM on 2.5 Flash;
# we cap at 4 to leave a buffer for clock skew between us and Google.
_rpm_lock = threading.Lock()
_rpm_window: deque = deque()

GEMINI_TIMEOUT = 20         # seconds for a single API call
GEMINI_MAX_ATTEMPTS = 2     # tries per model before moving to the next
GEMINI_BUDGET = 45          # hard cap on total wall time across all attempts
GEMINI_RPM_LIMIT = 4        # in-process cap (free tier limit is 5 RPM)
GEMINI_RPM_WINDOW = 60      # seconds in the sliding window

# Model fallback chain. If the primary keeps 429ing or is unavailable, the
# next one is tried. Order is "newest first" since 2.5 Flash has the best
# quality but tightest free-tier quota; 1.5 Flash has more headroom.
GEMINI_MODELS = (
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
)

_RETRY_AFTER_RE = re.compile(r"retry[-_ ]?after[^\d]*(\d+)", re.IGNORECASE)


def _get_gemini():
    global _gemini_model
    if _gemini_model is not None:
        return _gemini_model
    if not GEMINI_API_KEY:
        return None
    try:
        from google import genai
        client = genai.Client(api_key=GEMINI_API_KEY)
        _gemini_model = client
        return client
    except Exception:
        traceback.print_exc()
        return None


def _throttle_rpm(deadline: float) -> bool:
    """Reserve a slot in the sliding RPM window. Returns False if we'd have
    to wait past the deadline (caller should give up)."""
    while True:
        with _rpm_lock:
            now = time.time()
            while _rpm_window and now - _rpm_window[0] >= GEMINI_RPM_WINDOW:
                _rpm_window.popleft()
            if len(_rpm_window) < GEMINI_RPM_LIMIT:
                _rpm_window.append(now)
                print(f"[GEMINI] RPM slot acquired ({len(_rpm_window)}/{GEMINI_RPM_LIMIT} in last {GEMINI_RPM_WINDOW}s)")
                return True
            wait = GEMINI_RPM_WINDOW - (now - _rpm_window[0]) + 0.05
        if time.time() + wait > deadline:
            print(f"[GEMINI] RPM throttle would wait {wait:.1f}s past budget — aborting")
            return False
        print(f"[GEMINI] RPM full ({len(_rpm_window)}/{GEMINI_RPM_LIMIT}), sleeping {min(wait, 5.0):.1f}s")
        time.sleep(min(wait, 5.0))


def _classify_error(exc: Exception):
    """Inspect a Gemini exception and return (status_code, retry_after_s, kind).
    kind ∈ {'rate_limit', 'server', 'client', 'unknown'}."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    msg = str(exc)

    if not isinstance(code, int):
        for token in ("429", "503", "504", "502", "500", "404", "403", "400"):
            if token in msg:
                code = int(token)
                break

    retry_after = None
    blob = msg
    for attr in ("details", "response", "body"):
        v = getattr(exc, attr, None)
        if v is not None:
            blob += " " + str(v)
    m = _RETRY_AFTER_RE.search(blob)
    if m:
        try:
            retry_after = int(m.group(1))
        except ValueError:
            retry_after = None

    if code == 429:
        return code, retry_after, "rate_limit"
    if isinstance(code, int) and 500 <= code < 600:
        return code, retry_after, "server"
    if isinstance(code, int) and 400 <= code < 500:
        return code, retry_after, "client"
    return code, retry_after, "unknown"


def _call_gemini_sync(client, model: str, prompt: str) -> dict:
    """Blocking generate_content call. Raises on any error."""
    response = client.models.generate_content(model=model, contents=prompt)
    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
    return json.loads(text)


def _try_once(client, model: str, prompt: str, attempt_no: int):
    """Run one timed attempt. Returns (result, error_tuple). Exactly one is
    non-None. error_tuple is (kind, code, retry_after)."""
    t0 = time.time()
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_call_gemini_sync, client, model, prompt)
            result = future.result(timeout=GEMINI_TIMEOUT)
        print(f"[GEMINI] attempt#{attempt_no} OK on {model} in {time.time()-t0:.1f}s")
        return result, None
    except FuturesTimeout:
        elapsed = time.time() - t0
        print(f"[GEMINI] attempt#{attempt_no} TIMEOUT on {model} after {elapsed:.1f}s "
              f"(per-call timeout = {GEMINI_TIMEOUT}s)")
        return None, ("timeout", None, None)
    except Exception as e:
        code, retry_after, kind = _classify_error(e)
        elapsed = time.time() - t0
        print(f"[GEMINI] attempt#{attempt_no} FAIL on {model} in {elapsed:.1f}s — "
              f"kind={kind} http={code} retry_after={retry_after} "
              f"err={type(e).__name__}: {str(e)[:200]}")
        return None, (kind, code, retry_after)


def gemini_character_analysis(metrics: dict) -> dict | None:
    """Call Gemini with a strict timeout. Returns None on any failure."""
    if SKIP_GEMINI:
        print("[GEMINI] SKIP_GEMINI=1 — not calling API")
        return None
    client = _get_gemini()
    if not client:
        print("[GEMINI] No API key or client init failed — skipping")
        return None

    summary = {
        "toplam_saat": metrics["bizim_rapor"]["toplam_saat"],
        "toplam_kayit": metrics["bizim_rapor"]["toplam_kayit"],
        "seviye": metrics["level"]["title"],
        "arketip": metrics["archetype"]["name"],
        "sabırsızlık": metrics["metrikler"]["impatience_score_pct"],
        "tamamlama": metrics["metrikler"]["completion_rate_pct"],
        "keşif_skoru": metrics["metrikler"]["exploration_score"],
        "sanatçı_çeşitliliği": metrics["metrikler"]["artist_diversity_entropy"],
        "gece_dinleme": metrics["metrikler"]["night_listening_ratio_pct"],
        "odak_skoru": metrics["metrikler"]["focus_session_score_pct"],
        "sadakat": metrics["metrikler"]["artist_loyalty_score_pct"],
        "alışkanlık": metrics["metrikler"]["habit_loop_score_pct"],
        "günlük_yoğunluk_saat": metrics["metrikler"]["listening_intensity_h_per_day"],
        "mobil_oran": metrics["metrikler"]["mobile_usage_ratio_pct"],
        "shuffle_oranı": metrics["bizim_rapor"]["shuffle_orani_pct"],
        "yeni_parça_oranı": metrics["metrikler"]["music_novelty_rate_pct"],
        "erken_atlama": metrics["metrikler"]["early_skip_rate_pct"],
        "top5_sanatçı": [a["sanatci"] for a in metrics["top_sanatcilar"][:5]],
        "top5_şarkı": [s["sarki"] for s in metrics["top_sarkilar"][:5]],
        "rozetler": [b["name"] for b in metrics["badges"] if b["earned"]],
        "radar": metrics["radar"],
    }

    prompt = f"""Sen yaratıcı bir müzik psikoloğu ve karakter analistisin.
Aşağıdaki Spotify dinleme verilerini kullanarak kullanıcının müzik kişiliğini
derinlemesine, kişiselleştirilmiş ve yaratıcı şekilde analiz et.

## Dinleme Verileri
```json
{json.dumps(summary, ensure_ascii=False, indent=2)}
```

## Görev
Yanıtını SADECE aşağıdaki JSON formatında ver (başka metin ekleme):
{{
  "title": "Yaratıcı 2-4 kelimelik karakter başlığı (Türkçe)",
  "summary": "4-5 cümlelik derinlemesine karakter analizi. Dinleme alışkanlıklarını psikolojik açıdan yorumla, kişilik özelliklerini çıkar. Edebi ve etkileyici bir dil kullan. (Türkçe)",
  "traits": ["Kişilik özelliği 1", "Kişilik özelliği 2", "Kişilik özelliği 3", "Kişilik özelliği 4"],
  "insights": ["Şaşırtıcı gözlem 1 (veri destekli)", "Şaşırtıcı gözlem 2", "Şaşırtıcı gözlem 3"],
  "prediction": "Müzik zevkine dayalı yaratıcı bir tahmin (1-2 cümle, Türkçe)"
}}"""

    t_start = time.time()
    deadline = t_start + GEMINI_BUDGET
    print(f"[GEMINI] starting (budget={GEMINI_BUDGET}s, models={list(GEMINI_MODELS)}, "
          f"per-call timeout={GEMINI_TIMEOUT}s, max attempts/model={GEMINI_MAX_ATTEMPTS})")

    attempt_no = 0
    last_kind = None
    last_code = None

    for model in GEMINI_MODELS:
        for retry_idx in range(GEMINI_MAX_ATTEMPTS):
            attempt_no += 1

            if time.time() >= deadline:
                print(f"[GEMINI] budget exhausted before attempt#{attempt_no} on {model} — stopping")
                break

            if not _throttle_rpm(deadline):
                # Can't get an RPM slot in time — no point in trying further models.
                print(f"[GEMINI] giving up: cannot acquire RPM slot within budget "
                      f"(elapsed={time.time()-t_start:.1f}s)")
                return None

            result, err = _try_once(client, model, prompt, attempt_no)
            if result is not None:
                print(f"[GEMINI] DONE in {time.time()-t_start:.1f}s "
                      f"(model={model}, attempts={attempt_no})")
                return result

            kind, code, retry_after = err
            last_kind, last_code = kind, code

            # Non-retryable client errors (e.g. 400 bad request, 403 forbidden,
            # 404 model not found): switch model immediately, no backoff.
            if kind == "client" and code != 429:
                print(f"[GEMINI] non-retryable client error (http={code}) on {model} — switching model")
                break

            # Last attempt on this model? Switch to next model.
            if retry_idx == GEMINI_MAX_ATTEMPTS - 1:
                print(f"[GEMINI] {GEMINI_MAX_ATTEMPTS} attempts exhausted on {model} "
                      f"(last kind={kind}, http={code}) — switching model")
                break

            # Compute backoff delay.
            if retry_after is not None:
                delay = retry_after + random.uniform(0, 1.0)
                reason = f"server-supplied Retry-After={retry_after}s"
            elif kind == "rate_limit":
                # 429 with no Retry-After — back off hard since RPM window is 60s.
                base = 15 * (2 ** retry_idx)        # 15s, 30s
                delay = base + random.uniform(0, 5)
                reason = f"429 backoff base={base}s+jitter"
            elif kind == "server" or kind == "timeout":
                base = 2 ** retry_idx               # 1s, 2s
                delay = base + random.uniform(0, base)
                reason = f"{kind} backoff base={base}s+jitter"
            else:
                base = 2 ** retry_idx
                delay = base + random.uniform(0, base)
                reason = f"unknown-error backoff base={base}s+jitter"

            # Don't sleep past the budget.
            remaining = deadline - time.time()
            if delay >= remaining:
                print(f"[GEMINI] backoff {delay:.1f}s ({reason}) exceeds remaining budget "
                      f"{remaining:.1f}s — switching model instead")
                break

            print(f"[GEMINI] retrying on {model} in {delay:.1f}s ({reason}, "
                  f"next attempt {retry_idx+2}/{GEMINI_MAX_ATTEMPTS})")
            time.sleep(delay)

        if time.time() >= deadline:
            print(f"[GEMINI] budget exhausted after {attempt_no} attempts — stopping model loop")
            break

    elapsed = time.time() - t_start
    print(f"[GEMINI] all paths failed after {attempt_no} attempts in {elapsed:.1f}s "
          f"(last kind={last_kind}, http={last_code}) — dashboard will render without AI analysis")
    return None


# ─── Database (benchmark) ────────────────────────────────────────────────────

METRIC_KEYS = [
    "impatience_score_pct", "completion_rate_pct", "exploration_score",
    "artist_diversity_entropy", "early_skip_rate_pct",
    "listening_intensity_h_per_day", "night_listening_ratio_pct",
    "mobile_usage_ratio_pct", "focus_session_score_pct",
    "music_novelty_rate_pct", "artist_loyalty_score_pct",
    "habit_loop_score_pct", "listening_fragmentation_index",
    "total_hours", "shuffle_pct",
]


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


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        cols = ", ".join(f"{k} REAL DEFAULT 0" for k in METRIC_KEYS)
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS submissions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                {cols}
            )
        """)
        conn.commit()


init_db()

METRIC_LABELS = {
    "impatience_score_pct": ("Sabırsızlık", "kullanıcıdan daha sabırsız"),
    "completion_rate_pct": ("Tamamlama", "kullanıcıdan daha fazla şarkı bitiriyor"),
    "exploration_score": ("Keşif", "kullanıcıdan daha fazla keşfediyor"),
    "artist_diversity_entropy": ("Çeşitlilik", "kullanıcıdan daha eklektik"),
    "early_skip_rate_pct": ("Erken Atlama", "kullanıcıdan daha hızlı atlıyor"),
    "listening_intensity_h_per_day": ("Yoğunluk", "kullanıcıdan daha yoğun dinliyor"),
    "night_listening_ratio_pct": ("Gece Kuşu", "kullanıcıdan daha çok gece dinliyor"),
    "mobile_usage_ratio_pct": ("Mobil", "kullanıcıdan daha çok mobil kullanıyor"),
    "focus_session_score_pct": ("Odak", "kullanıcıdan daha odaklı"),
    "music_novelty_rate_pct": ("Yenilik", "kullanıcıdan daha çok yeni parça keşfediyor"),
    "artist_loyalty_score_pct": ("Sadakat", "kullanıcıdan daha sadık"),
    "habit_loop_score_pct": ("Alışkanlık", "kullanıcıdan daha alışkanlık odaklı"),
    "listening_fragmentation_index": ("Parçalılık", "kullanıcıdan daha parçalı dinliyor"),
    "total_hours": ("Toplam Süre", "kullanıcıdan daha çok dinlemiş"),
    "shuffle_pct": ("Shuffle", "kullanıcıdan daha çok shuffle kullanıyor"),
}


def _generate_labels(percentiles: dict) -> dict:
    labels = {}
    for k, pct in percentiles.items():
        _, suffix = METRIC_LABELS.get(k, (k, ""))
        labels[k] = f"%{pct:.0f} {suffix}"
    return labels


def compute_percentiles(user_values: dict) -> dict:
    with get_db() as conn:
        count = conn.execute("SELECT COUNT(*) FROM submissions").fetchone()[0]
        if count < 2:
            p = {k: 50.0 for k in METRIC_KEYS}
            return {"total_users": count, "percentiles": p, "labels": _generate_labels(p)}
        percentiles = {}
        for k in METRIC_KEYS:
            rows = conn.execute(f"SELECT {k} FROM submissions ORDER BY {k}").fetchall()
            all_vals = [r[0] for r in rows]
            below = sum(1 for v in all_vals if v < user_values[k])
            percentiles[k] = round(100 * below / len(all_vals), 1)
    return {"total_users": count, "percentiles": percentiles, "labels": _generate_labels(percentiles)}


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
        with get_db() as conn:
            rows = [dict(r) for r in conn.execute(
                f"SELECT {', '.join(CLUSTER_METRIC_KEYS)} FROM submissions"
            ).fetchall()]
        user_vec = {
            "impatience_score_pct": metrics["metrikler"]["impatience_score_pct"],
            "completion_rate_pct": metrics["metrikler"]["completion_rate_pct"],
            "exploration_score": metrics["metrikler"]["exploration_score"],
            "artist_diversity_entropy": metrics["metrikler"]["artist_diversity_entropy"],
            "early_skip_rate_pct": metrics["metrikler"]["early_skip_rate_pct"],
            "listening_intensity_h_per_day": metrics["metrikler"]["listening_intensity_h_per_day"],
            "night_listening_ratio_pct": metrics["metrikler"]["night_listening_ratio_pct"],
            "mobile_usage_ratio_pct": metrics["metrikler"]["mobile_usage_ratio_pct"],
            "focus_session_score_pct": metrics["metrikler"]["focus_session_score_pct"],
            "music_novelty_rate_pct": metrics["metrikler"]["music_novelty_rate_pct"],
            "artist_loyalty_score_pct": metrics["metrikler"]["artist_loyalty_score_pct"],
            "habit_loop_score_pct": metrics["metrikler"]["habit_loop_score_pct"],
            "listening_fragmentation_index": metrics["metrikler"]["listening_fragmentation_index"],
            "total_hours": metrics["bizim_rapor"]["toplam_saat"],
            "shuffle_pct": metrics["bizim_rapor"]["shuffle_orani_pct"],
        }
        metrics["clustering"] = cluster_users(rows, user_vector=user_vec)
        print(f"[CLUSTER] {metrics['clustering'].get('status')}, "
              f"k={metrics['clustering'].get('k')}, "
              f"silhouette={metrics['clustering'].get('silhouette')}")
    except Exception as e:
        print(f"[CLUSTER] FAILED: {e}")
        traceback.print_exc()
        metrics["clustering"] = {"status": "error", "error": str(e)}

    # Gemini (with timeout — dashboard works without it)
    ai = gemini_character_analysis(metrics)
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
    with get_db() as conn:
        cols = ", ".join(METRIC_KEYS)
        ph = ", ".join(["?"] * len(METRIC_KEYS))
        conn.execute(f"INSERT INTO submissions ({cols}) VALUES ({ph})", [values[k] for k in METRIC_KEYS])
        conn.commit()
    return compute_percentiles(values)


@app.post("/api/percentiles")
def get_percentiles(data: MetricsSubmission):
    values = {k: getattr(data, k) for k in METRIC_KEYS}
    return compute_percentiles(values)


@app.get("/api/stats")
def get_stats():
    with get_db() as conn:
        count = conn.execute("SELECT COUNT(*) FROM submissions").fetchone()[0]
        if count == 0:
            return {"total_users": 0, "averages": {}}
        avgs = {}
        for k in METRIC_KEYS:
            row = conn.execute(f"SELECT AVG({k}) as avg_val FROM submissions").fetchone()
            avgs[k] = round(row["avg_val"], 2) if row["avg_val"] is not None else 0
    return {"total_users": count, "averages": avgs}


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
    """Cluster all submitted benchmark users (no current-user vector)."""
    with get_db() as conn:
        rows = [dict(r) for r in conn.execute(
            f"SELECT {', '.join(CLUSTER_METRIC_KEYS)} FROM submissions"
        ).fetchall()]
    return cluster_users(rows)


@app.get("/api/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
