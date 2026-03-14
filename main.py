# -*- coding: utf-8 -*-
"""Datatify — Spotify Listening DNA web app (FastAPI + Gemini)."""
import os
import json
import sqlite3
import traceback
from contextlib import contextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from analyzer import analyze

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES = BASE_DIR / "templates"
STATIC = BASE_DIR / "static"
DB_PATH = os.environ.get("DB_PATH", str(BASE_DIR / "benchmark.db"))
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

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


def gemini_character_analysis(metrics: dict) -> dict | None:
    """Call Gemini to generate a creative character analysis from metrics."""
    client = _get_gemini()
    if not client:
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

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
        return json.loads(text)
    except Exception:
        traceback.print_exc()
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
    all_records: list[dict] = []
    for f in files:
        try:
            raw = await f.read()
            data = json.loads(raw.decode("utf-8"))
            if isinstance(data, list):
                all_records.extend(data)
        except Exception:
            continue

    if not all_records:
        raise HTTPException(status_code=400, detail="Geçerli Spotify JSON dosyası bulunamadı.")

    metrics = analyze(all_records)
    if "error" in metrics:
        raise HTTPException(status_code=400, detail=metrics["error"])

    # Gemini character analysis (non-blocking — skipped if unavailable)
    ai = gemini_character_analysis(metrics)
    if ai:
        metrics["gemini_analysis"] = ai

    # Render dashboard
    template = (TEMPLATES / "dashboard.html").read_text(encoding="utf-8")
    html = template.replace("SPOTIFY_DATA_PLACEHOLDER", json.dumps(metrics, ensure_ascii=False))
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


@app.get("/api/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
