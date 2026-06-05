# -*- coding: utf-8 -*-
"""Gemini AI character analysis: client init, RPM throttle, retry/fallback engine."""
import json
import os
import random
import re
import threading
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

GEMINI_TIMEOUT = 25
GEMINI_MAX_ATTEMPTS = 2
GEMINI_BUDGET = 60
GEMINI_RPM_LIMIT = 4
GEMINI_RPM_WINDOW = 60

GEMINI_MODELS = (
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-flash-latest",
)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
SKIP_GEMINI = os.environ.get("SKIP_GEMINI", "").strip().lower() in ("1", "true", "yes")

_client = None
_rpm_lock = threading.Lock()
_rpm_window: deque = deque()
_RETRY_AFTER_RE = re.compile(r"retry[-_ ]?after[^\d]*(\d+)", re.IGNORECASE)


def _get_client():
    global _client
    if _client is not None:
        return _client
    if not GEMINI_API_KEY:
        return None
    try:
        from google import genai
        _client = genai.Client(api_key=GEMINI_API_KEY)
        return _client
    except Exception:
        traceback.print_exc()
        return None


def _throttle_rpm(deadline: float) -> bool:
    """Reserve a slot in the sliding RPM window. Returns False if wait would exceed deadline."""
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
    """Return (status_code, retry_after_s, kind). kind ∈ {'rate_limit','server','client','unknown'}."""
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
    """Blocking generate_content call. Requests application/json directly."""
    config = {"response_mime_type": "application/json"}
    try:
        response = client.models.generate_content(model=model, contents=prompt, config=config)
    except TypeError:
        response = client.models.generate_content(model=model, contents=prompt)

    text = (response.text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        cleaned = text
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
        if not cleaned.startswith("{"):
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end > start:
                cleaned = cleaned[start:end + 1]
        return json.loads(cleaned)


def _try_once(client, model: str, prompt: str, attempt_no: int):
    """Run one timed attempt. Returns (result, error_tuple); exactly one is non-None."""
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


def _build_prompt(metrics: dict) -> str:
    M = metrics["metrikler"]
    R = metrics["bizim_rapor"]
    summary = {
        "toplam_saat":          R["toplam_saat"],
        "toplam_kayit":         R["toplam_kayit"],
        "seviye":               metrics["level"]["title"],
        "arketip":              metrics["archetype"]["name"],
        "sabırsızlık":          M["impatience_score_pct"],
        "tamamlama":            M["completion_rate_pct"],
        "keşif_skoru":          M["exploration_score"],
        "sanatçı_çeşitliliği":  M["artist_diversity_entropy"],
        "gece_dinleme":         M["night_listening_ratio_pct"],
        "odak_skoru":           M["focus_session_score_pct"],
        "sadakat":              M["artist_loyalty_score_pct"],
        "alışkanlık":           M["habit_loop_score_pct"],
        "günlük_yoğunluk_saat": M["listening_intensity_h_per_day"],
        "mobil_oran":           M["mobile_usage_ratio_pct"],
        "shuffle_oranı":        R["shuffle_orani_pct"],
        "yeni_parça_oranı":     M["music_novelty_rate_pct"],
        "erken_atlama":         M["early_skip_rate_pct"],
        "top5_sanatçı":         [a["sanatci"] for a in metrics["top_sanatcilar"][:5]],
        "top5_şarkı":           [s["sarki"] for s in metrics["top_sarkilar"][:5]],
        "rozetler":             [b["name"] for b in metrics["badges"] if b["earned"]],
        "radar":                metrics["radar"],
    }
    return f"""Sen yaratıcı bir müzik psikoloğu ve karakter analistisin.
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


def analyze_character(metrics: dict) -> dict | None:
    """Call Gemini for AI character analysis. Returns None on any failure."""
    if SKIP_GEMINI:
        print("[GEMINI] SKIP_GEMINI=1 — not calling API")
        return None
    client = _get_client()
    if not client:
        print("[GEMINI] No API key or client init failed — skipping")
        return None

    prompt = _build_prompt(metrics)
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

            if kind == "client" and code != 429:
                print(f"[GEMINI] non-retryable client error (http={code}) on {model} — switching model")
                break

            if retry_idx == GEMINI_MAX_ATTEMPTS - 1:
                print(f"[GEMINI] {GEMINI_MAX_ATTEMPTS} attempts exhausted on {model} "
                      f"(last kind={kind}, http={code}) — switching model")
                break

            if retry_after is not None:
                delay = retry_after + random.uniform(0, 1.0)
                reason = f"server-supplied Retry-After={retry_after}s"
            elif kind == "rate_limit":
                base = 15 * (2 ** retry_idx)
                delay = base + random.uniform(0, 5)
                reason = f"429 backoff base={base}s+jitter"
            elif kind in ("server", "timeout"):
                base = 2 ** retry_idx
                delay = base + random.uniform(0, base)
                reason = f"{kind} backoff base={base}s+jitter"
            else:
                base = 2 ** retry_idx
                delay = base + random.uniform(0, base)
                reason = f"unknown-error backoff base={base}s+jitter"

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
