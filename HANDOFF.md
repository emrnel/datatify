# Datatify — Proje Handoff Dokümanı

> Bu doküman, önceki sohbetten aktarılan tüm bağlam bilgilerini içerir.
> Yeni sohbete bu dosyayı referans vererek devam et.

---

## 1. Proje Özeti

**Datatify** — Spotify Extended Streaming History verilerini analiz edip, davranışsal içgörüler, gamification, graf analizi ve AI karakter profili sunan bir web uygulaması.

- **Ders:** CSE 458 — Big Data Analytics
- **Takım:** Soner Güneş, Mehmet Alp Atay, Emre İlhan Şenel, H. Muhammet Çengelci
- **Repo:** https://github.com/emrnel/datatify.git
- **Deploy:** https://web-production-65055.up.railway.app/ (Railway)
- **Proje dizini:** `d:\Downloads\datatify`
- **Ham veri dizini:** `d:\Downloads\my_spotify_data\Spotify Extended Streaming History` (9 JSON, ~131K kayıt, 2018–2026)

---

## 2. Proje Dosya Yapısı

```
datatify/
├── analyzer.py           # Ana analiz motoru (25+ metrik, badge, level, archetype)
├── main.py               # FastAPI backend (upload, Gemini, benchmark API)
├── requirements.txt      # fastapi, uvicorn[standard], python-multipart, google-genai
├── Procfile              # web: uvicorn main:app --host 0.0.0.0 --port $PORT
├── railway.json          # Railway deploy config
├── .gitignore
├── benchmark.db          # SQLite — anonim benchmark verileri
├── PROJECT_PROPOSAL_V5.md
├── templates/
│   ├── index.html        # Landing page (dosya yükleme)
│   └── dashboard.html    # Premium animated dashboard
└── static/               # (boş, gelecekte statik asset'ler için)
```

---

## 3. Mevcut Tech Stack

| Katman | Teknoloji |
|--------|-----------|
| Backend | FastAPI + Uvicorn |
| Analiz | Pure Python (collections, datetime, math) |
| AI | Gemini 2.0 Flash (`google-genai`) |
| Frontend | Vanilla HTML/CSS/JS, Chart.js, SVG |
| Veritabanı | SQLite (benchmark) |
| Deploy | Railway |

---

## 4. Git Durumu (5 commit, main branch, push edilmiş)

```
0768803 feat: add SKIP_GEMINI env var to skip Gemini API call when quota exhausted
bf1b67a feat: add styled placeholder when Gemini AI analysis is unavailable
3b2c9a4 fix: unicode encode error in logging on Windows
e943b0f fix: add Gemini timeout, request logging, client-side timeout, clearer upload instructions
dbac9e7 feat: Datatify - Spotify Listening DNA web app with FastAPI, Gemini AI, premium dashboard, and Railway deployment
```

---

## 5. Hocaya Gönderilen Proposal (PDF) — Önemli Taahhütler

PDF dosyası: `d:\Downloads\my_spotify_data\Spotify Extended Streaming History\CSE458_Project_Proposal.pdf` (9 sayfa)

### Proposal'da vaat edilen bileşenler ve mevcut durum:

| # | Vaat | Durum | Not |
|---|------|-------|-----|
| 1 | Pandas/Polars ile veri işleme | ⚠️ Kısmen | Şu an pure Python collections; Pandas'a geçiş kolay |
| 2 | 25+ behavioral metrics | ✅ Tamam | analyzer.py'da tamamı var |
| 3 | PageRank + Community Detection (NetworkX) | ❌ Yok | **Proposal'da söz verildi, yapılması şart** |
| 4 | K-Means clustering (scikit-learn) | ❌ Yok | **Proposal'da söz verildi, yapılması şart** |
| 5 | Gemini LLM entegrasyonu | ✅ Tamam | Rate limit sorunu var, placeholder UI mevcut |
| 6 | Web dashboard | ✅ Tamam | Chart.js, SVG gauge, animasyonlar |
| 7 | Cloud deploy (GCP/AWS) | ⚠️ Kısmen | Railway'de, proposal GCP/AWS diyor |
| 8 | Parquet dönüşümü | ❌ Yok | Proposal'da bahsedildi |
| 9 | Ölçek testi (130K → 5M) | ❌ Yok | Sentetik veri ile benchmark gerekli |
| 10 | Gamification (badge, level, archetype) | ✅ Tamam | 15 badge, 10 level |
| 11 | Graf görselleştirmesi (dashboard'da) | ❌ Yok | Proposal'da bahsedildi |
| 12 | Benchmark karşılaştırma | ✅ Tamam | SQLite + API endpoint'leri mevcut |

### PDF'deki tutarsızlık (farkında ol):
- **Methods** bölümünde "Pandas/Polars, NetworkX, scikit-learn" deniyor
- **Deliverables** bölümünde hâlâ "PySpark processing pipeline" yazıyor
- Bu tutarsızlık düzeltilmeli (ya hep PySpark ya hep Pandas/NetworkX)

---

## 6. Ders Gereksinimleri — Midterm (Hafta 9)

Midterm 4 bileşen istiyor:

1. **Revised Proposal (PP+):** Hoca geri bildirimi gelince güncellenmeli
2. **System Architecture & Design:** Pipeline blok diyagramı gerekli (henüz yok)
3. **Implementation Status ("Proof of Life"):** Güçlü durumda — çalışan uygulama var
4. **Preliminary Results:** Mevcut metrikler gösterilebilir, graf/clustering sonuçları eklenirse çok güçlü

---

## 7. Midterm'e Kadar Yapılması Gerekenler (Öncelik Sırasıyla)

| Öncelik | Görev | Detay | Süre |
|---------|-------|-------|------|
| **1** | NetworkX graf analizi | PageRank + Label Propagation (artist geçiş grafı) — **proposal'da söz verildi** | 1-2 gün |
| **2** | Graf görselleştirmesi | Dashboard'a artist community graph ekle | 1 gün |
| **3** | scikit-learn K-Means clustering | Benchmark verileri üzerinden kullanıcı kümeleme | 1 gün |
| **4** | Mimari diyagramı | Pipeline blok diyagramı (midterm zorunlu bileşen) | Birkaç saat |
| **5** | Cloud'a taşıma (GCP/AWS) | Proposal'da söz verildi, Railway yerine | 1 gün |
| **6** | Ölçek testi | Sentetik veri üretimi (500K-1M-5M) + benchmark | Birkaç saat |
| **7** | Pandas'a geçiş | Pure Python → Pandas DataFrame (proposal tutarlılığı) | Birkaç saat |

---

## 8. Mevcut Bilinen Sorunlar ve Çözümleri

### Gemini API Rate Limit
- Günde 20 istek hakkı var
- `SKIP_GEMINI=1` env var ile bypass edilebilir
- Dashboard'da Gemini yoksa placeholder kart gösteriliyor
- `GEMINI_TIMEOUT = 20` saniye server-side timeout uygulanıyor

### Railway Deploy
- Env var: `GEMINI_API_KEY` Railway'de tanımlı
- Env var: `SKIP_GEMINI` opsiyonel
- Port: `$PORT` env var'dan alınıyor (Procfile'da)
- Deploy komutu: `git push origin main` sonrası Railway otomatik build eder

### Windows Konsol Encoding
- `print()` içinde Unicode karakterler (`→` gibi) `UnicodeEncodeError` veriyor
- Çözüm: ASCII-safe karakterler kullan (`->` gibi)

### Client-Side Timeout
- `index.html`'de 120 saniye `AbortController` timeout var
- Büyük dosyalarda zaman aşımı mesajı gösterilir

---

## 9. Analyzer.py — Hesaplanan Metrikler

### Temel Metrikler
- `total_plays`, `total_hours`, `unique_tracks`, `unique_artists`
- `top_artists` (top 15), `top_songs` (top 15), `top_albums` (top 10)

### Davranışsal Metrikler (25+)
- `impatience_score` = skipped / total
- `completion_rate` = completed / total
- `early_skip_rate` = skips < 30s / total skips
- `artist_diversity_entropy` = Shannon entropy
- `exploration_score` = unique tracks / total plays
- `listening_intensity` = hours / calendar days
- `yearly_growth_pct` = yıllık değişim
- `night_listening_ratio_pct` = 00:00-06:00 local / total (timezone: conn_country)
- `mobile_usage_ratio_pct` = mobile playtime / total
- `album_listening_ratio_pct` = album tracks / total
- `focus_session_score` = long uninterrupted / total
- `music_novelty_rate` = monthly new tracks avg
- `artist_loyalty_score` = top 10 artists plays / total
- `song_replay_density` = top song plays / total
- `listening_fragmentation` = skips / sessions
- `habit_loop_score` = repeated consecutive pairs / total pairs

### Gamification
- **15 badge** (Night Owl, Marathon Runner, Centurion, vb.) — her birinin koşulu var
- **10 level** (Newbie → Legendary) — toplam dinleme saatine göre
- **Archetype** (Explorer, Loyalist, Night Owl, Casual, Balanced) — rule-based

### Gece Kuşu Badge Detayı
- Koşul: `night_listening_ratio_pct > 25`
- "Gece" = yerel saat 00:00–06:00
- Timezone `conn_country`'den çıkarılıyor (TR → UTC+3)
- Örnek: UTC 21:00 → TR yerel 00:00 = gece sayılır

---

## 10. Main.py — API Endpoint'leri

| Endpoint | Method | Açıklama |
|----------|--------|----------|
| `/` | GET | Landing page (index.html) |
| `/analyze` | POST | JSON dosya upload → analiz → dashboard render |
| `/api/submit` | POST | Anonim benchmark verisi gönder |
| `/api/percentiles` | GET | Percentile hesapla |
| `/api/stats` | GET | Genel istatistikler |

---

## 11. Dashboard Özellikleri

- **Hero:** Dinamik başlık
- **Profile:** Archetype, AI Analysis (Gemini veya placeholder), Level, Radar chart (6 boyut)
- **Badges:** 15 badge kartı (kazanılan/kazanılmayan)
- **Overview:** Temel istatistikler (animated counter)
- **Behavioral Metrics:** SVG gauge rings
- **Time Analysis:** Chart.js (circadian, weekday, yearly trends)
- **Top Artists/Songs:** Liste
- **Benchmark:** Diğer kullanıcılarla karşılaştırma

### UI Özellikleri
- Dark theme, glassmorphism
- CSS animasyonları: orb glow, fade-up, spin, shimmer, pulse
- Scroll-triggered reveals (IntersectionObserver)
- Responsive design

---

## 12. Proposal'daki Formül ve Algoritma Detayları (PDF'den)

### Graf Analizi (yapılacak)
- **Nodes:** Sanatçılar
- **Edges:** Aynı session'da A'dan sonra B dinlendiğinde A→B (weight = occurrence count)
- **PageRank:** En "merkezi" sanatçıları bul
- **Label Propagation:** Co-listened artist cluster'ları keşfet
- **Connected Components:** İzole dinleme adaları

### Clustering (yapılacak)
- **15 boyutlu vektör:** impatience, completion, exploration, diversity, early skip, intensity, night ratio, mobile, focus, novelty, loyalty, replay density, fragmentation, habit loop, shuffle ratio
- **K-Means** (scikit-learn) ile doğal gruplamalar
- **Elbow method + silhouette score** ile optimal k

### Evaluation Kriterleri (proposal'dan)
| Kriter | Hedef |
|--------|-------|
| Formül doğruluğu | 100% match |
| Graf analizi geçerliliği | Pattern'larla uyumlu community'ler |
| Clustering kalitesi | Silhouette > 0.3 |
| Ölçeklenebilirlik | < 30s for 5M records |
| Insight kalitesi | ≥ 3.5/5 (5+ katılımcı) |
| Kullanıcı deneyimi | > 80% task completion |

---

## 13. Önceki Sohbetin Transcript'i

Tam konuşma geçmişi:
`C:\Users\Emre\.cursor\projects\d-Downloads-my-spotify-data-Spotify-Extended-Streaming-History/agent-transcripts/3b2574d3-5474-455f-aee5-a0f5ff359e85/3b2574d3-5474-455f-aee5-a0f5ff359e85.jsonl`

---

## 14. Hızlı Başlangıç (Yeni Sohbet İçin)

```powershell
# Lokalde çalıştırmak için:
cd d:\Downloads\datatify
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000

# Deploy için:
git add . && git commit -m "message" && git push origin main
# Railway otomatik build eder
```

### Railway Env Vars
- `GEMINI_API_KEY` = (Gemini API anahtarı)
- `SKIP_GEMINI` = "1" (opsiyonel, Gemini'yi atlamak için)

---

## 15. Sonraki Adımlar Özet

1. **NetworkX graf analizi** → `analyzer.py`'a ekle, dashboard'a görselleştir
2. **scikit-learn clustering** → benchmark verileri üzerinden
3. **Mimari diyagramı** → midterm sunumu için
4. **Cloud taşıma** → GCP Cloud Run veya AWS App Runner
5. **Ölçek testi** → sentetik veri + benchmark
6. **Proposal tutarsızlığı** → PySpark vs Pandas/NetworkX netleştir
