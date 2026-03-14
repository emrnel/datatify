# Personal Digital Data Analysis for Behavioral Insights

**A Project Proposal for Big Data Analytics**

---

**Abstract** — This project analyzes personal digital data exported from multiple platforms to extract behavioral insights, daily routines, and psychological traits. Using a pipeline of Apache Spark for distributed processing, hardcoded mathematical formulas for behavioral scoring, and a Large Language Model (LLM) for narrative generation, the system transforms raw export files into a comprehensive personality profile. A working prototype has been developed that processes over 130,000 Spotify streaming records, computes 25+ behavioral metrics, and generates an interactive premium dashboard with AI-powered character analysis. The project will extend this foundation to incorporate Google and Discord data and scale the pipeline with Big Data technologies.

---

## I. INTRODUCTION

Users generate significant volumes of digital data every day across platforms such as Spotify, YouTube, Google Maps, and Discord. Each platform captures a different dimension of behavior: music consumption patterns reveal attention span and emotional tendencies, location data exposes daily routines, and messaging timestamps indicate social habits. However, this data remains siloed within each application. No existing tool allows a user to combine their exported data into a unified behavioral profile with mathematically grounded metrics and AI-generated insights.

This project addresses this gap by building a full-stack data analytics pipeline that:

1. Ingests raw JSON/CSV export files from multiple platforms.
2. Processes them at scale using Apache Spark (PySpark).
3. Computes 25+ hardcoded behavioral metrics using statistical and information-theoretic formulas.
4. Uses a Large Language Model (Gemini 2.0 Flash) to generate personalized psychological insights.
5. Presents results through an interactive web dashboard with gamification elements.

A working prototype has already been developed for the Spotify data source, demonstrating the feasibility of the approach. The prototype processes 130,000+ streaming records in under 2 seconds and produces a full behavioral profile.

---

## II. LITERATURE REVIEW

**Digital Phenotyping and Behavioral Modeling.** Insel (2017) introduced the concept of "digital phenotyping," using passively collected smartphone data to infer behavioral and psychological traits. Our project extends this idea to voluntarily exported platform data, where richer metadata (e.g., skip behavior, listening duration, offline usage) enables more granular behavioral modeling.

**Music Listening Behavior Analysis.** Spotify's own research team (Anderson et al., 2020) demonstrated that streaming data can reveal listener archetypes based on exploration vs. exploitation patterns. Schedl et al. (2018) provided a comprehensive survey of music information retrieval, establishing metrics such as artist diversity (Shannon entropy) and listening intensity as standard behavioral indicators. Our project builds on these established metrics and adds novel composite scores such as Habit Loop Score and Focus Session Score.

**Big Data Processing Frameworks.** Zaharia et al. (2016) presented Apache Spark as a unified engine for large-scale data processing, demonstrating 10-100x speedups over Hadoop MapReduce. Our pipeline uses PySpark to parallelize metric computation across partitioned user data, enabling horizontal scaling as the user base grows.

**LLMs for Data Interpretation.** Wei et al. (2022) showed that chain-of-thought prompting enables LLMs to perform complex reasoning tasks. Recent work on retrieval-augmented generation (RAG) demonstrates effective use of LLMs for summarizing structured data into natural language narratives (Lewis et al., 2020). Our project uses structured metric summaries as prompts to generate personalized psychological insights via the Gemini API.

**Gamification in Data Visualization.** Deterding et al. (2011) established a framework for gamification in non-game contexts. Our dashboard incorporates badges, levels, archetypes, and radar charts to increase user engagement with their behavioral data.

---

## III. DATA SOURCES

The project uses user-exported data files. No API access or scraping is required — all data comes from platform-provided export features (GDPR Article 15 compliance).

| Platform | Data Type | Format | Key Fields |
|----------|-----------|--------|------------|
| **Spotify** | Extended Streaming History | JSON | `ts`, `ms_played`, `master_metadata_track_name`, `master_metadata_album_artist_name`, `skipped`, `shuffle`, `reason_start`, `reason_end`, `conn_country`, `platform`, `offline`, `incognito_mode` |
| **Google** | YouTube watch history, Maps location history | JSON/HTML | Watch timestamps, video titles, location coordinates, visit durations |
| **Discord** | Message history, activity logs | JSON/CSV | Message timestamps, channel IDs, online status |

**Current Progress:** The Spotify pipeline is fully operational with 9 JSON files containing 131,720 streaming records spanning 2018–2026. Each record contains 20 fields including precise millisecond timestamps, skip indicators, platform identifiers, and country codes.

**Storage Optimization:** Raw JSON files will be converted to Apache Parquet format for columnar storage efficiency, enabling faster analytical queries in PySpark.

---

## IV. METHODOLOGY AND TECHNIQUES

### A. Data Ingestion and Processing Pipeline

The system accepts raw JSON export files via a web upload interface built with FastAPI. Files are parsed, validated, and combined into a unified record set. For distributed processing at scale, we use **Apache PySpark** to:

- Partition data by year/month for parallel processing.
- Compute aggregate statistics (per-artist, per-track, per-hour, per-day distributions).
- Calculate session boundaries using a 30-minute gap heuristic.
- Generate time-series features (monthly novelty rates, yearly growth).

### B. Hardcoded Behavioral Metrics (25+ Formulas)

All metrics are computed using deterministic mathematical formulas — no ML training required. The following categories are implemented:

**Attention & Patience:**
- *Impatience Score* = `skipped_tracks / total_tracks`
- *Completion Rate* = `completed_tracks / total_tracks`
- *Early Skip Rate* = `skips_before_30s / total_skips`
- *Skip Latency Ratio* = `avg_skip_time / avg_track_duration`

**Diversity & Exploration:**
- *Artist Diversity* = Shannon entropy: `H = -Σ p(x) · log₂(p(x))` over artist play distribution
- *Exploration Score* = `unique_tracks / total_plays`
- *Music Novelty Rate* = monthly average of `new_tracks / total_tracks_that_month`

**Habit & Routine:**
- *Circadian Listening Profile* = hourly play distribution (timezone-corrected using `conn_country`)
- *Night Listening Ratio* = `plays_00:00-06:00_local / total_plays`
- *Habit Loop Score* = `repeated_consecutive_pairs / total_pairs`
- *Listening Intensity* = `total_hours / calendar_days`
- *Yearly Growth* = `(hours_year_n - hours_year_n-1) / hours_year_n-1`

**Loyalty & Focus:**
- *Artist Loyalty Score* = `plays_top10_artists / total_plays`
- *Song Replay Density* = `plays_top_song / total_plays`
- *Focus Session Score* = `long_uninterrupted_plays / total_plays`
- *Listening Fragmentation Index* = `total_skips / total_sessions`

**Platform & Context:**
- *Mobile Usage Ratio* = `mobile_playtime / total_playtime`
- *Offline Ratio* = `offline_hours / total_hours`
- *Shuffle Ratio* = `shuffle_plays / total_plays`

### C. Gamification Engine

The system includes a rule-based gamification layer:

- **15 Badges** with specific unlock conditions (e.g., "Night Owl" for >25% night listening, "Centurion" for 100K+ plays, "Explorer" for 1000+ unique artists).
- **10-Level Progression System** based on total listening hours (50h → Casual, up to 5000h → Transcendent).
- **Archetype Classification** using rule-based decision trees over computed metrics (e.g., "The Night Diver," "The Restless Explorer," "The Loyal Guardian").
- **Radar Chart Profile** with 6 normalized dimensions: Patience, Exploration, Loyalty, Focus, Diversity, Night Owl.

### D. LLM Integration (Gemini 2.0 Flash)

The system sends a structured JSON summary of all computed metrics, top artists/songs, earned badges, and radar scores to the **Google Gemini 2.0 Flash** API. The LLM generates:

1. A creative character title (e.g., "Gecenin Sessiz Filozofu").
2. A 4-5 sentence psychological character analysis.
3. Four personality traits derived from listening data.
4. Three data-supported surprising insights.
5. A creative prediction about the user's personality.

The Gemini call operates with a **20-second timeout** and graceful fallback — the dashboard renders fully even without AI output.

### E. Benchmark and Comparison System

An anonymous benchmarking API allows users to compare their metrics against other users:

- Metrics are submitted anonymously (no track/artist data shared).
- Percentile rankings are computed across all submissions.
- SQLite stores anonymous metric vectors for percentile calculation.

### F. Web Dashboard

The frontend is a single-page premium dashboard with:
- Animated background effects and glassmorphism design.
- Interactive Chart.js visualizations (circadian, weekday, yearly trends, growth).
- SVG gauge rings for behavioral scores.
- Scroll-triggered animations and animated counters.
- Fully responsive design (mobile/tablet/desktop).

---

## V. EVALUATION

Success will be measured across three dimensions:

| Criterion | Method | Target |
|-----------|--------|--------|
| **Formula Accuracy** | Unit tests comparing PySpark metric outputs against manually verified samples from raw JSON | 100% match on all 25+ metrics |
| **Insight Quality** | Blind evaluation of LLM-generated character analyses by 5+ participants rating relevance (1-5 scale) | Average score ≥ 3.5/5 |
| **System Performance** | Benchmark processing time for datasets of 50K, 100K, 500K, and 1M records using PySpark | Sub-10s for 500K records |
| **User Experience** | Usability testing with 5+ participants measuring task completion and satisfaction | >80% task completion rate |

---

## VI. DELIVERABLES

1. **Working Web Application** — A deployed web dashboard (Railway) where users upload their Spotify JSON files and receive a full behavioral analysis with AI insights.
2. **PySpark Processing Pipeline** — Spark scripts for distributed metric computation, partitioned by time period.
3. **Technical Report** — Documenting the Big Data architecture, all mathematical formulas, LLM prompt engineering, and evaluation results.
4. **Source Code Repository** — Complete codebase on GitHub (`github.com/emrnel/datatify`) with documentation.
5. **Presentation** — Final presentation demonstrating the system with live data.

---

## VII. CURRENT PROGRESS

| Component | Status |
|-----------|--------|
| Spotify data ingestion (9 files, 131K records) | ✅ Complete |
| 25+ behavioral metric computation | ✅ Complete |
| Gamification (badges, levels, archetypes, radar) | ✅ Complete |
| Gemini LLM character analysis | ✅ Complete |
| Premium web dashboard (HTML/CSS/JS + Chart.js) | ✅ Complete |
| FastAPI backend with file upload | ✅ Complete |
| Anonymous benchmark API | ✅ Complete |
| Railway deployment | ✅ Complete |
| Migration to PySpark | 🔄 In Progress |
| Google data integration | 📋 Planned |
| Discord data integration | 📋 Planned |

---

## VIII. REFERENCES

1. Anderson, A., et al. (2020). "Algorithmic Effects on the Diversity of Consumption on Spotify." *Proceedings of The Web Conference (WWW)*.
2. Deterding, S., et al. (2011). "From Game Design Elements to Gamefulness: Defining Gamification." *Proceedings of MindTrek*.
3. Insel, T. R. (2017). "Digital Phenotyping: Technology for a New Science of Behavior." *JAMA*, 318(13), 1215–1216.
4. Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *NeurIPS*.
5. Schedl, M., et al. (2018). "Current Challenges and Visions in Music Recommender Systems Research." *International Journal of Multimedia Information Retrieval*, 7(2), 95–116.
6. Wei, J., et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." *NeurIPS*.
7. Zaharia, M., et al. (2016). "Apache Spark: A Unified Engine for Big Data Processing." *Communications of the ACM*, 59(11), 56–65.
8. Google. (2024). "Gemini API Documentation." https://ai.google.dev/docs
9. Apache Spark. (2024). "PySpark Documentation." https://spark.apache.org/docs/latest/api/python/
