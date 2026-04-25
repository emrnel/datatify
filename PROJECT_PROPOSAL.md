# Personal Digital Data Analysis for Behavioral Insights
## CSE 458 — Big Data Analytics — Project Proposal

**Team Members:** Soner Güneş · Mehmet Alp Atay · Emre İlhan Şenel · H. Muhammet Çengelci

---

## Abstract

This proposal describes a semester project that will analyze personal digital data exported from Spotify to extract behavioral insights using Big Data technologies. The semester work will focus on migrating the pipeline to PySpark, deploying on cloud infrastructure, and implementing graph and clustering analyses. As a future extension, the system could incorporate additional data sources such as Google Takeout exports to build a cross-platform behavioral profile.

---

## I. Introduction

### Problem

Users generate significant volumes of digital data on streaming platforms such as Spotify. A single user's Extended Streaming History can contain over 100,000 records spanning multiple years, with rich metadata including timestamps, skip behavior, device information, and geographic context. However, there is no tool that processes this data efficiently to produce a comprehensive behavioral profile with mathematically grounded metrics, graph-based relationship analysis, and AI-generated psychological insights.

The core data analysis challenge is twofold: (1) processing and analyzing structured streaming data efficiently using modern Python libraries, and (2) applying graph algorithms and clustering techniques to discover non-obvious patterns in listening behavior.

### Objectives

We propose to build a data analytics pipeline that will:

1. Process data efficiently using Python libraries (e.g., Pandas or Polars) on a standard cloud server.
2. Compute 25+ behavioral metrics using deterministic formulas (entropy, ratios, time-series).
3. Build a listening transition graph and apply PageRank and community detection using Spark GraphFrames to identify central artists and listening communities.
4. Cluster users by behavioral metric vectors using Spark MLlib (K-Means) to discover listener archetypes from data.
5. Use an LLM (Gemini 2.0 Flash) to generate personalized narrative insights from the computed metrics.
6. Present results through an interactive web dashboard deployed on cloud infrastructure.

---

## II. Literature Review

**Digital phenotyping and behavioral modeling.** Insel (2017) introduced "digital phenotyping" — using passively collected digital data to infer behavioral and psychological traits. Our project applies this concept to voluntarily exported platform data, where rich metadata (skip behavior, listening duration, offline usage) enables fine-grained behavioral modeling.

**Music listening behavior analysis.** Anderson et al. (2020) demonstrated that streaming data can reveal listener archetypes based on exploration vs. exploitation patterns. Schedl et al. (2018) provided a comprehensive survey of music information retrieval, establishing metrics such as artist diversity (Shannon entropy) and listening intensity as standard behavioral indicators. We build on these established metrics and add novel composite scores such as Habit Loop Score and Focus Session Score.

**Graph-based analysis of listening behavior.** Page et al. (1999) introduced PageRank for measuring node importance in directed graphs. We will apply PageRank to a listening transition graph (artist-to-artist edges weighted by consecutive play frequency) to identify the most "central" artists in a user's listening flow. Girvan and Newman (2002) introduced community detection via edge betweenness; we will use label propagation on the same graph to discover clusters of co-listened artists.

**Clustering for user segmentation.** Arthur and Vassilvitskii (2007) introduced k-means++ initialization for efficient clustering. We will use scikit-learn's K-Means implementation to cluster users based on their behavioral metric vectors, discovering data-driven listener archetypes across multiple users.

**LLMs for data interpretation.** Wei et al. (2022) demonstrated chain-of-thought prompting for reasoning tasks. We will use structured metric summaries as prompts to generate personalized psychological narratives via the Gemini API.

**Gamification in data visualization.** Deterding et al. (2011) established a framework for gamification in non-game contexts. Our dashboard will include badges, levels, and archetypes to increase user engagement with behavioral data.

---

## III. Data

### What data we will use

The primary data source is Spotify Extended Streaming History, exported by users via Spotify's account settings (GDPR Article 15 compliance). No API access or scraping is required.

| Field | Description | Use |
|-------|-------------|-----|
| `ts` | Timestamp when track stopped playing (UTC) | Temporal analysis, circadian profile, sessions |
| `ms_played` | Milliseconds the stream was played | Duration metrics, skip detection |
| `master_metadata_track_name` | Track name | Diversity, novelty, replay metrics |
| `master_metadata_album_artist_name` | Artist name | Artist graph, loyalty, entropy |
| `master_metadata_album_album_name` | Album name | Album listening ratio |
| `skipped` | Whether user skipped the track | Impatience, early skip, fragmentation |
| `shuffle` | Whether shuffle mode was active | Shuffle ratio |
| `reason_start` / `reason_end` | Why track started/ended | Completion rate, behavior patterns |
| `conn_country` | Country code during playback | Timezone correction, geographic analysis |
| `platform` | Device/app used | Mobile vs. desktop ratio |
| `offline` | Whether played offline | Offline usage ratio |
| `incognito_mode` | Whether in private session | Incognito behavior |

For development we have obtained sample Spotify exports (9 JSON files, ~131K records, spanning 2018–2026). For scalability testing, we will generate synthetic datasets of 500K, 1M, and 5M records to benchmark PySpark performance on cloud clusters.

### Future extension

As a potential extension beyond the semester scope, the system could incorporate additional data sources such as Google Takeout (YouTube watch history, Maps location data) to build a cross-platform behavioral profile. The pipeline architecture is designed to accommodate multiple data sources through a common ingestion and metric computation interface.

---

## IV. Methods

### A. Cloud Infrastructure

The project will be deployed on a public cloud platform (GCP or AWS):

- **Processing:** GCP Dataproc or AWS EMR for managed Spark clusters.
- **Storage:** GCS or S3 for Parquet data files.
- **Web application:** GCP Cloud Run or AWS App Runner for the FastAPI dashboard.
- **Database:** Cloud-hosted PostgreSQL or SQLite for anonymous benchmark data.

### B. Data Processing Pipeline

The metric computation pipeline will be implemented using Python data manipulation libraries (e.g., Pandas or Polars):

- **Ingestion:** Read JSON exports, validate schema, convert to Parquet, partition by year/month.
- **Aggregation:** GroupBy operations for per-artist, per-track, per-hour, per-day, per-platform distributions using Pandas/Polars transformations.
- **Session detection:** Window functions to identify listening sessions based on a 30-minute inactivity gap.
- **Time-series features:** Monthly novelty rates, yearly growth, circadian profiles with timezone correction derived from `conn_country`.

Scalability will be tested by processing synthetic datasets of increasing size (130K → 500K → 1M → 5M records) and measuring execution time.

### C. Behavioral Metrics (25+ Formulas)

All metrics are computed using deterministic mathematical formulas — no ML training required:

**Attention and patience**
- Impatience Score = `skipped_tracks / total_tracks`
- Completion Rate = `completed_tracks / total_tracks`
- Early Skip Rate = `skips_before_30s / total_skips`
- Skip Latency Ratio = `avg_skip_time / avg_track_duration`

**Diversity and exploration**
- Artist Diversity = H = −∑ p(x) log₂ p(x) over artist play distribution
- Exploration Score = `unique_tracks / total_plays`
- Music Novelty Rate = monthly avg of `new_tracks / total_tracks_that_month`

**Habit and routine**
- Circadian Listening Profile = hourly play distribution (timezone-corrected)
- Night Listening Ratio = `plays_00:00–06:00_local / total_plays`
- Habit Loop Score = `repeated_consecutive_track_pairs / total_pairs`
- Listening Intensity = `total_hours / calendar_days`
- Yearly Growth = `(hours_year_n – hours_year_n−1) / hours_year_n−1`

**Loyalty and focus**
- Artist Loyalty Score = `plays_top10_artists / total_plays`
- Song Replay Density = `plays_top_song / total_plays`
- Focus Session Score = `long_uninterrupted_plays / total_plays`
- Listening Fragmentation Index = `total_skips / total_sessions`

**Platform and context**
- Mobile Usage Ratio = `mobile_playtime / total_playtime`
- Offline Ratio = `offline_hours / total_hours`
- Shuffle Ratio = `shuffle_plays / total_plays`

### D. Graph Analysis

We will construct a listening transition graph from the streaming data:

- **Nodes:** Artists (each artist is a vertex).
- **Edges:** Directed edge from artist A to artist B when B is played immediately after A in the same session. Edge weight = number of occurrences.

Algorithms to apply:

- **PageRank:** Identify the most "central" artists in the user's listening flow — artists that act as hubs connecting different listening patterns. This provides a different ranking than simple play count.
- **Label Propagation (Community Detection):** Discover clusters of artists that are frequently listened to together, revealing the user's distinct "music communities" (e.g., a workout playlist cluster vs. a late-night ambient cluster).
- **Connected Components:** Identify isolated listening islands — groups of artists the user only listens to in specific contexts.

This analysis will be performed using NetworkX, which integrates seamlessly with Python data structures.

### E. User Clustering (K-Means)

When multiple users submit their data through the web dashboard, we will collect anonymous behavioral metric vectors (15 dimensions: impatience, completion rate, exploration score, diversity entropy, early skip rate, listening intensity, night listening ratio, mobile usage, focus score, novelty rate, loyalty, replay density, fragmentation, habit loop, shuffle ratio).

- **K-Means clustering (scikit-learn)** will be applied to discover natural groupings of listeners.
- Cluster characteristics will be analyzed to produce data-driven archetype labels (e.g., "Night Explorers," "Loyal Repeaters," "Impatient Skippers").
- The optimal number of clusters will be determined using the elbow method and silhouette scores.

### F. LLM Integration (Gemini 2.0 Flash)

The system will send a structured JSON summary of computed metrics, graph analysis results, top artists/songs, and earned badges to the Gemini 2.0 Flash API. The LLM will generate:

- A creative character title.
- A 4–5 sentence psychological character analysis.
- Personality traits derived from listening data.
- Data-supported surprising insights.
- A creative prediction about the user's personality.

The Gemini call will operate with a strict timeout and graceful fallback — the dashboard renders fully even without AI output.

### G. Gamification

A rule-based gamification layer will provide:

- 15 badges with specific unlock conditions (e.g., "Night Owl" for >25% night listening, "Centurion" for 100K+ plays).
- 10-level progression based on total listening hours.
- Archetype classification using rule-based decision trees over computed metrics.
- Radar chart profile with 6 normalized dimensions.

### H. Web Dashboard

The frontend will be a single-page interactive dashboard with:

- Chart.js visualizations (circadian, weekday, yearly trends, growth charts).
- SVG gauge rings for behavioral scores.
- Graph visualization of artist communities.
- Animated counters, scroll-triggered reveals, responsive design.
- Benchmark comparison against other anonymous users.

---

## V. Evaluation

| Criterion | Method | Target |
|-----------|--------|--------|
| Formula accuracy | Unit tests: Python metric outputs vs. manually verified samples from raw JSON | 100% match on all 25+ metrics |
| Graph analysis validity | Verify PageRank rankings and communities against manual inspection of top-artist listening sequences | Communities align with identifiable listening patterns |
| Clustering quality | Silhouette score and elbow method for K-Means on multi-user metric vectors | Silhouette score > 0.3 |
| Scalability | Processing time for 130K, 500K, 1M, 5M records on a standard cloud server | Sub-linear scaling; < 30s for 5M records |
| Insight quality | Blind evaluation of LLM narratives by 5+ participants (1–5 scale) | Average ≥ 3.5/5 |
| User experience | Usability test: task completion and satisfaction with 5+ participants | > 80% task completion |

---

## VI. Deliverables

By the end of the quarter we expect to submit:

1. **Working web application** — Cloud-deployed dashboard (GCP/AWS) where users upload Spotify JSON files and receive behavioral analysis with graph insights, clustering results, and LLM narratives.
2. **PySpark processing pipeline** — Spark scripts for distributed metric computation, graph analysis (GraphFrames), and user clustering (MLlib), running on a managed cloud cluster.
3. **Scalability report** — Benchmark results showing processing times across dataset sizes (130K to 5M records) on different cluster configurations.
4. **Technical report** — Conference-format paper documenting the Big Data architecture, mathematical formulas, graph algorithms, clustering methodology, LLM integration, and evaluation results.
5. **Source code repository** — GitHub with documentation and cloud deployment instructions.
6. **Final presentation** — Live demonstration of the system with real data.

---

## VII. References

1. Anderson, A., et al. (2020). "Algorithmic Effects on the Diversity of Consumption on Spotify." *Proceedings of The Web Conference (WWW).*
2. Arthur, D. and Vassilvitskii, S. (2007). "k-means++: The Advantages of Careful Seeding." *Proceedings of SODA.*
3. Deterding, S., et al. (2011). "From Game Design Elements to Gamefulness: Defining Gamification." *Proceedings of MindTrek.*
4. Girvan, M. and Newman, M.E.J. (2002). "Community Structure in Social and Biological Networks." *Proceedings of the National Academy of Sciences, 99*(12), 7821–7826.
5. Insel, T. R. (2017). "Digital Phenotyping: Technology for a New Science of Behavior." *JAMA, 318*(13), 1215–1216.
6. Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *NeurIPS.*
7. Page, L., et al. (1999). "The PageRank Citation Ranking: Bringing Order to the Web." Stanford InfoLab Technical Report.
8. Schedl, M., et al. (2018). "Current Challenges and Visions in Music Recommender Systems Research." *International Journal of Multimedia Information Retrieval, 7*(2), 95–116.
9. Wei, J., et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." *NeurIPS.*
