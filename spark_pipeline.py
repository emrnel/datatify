# -*- coding: utf-8 -*-
"""
PySpark Processing Pipeline (Deliverable §VI.2 of proposal).

Distributed implementation of the same metric pipeline as ``data_pipeline.py``
plus Spark MLlib K-Means clustering. Designed to run on a managed Spark
cluster (GCP Dataproc / AWS EMR) but also works locally for development /
benchmark via Spark's built-in local mode (``master=local[*]``).

Modules
-------
    * ``spark_session(...)``         create a configured SparkSession
    * ``ingest_to_parquet(...)``     JSON exports → Parquet (year/month)
    * ``compute_metrics_spark(...)`` distributed groupBy aggregations
    * ``build_transition_edges(...)``artist→artist edges (used by GraphFrames)
    * ``cluster_users_spark(...)``   Spark MLlib K-Means on metric vectors

GraphFrames (PageRank + Label Propagation) is loaded conditionally — it
requires the JAR package to be supplied via ``--packages graphframes:...``
when launching ``spark-submit``. ``build_transition_edges`` is always
available so the same edge DataFrame can be analyzed by NetworkX locally
when GraphFrames is not installed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable

# Windows + PySpark: workers must use the *same* interpreter as the driver
# or Spark prints "Python not found" and the Python UDF/Arrow path fails.
if "PYSPARK_PYTHON" not in os.environ:
    os.environ["PYSPARK_PYTHON"] = sys.executable
if "PYSPARK_DRIVER_PYTHON" not in os.environ:
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

GRAPHFRAMES_PACKAGE = "graphframes:graphframes:0.8.3-spark3.5-s_2.12"


def spark_session(
    app_name: str = "datatify",
    master: str | None = None,
    with_graphframes: bool = False,
    local_threads: int | None = None,
):
    """Create (or reuse) a SparkSession.

    The function imports PySpark lazily so that the FastAPI app does not pull
    Spark into memory when it is only running the Pandas pipeline.
    """
    from pyspark.sql import SparkSession

    builder = SparkSession.builder.appName(app_name)
    builder = builder.config(
        "spark.sql.session.timeZone", "UTC"
    ).config("spark.sql.shuffle.partitions", "32").config(
        "spark.sql.execution.arrow.pyspark.enabled", "false"
    )
    # Local dev (esp. Windows): avoid driver/worker binding to a wrong NIC
    if not os.environ.get("SPARK_LOCAL_IP"):
        os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
    if master:
        builder = builder.master(master)
    elif local_threads is not None:
        builder = builder.master(f"local[{local_threads}]").config(
            "spark.sql.shuffle.partitions", str(max(2, local_threads * 2))
        )
    elif not os.environ.get("SPARK_MASTER"):
        builder = builder.master("local[*]")
    if with_graphframes:
        builder = builder.config("spark.jars.packages", GRAPHFRAMES_PACKAGE)
    return builder.getOrCreate()


# ───────────────────────── Ingestion ───────────────────────────────────────

def ingest_to_parquet(
    spark, json_glob: str, out_root: str | Path
) -> str:
    """Read JSON exports, validate, partition by year/month, write Parquet."""
    from pyspark.sql import functions as F
    from pyspark.sql.types import IntegerType, LongType, BooleanType, StringType

    df = spark.read.json(json_glob)

    df = df.filter(
        F.col("ts").isNotNull()
        & F.col("master_metadata_track_name").isNotNull()
    )

    df = (
        df.withColumn("ts", F.to_timestamp("ts"))
        .withColumn("ms_played", F.col("ms_played").cast(LongType()))
        .withColumn("year", F.year("ts").cast(IntegerType()))
        .withColumn("month", F.month("ts").cast(IntegerType()))
        .withColumn("hour_utc", F.hour("ts").cast(IntegerType()))
        .withColumn("weekday", F.dayofweek("ts").cast(IntegerType()))
    )
    for col in ("skipped", "shuffle", "offline", "incognito_mode"):
        if col in df.columns:
            df = df.withColumn(col, F.col(col).cast(BooleanType()))
        else:
            df = df.withColumn(col, F.lit(False))
    for col in (
        "master_metadata_album_artist_name",
        "master_metadata_album_album_name",
        "reason_start",
        "reason_end",
        "conn_country",
        "platform",
    ):
        if col not in df.columns:
            df = df.withColumn(col, F.lit("Unknown"))
        df = df.withColumn(
            col, F.coalesce(F.col(col).cast(StringType()), F.lit("Unknown"))
        )

    out_root = str(out_root)
    df.write.mode("overwrite").partitionBy("year", "month").parquet(out_root)
    return out_root


# ───────────────────────── Metric computation ──────────────────────────────

def compute_metrics_spark_dataframe(spark, df) -> dict:
    """Run metric aggregation on an existing Spark DataFrame (no file I/O)."""
    from pyspark.sql import functions as F

    df = df.filter(F.col("master_metadata_track_name").isNotNull())

    total_plays = df.count()
    if total_plays == 0:
        return {"error": "no records"}

    agg_row = df.agg(
        F.sum("ms_played").alias("total_ms"),
        F.sum(F.when(F.col("skipped"), 1).otherwise(0)).alias("skipped_count"),
        F.sum(
            F.when(F.lower(F.col("reason_end")) == "endplay", 1).otherwise(0)
        ).alias("completed_count"),
        F.sum(F.when(F.col("shuffle"), 1).otherwise(0)).alias("shuffle_count"),
        F.sum(F.when(F.col("offline"), F.col("ms_played")).otherwise(0)).alias("offline_ms"),
        F.sum(F.when(F.col("incognito_mode"), 1).otherwise(0)).alias("incognito_count"),
        F.sum(F.when(F.col("skipped") & (F.col("ms_played") < 30000), 1).otherwise(0)).alias("early_skip_count"),
        F.sum(F.when(~F.col("skipped") & (F.col("ms_played") >= 240000), 1).otherwise(0)).alias("focus_count"),
        F.min("ts").alias("first_ts"),
        F.max("ts").alias("last_ts"),
    ).collect()[0]

    total_ms = int(agg_row["total_ms"] or 0)
    total_hours = total_ms / (1000 * 3600)
    skipped_count = int(agg_row["skipped_count"] or 0)
    completed_count = int(agg_row["completed_count"] or 0)

    by_artist = (
        df.groupBy("master_metadata_album_artist_name")
        .agg(
            F.sum("ms_played").alias("ms"),
            F.count("*").alias("plays"),
            F.sum(F.when(F.col("skipped"), 1).otherwise(0)).alias("skips"),
        )
        .orderBy(F.desc("ms"))
    )
    by_song = (
        df.groupBy(
            "master_metadata_track_name", "master_metadata_album_artist_name"
        )
        .agg(F.sum("ms_played").alias("ms"), F.count("*").alias("plays"))
        .orderBy(F.desc("ms"))
    )
    by_year = df.groupBy("year").agg(
        F.sum("ms_played").alias("ms"), F.count("*").alias("plays")
    ).orderBy("year")
    by_hour = df.groupBy("hour_utc").agg(F.sum("ms_played").alias("ms")).orderBy("hour_utc")
    by_weekday = df.groupBy("weekday").agg(F.sum("ms_played").alias("ms")).orderBy("weekday")
    by_country = df.groupBy("conn_country").agg(F.sum("ms_played").alias("ms"))

    top_artists = by_artist.limit(20).collect()
    top_songs = by_song.limit(20).collect()
    years = [(r["year"], r["plays"], r["ms"]) for r in by_year.collect()]
    hours = {r["hour_utc"]: int(r["ms"] or 0) for r in by_hour.collect()}
    weekdays = {r["weekday"]: int(r["ms"] or 0) for r in by_weekday.collect()}
    countries = {r["conn_country"]: int(r["ms"] or 0) for r in by_country.collect()}

    artist_count = by_artist.count()
    track_count = by_song.count()

    calendar_first = agg_row["first_ts"]
    calendar_last = agg_row["last_ts"]
    calendar_days = max(1, (calendar_last - calendar_first).days) if calendar_first and calendar_last else 1

    metrics = {
        "engine": "pyspark",
        "summary": {
            "total_plays": total_plays,
            "total_hours": round(total_hours, 2),
            "unique_artists": artist_count,
            "unique_tracks": track_count,
            "calendar_days": calendar_days,
            "first_year": int(calendar_first.year) if calendar_first else None,
            "last_year": int(calendar_last.year) if calendar_last else None,
        },
        "metrics": {
            "impatience_score_pct": round(100 * skipped_count / total_plays, 2),
            "completion_rate_pct": round(100 * completed_count / total_plays, 2),
            "exploration_score": round(100 * track_count / total_plays, 2),
            "early_skip_rate_pct": round(100 * int(agg_row["early_skip_count"] or 0) / max(skipped_count, 1), 2),
            "focus_session_score_pct": round(100 * int(agg_row["focus_count"] or 0) / total_plays, 2),
            "shuffle_pct": round(100 * int(agg_row["shuffle_count"] or 0) / total_plays, 2),
            "offline_ratio_pct": round(100 * int(agg_row["offline_ms"] or 0) / max(total_ms, 1), 2),
            "listening_intensity_h_per_day": round(total_hours / calendar_days, 3),
        },
        "top_artists": [
            {"artist": r["master_metadata_album_artist_name"], "ms": int(r["ms"]), "plays": int(r["plays"])}
            for r in top_artists
        ],
        "top_songs": [
            {
                "track": r["master_metadata_track_name"],
                "artist": r["master_metadata_album_artist_name"],
                "ms": int(r["ms"]),
                "plays": int(r["plays"]),
            }
            for r in top_songs
        ],
        "by_year": [{"year": int(y), "plays": int(p), "ms": int(m)} for y, p, m in years],
        "by_hour": hours,
        "by_weekday": weekdays,
        "by_country": countries,
    }
    return metrics


def compute_metrics_spark(spark, parquet_root: str | Path) -> dict:
    """Read partitioned Parquet from ``parquet_root`` and run ``compute_metrics_spark_dataframe``."""
    df = spark.read.parquet(str(parquet_root))
    return compute_metrics_spark_dataframe(spark, df)


def prepare_metrics_dataframe_from_records(spark, records: list[dict]):
    """Build the Spark DataFrame for ``compute_metrics_spark_dataframe`` from
    raw export dicts.  Uses a temp JSONL + ``spark.read.json`` so the ingest
    path is JVM-only and works on Windows (``createDataFrame`` from Pandas
    can crash Python workers on some laptops / Docker network setups).
    """
    import json
    import tempfile
    from pathlib import Path

    from pyspark.sql import functions as F
    from pyspark.sql.types import BooleanType, LongType, StringType

    tmp = Path(
        tempfile.mkstemp(suffix=".jsonl", prefix="datatify_bench_")[1]
    )
    with tmp.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    p = str(tmp.resolve()).replace("\\", "/")
    if not p.startswith("/"):
        p = "/" + p
    uri = f"file://{p}"
    try:
        df = spark.read.json(uri)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass

    df = (
        df.withColumn("ts", F.to_timestamp("ts"))
        .withColumn("ms_played", F.col("ms_played").cast(LongType()))
        .withColumn("year", F.year("ts"))
        .withColumn("month", F.month("ts"))
        .withColumn("hour_utc", F.hour("ts"))
        .withColumn("weekday", F.date_format("ts", "E"))
    )
    # `dayofweek` in Spark: use day_of_week in Spark 3.5+ or map from E
    df = df.withColumn(
        "weekday", (F.dayofweek("ts").cast("int") - 1) % 7
    )  # Mon=0 .. Sun=6 to match Pandas
    for col in ("skipped", "shuffle", "offline", "incognito_mode"):
        if col not in df.columns:
            df = df.withColumn(col, F.lit(False))
        else:
            df = df.withColumn(col, F.col(col).cast(BooleanType()))
    for col in (
        "master_metadata_album_artist_name",
        "master_metadata_album_album_name",
        "reason_start",
        "reason_end",
        "conn_country",
        "platform",
    ):
        if col not in df.columns:
            df = df.withColumn(col, F.lit("Unknown"))
        else:
            df = df.withColumn(
                col, F.coalesce(F.col(col).cast(StringType()), F.lit("Unknown"))
            )
    return df


def compute_metrics_spark_from_records(spark, records: list[dict]) -> dict:
    """End-to-end Spark metrics for raw streaming-history dict rows."""
    df = prepare_metrics_dataframe_from_records(spark, records)
    return compute_metrics_spark_dataframe(spark, df)


def compute_metrics_spark_from_pandas(spark, pdf) -> dict:
    """Pandas -> Spark (fallback, mainly for Linux CI).  Prefer
    :func:`compute_metrics_spark_from_records` for benchmarks.
    """
    for col in ("year", "month", "hour_utc", "weekday"):
        if col not in pdf.columns:
            raise ValueError(
                f"DataFrame must include {col!r} (use ts-derived columns in benchmark)."
            )
    sdf = spark.createDataFrame(pdf)
    return compute_metrics_spark_dataframe(spark, sdf)


# ───────────────────────── Graph (transition edges) ────────────────────────

def build_transition_edges(spark, parquet_root: str | Path):
    """Return a DataFrame of (src_artist, dst_artist, weight) transitions.

    Two consecutive plays form an edge only if they fall inside the same
    30-minute session window. Implemented with ``Window.lag`` for
    distributed ordering.
    """
    from pyspark.sql import functions as F, Window

    df = spark.read.parquet(str(parquet_root))
    df = df.filter(F.col("master_metadata_album_artist_name").isNotNull())

    w = Window.orderBy("ts")
    df2 = (
        df.withColumn("artist", F.col("master_metadata_album_artist_name"))
        .withColumn("prev_artist", F.lag("artist").over(w))
        .withColumn("prev_ts", F.lag("ts").over(w))
        .withColumn(
            "gap_min",
            (F.col("ts").cast("long") - F.col("prev_ts").cast("long")) / 60.0,
        )
    )
    transitions = df2.filter(
        F.col("prev_artist").isNotNull() & (F.col("gap_min") <= 30)
    )
    edges = (
        transitions.groupBy(F.col("prev_artist").alias("src"), F.col("artist").alias("dst"))
        .agg(F.count("*").alias("weight"))
        .orderBy(F.desc("weight"))
    )
    return edges


def run_pagerank_graphframes(spark, edges_df, top_k: int = 20) -> list[dict]:
    """Compute PageRank using GraphFrames (requires ``graphframes`` JAR)."""
    try:
        from graphframes import GraphFrame
    except ImportError:
        return []
    from pyspark.sql import functions as F

    vertices = (
        edges_df.select(F.col("src").alias("id"))
        .union(edges_df.select(F.col("dst").alias("id")))
        .distinct()
    )
    g = GraphFrame(vertices, edges_df.withColumnRenamed("weight", "relationship"))
    pr = g.pageRank(resetProbability=0.15, maxIter=20)
    rows = (
        pr.vertices.orderBy(F.desc("pagerank"))
        .limit(top_k)
        .collect()
    )
    return [
        {"rank": i + 1, "artist": r["id"], "score": round(float(r["pagerank"]), 6)}
        for i, r in enumerate(rows)
    ]


def run_label_propagation_graphframes(spark, edges_df, max_iter: int = 5) -> list[dict]:
    """Community detection via GraphFrames Label Propagation."""
    try:
        from graphframes import GraphFrame
    except ImportError:
        return []
    from pyspark.sql import functions as F

    vertices = (
        edges_df.select(F.col("src").alias("id"))
        .union(edges_df.select(F.col("dst").alias("id")))
        .distinct()
    )
    g = GraphFrame(vertices, edges_df.withColumnRenamed("weight", "relationship"))
    lp = g.labelPropagation(maxIter=max_iter)
    grouped = (
        lp.groupBy("label")
        .agg(F.count("*").alias("size"), F.collect_list("id").alias("members"))
        .orderBy(F.desc("size"))
    )
    rows = grouped.limit(20).collect()
    return [
        {"community": int(r["label"]), "size": int(r["size"]), "top_artists": list(r["members"])[:8]}
        for r in rows
    ]


# ───────────────────────── User clustering ─────────────────────────────────

def cluster_users_spark(spark, vectors_df, k: int = 4) -> dict:
    """Spark MLlib K-Means on a DataFrame of behavioural metric vectors.

    ``vectors_df`` must contain a numeric column for each metric in the
    canonical 15-dimensional list (see ``clustering.METRIC_KEYS``).
    """
    from pyspark.ml.clustering import KMeans
    from pyspark.ml.evaluation import ClusteringEvaluator
    from pyspark.ml.feature import VectorAssembler, StandardScaler

    feature_cols = [c for c in vectors_df.columns if c not in ("user_id", "id")]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
    scaler = StandardScaler(
        inputCol="features_raw", outputCol="features", withMean=True, withStd=True
    )

    assembled = assembler.transform(vectors_df)
    scaler_model = scaler.fit(assembled)
    scaled = scaler_model.transform(assembled)

    kmeans = KMeans(k=k, seed=42, featuresCol="features", predictionCol="cluster")
    model = kmeans.fit(scaled)
    predictions = model.transform(scaled)

    evaluator = ClusteringEvaluator(
        featuresCol="features", predictionCol="cluster", metricName="silhouette"
    )
    silhouette = float(evaluator.evaluate(predictions))

    sizes = (
        predictions.groupBy("cluster").count().orderBy("cluster").collect()
    )
    centers = [list(map(float, c)) for c in model.clusterCenters()]

    return {
        "engine": "pyspark.mllib",
        "k": k,
        "silhouette": round(silhouette, 4),
        "cluster_sizes": [{"cluster": int(r["cluster"]), "size": int(r["count"])} for r in sizes],
        "centers_scaled": centers,
        "feature_cols": feature_cols,
    }


# ───────────────────────── CLI driver ──────────────────────────────────────

def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Datatify PySpark pipeline")
    ap.add_argument("command", choices=["ingest", "metrics", "graph", "all"])
    ap.add_argument("--json", help="Glob pattern for input JSON files")
    ap.add_argument("--parquet", required=True, help="Parquet output / input root")
    ap.add_argument("--graphframes", action="store_true", help="Enable GraphFrames")
    args = ap.parse_args()

    spark = spark_session(with_graphframes=args.graphframes)

    if args.command in ("ingest", "all"):
        if not args.json:
            raise SystemExit("--json is required for ingest/all")
        print(f"[spark] ingesting {args.json} -> {args.parquet}")
        ingest_to_parquet(spark, args.json, args.parquet)

    if args.command in ("metrics", "all"):
        print(f"[spark] computing metrics from {args.parquet}")
        m = compute_metrics_spark(spark, args.parquet)
        print(m["summary"])
        print(m["metrics"])

    if args.command in ("graph", "all"):
        print(f"[spark] building transition edges")
        edges = build_transition_edges(spark, args.parquet)
        edges.show(20, truncate=False)
        if args.graphframes:
            pr = run_pagerank_graphframes(spark, edges)
            print("PageRank (top 20):", pr)
            comms = run_label_propagation_graphframes(spark, edges)
            print("Communities:", comms)

    spark.stop()


if __name__ == "__main__":
    _main()
