# -*- coding: utf-8 -*-
"""
Scalability Benchmark (Methods §B + Evaluation §V of proposal).

Measures end-to-end processing time for the same metric pipeline at
increasing dataset sizes (130K → 500K → 1M → 5M records) under three
execution engines:

    * pure-Python (``analyzer.analyze``)
    * Pandas      (``data_pipeline.compute_metrics_pandas``)
    * PySpark     (``spark_pipeline.compute_metrics_spark``)

Generates synthetic input via ``synthetic_data.generate_records`` and emits
a JSON + CSV report to ``benchmark_results/``.

Usage
-----
    python benchmark.py                                # default scales
    python benchmark.py --scales 100000 500000 1000000 # custom
    python benchmark.py --no-spark                     # skip PySpark
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from synthetic_data import generate_records

DEFAULT_SCALES = [130_000, 500_000, 1_000_000]
RESULTS_DIR = Path(__file__).resolve().parent / "benchmark_results"


def _time_it(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return time.perf_counter() - t0, result


def benchmark_python(records: list[dict]) -> float:
    from analyzer import analyze

    elapsed, _ = _time_it(analyze, records)
    return elapsed


def benchmark_pandas(records: list[dict]) -> float:
    from data_pipeline import load_records, compute_metrics_pandas

    df = load_records(records)
    elapsed, _ = _time_it(compute_metrics_pandas, df)
    return elapsed


def benchmark_spark(records: list[dict]) -> float:
    """Bench Spark aggregations on the same rows as Pandas.

    Uses a temp JSONL + ``spark.read.json`` so the ingest path stays
    JVM-only.  This avoids two Windows-PySpark pitfalls:
      * ``spark.read.parquet`` requires ``winutils.exe`` / Hadoop native I/O.
      * ``spark.createDataFrame(pandas_df)`` opens a Python worker, which
        crashes on machines where Spark binds to ``kubernetes.docker.internal``.
    On GCP Dataproc, use ``compute_metrics_spark`` with a GCS Parquet URI.
    """
    from spark_pipeline import (
        spark_session,
        compute_metrics_spark_from_records,
    )

    spark = spark_session(app_name="datatify-bench", local_threads=1)
    try:
        elapsed, _ = _time_it(compute_metrics_spark_from_records, spark, records)
    finally:
        spark.stop()
    return elapsed


def run_benchmark(scales: list[int], skip_spark: bool = False) -> list[dict]:
    RESULTS_DIR.mkdir(exist_ok=True)
    rows: list[dict] = []

    for n in scales:
        print(f"\n=== scale: {n:,} records ===")
        print("  generating synthetic data...")
        records = generate_records(n, seed=42)

        print("  [python]   running...")
        py_t = benchmark_python(records)
        print(f"  [python]   {py_t:.2f}s")

        print("  [pandas]   running...")
        pd_t = benchmark_pandas(records)
        print(f"  [pandas]   {pd_t:.2f}s")

        sp_t: float | None = None
        if not skip_spark:
            try:
                print("  [pyspark]  running...")
                sp_t = benchmark_spark(records)
                print(f"  [pyspark]  {sp_t:.2f}s")
            except Exception as e:
                print(f"  [pyspark]  SKIPPED ({e})")

        rows.append(
            {
                "records": n,
                "python_seconds": round(py_t, 3),
                "pandas_seconds": round(pd_t, 3),
                "pyspark_seconds": round(sp_t, 3) if sp_t is not None else None,
            }
        )

    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Datatify scalability benchmark")
    ap.add_argument(
        "--scales",
        type=int,
        nargs="+",
        default=DEFAULT_SCALES,
        help="Record counts to benchmark (default: 130K, 500K, 1M)",
    )
    ap.add_argument("--no-spark", action="store_true", help="Skip the PySpark run")
    args = ap.parse_args()

    rows = run_benchmark(args.scales, skip_spark=args.no_spark)

    RESULTS_DIR.mkdir(exist_ok=True)
    json_path = RESULTS_DIR / "benchmark_results.json"
    csv_path = RESULTS_DIR / "benchmark_results.csv"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print("\n=== summary ===")
    for r in rows:
        print(r)
    print(f"\nresults written to:\n  {json_path}\n  {csv_path}")


if __name__ == "__main__":
    main()
