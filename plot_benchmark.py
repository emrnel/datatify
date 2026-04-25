# -*- coding: utf-8 -*-
"""Render the scalability curve from ``benchmark_results/benchmark_results.json``.

Usage:
    python plot_benchmark.py

Writes:
    benchmark_results/scalability.png  (matplotlib log-log)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "benchmark_results" / "benchmark_results.json"
OUT_LINEAR = ROOT / "benchmark_results" / "scalability_linear.png"
OUT_LOG = ROOT / "benchmark_results" / "scalability_loglog.png"


def main() -> None:
    rows = json.loads(RESULTS.read_text(encoding="utf-8"))
    rows.sort(key=lambda r: r["records"])
    xs = [r["records"] for r in rows]
    py = [r["python_seconds"] for r in rows]
    pd = [r["pandas_seconds"] for r in rows]
    sp = [r["pyspark_seconds"] for r in rows]

    for path, scale in ((OUT_LINEAR, "linear"), (OUT_LOG, "log")):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(xs, py, marker="o", label="Pure Python", color="#3b82f6", linewidth=2)
        ax.plot(xs, pd, marker="s", label="Pandas", color="#10b981", linewidth=2)
        ax.plot(xs, sp, marker="^", label="PySpark (local[1])", color="#ef4444", linewidth=2)
        ax.set_xlabel("Records (synthetic Spotify-like)")
        ax.set_ylabel("Wall-clock seconds (lower is better)")
        ax.set_title(
            "Datatify scalability benchmark — pure Python vs. Pandas vs. PySpark\n"
            "Single Windows laptop, identical metric pipeline, seed=42"
        )
        if scale == "log":
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
