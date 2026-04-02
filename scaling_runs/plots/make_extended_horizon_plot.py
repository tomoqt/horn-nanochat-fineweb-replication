#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot extended token-horizon results.")
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--out-png", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.summary_json.read_text(encoding="utf-8"))
    horizons = payload["horizons"]
    labels = [str(h["block_size"]) for h in horizons]
    baseline = np.array([h["baseline_final_val_loss_mean"] for h in horizons], dtype=float)
    horn = np.array([h["horn_final_val_loss_mean"] for h in horizons], dtype=float)
    delta = horn - baseline

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.bar(x - width / 2, baseline, width, color="#1f77b4", label="baseline")
    ax.bar(x + width / 2, horn, width, color="#d62728", label="horn")

    for i, d in enumerate(delta):
        ax.text(
            x[i],
            max(baseline[i], horn[i]) + 0.005,
            f"Δ={d:+.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title("Extended Token Horizon Sweep (FineWeb)")
    ax.set_xlabel("Block Size (Token Horizon)")
    ax.set_ylabel("Final Validation Loss (mean over seeds)")
    ax.set_xticks(x, labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
