#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_variant(runs: list[dict[str, Any]], variant: str) -> dict[str, float]:
    rows = [r for r in runs if r.get("variant") == variant]
    finals = [float(r["final_val_loss"]) for r in rows]
    walls = [float(r["wall_seconds"]) for r in rows]
    if not finals:
        raise RuntimeError(f"No runs found for variant={variant}")
    return {
        "final_val_loss_mean": mean(finals),
        "final_val_loss_std": pstdev(finals) if len(finals) > 1 else 0.0,
        "wall_seconds_mean": mean(walls),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize token-horizon branch outputs.")
    parser.add_argument("--root-outdir", type=Path, required=True)
    parser.add_argument("--plan-json", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    args = parser.parse_args()

    plan = load_json(args.plan_json)
    horizons = []
    dataset_ok = True
    not_tiny = True
    evidence: list[str] = []

    for branch in plan["branches"]:
        outdir = args.root_outdir / branch["outdir_name"]
        bench = outdir / "benchmark_summary.json"
        if not bench.exists():
            raise FileNotFoundError(f"Missing benchmark summary: {bench}")
        payload = load_json(bench)
        runs = payload["runs"]
        baseline = summarize_variant(runs, "baseline")
        horn = summarize_variant(runs, "horn")
        cfg = payload["config"]
        data_tag = str(cfg.get("dataset", "unknown"))
        source_tag = str(cfg.get("dataset_source", "unknown"))
        evidence.append(
            f"{branch['name']}: dataset={data_tag} dataset_source={source_tag} data_path={cfg.get('data_path')}"
        )
        if data_tag != "fineweb":
            dataset_ok = False
        if "tinyshakespeare" in source_tag.lower():
            not_tiny = False

        delta = horn["final_val_loss_mean"] - baseline["final_val_loss_mean"]
        rel = -100.0 * delta / baseline["final_val_loss_mean"]
        horizons.append(
            {
                "name": branch["name"],
                "block_size": int(branch["block_size"]),
                "batch_size": int(branch["batch_size"]),
                "tokens_per_step": int(branch["tokens_per_step"]),
                "dataset": data_tag,
                "dataset_source": source_tag,
                "data_path": cfg.get("data_path"),
                "baseline_final_val_loss_mean": baseline["final_val_loss_mean"],
                "horn_final_val_loss_mean": horn["final_val_loss_mean"],
                "horn_delta_final_vs_baseline": delta,
                "horn_relative_improvement_final_pct": rel,
                "baseline_wall_seconds_mean": baseline["wall_seconds_mean"],
                "horn_wall_seconds_mean": horn["wall_seconds_mean"],
            }
        )

    best_by_horn = min(horizons, key=lambda h: h["horn_final_val_loss_mean"])
    best_by_delta = min(horizons, key=lambda h: h["horn_delta_final_vs_baseline"])

    out = {
        "experiment_name": plan["experiment_name"],
        "dataset": "fineweb",
        "dataset_verification_passed": bool(dataset_ok and not_tiny),
        "verification_evidence": evidence,
        "horizons": horizons,
        "best_horn_horizon": best_by_horn["name"],
        "best_horn_block_size": best_by_horn["block_size"],
        "best_horn_final_val_loss_mean": best_by_horn["horn_final_val_loss_mean"],
        "best_delta_horizon": best_by_delta["name"],
        "best_delta_horn_minus_baseline": best_by_delta["horn_delta_final_vs_baseline"],
    }

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = [
        f"# {plan['experiment_name']} Summary",
        "",
        f"- Dataset verification passed: `{out['dataset_verification_passed']}`",
        f"- Best horizon by horn absolute loss: `{out['best_horn_horizon']}`",
        f"- Best horizon by horn-vs-baseline delta: `{out['best_delta_horizon']}`",
        "",
        "| horizon | block | batch | baseline | horn | delta (horn-baseline) | rel_impr_pct |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for h in horizons:
        lines.append(
            f"| {h['name']} | {h['block_size']} | {h['batch_size']} | "
            f"{h['baseline_final_val_loss_mean']:.4f} | {h['horn_final_val_loss_mean']:.4f} | "
            f"{h['horn_delta_final_vs_baseline']:+.4f} | {h['horn_relative_improvement_final_pct']:+.3f}% |"
        )
    lines.append("")
    lines.append("## Verification Evidence")
    for ev in evidence:
        lines.append(f"- {ev}")
    lines.append("")

    args.summary_md.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
