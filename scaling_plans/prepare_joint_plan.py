#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_experiment_summaries(root: Path) -> list[tuple[str, dict]]:
    pairs: list[tuple[str, dict]] = []
    for p in sorted(root.rglob("benchmark_summary.json")):
        exp_name = p.parent.name
        payload = _load_summary(p)
        pairs.append((exp_name, payload))
    return pairs


def _horn_score(payload: dict) -> float:
    # Lower is better. Support both aggregated and single-run payloads.
    if "summary" in payload and "horn" in payload["summary"]:
        return float(payload["summary"]["horn"]["final_val_loss_mean"])
    if payload.get("variant") == "horn" and "final_val_loss" in payload:
        return float(payload["final_val_loss"])
    raise KeyError("No horn score found in payload")


def _extract_horn_rows_from_branch_summary(payload: dict) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # Expected flexible shapes:
    # - payload["rows"] -> list dicts
    # - payload["results"] -> list dicts
    # - payload["experiments"] -> dict exp_name -> dict(variants -> ...)
    if isinstance(payload.get("rows"), list):
        rows.extend([r for r in payload["rows"] if isinstance(r, dict)])
    if isinstance(payload.get("results"), list):
        rows.extend([r for r in payload["results"] if isinstance(r, dict)])
    if isinstance(payload.get("aggregate_results"), list):
        # Common format emitted by branch workers.
        cfg_by_exp: dict[str, dict[str, Any]] = {}
        for run in payload.get("per_run_results", []):
            if not isinstance(run, dict):
                continue
            exp = run.get("experiment") or run.get("exp_name")
            if not exp:
                continue
            cfg_by_exp.setdefault(
                str(exp),
                {
                    "n_layer": run.get("n_layer") or run.get("config", {}).get("n_layer"),
                    "n_head": run.get("n_head") or run.get("config", {}).get("n_head"),
                    "n_embd": run.get("n_embd") or run.get("config", {}).get("n_embd"),
                    "block_size": run.get("block_size") or run.get("config", {}).get("block_size"),
                    "batch_size": run.get("batch_size") or run.get("config", {}).get("batch_size"),
                },
            )
        for r in payload["aggregate_results"]:
            if not isinstance(r, dict):
                continue
            exp_name = str(r.get("experiment") or r.get("exp_name") or "unknown")
            row_cfg = cfg_by_exp.get(exp_name, {})
            rows.append(
                {
                    "exp_name": exp_name,
                    "variant": r.get("variant"),
                    "mean_final_val_loss": r.get("final_val_loss_mean"),
                    "n_layer": row_cfg.get("n_layer"),
                    "n_head": row_cfg.get("n_head"),
                    "n_embd": row_cfg.get("n_embd"),
                    "block_size": row_cfg.get("block_size"),
                    "batch_size": row_cfg.get("batch_size"),
                }
            )
    if isinstance(payload.get("experiments"), dict):
        for exp_name, exp_payload in payload["experiments"].items():
            if not isinstance(exp_payload, dict):
                continue
            variants = exp_payload.get("variants")
            if isinstance(variants, dict) and "horn" in variants:
                horn_entry = variants["horn"]
                if isinstance(horn_entry, dict):
                    rows.append(
                        {
                            "exp_name": exp_name,
                            "variant": "horn",
                            "mean_final_val_loss": horn_entry.get("mean_final_val_loss")
                            or horn_entry.get("final_val_loss_mean"),
                            "n_layer": exp_payload.get("n_layer"),
                            "n_head": exp_payload.get("n_head"),
                            "n_embd": exp_payload.get("n_embd"),
                            "block_size": exp_payload.get("block_size"),
                            "batch_size": exp_payload.get("batch_size"),
                        }
                    )
    return rows


def _choose_best_from_branch_summary(path: Path, branch_kind: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = _load_summary(path)
    rows = _extract_horn_rows_from_branch_summary(payload)
    norm_rows = []
    for r in rows:
        if r.get("variant") not in (None, "horn"):
            continue
        score = r.get("mean_final_val_loss") or r.get("final_val_loss")
        if score is None:
            continue
        score_f = float(score)
        record = {"score": score_f, **r}
        norm_rows.append(record)
    if not norm_rows:
        return None
    best = min(norm_rows, key=lambda x: x["score"])
    if branch_kind == "model":
        return {
            "name": str(best.get("exp_name", "unknown_model_exp")),
            "score": best["score"],
            "model_cfg": {
                "n_layer": int(best["n_layer"]),
                "n_head": int(best["n_head"]),
                "n_embd": int(best["n_embd"]),
            },
        }
    return {
        "name": str(best.get("exp_name", "unknown_horizon_exp")),
        "score": best["score"],
        "horizon_cfg": {
            "block_size": int(best["block_size"]),
            "batch_size": int(best["batch_size"]),
        },
    }


def _extract_model_cfg(payload: dict) -> dict:
    cfg = payload["config"]
    return {
        "n_layer": int(cfg["n_layer"]),
        "n_head": int(cfg["n_head"]),
        "n_embd": int(cfg["n_embd"]),
    }


def _extract_horizon_cfg(payload: dict) -> dict:
    cfg = payload["config"]
    return {
        "block_size": int(cfg["block_size"]),
        "batch_size": int(cfg["batch_size"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Select best branch results and create joint plan.")
    parser.add_argument("--model-branch-dir", type=Path, required=True)
    parser.add_argument("--horizon-branch-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=30)
    args = parser.parse_args()

    model_best = _choose_best_from_branch_summary(
        args.model_branch_dir / "model_branch_summary.json",
        branch_kind="model",
    )
    horizon_best = _choose_best_from_branch_summary(
        args.horizon_branch_dir / "horizon_branch_summary.json",
        branch_kind="horizon",
    )

    if model_best is None:
        model_runs = _find_experiment_summaries(args.model_branch_dir)
        # Keep only payloads that actually have horn aggregate.
        model_runs = [r for r in model_runs if "summary" in r[1] and "horn" in r[1]["summary"]]
        if not model_runs:
            raise SystemExit(f"No horn benchmark_summary.json files under {args.model_branch_dir}")
        model_name, model_payload = min(model_runs, key=lambda x: _horn_score(x[1]))
        model_cfg = _extract_model_cfg(model_payload)
        model_score = _horn_score(model_payload)
    else:
        model_name = model_best["name"]
        model_cfg = model_best["model_cfg"]
        model_score = float(model_best["score"])

    if horizon_best is None:
        horizon_runs = _find_experiment_summaries(args.horizon_branch_dir)
        horizon_runs = [r for r in horizon_runs if "summary" in r[1] and "horn" in r[1]["summary"]]
        if not horizon_runs:
            raise SystemExit(f"No horn benchmark_summary.json files under {args.horizon_branch_dir}")
        horizon_name, horizon_payload = min(horizon_runs, key=lambda x: _horn_score(x[1]))
        horizon_cfg = _extract_horizon_cfg(horizon_payload)
        horizon_score = _horn_score(horizon_payload)
    else:
        horizon_name = horizon_best["name"]
        horizon_cfg = horizon_best["horizon_cfg"]
        horizon_score = float(horizon_best["score"])

    joint_plan = {
        "experiment_name": "joint_scaled_from_branch_winners",
        "selection": {
            "model_branch_winner": {
                "name": model_name,
                "horn_final_val_loss_mean": model_score,
            },
            "horizon_branch_winner": {
                "name": horizon_name,
                "horn_final_val_loss_mean": horizon_score,
            },
        },
        "run_config": {
            "variants": ["baseline", "horn"],
            "seeds": [1337, 2027],
            "steps": args.steps,
            "eval_interval": args.eval_interval,
            "eval_iters": args.eval_iters,
            "horn_m_init": 0.5,
            **model_cfg,
            **horizon_cfg,
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(joint_plan, indent=2), encoding="utf-8")

    md = [
        "# Joint Scaling Plan Selected from Branch Winners",
        "",
        f"- Model-size winner: `{model_name}` (horn final mean={model_score:.6f})",
        f"- Token-horizon winner: `{horizon_name}` (horn final mean={horizon_score:.6f})",
        "",
        "## Joint Run Config",
        f"- n_layer: `{model_cfg['n_layer']}`",
        f"- n_head: `{model_cfg['n_head']}`",
        f"- n_embd: `{model_cfg['n_embd']}`",
        f"- block_size: `{horizon_cfg['block_size']}`",
        f"- batch_size: `{horizon_cfg['batch_size']}`",
        f"- seeds: `{joint_plan['run_config']['seeds']}`",
        f"- steps: `{joint_plan['run_config']['steps']}`",
        "",
        "Generated by `prepare_joint_plan.py`.",
    ]
    args.out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(joint_plan, indent=2))


if __name__ == "__main__":
    main()
