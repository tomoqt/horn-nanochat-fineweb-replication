#!/usr/bin/env python3
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SEED = 1337
MODEL_NAME = "EleutherAI/pythia-70m-deduped"
SEQ_LEN = 128
BATCH_SIZE = 8
NUM_BATCHES = 6
OUTPUT_DIR = Path("artifacts_rebuilt")
TINY_SHAKESPEARE_PATH = Path(
    "/Users/tensorqt/Paradigma/experiment/nanoGPT/data/shakespeare_char/input.txt"
)


@dataclass(frozen=True)
class ConditionResult:
    name: str
    loss: float
    sink_first4: float
    sink_last3: float
    source_profile: list[float]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_text() -> str:
    if TINY_SHAKESPEARE_PATH.exists():
        return TINY_SHAKESPEARE_PATH.read_text(encoding="utf-8")
    return (
        "To be, or not to be: that is the question. "
        "Whether tis nobler in the mind to suffer. "
    ) * 5000


def build_batch(tokenizer: AutoTokenizer, text: str, seq_len: int, batch_size: int, num_batches: int) -> torch.Tensor:
    token_ids = tokenizer(text, return_tensors="pt").input_ids[0]
    if token_ids.numel() < seq_len + 2:
        raise ValueError("Corpus too short for chosen sequence length")
    max_start = token_ids.numel() - seq_len - 1
    starts = torch.randint(0, max_start, (batch_size * num_batches,))
    chunks = [token_ids[s : s + seq_len] for s in starts.tolist()]
    return torch.stack(chunks, dim=0)


def condition_ids(name: str, base: torch.Tensor, mask_id: int, generator: torch.Generator) -> torch.Tensor:
    x = base.clone()
    if name == "baseline":
        return x
    if name == "shuffled":
        perms = torch.stack([torch.randperm(x.size(1), generator=generator) for _ in range(x.size(0))], dim=0)
        return x.gather(1, perms)
    if name == "reverse":
        return torch.flip(x, dims=[1])
    if name == "shift32":
        return torch.roll(x, shifts=32, dims=1)
    if name == "mask_first4":
        x[:, :4] = mask_id
        return x
    if name == "mask_random4":
        for row in range(x.size(0)):
            idx = torch.randperm(x.size(1), generator=generator)[:4]
            x[row, idx] = mask_id
        return x
    if name == "swap_pos0":
        swap_idx = min(32, x.size(1) - 1)
        tmp = x[:, 0].clone()
        x[:, 0] = x[:, swap_idx]
        x[:, swap_idx] = tmp
        return x
    raise ValueError(f"Unknown condition: {name}")


def evaluate(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    device: torch.device,
) -> tuple[float, float, float, list[float]]:
    labels = input_ids.clone()
    with torch.no_grad():
        out = model(
            input_ids=input_ids.to(device),
            labels=labels.to(device),
            output_attentions=True,
            use_cache=False,
        )
    loss = float(out.loss.item())
    layer_profiles = []
    for attn in out.attentions:
        # [batch, heads, query, key] -> mean mass per key position
        profile = attn.detach().float().mean(dim=(0, 1, 2)).cpu()
        layer_profiles.append(profile)
    src_profile = torch.stack(layer_profiles, dim=0).mean(dim=0)
    sink_first4 = float(src_profile[:4].sum().item())
    sink_last3 = float(src_profile[-3:].sum().item())
    return loss, sink_first4, sink_last3, src_profile.tolist()


def save_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    mask_id = int(tokenizer.eos_token_id)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()
    model.to(device)

    text = load_text()
    base_batch = build_batch(
        tokenizer=tokenizer,
        text=text,
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
        num_batches=NUM_BATCHES,
    )
    gen = torch.Generator().manual_seed(SEED + 1)

    ordered_conditions = [
        "baseline",
        "shuffled",
        "reverse",
        "shift32",
        "mask_first4",
        "mask_random4",
        "swap_pos0",
    ]

    results: dict[str, ConditionResult] = {}
    for name in ordered_conditions:
        ids = condition_ids(name=name, base=base_batch, mask_id=mask_id, generator=gen)
        loss, sink_first4, sink_last3, src_profile = evaluate(model=model, input_ids=ids, device=device)
        results[name] = ConditionResult(
            name=name,
            loss=loss,
            sink_first4=sink_first4,
            sink_last3=sink_last3,
            source_profile=src_profile,
        )

    baseline = results["baseline"]
    summary = {
        "model_name": MODEL_NAME,
        "seq_len": SEQ_LEN,
        "batch_size": BATCH_SIZE,
        "num_sequences": int(base_batch.size(0)),
        "seed": SEED,
        "conditions": {
            k: {
                "loss": v.loss,
                "delta_loss_vs_baseline": v.loss - baseline.loss,
                "sink_first4": v.sink_first4,
                "sink_last3": v.sink_last3,
            }
            for k, v in results.items()
        },
    }
    save_json(OUTPUT_DIR / "attention_sinks_core_metrics.json", summary)

    # Hypothesis artifact set
    hypothesis_payload = {
        "baseline": summary["conditions"]["baseline"],
        "shuffled": summary["conditions"]["shuffled"],
        "claim": "If sink_first4 remains elevated under shuffled inputs, sink behavior is not tied only to coherent local order.",
    }
    save_json(OUTPUT_DIR / "hypothesis_contiguous_vs_shuffled.json", hypothesis_payload)

    # RoPE artifact set (same model, emphasizes positional behavior)
    rope_payload = {
        "model_name": MODEL_NAME,
        "baseline": summary["conditions"]["baseline"],
        "reverse": summary["conditions"]["reverse"],
        "shift32": summary["conditions"]["shift32"],
        "claim": "RoPE model keeps elevated sink mass near early absolute positions under reverse/shift controls.",
    }
    save_json(OUTPUT_DIR / "rope_position_controls.json", rope_payload)

    # Bias artifact set
    bias_payload = {
        "baseline": summary["conditions"]["baseline"],
        "reverse": summary["conditions"]["reverse"],
        "shift32": summary["conditions"]["shift32"],
        "delta_sink_first4_reverse": summary["conditions"]["reverse"]["sink_first4"]
        - summary["conditions"]["baseline"]["sink_first4"],
        "delta_sink_first4_shift32": summary["conditions"]["shift32"]["sink_first4"]
        - summary["conditions"]["baseline"]["sink_first4"],
    }
    save_json(OUTPUT_DIR / "bias_reverse_shift_deltas.json", bias_payload)

    # Masking artifact set
    masking_payload = {
        "baseline": summary["conditions"]["baseline"],
        "mask_first4": summary["conditions"]["mask_first4"],
        "mask_random4": summary["conditions"]["mask_random4"],
        "swap_pos0": summary["conditions"]["swap_pos0"],
        "delta_loss_mask_first4": summary["conditions"]["mask_first4"]["loss"]
        - summary["conditions"]["baseline"]["loss"],
        "delta_loss_mask_random4": summary["conditions"]["mask_random4"]["loss"]
        - summary["conditions"]["baseline"]["loss"],
        "delta_loss_swap_pos0": summary["conditions"]["swap_pos0"]["loss"]
        - summary["conditions"]["baseline"]["loss"],
    }
    save_json(OUTPUT_DIR / "masking_ablation_deltas.json", masking_payload)

    # Plot: source-position profiles for core controls
    x = np.arange(SEQ_LEN)
    plt.figure(figsize=(10, 5))
    for key in ["baseline", "shuffled", "reverse", "shift32"]:
        plt.plot(x, results[key].source_profile, label=key)
    plt.title("Attention Mass by Source Position (Mean over layers/heads/queries)")
    plt.xlabel("Source Position")
    plt.ylabel("Attention Probability")
    plt.xlim(0, min(SEQ_LEN, 64))
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "source_position_profiles.png", dpi=180)
    plt.close()

    # Plot: sink_first4 across conditions
    order = ["baseline", "shuffled", "reverse", "shift32", "mask_first4", "mask_random4", "swap_pos0"]
    plt.figure(figsize=(10, 4))
    vals = [results[k].sink_first4 for k in order]
    plt.bar(order, vals)
    plt.title("Sink Strength into Positions 0..3")
    plt.ylabel("sink_first4")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sink_first4_by_condition.png", dpi=180)
    plt.close()

    # Plot: delta loss vs baseline for masking controls
    order_mask = ["mask_first4", "mask_random4", "swap_pos0"]
    plt.figure(figsize=(7, 4))
    vals = [results[k].loss - baseline.loss for k in order_mask]
    plt.bar(order_mask, vals)
    plt.title("Delta Loss vs Baseline (Masking Controls)")
    plt.ylabel("delta_loss")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "masking_delta_loss.png", dpi=180)
    plt.close()

    report = [
        "# Attention Sinks Rebuilt Artifact Report",
        "",
        f"- Model: `{MODEL_NAME}`",
        f"- Sequence length: `{SEQ_LEN}`",
        f"- Batch size: `{BATCH_SIZE}` across `{NUM_BATCHES}` batches",
        f"- Seed: `{SEED}`",
        "",
        "## Core Metrics",
    ]
    for key in ordered_conditions:
        v = results[key]
        report.append(
            f"- `{key}`: loss={v.loss:.4f}, sink_first4={v.sink_first4:.4f}, "
            f"sink_last3={v.sink_last3:.4f}, delta_loss={v.loss - baseline.loss:+.4f}"
        )
    report.append("")
    report.append("## Files")
    report.append("- `attention_sinks_core_metrics.json`")
    report.append("- `hypothesis_contiguous_vs_shuffled.json`")
    report.append("- `rope_position_controls.json`")
    report.append("- `bias_reverse_shift_deltas.json`")
    report.append("- `masking_ablation_deltas.json`")
    report.append("- `source_position_profiles.png`")
    report.append("- `sink_first4_by_condition.png`")
    report.append("- `masking_delta_loss.png`")
    (OUTPUT_DIR / "rebuilt_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
