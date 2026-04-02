#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False


TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
FINEWEB_CANDIDATES: list[tuple[str, str | None]] = [
    ("HuggingFaceFW/fineweb", "sample-10BT"),
    ("HuggingFaceFW/fineweb-edu", "sample-10BT"),
    ("HuggingFaceFW/fineweb", None),
    ("HuggingFaceFW/fineweb-edu", None),
]
FALLBACK_TEXT = (
    "<|user|> Hello!\n<|assistant|> Hi, how can I help?\n"
    "<|user|> Explain residual networks simply.\n"
    "<|assistant|> A residual block adds a learned correction to its input.\n"
) * 4000


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_tinyshakespeare_text(data_path: Path) -> str:
    if data_path.exists():
        return data_path.read_text(encoding="utf-8")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(TINY_SHAKESPEARE_URL, timeout=30) as resp:
            body = resp.read().decode("utf-8")
        data_path.write_text(body, encoding="utf-8")
        return body
    except Exception:
        return FALLBACK_TEXT


def load_fineweb_text(
    cache_path: Path,
    target_chars: int,
    max_docs: int,
) -> tuple[str, str]:
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8"), f"cache:{cache_path}"

    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(
            "datasets package is required for --dataset fineweb. "
            "Install with: pip install datasets"
        ) from exc

    last_error: Exception | None = None
    selected_ds = None
    selected_name = None
    selected_cfg = None

    for ds_name, cfg in FINEWEB_CANDIDATES:
        try:
            kwargs: dict[str, Any] = {
                "path": ds_name,
                "split": "train",
                "streaming": True,
            }
            if cfg is not None:
                kwargs["name"] = cfg
            selected_ds = load_dataset(**kwargs)
            selected_name = ds_name
            selected_cfg = cfg
            break
        except Exception as exc:
            last_error = exc

    if selected_ds is None:
        raise RuntimeError(
            "Unable to stream FineWeb from known configs. "
            "Tried candidates: "
            + ", ".join([f"{n}/{c or 'default'}" for n, c in FINEWEB_CANDIDATES])
        ) from last_error

    docs: list[str] = []
    char_count = 0
    for i, item in enumerate(selected_ds):
        txt = item.get("text") or item.get("content") or item.get("raw_content") or ""
        if not isinstance(txt, str) or not txt.strip():
            continue
        docs.append(txt.strip() + "\n")
        char_count += len(txt) + 1
        if char_count >= target_chars:
            break
        if i + 1 >= max_docs:
            break

    if not docs:
        raise RuntimeError("FineWeb stream yielded no usable text.")

    text = "".join(docs)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text, encoding="utf-8")
    cfg_s = selected_cfg if selected_cfg is not None else "default"
    source = f"hf:{selected_name}/{cfg_s}"
    return text, source


def load_local_text(data_path: Path) -> str:
    if not data_path.exists():
        raise FileNotFoundError(f"Local dataset path does not exist: {data_path}")
    return data_path.read_text(encoding="utf-8")


def build_chat_like_text(raw_text: str, max_pairs: int = 50_000) -> str:
    lines = [x.strip() for x in raw_text.splitlines() if x.strip()]
    if len(lines) < 3:
        return FALLBACK_TEXT
    pairs = min(max_pairs, len(lines) - 1)
    chunks = []
    for i in range(pairs):
        user = lines[i]
        assistant = lines[i + 1]
        chunks.append(
            f"<|user|> Continue Shakespeare style:\n{user}\n<|assistant|> {assistant}\n"
        )
    return "".join(chunks)


@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float
    variant: str
    horn_m_init: float
    horn_eta_init: float


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(
            1, 1, cfg.block_size, cfg.block_size
        )
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, t, emb = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(emb, dim=2)
        q = q.view(bsz, t, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, t, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.mask[:, :, :t, :t] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, t, emb)
        y = self.resid_dropout(self.proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = F.gelu(x)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class ResidualCore(nn.Module):
    """Core residual function u_l = f_l(LN(x_l)) used by both baseline and HORN."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(self.ln_1(x)) + self.mlp(self.ln_2(x))


def inv_softplus(y: float) -> float:
    return math.log(math.expm1(y))


def safe_logit(p: float) -> float:
    eps = 1e-5
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


class BaselineBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.core = ResidualCore(cfg)

    def forward(
        self, x: torch.Tensor, v: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, float]]:
        del v
        u = self.core(x)
        return x + u, None, {}


class HornBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.core = ResidualCore(cfg)
        self.is_no_momentum = cfg.variant == "horn_no_momentum"
        if self.is_no_momentum:
            self.register_buffer("fixed_m", torch.tensor(0.0))
            self.m_logit = None
        else:
            self.m_logit = nn.Parameter(torch.full((cfg.n_embd,), safe_logit(cfg.horn_m_init)))
            self.fixed_m = None
        self.eta_param = nn.Parameter(
            torch.full((cfg.n_embd,), inv_softplus(cfg.horn_eta_init))
        )

    def forward(
        self, x: torch.Tensor, v: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        if v is None:
            v = torch.zeros_like(x)
        u = self.core(x)
        eta = F.softplus(self.eta_param).view(1, 1, -1)
        if self.is_no_momentum:
            m = self.fixed_m.view(1, 1, 1)
        else:
            assert self.m_logit is not None
            m = torch.sigmoid(self.m_logit).view(1, 1, -1)
        v_next = m * v + eta * u
        x_next = x + v_next
        stats = {
            "m_mean": float(m.mean().detach().cpu().item()),
            "eta_mean": float(eta.mean().detach().cpu().item()),
            "v_rms": float(v_next.detach().pow(2).mean().sqrt().cpu().item()),
        }
        return x_next, v_next, stats


class NanoChatLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        if cfg.variant == "baseline":
            self.blocks = nn.ModuleList([BaselineBlock(cfg) for _ in range(cfg.n_layer)])
        else:
            self.blocks = nn.ModuleList([HornBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, float]]:
        bsz, t = idx.shape
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)
        v_state: torch.Tensor | None = None
        stat_items: list[dict[str, float]] = []
        for block in self.blocks:
            x, v_state, stats = block(x, v_state)
            if stats:
                stat_items.append(stats)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        aux: dict[str, float] = {}
        if stat_items:
            aux["m_mean"] = float(np.mean([s["m_mean"] for s in stat_items]))
            aux["eta_mean"] = float(np.mean([s["eta_mean"] for s in stat_items]))
            aux["v_rms"] = float(np.mean([s["v_rms"] for s in stat_items]))
        return logits, loss, aux


@dataclass
class RunResult:
    variant: str
    seed: int
    train_losses: list[float]
    val_steps: list[int]
    val_losses: list[float]
    horn_m_mean: list[float]
    horn_eta_mean: list[float]
    horn_v_rms: list[float]
    final_val_loss: float
    best_val_loss: float
    wall_seconds: float
    tokens_seen: int


def get_batch(
    data: torch.Tensor, block_size: int, batch_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(
    model: NanoChatLM,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    block_size: int,
    batch_size: int,
    eval_iters: int,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    split_losses: dict[str, list[float]] = {"train": [], "val": []}
    for split_name, split_data in (("train", train_data), ("val", val_data)):
        for _ in range(eval_iters):
            x, y = get_batch(split_data, block_size, batch_size, device)
            _, loss, _ = model(x, y)
            assert loss is not None
            split_losses[split_name].append(float(loss.item()))
    model.train()
    return float(np.mean(split_losses["train"])), float(np.mean(split_losses["val"]))


def train_one(
    model_cfg: ModelConfig,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    device: torch.device,
    seed: int,
    steps: int,
    eval_interval: int,
    eval_iters: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
) -> RunResult:
    set_seed(seed)
    model = NanoChatLM(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
    )

    train_losses: list[float] = []
    val_steps: list[int] = []
    val_losses: list[float] = []
    horn_m_mean: list[float] = []
    horn_eta_mean: list[float] = []
    horn_v_rms: list[float] = []

    start = time.time()
    ema_loss = None

    for step in range(steps + 1):
        if step % eval_interval == 0 or step == steps:
            train_eval, val_eval = estimate_loss(
                model,
                train_data=train_data,
                val_data=val_data,
                block_size=model_cfg.block_size,
                batch_size=batch_size,
                eval_iters=eval_iters,
                device=device,
            )
            train_losses.append(train_eval)
            val_losses.append(val_eval)
            val_steps.append(step)
            print(
                f"[{model_cfg.variant}] seed={seed} step={step}/{steps} "
                f"train={train_eval:.4f} val={val_eval:.4f}",
                flush=True,
            )
        if step == steps:
            break

        x, y = get_batch(train_data, model_cfg.block_size, batch_size, device)
        _, loss, aux = model(x, y)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        loss_f = float(loss.item())
        if ema_loss is None:
            ema_loss = loss_f
        else:
            ema_loss = 0.95 * ema_loss + 0.05 * loss_f

        if aux:
            horn_m_mean.append(aux.get("m_mean", 0.0))
            horn_eta_mean.append(aux.get("eta_mean", 0.0))
            horn_v_rms.append(aux.get("v_rms", 0.0))

    wall = time.time() - start
    return RunResult(
        variant=model_cfg.variant,
        seed=seed,
        train_losses=train_losses,
        val_steps=val_steps,
        val_losses=val_losses,
        horn_m_mean=horn_m_mean,
        horn_eta_mean=horn_eta_mean,
        horn_v_rms=horn_v_rms,
        final_val_loss=val_losses[-1],
        best_val_loss=min(val_losses),
        wall_seconds=wall,
        tokens_seen=steps * batch_size * model_cfg.block_size,
    )


def aggregate_results(results: list[RunResult]) -> dict[str, Any]:
    grouped: dict[str, list[RunResult]] = {}
    for run in results:
        grouped.setdefault(run.variant, []).append(run)

    agg: dict[str, Any] = {}
    ref_variant = "baseline" if "baseline" in grouped else sorted(grouped.keys())[0]
    ref_final_mean = float(np.mean([r.final_val_loss for r in grouped[ref_variant]]))
    ref_best_mean = float(np.mean([r.best_val_loss for r in grouped[ref_variant]]))

    for variant, runs in grouped.items():
        final_vals = np.array([r.final_val_loss for r in runs], dtype=np.float64)
        best_vals = np.array([r.best_val_loss for r in runs], dtype=np.float64)
        wall_vals = np.array([r.wall_seconds for r in runs], dtype=np.float64)
        agg[variant] = {
            "n_runs": len(runs),
            "final_val_loss_mean": float(final_vals.mean()),
            "final_val_loss_std": float(final_vals.std(ddof=0)),
            "best_val_loss_mean": float(best_vals.mean()),
            "best_val_loss_std": float(best_vals.std(ddof=0)),
            "wall_seconds_mean": float(wall_vals.mean()),
            "delta_final_vs_baseline": float(final_vals.mean() - ref_final_mean),
            "delta_best_vs_baseline": float(best_vals.mean() - ref_best_mean),
        }
        if variant != ref_variant:
            agg[variant]["relative_improvement_final_pct"] = float(
                100.0 * (ref_final_mean - final_vals.mean()) / ref_final_mean
            )
    return agg


def save_curves_plot(results: list[RunResult], out_path: Path) -> None:
    if not HAS_MATPLOTLIB:
        return
    grouped: dict[str, list[RunResult]] = {}
    for run in results:
        grouped.setdefault(run.variant, []).append(run)

    plt.figure(figsize=(9, 5))
    for variant, runs in grouped.items():
        steps = np.array(runs[0].val_steps)
        curves = np.array([r.val_losses for r in runs], dtype=np.float64)
        mean_curve = curves.mean(axis=0)
        std_curve = curves.std(axis=0)
        plt.plot(steps, mean_curve, label=variant)
        plt.fill_between(steps, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)
    plt.title("Validation Loss During Scaled NanoChat Pretraining")
    plt.xlabel("Step")
    plt.ylabel("Val loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_bar_plot(summary: dict[str, Any], out_path: Path) -> None:
    if not HAS_MATPLOTLIB:
        return
    variants = list(summary.keys())
    means = [summary[v]["final_val_loss_mean"] for v in variants]
    stds = [summary[v]["final_val_loss_std"] for v in variants]
    plt.figure(figsize=(8, 4))
    plt.bar(variants, means, yerr=stds, capsize=4)
    plt.title("Final Validation Loss (mean ± std across seeds)")
    plt.ylabel("Final val loss")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_horn_stats_plot(results: list[RunResult], out_path: Path) -> None:
    if not HAS_MATPLOTLIB:
        return
    horn_runs = [r for r in results if r.variant in {"horn", "horn_no_momentum"}]
    if not horn_runs:
        return
    plt.figure(figsize=(10, 4))
    for run in horn_runs:
        if not run.horn_m_mean:
            continue
        x = np.arange(1, len(run.horn_m_mean) + 1)
        plt.plot(x, run.horn_m_mean, alpha=0.5, label=f"{run.variant}-seed{run.seed}-m")
    plt.title("HORN Momentum Coefficient Mean During Training")
    plt.xlabel("Train step")
    plt.ylabel("m mean")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline vs HORN nanochat pretraining benchmark")
    parser.add_argument(
        "--dataset",
        type=str,
        default="fineweb",
        choices=["fineweb", "tinyshakespeare", "local"],
    )
    parser.add_argument("--fineweb-target-chars", type=int, default=8_000_000)
    parser.add_argument("--fineweb-max-docs", type=int, default=40_000)
    parser.add_argument(
        "--chat-format",
        action="store_true",
        help="Wrap dataset into user/assistant chat-like records (default off for fineweb).",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["baseline", "horn", "horn_no_momentum"],
        choices=["baseline", "horn", "horn_no_momentum"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1337, 2027])
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--horn-m-init", type=float, default=0.0)
    parser.add_argument("--horn-eta-init", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, default=Path("horn_benchmark_artifacts"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    if args.data_path is None:
        if args.dataset == "fineweb":
            data_path = Path("data/fineweb_sample.txt")
        elif args.dataset == "tinyshakespeare":
            data_path = Path("data/tinyshakespeare.txt")
        else:
            data_path = Path("data/local_corpus.txt")
    else:
        data_path = args.data_path

    dataset_source = f"local:{data_path}"
    if args.dataset == "fineweb":
        try:
            raw_text, dataset_source = load_fineweb_text(
                cache_path=data_path,
                target_chars=args.fineweb_target_chars,
                max_docs=args.fineweb_max_docs,
            )
        except Exception as exc:
            raise RuntimeError(
                "fineweb load failed and no cache could be used; "
                "not falling back to tinyshakespeare."
            ) from exc
    elif args.dataset == "tinyshakespeare":
        raw_text = load_tinyshakespeare_text(data_path)
    else:
        raw_text = load_local_text(data_path)

    use_chat_format = bool(args.chat_format)
    if args.dataset == "tinyshakespeare":
        # Keep previous behavior on tinyshakespeare unless explicitly disabled by using local/fineweb.
        use_chat_format = True
    corpus_text = build_chat_like_text(raw_text) if use_chat_format else raw_text

    chars = sorted(list(set(corpus_text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    encoded = torch.tensor([stoi[c] for c in corpus_text], dtype=torch.long)
    split_idx = int(0.9 * len(encoded))
    train_data = encoded[:split_idx]
    val_data = encoded[split_idx:]

    device = pick_device(args.device)
    print(f"device={device}", flush=True)
    print(f"vocab_size={len(chars)} train_tokens={len(train_data)} val_tokens={len(val_data)}", flush=True)

    all_results: list[RunResult] = []
    run_idx = 0
    total_runs = len(args.variants) * len(args.seeds)

    for variant in args.variants:
        for seed in args.seeds:
            run_idx += 1
            print(f"\n=== Run {run_idx}/{total_runs}: variant={variant} seed={seed} ===", flush=True)
            cfg = ModelConfig(
                vocab_size=len(chars),
                block_size=args.block_size,
                n_layer=args.n_layer,
                n_head=args.n_head,
                n_embd=args.n_embd,
                dropout=args.dropout,
                variant=variant,
                horn_m_init=args.horn_m_init,
                horn_eta_init=args.horn_eta_init,
            )
            result = train_one(
                model_cfg=cfg,
                train_data=train_data,
                val_data=val_data,
                device=device,
                seed=seed,
                steps=args.steps,
                eval_interval=args.eval_interval,
                eval_iters=args.eval_iters,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                grad_clip=args.grad_clip,
            )
            all_results.append(result)

    summary = aggregate_results(all_results)
    payload = {
        "config": {
            "variants": args.variants,
            "seeds": args.seeds,
            "steps": args.steps,
            "eval_interval": args.eval_interval,
            "eval_iters": args.eval_iters,
            "batch_size": args.batch_size,
            "block_size": args.block_size,
            "n_layer": args.n_layer,
            "n_head": args.n_head,
            "n_embd": args.n_embd,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "horn_m_init": args.horn_m_init,
            "horn_eta_init": args.horn_eta_init,
            "device": str(device),
            "dataset": args.dataset,
            "dataset_source": dataset_source,
            "data_path": str(data_path),
            "is_chat_like_dataset": use_chat_format,
            "fineweb_target_chars": args.fineweb_target_chars,
            "fineweb_max_docs": args.fineweb_max_docs,
        },
        "summary": summary,
        "runs": [asdict(r) for r in all_results],
    }
    (outdir / "benchmark_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    (outdir / "vocab.json").write_text(
        json.dumps({"vocab_size": len(chars), "chars": chars}, indent=2),
        encoding="utf-8",
    )
    save_curves_plot(all_results, outdir / "val_loss_curves.png")
    save_bar_plot(summary, outdir / "final_val_bar.png")
    save_horn_stats_plot(all_results, outdir / "horn_momentum_evolution.png")

    top_lines = [
        "# HORN vs Baseline NanoChat Benchmark",
        "",
        f"- Device: `{device}`",
        f"- Variants: `{', '.join(args.variants)}`",
        f"- Seeds: `{args.seeds}`",
        f"- Steps per run: `{args.steps}`",
        "",
        "## Final Validation Loss (mean +/- std)",
    ]
    for variant, row in summary.items():
        top_lines.append(
            f"- `{variant}`: {row['final_val_loss_mean']:.4f} +/- {row['final_val_loss_std']:.4f} "
            f"(delta vs baseline: {row['delta_final_vs_baseline']:+.4f})"
        )
    top_lines.append("")
    top_lines.append("## Files")
    top_lines.append("- `benchmark_summary.json`")
    top_lines.append("- `vocab.json`")
    if HAS_MATPLOTLIB:
        top_lines.append("- `val_loss_curves.png`")
        top_lines.append("- `final_val_bar.png`")
        top_lines.append("- `horn_momentum_evolution.png`")
    (outdir / "report.md").write_text("\n".join(top_lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
