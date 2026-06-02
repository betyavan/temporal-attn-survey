#!/usr/bin/env python
"""Phase 3: батч-аблация голов temporal attention на CogVideoX.

Гоняет гипотезы из src.ablation.HYPOTHESES через run_ablation -> summarize_ablation,
пишет results/{metrics.json, ablation_table.md, ablation_deltas.png}.
Наука в src/, тут только оркестрация.

    poetry run python scripts/03_ablation_experiments.py                       # все гипотезы
    poetry run python scripts/03_ablation_experiments.py h1 h3 --save-videos   # выбранные + видео

Стоимость: 2·n_videos генераций на гипотезу (+2 с --save-videos).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.ablation import HYPOTHESES, ablated_heads, run_ablation, summarize_ablation  # noqa: E402
from src.metrics import load_clip_model  # noqa: E402

MODEL = os.environ.get("COGVIDEOX_PATH", "THUDM/CogVideoX-5b")
CONFIG = REPO_ROOT / "configs" / "baseline_prompts.yaml"
RESULTS = REPO_ROOT / "results"
NUM_FRAMES, SEED, DEVICE = 49, 0, "cuda"


def load_pipeline(cpu_offload: bool) -> Any:
    from diffusers import CogVideoXPipeline

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    pipe = CogVideoXPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16, use_safetensors=True)
    pipe.enable_model_cpu_offload() if cpu_offload else pipe.to(DEVICE)
    pipe.vae.enable_tiling()
    return pipe


def render_pair(pipe: Any, prompt: str, targets: list[tuple[int, list[int]]], fps: int, name: str) -> None:
    """baseline и ablated mp4 на один промпт; один seed -> разница только в аблации."""
    from diffusers.utils import export_to_video

    out = RESULTS / "ablation_videos"
    out.mkdir(parents=True, exist_ok=True)

    baseline = pipe(prompt, num_frames=NUM_FRAMES, generator=torch.manual_seed(SEED)).frames[0]
    export_to_video(baseline, str(out / f"{name}_baseline.mp4"), fps=fps)
    with ablated_heads(pipe.transformer, targets):
        ablated = pipe(prompt, num_frames=NUM_FRAMES, generator=torch.manual_seed(SEED)).frames[0]
    export_to_video(ablated, str(out / f"{name}_ablated.mp4"), fps=fps)
    print(f"      видео -> {out}/{name}_{{baseline,ablated}}.mp4")


def fmt_targets(targets: list[tuple[int, list[int]]]) -> str:
    return ", ".join(f"L{lyr}·all" if len(h) >= 48 else f"L{lyr}H{'/'.join(map(str, h))}" for lyr, h in targets)


def build_table(summaries: dict[str, dict[str, float]]) -> str:
    rows = ["| Hypothesis | Targets | motion (Δ%) | consistency (Δ%) |", "|---|---|---|---|"]
    for name, s in summaries.items():
        m = f"{s['motion_score_baseline']:.3f}→{s['motion_score_ablated']:.3f} ({s['motion_score_delta_pct']:+.1f}%)"
        c = f"{s['temporal_consistency_baseline']:.4f}→{s['temporal_consistency_ablated']:.4f} ({s['temporal_consistency_delta_pct']:+.1f}%)"
        rows.append(f"| `{name}` | {fmt_targets(HYPOTHESES[name].targets)} | {m} | {c} |")
    return "\n".join(rows)


def plot_deltas(summaries: dict[str, dict[str, float]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(summaries)
    x, w = range(len(names)), 0.38
    fig, ax = plt.subplots(figsize=(max(6.0, 1.8 * len(names)), 4.5))
    for off, metric, color in ((-w / 2, "motion_score", "#c44e52"), (w / 2, "temporal_consistency", "#4c72b0")):
        vals = [summaries[n][f"{metric}_delta_pct"] for n in names]
        ax.bar_label(ax.bar([i + off for i in x], vals, w, label=metric, color=color), fmt="%+.1f", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_title("Эффект аблации голов на метрики движения")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "ablation_deltas.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("hypotheses", nargs="*", help="имена из HYPOTHESES (пусто = все)")
    ap.add_argument("--n-videos", type=int, default=10)
    ap.add_argument("--save-videos", action="store_true")
    ap.add_argument("--cpu-offload", action="store_true")
    args = ap.parse_args()

    names = args.hypotheses or list(HYPOTHESES)
    unknown = [n for n in names if n not in HYPOTHESES]
    if unknown:
        ap.error(f"неизвестные гипотезы {unknown}; доступны {list(HYPOTHESES)}")

    cfg = OmegaConf.load(CONFIG)
    prompts = [str(p.text) for p in cfg.prompts]
    fps = int(cfg.generation.fps)

    pipe = load_pipeline(args.cpu_offload)
    clip_model, preprocess = load_clip_model(device=DEVICE)

    summaries: dict[str, dict[str, float]] = {}
    report: dict[str, Any] = {}
    for name in names:
        hyp = HYPOTHESES[name]
        print(f"=== {name}: {hyp.description} ===")
        raw = run_ablation(
            pipe, prompts, hyp.targets, clip_model, preprocess,
            num_frames=NUM_FRAMES, n_videos=args.n_videos, seed=SEED,
        )
        summaries[name] = summarize_ablation(raw)
        report[name] = {"targets": [[lyr, h] for lyr, h in hyp.targets], "summary": summaries[name], "raw": raw}
        print(
            f"      motion {summaries[name]['motion_score_delta_pct']:+.1f}% | "
            f"consistency {summaries[name]['temporal_consistency_delta_pct']:+.1f}%\n"
        )
        if args.save_videos:
            render_pair(pipe, prompts[0], hyp.targets, fps, name)

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    table = build_table(summaries)
    (RESULTS / "ablation_table.md").write_text(table + "\n", encoding="utf-8")
    plot_deltas(summaries)
    print(f"\n{table}\n\nresults/ <- metrics.json, ablation_table.md, ablation_deltas.png")


if __name__ == "__main__":
    main()
