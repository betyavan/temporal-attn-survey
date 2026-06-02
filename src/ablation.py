"""
src/ablation.py — хирургическая аблация отдельных голов temporal attention.

Идея: вклад головы h в выход блока — это softmax(QKᵀ/√d)·V для её слайса,
спроецированный через `to_out`. Если занулить столбцы `to_out`, отвечающие за
слайс головы [h·head_dim : (h+1)·head_dim], голова перестаёт влиять на выход —
а Q/K/V и кэш пре-аттеншн активаций остаются нетронутыми.

⚠️  ВАЖНО про модуль внимания
-----------------------------
Скетч в README использует `block.temporal_attn`, но в РЕАЛЬНОМ CogVideoX-5B
такого модуля НЕТ: это унифицированный `block.attn1` (full 3D attention),
тот же, на котором висят хуки в src/hooks.py. Поэтому зануляем
`block.attn1.to_out[0].weight`, а не `temporal_attn`.

⚠️  ВАЖНО про имена в diffusers
--------------------------------
У diffusers `Attention`: число голов — `attn.heads` (НЕ `num_heads`).
`to_out` — это `nn.ModuleList([Linear, Dropout])`, веса в `to_out[0].weight`
формы (query_dim, inner_dim). Надёжно считать:
    head_dim = attn.to_out[0].weight.shape[1] // attn.heads

Публичное API:
    Hypothesis, SPECIALIST_HEADS, HYPOTHESES — конфиг из reports/phase1_profiles.md
    zero_out_heads / restore_heads — занулить/восстановить головы одного слоя
    ablated_heads                  — контекстный менеджер вокруг нескольких слоёв
    run_ablation                   — протокол baseline vs ablated
    summarize_ablation             — свернуть результаты в дельты для таблицы Phase 3
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator
from collections import defaultdict

import torch
from torch import Tensor
from src.metrics import temporal_consistency, motion_score

# --- Геометрия CogVideoX-5B -------------------------------------------------
NUM_LAYERS = 42
NUM_HEADS = 48  # attn1: hidden 3072 / 48 = head_dim 64

# Цель аблации внутри блока. Меняй здесь, если тестируешь не attn1.
ATTN_MODULE = "attn1"

# (layer_idx, [head_idx, ...]) — какие головы какого слоя занулить.
LayerHeads = tuple[int, list[int]]


# --- Специалисты из reports/phase1_profiles.md (Находка 4) ------------------
# |mean_dom_d| > 3 и σ < 3 кадра. Жирные в отчёте (σ=0) помечены True.
# label -> (layer, head, mean_dom_d, hardwired_sigma0)
SPECIALIST_HEADS: dict[str, tuple[int, int, float, bool]] = {
    "L41H5": (41, 5, +6.0, True),
    "L6H11": (6, 11, +5.9, False),
    "L8H46": (8, 46, -5.7, False),
    "L10H38": (10, 38, -5.5, False),
    "L1H21": (1, 21, -5.0, True),
    "L25H21": (25, 21, -4.1, False),
    "L41H36": (41, 36, +4.0, True),
    "L41H32": (41, 32, +4.0, True),
    "L0H42": (0, 42, +4.0, True),
    "L11H44": (11, 44, +3.8, False),
    "L0H17": (0, 17, -3.5, False),
    "L4H0": (4, 0, -3.1, False),
}


@dataclass(frozen=True)
class Hypothesis:
    """Один эксперимент аблации: что зануляем и что ожидаем."""

    description: str
    targets: list[LayerHeads]


# --- 4 гипотезы из «Гипотезы для Phase 2» отчёта (по убыванию приоритета) ----
HYPOTHESES: dict[str, Hypothesis] = {
    "h1_future_specialist": Hypothesis(
        description="L41H5 (σ=0, dom_d=+6). Ожидание: слабый эффект на temporal_consistency.",
        targets=[(41, [5])],
    ),
    "h2_past_specialists": Hypothesis(
        description="L1H21 + L8H46 (в прошлое). Ожидание: ломает ближнюю когерентность в начале сети.",
        targets=[(1, [21]), (8, [46])],
    ),
    "h3_global_phase": Hypothesis(
        description="Все головы L30–L41 (поздняя глобальная фаза). Ожидание: макс. падение temporal_consistency.",
        targets=[(layer, list(range(NUM_HEADS))) for layer in range(30, 42)],
    ),
    "h4_transition_zone": Hypothesis(
        description="Все головы L24–L28 (зона фазового перехода). Ожидание: макс. эффект на motion_score.",
        targets=[(layer, list(range(NUM_HEADS))) for layer in range(24, 29)],
    ),
}


# --- Хирургия весов ---------------------------------------------------------


def zero_out_heads(model: Any, layer_idx: int, head_indices: list[int]) -> dict[int, Tensor]:
    """Занулить столбцы `to_out`, соответствующие указанным головам одного слоя.

    Args:
        model:        CogVideoXTransformer3DModel (есть .transformer_blocks).
        layer_idx:    индекс блока 0..NUM_LAYERS-1.
        head_indices: какие головы занулить.

    Returns:
        saved — {head_idx: клон оригинального слайса весов} для restore_heads.

    Алгоритм:
        block = model.transformer_blocks[layer_idx]
        attn  = getattr(block, ATTN_MODULE)
        head_dim = attn.to_out[0].weight.shape[1] // attn.heads
        для каждого h:
            start, end = h*head_dim, (h+1)*head_dim
            saved[h] = attn.to_out[0].weight.data[:, start:end].clone()
            attn.to_out[0].weight.data[:, start:end] = 0
        вернуть saved.

    TODO: реализовать. Клонируй ДО зануления, иначе сохранишь нули.
    """
    saved = dict()
    block = model.transformer_blocks[layer_idx]
    attn = getattr(block, ATTN_MODULE)
    head_dim = attn.to_out[0].weight.shape[1] // attn.heads
    for h in head_indices:
        start, end = h*head_dim, (h+1)*head_dim
        saved[h] = attn.to_out[0].weight.data[:, start:end].clone()
        attn.to_out[0].weight.data[:, start:end] = 0
    return saved


def restore_heads(model: Any, layer_idx: int, saved: dict[int, Tensor]) -> None:
    """Вернуть веса голов, сохранённые zero_out_heads, на место (in-place).

    TODO: для каждого (head_idx, weights) положить weights обратно в тот же
          слайс [head_idx*head_dim : (head_idx+1)*head_dim] того же attn.to_out[0].
    """
    block = model.transformer_blocks[layer_idx]
    attn = getattr(block, ATTN_MODULE)
    head_dim = attn.to_out[0].weight.shape[1] // attn.heads
    for h, weights in saved.items():
        start, end = h*head_dim, (h+1)*head_dim
        attn.to_out[0].weight.data[:, start:end] = weights.clone()


@contextmanager
def ablated_heads(model: Any, targets: list[LayerHeads]) -> Generator[None, None, None]:
    """Контекст: занулить головы по всем (layer, heads) на входе, восстановить на выходе.

    Гарантирует восстановление даже при исключении внутри блока (try/finally),
    чтобы тот же экземпляр модели можно было гонять baseline → ablated → baseline.

    Использование:
        with ablated_heads(pipe.transformer, HYPOTHESES["h1_future_specialist"].targets):
            video = pipe(prompt, num_frames=49).frames[0]
    """
    saved: dict[int, dict[int, Tensor]] = {}
    try:
        for layer_idx, head_indices in targets:
            saved[layer_idx] = zero_out_heads(model, layer_idx, head_indices)
        yield
    finally:
        for layer_idx, layer_saved in saved.items():
            restore_heads(model, layer_idx, layer_saved)


# --- Протокол эксперимента --------------------------------------------------


def run_ablation(
    pipe: Any,
    prompts: list[str],
    targets: list[LayerHeads],
    clip_model: Any,
    clip_preprocess: Any,
    *,
    num_frames: int = 49,
    n_videos: int = 10,
    seed: int = 0,
    save_dir: Path | None = None,
    fps: int = 8,
    tag: str = "",
) -> dict[str, list[dict[str, float]]]:
    """Сгенерировать baseline и ablated видео, посчитать метрики на каждом.

    Args:
        pipe:            CogVideoX pipeline; модель внутри — pipe.transformer.
        prompts:         список промптов (берём первые n_videos).
        targets:         что занулять — обычно HYPOTHESES[name].targets.
        clip_model,
        clip_preprocess: для metrics.temporal_consistency (см. load_clip_model).
        num_frames:      длина видео (49 → латент T=13).
        n_videos:        сколько промптов прогнать.
        seed:            ОДИН и тот же seed на промпт для baseline и ablated —
                         тогда единственное отличие между ними это аблация
                         (низкодисперсные дельты).
        save_dir:        если задан — кадры пишутся в .mp4 сразу после генерации
                         (без повторного прохода пайплайна), файлы
                         {save_dir}/{tag}p{i}_{baseline,ablated}.mp4.
        fps:             частота кадров для export_to_video.
        tag:             префикс имён файлов (обычно имя гипотезы).

    Returns:
        {"baseline": [{"motion_score": .., "temporal_consistency": ..}, ...],
         "ablated":  [...]}  — по одному dict на промпт в каждом списке.

    Протокол:
        from src.metrics import motion_score, temporal_consistency
        model = pipe.transformer
        для каждого промпта (фиксируя generator=torch.Generator(...).manual_seed(seed)):
            1. baseline = pipe(prompt, num_frames=num_frames, generator=...).frames[0]
               → записать motion_score / temporal_consistency.
            2. with ablated_heads(model, targets):
                   ablated = pipe(prompt, num_frames=num_frames, generator=...).frames[0]
               → записать те же две метрики.
        (Веса восстанавливаются контекстом ablated_heads после каждого промпта.)

    TODO: реализовать по протоколу выше.
    """
    metrics: dict[str, list[dict[str, float]]] = defaultdict(list)
    model = pipe.transformer
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
    for i, prompt in enumerate(prompts[:n_videos]):
        gen = torch.Generator("cpu").manual_seed(seed + i)
        n_base = _targeted_norm(model, targets)  # должна быть >0 и одинаковой каждый промпт
        baseline = pipe(prompt, num_frames=num_frames, generator=gen).frames[0]
        _save_video(baseline, save_dir, f"{tag}p{i}_baseline", fps)
        metrics["baseline"].append({
            "motion_score": motion_score(baseline),
            "temporal_consistency": temporal_consistency(baseline, clip_model, clip_preprocess),
        })
        gen = torch.Generator("cpu").manual_seed(seed + i)  # тот же шум, что и baseline
        with ablated_heads(model, targets):
            n_abl = _targeted_norm(model, targets)  # должна быть ~0 (головы занулены)
            ablated = pipe(prompt, num_frames=num_frames, generator=gen).frames[0]
            _save_video(ablated, save_dir, f"{tag}p{i}_ablated", fps)
            metrics["ablated"].append({
                "motion_score": motion_score(ablated),
                "temporal_consistency": temporal_consistency(ablated, clip_model, clip_preprocess),
            })
        print(f"      [p{i}] ‖to_out_targeted‖: baseline={n_base:.3f} ablated={n_abl:.3f}")
        if n_base < 1e-6:
            print(f"      ⚠️  baseline-норма ≈0 на p{i}: restore НЕ сработал (веса остались занулены)")
    return metrics


def _targeted_norm(model: Any, targets: list[LayerHeads]) -> float:
    """L2-норма всех зануляемых столбцов `to_out` суммарно — диагностика restore.

    Перед baseline-генерацией должна быть >0 и стабильной от промпта к промпту;
    внутри ablated_heads — ≈0. Если перед baseline она ≈0, головы остались
    занулены с прошлого промпта (restore не сработал в этом окружении, напр.
    из-за cpu-offload).
    """
    total = 0.0
    for layer_idx, head_indices in targets:
        attn = getattr(model.transformer_blocks[layer_idx], ATTN_MODULE)
        w = attn.to_out[0].weight.data
        head_dim = w.shape[1] // attn.heads
        for h in head_indices:
            total += float(w[:, h * head_dim : (h + 1) * head_dim].float().norm() ** 2)
    return total**0.5


def _save_video(frames: list[Any], save_dir: Path | None, name: str, fps: int) -> None:
    """Записать кадры в {save_dir}/{name}.mp4; no-op если save_dir не задан."""
    if save_dir is None:
        return
    from diffusers.utils import export_to_video

    path = save_dir / f"{name}.mp4"
    export_to_video(frames, str(path), fps=fps)
    print(f"      видео -> {path}")


def _mean(values: list[float]) -> float:
    """Среднее по списку метрик; nan для пустого (усреднять нечего)."""
    return float(sum(values) / len(values)) if values else float("nan")


def summarize_ablation(results: dict[str, list[dict[str, float]]]) -> dict[str, float]:
    """Свернуть выход run_ablation в средние и относительные дельты для таблицы Phase 3.

    Returns:
        {"motion_score_baseline": ..,  "motion_score_ablated": ..,
         "motion_score_delta_pct": ..,
         "temporal_consistency_baseline": .., "..._ablated": .., "..._delta_pct": ..}
        где delta_pct = (ablated_mean - baseline_mean) / baseline_mean * 100.

    Пустой список метрик → nan; baseline_mean == 0 → delta_pct = nan
    (например, near_static, где motion_score ≈ 0).
    """
    summary: dict[str, float] = {}
    for metric in ("motion_score", "temporal_consistency"):
        baseline_mean = _mean([row[metric] for row in results["baseline"]])
        ablated_mean = _mean([row[metric] for row in results["ablated"]])

        if baseline_mean == 0.0:
            delta_pct = float("nan")
        else:
            delta_pct = (ablated_mean - baseline_mean) / baseline_mean * 100.0

        summary[f"{metric}_baseline"] = baseline_mean
        summary[f"{metric}_ablated"] = ablated_mean
        summary[f"{metric}_delta_pct"] = delta_pct
    return summary
