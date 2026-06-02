"""
src/metrics.py — метрики качества видео для Phase 2 (targeted ablation).

Сравниваем baseline vs ablated по двум осям движения:

- motion_score        — сколько в видео движения (величина оптического потока).
                        Падение после аблации → голова отвечала за порождение движения.
- temporal_consistency — насколько соседние кадры семантически согласованы.
                        Падение после аблации → голова отвечала за когерентность кадров.

Обе функции принимают список PIL-кадров ровно в том виде, в каком их отдаёт
diffusers-пайплайн: `frames = pipe(prompt, num_frames=49).frames[0]`.

Зависимости (добавь импорты внутри тел при реализации):
    import cv2            # Farneback optical flow
    import numpy as np    # массивы / агрегация
    import torch          # CLIP-инференс
    import open_clip      # модель для temporal_consistency
"""

from __future__ import annotations

from typing import Any, List

from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import open_clip


def load_clip_model(
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    device: str = "cuda",
) -> tuple[Any, Any]:
    """Загрузить open_clip модель и её val-препроцесс для temporal_consistency.

    Returns:
        (clip_model, preprocess) — модель в eval()/на device и трансформ
        для одного PIL-кадра → нормированный тензор (C, H, W).

    Hint:
        open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        возвращает (model, preprocess_train, preprocess_val) — нужен model и
        preprocess_val. Не забудь model.eval() и model.to(device).

    TODO: реализовать загрузку.
    """
    model, _, preprocess_val = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model.eval()
    model.to(device)
    return model, preprocess_val


def motion_score(video_frames: list[Image.Image]) -> float:
    """Средняя величина оптического потока Farneback между соседними кадрами.

    Args:
        video_frames: список PIL-кадров (как из pipe(...).frames[0]).

    Returns:
        Скаляр ≥ 0 — среднее по всем парам (i, i+1) от mean‖flow‖₂.
        Больше → больше движения в видео.

    Алгоритм:
        Для каждой пары соседних кадров:
          1. PIL → np.array → cv2.cvtColor(..., COLOR_RGB2GRAY).
          2. flow = cv2.calcOpticalFlowFarneback(f1, f2, None,
                       0.5, 3, 15, 3, 5, 1.2, 0)  → (H, W, 2).
          3. magnitude = sqrt(flow[...,0]² + flow[...,1]²); взять .mean().
        Вернуть среднее magnitude по всем парам как float.

    TODO: реализовать. Пустой/одно-кадровый вход → вернуть 0.0.
    """
    if len(video_frames) < 2:
        return 0.0
    scores: List[float] = []
    for i in range(len(video_frames) - 1):
        f1 = cv2.cvtColor(np.array(video_frames[i]), cv2.COLOR_RGB2GRAY)
        f2 = cv2.cvtColor(np.array(video_frames[i + 1]), cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(f1, f2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        scores.append(float(np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2).mean()))
    return float(np.mean(scores))


def temporal_consistency(
    video_frames: list[Image.Image],
    clip_model: Any,
    preprocess: Any,
    device: str = "cuda",
) -> float:
    """Средняя косинусная близость CLIP-эмбеддингов соседних кадров.

    Args:
        video_frames: список PIL-кадров.
        clip_model:   open_clip модель (см. load_clip_model).
        preprocess:   val-трансформ из load_clip_model.
        device:       куда класть тензоры для инференса.

    Returns:
        Скаляр в [-1, 1] — среднее cos(emb_i, emb_{i+1}).
        Больше → кадры семантически стабильнее (меньше «мерцания»).

    Алгоритм:
        1. frames_tensor = torch.stack([preprocess(f) for f in video_frames]).to(device)
        2. with torch.no_grad(): feats = clip_model.encode_image(frames_tensor)
        3. feats = F.normalize(feats, dim=-1)
        4. sims = (feats[:-1] * feats[1:]).sum(-1)  → вернуть sims.mean().item()

    TODO: реализовать. Оборачивай инференс в torch.no_grad().
    """
    if len(video_frames) < 2:
        return 1.0
    frames_tensor = torch.stack([preprocess(f) for f in video_frames]).to(device)
    with torch.no_grad():
        feats = clip_model.encode_image(frames_tensor)
        feats = F.normalize(feats, dim=-1)
        sims = (feats[:-1] * feats[1:]).sum(-1)
        return sims.mean().item()
