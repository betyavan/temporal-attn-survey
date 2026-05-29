"""
src/hooks.py — forward hooks for CogVideoX temporal attention.

Задача: реализовать механизм перехвата весов temporal attention без
материализации полной матрицы (T*H*W, T*H*W) — она не влезает в память.

Ключевые факты:
- CogVideoX-5B: 42 блока, каждый с attn1 (full 3D unified attention, spatial+temporal)
- При 49 кадрах 480×720: T=13, H=60, W=90 → 70 200 токенов
- Flash Attention не возвращает веса → нужен monkey-patch F.scaled_dot_product_attention
- Профиль агрегируется на лету в (heads, 2T-1), полная матрица нигде не хранится

Публичное API:
    HookState                — хранит накопленные профили
    temporal_distance_profile  — чистая функция: (heads, S, S) → (heads, 2T-1)
    register_temporal_hooks  — патчит SDPA + регистрирует хуки
    remove_hooks             — отменяет всё и восстанавливает оригинальный SDPA
    temporal_hook_context    — контекстный менеджер вокруг register/remove

Использование в ноутбуке:
    from src.hooks import temporal_hook_context

    with temporal_hook_context(pipe.transformer, T=13, H=60, W=90) as state:
        pipe(prompt, ...)

    profiles = state.profiles  # {layer_idx: Tensor(heads, 2T-1)}
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Модульный стейт для monkey-patch (один на всё время регистрации)
# ---------------------------------------------------------------------------

_thread_local = threading.local()
_original_sdpa: Optional[Any] = None   # сохраняем оригинал до патча
_active_state: Optional["HookState"] = None


# ---------------------------------------------------------------------------
# Контейнер профилей
# ---------------------------------------------------------------------------


@dataclass
class HookState:
    """Накапливает temporal distance профили по всем слоям во время forward pass."""

    T: int
    H: int
    W: int
    _profiles: dict[int, Tensor] = field(default_factory=dict, repr=False, init=False)
    _counts: dict[int, int] = field(default_factory=dict, repr=False, init=False)

    @property
    def seq_len(self) -> int:
        return self.T * self.H * self.W

    def _accumulate(self, layer_idx: int, profile: Tensor) -> None:
        """Прибавить profile к накопленной сумме для layer_idx.

        Если layer_idx ещё не встречался — сохранить клон и поставить счётчик 1.
        Если уже есть — прибавить inplace и увеличить счётчик.
        """
        if layer_idx not in self._profiles:
            self._profiles[layer_idx] = profile.clone()
        else:
            self._profiles[layer_idx] += profile

        self._counts.setdefault(layer_idx, 0)
        self._counts[layer_idx] += 1

    @property
    def profiles(self) -> dict[int, Tensor]:
        """Вернуть усреднённые профили: {layer_idx: Tensor(heads, 2T-1)}.

        Разделить _profiles[idx] на _counts[idx] для каждого idx.
        """
        return {layer_idx: self._profiles[layer_idx] / self._counts[layer_idx] for layer_idx in self._profiles}

    def reset(self) -> None:
        """Сбросить накопленные данные (удобно между шагами денойзинга).

        Очистить _profiles и _counts.
        """
        self._profiles.clear()
        self._counts.clear()


# ---------------------------------------------------------------------------
# Основная математика: temporal distance profile
# ---------------------------------------------------------------------------


def temporal_distance_profile(attn_weights: Tensor, T: int, H: int, W: int) -> Tensor:
    """Свернуть полную матрицу внимания в гистограмму по temporal distance.

    Args:
        attn_weights: (heads, T*H*W, T*H*W) — уже softmax-нормированные веса
        T, H, W: размеры латентной сетки по времени и пространству

    Returns:
        profile: (heads, 2*T-1) — средний вес внимания при каждом смещении
                 d = t_k - t_q, индексированном как d + (T-1)

    Hint: для каждой пары (t_q, t_k) вырежи блок attn_weights[:, q_start:q_end, k_start:k_end],
          возьми среднее по spatial dims и прибавь в profile[:, dist].
          В конце нормируй на T.
    """
    heads = attn_weights.shape[0]
    profile = torch.zeros(heads, 2 * T - 1, device=attn_weights.device, dtype=torch.float32)
    for t_q in range(T):
        for t_k in range(T):
            dist = t_k - t_q + (T - 1)
            q_start, q_end = t_q * H * W, (t_q + 1) * H * W # (T*H*W, T*H*W)
            k_start, k_end = t_k * H * W, (t_k + 1) * H * W # (T*H*W, T*H*W)
            attn_block = attn_weights[:, q_start:q_end, k_start:k_end] # (heads, T*H*W, T*H*W)
            profile[:, dist] += attn_block.mean(dim=(-1, -2)) # (heads, 1)
    return profile / T

def _compute_temporal_profile_online(q: Tensor, k: Tensor, T: int, H: int, W: int) -> Tensor:
    """Вычислить temporal_distance_profile без материализации полной матрицы.

    Вместо (heads, S, S) итерируется по парам кадров (t_q, t_k), вычисляя блоки
    (B, heads, H*W, H*W) по одному — пиковая память O(heads * H*W * H*W).

    Args:
        q, k: (B, heads, T*H*W, d) — query и key из temporal_attn
        T, H, W: размеры латентной сетки

    Returns:
        profile: (heads, 2*T-1) на том же устройстве, dtype=float32

    Hint:
        1. Reshape q, k → (B, heads, T, H*W, d), cast to float32
        2. Для каждой пары (t_q, t_k):
           - scores = q_slice @ k_slice.T * scale    # (B, heads, HW, HW)
           - attn_block = softmax(scores, dim=-1)
           - profile[:, dist] += attn_block.mean(dim=(0, -2, -1))
        3. Нормируй на T
    """
    B, heads, _, d = q.shape
    q = q.reshape(B, heads, T, H * W, d).float()
    k = k.reshape(B, heads, T, H * W, d).float()
    profile = torch.zeros(heads, 2 * T - 1, device=q.device, dtype=torch.float32)
    for t_q in range(T):
        for t_k in range(T):
            dist = t_k - t_q + (T - 1)
            q_slice = q[:, :, t_q, :, :] # (B, heads, HW, d)
            k_slice = k[:, :, t_k, :, :] # (B, heads, HW, d)
            scores = q_slice @ k_slice.transpose(-2, -1) / d**0.5 # (B, heads, HW, HW)
            attn_block = F.softmax(scores, dim=-1) # (B, heads, HW, HW)
            profile[:, dist] += attn_block.mean(dim=(0, -2, -1))
    return profile / T


# ---------------------------------------------------------------------------
# Monkey-patched SDPA
# ---------------------------------------------------------------------------


def _patched_sdpa(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    **kwargs: Any,
) -> Tensor:
    """Замена F.scaled_dot_product_attention.

    Логика:
    1. Вызвать оригинальный SDPA (Flash Attention) → получить корректный output.
    2. Если _thread_local.temporal_layer_idx установлен И длина последовательности
       совпадает с _active_state.seq_len — вычислить temporal profile через
       _compute_temporal_profile_online и накопить в _active_state.

    TODO: реализовать описанную логику.
          Оборачивай шаг 2 в torch.no_grad() и .detach() для query/key.
    """
    assert _original_sdpa is not None, "hooks not registered"
    logger.info("patched_sdpa: layer %s", getattr(_thread_local, "temporal_layer_idx", None))
    output = _original_sdpa(query, key, value, **kwargs)
    with torch.no_grad():
        layer_idx = getattr(_thread_local, "temporal_layer_idx", None)
        if layer_idx is None:
            return output
        video_len = _active_state.seq_len  # T*H*W
        if query.shape[2] < video_len:
            return output
        # CogVideoX single-stream: text tokens precede video tokens in the sequence
        q_video = query.detach()[:, :, -video_len:, :]
        k_video = key.detach()[:, :, -video_len:, :]
        profile = _compute_temporal_profile_online(
            q=q_video,
            k=k_video,
            T=_active_state.T,
            H=_active_state.H,
            W=_active_state.W,
        )
        _active_state._accumulate(
            layer_idx=layer_idx,
            profile=profile,
        )
    return output


# ---------------------------------------------------------------------------
# Публичный API
# ---------------------------------------------------------------------------


def register_temporal_hooks(model: Any, T: int, H: int, W: int) -> tuple[HookState, list[Any]]:
    """Запатчить SDPA и зарегистрировать pre/post хуки на каждый temporal_attn.

    Args:
        model: CogVideoXTransformer3DModel (нужен атрибут .transformer_blocks)
        T, H, W: ожидаемые размеры латента во время инференса

    Returns:
        (state, handles) — state накапливает профили; handles передать в remove_hooks().

    Raises:
        RuntimeError: если хуки уже зарегистрированы.

    TODO:
        1. Проверить, что _original_sdpa is None (иначе RuntimeError).
        2. Сохранить F.scaled_dot_product_attention в _original_sdpa.
        3. Поставить F.scaled_dot_product_attention = _patched_sdpa.
        4. Создать HookState(T, H, W) и записать в _active_state.
        5. Для каждого block в model.transformer_blocks:
           - pre-hook: установить _thread_local.temporal_layer_idx = i
           - post-hook: сбросить _thread_local.temporal_layer_idx = None
           - добавить хэндлы в список.
        6. Вернуть (state, handles).
    """
    global _original_sdpa, _active_state
    if _original_sdpa is not None:
        raise RuntimeError("hooks already registered; call remove_hooks() first")
    _original_sdpa = F.scaled_dot_product_attention
    F.scaled_dot_product_attention = _patched_sdpa
    _active_state = HookState(T, H, W)
    handles = []
    for i, block in enumerate(model.transformer_blocks):
        def _pre_hook(module: Any, input: Any, _i: int = i) -> None:
            _thread_local.temporal_layer_idx = _i
        def _post_hook(module: Any, input: Any, output: Any) -> None:
            _thread_local.temporal_layer_idx = None
        handle = block.attn1.register_forward_pre_hook(_pre_hook)
        handles.append(handle)
        handle = block.attn1.register_forward_hook(_post_hook)
        handles.append(handle)
    return _active_state, handles


def remove_hooks(handles: list[Any]) -> None:
    """Удалить все хуки и восстановить оригинальный SDPA.

    TODO:
        1. Вызвать h.remove() для каждого handle.
        2. Восстановить F.scaled_dot_product_attention из _original_sdpa.
        3. Обнулить _original_sdpa и _active_state.
    """
    global _original_sdpa, _active_state
    for h in handles:
        h.remove()
    F.scaled_dot_product_attention = _original_sdpa
    _original_sdpa = None
    _active_state = None


@contextmanager
def temporal_hook_context(model: Any, T: int, H: int, W: int) -> Generator[HookState, None, None]:
    """Контекстный менеджер: register_temporal_hooks на входе, remove_hooks на выходе."""
    state, handles = register_temporal_hooks(model, T, H, W)
    try:
        yield state
    finally:
        remove_hooks(handles)
