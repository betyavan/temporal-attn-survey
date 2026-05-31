"""
src/hooks.py — forward hooks for CogVideoX temporal attention.

Задача: реализовать механизм перехвата весов temporal attention без
материализации полной матрицы (T*H*W, T*H*W) — она не влезает в память.

Ключевые факты:
- CogVideoX-5B: 42 блока, каждый с attn1 (full 3D unified attention, spatial+temporal)
- При 49 кадрах 480×720: после VAE T=13, H=60, W=90; после patch_size=2: T=13, H=30, W=45 → 17 550 видео-токенов + ~226 текстовых
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
    every_n_steps: int = 5  # собирать профиль только каждые N шагов денойзинга
    _profiles: dict[int, Tensor] = field(default_factory=dict, repr=False, init=False)
    _counts: dict[int, int] = field(default_factory=dict, repr=False, init=False)
    _step: int = field(default=0, repr=False, init=False)
    _collecting: bool = field(default=False, repr=False, init=False)

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
        self._step = 0
        self._collecting = False


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

_SPATIAL_STRIDE = 4  # субдискретизация пространственных позиций для ускорения


def _compute_temporal_profile_online(q: Tensor, k: Tensor, T: int, H: int, W: int) -> Tensor:
    """Вычислить temporal_distance_profile без материализации полной матрицы.

    Что измеряется
    --------------
    Для каждой головы и каждого расстояния d = t_k - t_q ∈ [-(T-1), T-1]:
    какую долю своего внимания токен из кадра t_q в среднем направляет
    на кадр t_k? Результат — гистограмма profile[heads, 2T-1], индексированная
    как d + (T-1).

    Почему softmax должен быть по всем T·HW ключам
    ------------------------------------------------
    Реальный attention для токена q_i вычисляет:

        attn[j] = exp(q_i · k_j / √d) / Σ_{all j} exp(q_i · k_j / √d)

    где сумма в знаменателе идёт по ВСЕМ T·HW позициям. Суммарная доля
    внимания, уделённая кадру t_k:

        mass(t_k) = Σ_{j ∈ t_k} attn[j]

    Если применять softmax только по HW ключам одного кадра (как делала
    старая реализация), mean(softmax(x)) = 1/HW — константа, не зависящая
    от Q и K. Любой профиль вырождается в треугольник с соотношением
    пиков T:1, одинаковый для всех голов, слоёв и промптов.

    Алгоритм
    ---------
    Итерируем по t_q. Для каждого запросного кадра:
      1. Один матмул q_slice @ k_all.T → (B, heads, HW', T·HW')
         где k_all — все T ключевых кадров конкатенированы.
      2. softmax(dim=-1) нормирует по всем T·HW' ключам — корректный знаменатель.
      3. Для каждого t_k срезаем соответствующий блок шириной HW' и суммируем
         по ключевым позициям → mass(t_k) ∈ [0, 1] на токен.
      4. Усредняем по батчу и запросным позициям → скаляр на голову.
    Делим на T чтобы усреднить по запросным кадрам.

    Пространственная субдискретизация (_SPATIAL_STRIDE) берёт каждую N-ю
    позицию в сетке HW. Профиль усредняется по позициям, поэтому выборка
    статистически эквивалентна полному набору при многократно меньшей памяти.
    Пиковая память: O(B · heads · HW' · T·HW'), ~280 MB при stride=4, T=13.

    Args:
        q, k: (B, heads, T*H*W, d) — query и key из temporal_attn
        T, H, W: размеры латентной сетки

    Returns:
        profile: (heads, 2*T-1) — доля внимания при каждом temporal смещении,
                 усреднённая по запросным кадрам и позициям.
    """
    B, heads, _, d = q.shape
    HW = H * W
    q = q.reshape(B, heads, T, HW, d).float()
    k = k.reshape(B, heads, T, HW, d).float()
    scale = d ** -0.5

    # Субдискретизация по пространству: профиль усредняется по позициям,
    # поэтому случайная выборка позиций даёт тот же результат при меньшей памяти.
    idx = torch.arange(0, HW, _SPATIAL_STRIDE, device=q.device)
    q = q[:, :, :, idx, :]  # (B, heads, T, HW', d)
    k = k[:, :, :, idx, :]
    HW_sub = len(idx)

    k_all = k.reshape(B, heads, T * HW_sub, d)  # (B, heads, T*HW', d) — все ключи
    profile = torch.zeros(heads, 2 * T - 1, device=q.device, dtype=torch.float32)

    for t_q in range(T):
        q_slice = q[:, :, t_q]  # (B, heads, HW', d)
        # Один матмул: scores по ВСЕМ T*HW' ключам → корректный softmax
        scores = q_slice @ k_all.transpose(-2, -1) * scale  # (B, heads, HW', T*HW')
        attn = F.softmax(scores, dim=-1)
        for t_k in range(T):
            dist = t_k - t_q + (T - 1)
            s, e = t_k * HW_sub, (t_k + 1) * HW_sub
            attn_mass = attn[:, :, :, s:e].sum(dim=-1).mean(dim=(0, -1))  # (heads,)
            profile[:, dist] += attn_mass

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
        # layer_idx==0 означает начало нового шага денойзинга
        if layer_idx == 0:
            _active_state._collecting = (_active_state._step % _active_state.every_n_steps == 0)
            _active_state._step += 1
        if not _active_state._collecting:
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


def register_temporal_hooks(model: Any, T: int, H: int, W: int, every_n_steps: int = 5) -> tuple[HookState, list[Any]]:
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
    _active_state = HookState(T, H, W, every_n_steps=every_n_steps)
    handles = []
    for i, block in enumerate(model.transformer_blocks):
        def _pre_hook(module: Any, input: Any, _i: int = i) -> None:
            _thread_local.temporal_layer_idx = _i
        def _post_hook(module: Any, input: Any, output: Any) -> None:
            _thread_local.temporal_layer_idx = None
        handle = block.register_forward_pre_hook(_pre_hook)
        handles.append(handle)
        handle = block.register_forward_hook(_post_hook)
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
def temporal_hook_context(model: Any, T: int, H: int, W: int, every_n_steps: int = 5) -> Generator[HookState, None, None]:
    """Контекстный менеджер: register_temporal_hooks на входе, remove_hooks на выходе."""
    state, handles = register_temporal_hooks(model, T, H, W, every_n_steps=every_n_steps)
    try:
        yield state
    finally:
        remove_hooks(handles)
