# Temporal Attention Survey (CogVideoX)

> [English version](README_en.md)

Исследуем внутреннюю механику **temporal attention** в CogVideoX — какие головы и слои отвечают за какой тип движения. Подход: сначала визуализируем что происходит, формулируем гипотезы, потом проверяем количественно через хирургическую аблацию конкретных голов.

Нарратив для портфолио: *«Не просто запускал модель — понял что происходит внутри, потом проверил количественно.»*

**Примерное время:** 10–14 дней (Фаза 1: 3–4 дня, Фаза 2: 4–5 дней, Фаза 3: 2–3 дня).

---

## Подготовка окружения

Зависимости и линтеры — в `pyproject.toml`, быстрые команды — в `Makefile`. Docker-образ собирается через `make docker`.

```bash
make prepare-environment   # venv + poetry (первый раз)
make install               # poetry install --no-root
```

**Веса модели.** Скачайте CogVideoX (например, `THUDM/CogVideoX-5b`) через Hugging Face и укажите путь в конфиге или переменной окружения (детали — в `configs/` и ноутбуках, по мере реализации). При использовании DVC для данных/весов: `make dvc-init`.

**Данные.** Референсные видео (baseline) и артефакты экспериментов кладите в `data/` и `results/` (см. `.gitignore` / DVC).

Опционально: `export.sh` в корне или `scripts/` для `HF_HOME`, путей к чекпоинтам и GPU — по аналогии с другими pet-projects.

---

## Quick start

Пока основной сценарий — ноутбуки и модули в `src/` (см. [Структура репозитория](#структура-репозитория)):

```bash
make install
poetry run jupyter lab notebooks/
```

Порядок работы:

1. `notebooks/01_baseline_generation.ipynb` — референсные видео  
2. `notebooks/02_attention_extraction.ipynb` — хуки и визуализация  
3. `notebooks/03_ablation_experiments.ipynb` — аблация и метрики  

Скрипты в `scripts/` — для пакетного запуска без ноутбуков (по мере добавления).

---

## Структура репозитория

```
temporal-attention-analysis/
├── notebooks/
│   ├── 01_baseline_generation.ipynb    # генерация референсных видео
│   ├── 02_attention_extraction.ipynb   # хуки и визуализация
│   └── 03_ablation_experiments.ipynb   # аблация и метрики
├── src/
│   ├── hooks.py        # register_attention_hooks, temporal_distance_profile
│   ├── ablation.py     # zero_out_heads, restore_heads, run_ablation
│   └── metrics.py      # motion_score, temporal_consistency
├── configs/            # пути к модели, промпты, параметры генерации
├── scripts/            # CLI / batch-запуск
├── data/               # входные данные, baseline-видео
├── results/
│   ├── attention_heatmaps/
│   ├── ablation_videos/
│   └── metrics.json
├── tests/
├── Dockerfile
├── Makefile
└── pyproject.toml
```

| Путь | Назначение |
|------|------------|
| `src/hooks.py` | Forward hooks, temporal distance profile, heatmaps |
| `src/ablation.py` | Обнуление / восстановление голов, протокол аблации |
| `src/metrics.py` | Optical flow motion score, CLIP temporal consistency |
| `results/` | Графики, JSON с метриками, GIF/MP4 baseline vs ablated |

Логирование экспериментов (ClearML и т.п.) — через зависимости в `pyproject.toml`, конфиги в `configs/` по необходимости.

---

## Архитектурный контекст

### CogVideoX и temporal attention

CogVideoX — DiT (Diffusion Transformer) с single-stream архитектурой. В отличие от image-моделей (FLUX, SD3), латент видео: `(B, C, T, H, W)` вместо `(B, C, H, W)`.

Ключевые параметры CogVideoX-5B:

- 42 трансформерных блока (`transformer_blocks[0..41]`)
- В каждом блоке **два attention**:
  - `attn1` — spatial (внутри кадра)
  - `temporal_attn` — temporal (между кадрами)
- Full 3D attention: все токены `T×H×W` через flash attention
- Позиции: 3D RoPE по осям `(t, x, y)`

### Как устроен один блок

```
CogVideoXBlock
├── norm1 + attn1 (spatial, inner_dim heads)
├── norm2 + temporal_attn (temporal, те же heads)
└── norm3 + ff (feed-forward)
```

Temporal attention: токен `(t1, h, w)` может смотреть на `(t2, h, w)` для любого `t2` — здесь кодируется движение.

### 3D RoPE

- `dim//4` — ось t  
- `dim//4` — ось x  
- `dim//2` — ось y  

RoPE на Q и K до attention (не к токенам). Паттерны внимания зависят от позиции `(t, x, y)`.

### Sequence length

После 3D VAE (4× по времени, 8× по пространству), 49 кадров, 480×720:

```
T=13, H=60, W=90  →  13 × 60 × 90 = 70 200 токенов
```

Полное хранение attention weights нереально — нужны хуки с усреднением на лету.

---

## Фаза 1: Visualization (3–4 дня)

### Шаг 1. Baseline — референсные видео

```python
prompts = [
    "A person walking in a park",
    "Ocean waves crashing on shore",
    "A car driving on a highway",
    "Leaves falling from a tree",
    "A dancing figure in a spotlight",
]
# Генерируем N=50 видео, сохраняем как reference
```

### Шаг 2. Извлечение attention maps через хуки

При 70K токенов матрица `(B, heads, 70200, 70200)` не влезает в память — усредняем на лету или берём только temporal slice.

```python
def register_attention_hooks(model):
    attention_maps = {}

    def hook_fn(name):
        def fn(module, input, output):
            # output[1] — attention weights (только если output_attentions=True)
            attn_weights = output[1].detach().cpu()  # (B, heads, T*H*W, T*H*W)
            attention_maps[name] = attn_weights.mean(0)  # (heads, seq, seq)
        return fn

    for i, block in enumerate(model.transformer_blocks):
        block.temporal_attn.register_forward_hook(hook_fn(f"layer_{i}"))
    return attention_maps
```

**Альтернатива:** `attn_implementation="eager"`, патч `scaled_dot_product_attention`, или кастомный attention processor (xformers).

### Шаг 3. Что визуализируем

**Temporal distance profile** — для каждой головы каждого слоя: на сколько кадров вперёд/назад смотрит внимание в среднем.

```python
def temporal_distance_profile(attn_weights, T, H, W):
    """
    attn_weights: (heads, T*H*W, T*H*W)
    Возвращает: (heads, 2*T-1) — распределение внимания по temporal distance
    """
    heads = attn_weights.shape[0]
    profile = torch.zeros(heads, 2 * T - 1)

    for t_q in range(T):
        for t_k in range(T):
            dist = t_k - t_q + (T - 1)
            q_start, q_end = t_q * H * W, (t_q + 1) * H * W
            k_start, k_end = t_k * H * W, (t_k + 1) * H * W
            profile[:, dist] += attn_weights[:, q_start:q_end, k_start:k_end].mean(dim=(-1, -2))

    return profile / T
```

**Layer-wise clustering** (KMeans по средним профилям):

```python
from sklearn.cluster import KMeans

features = np.stack([profiles[f"layer_{i}"].mean(0).numpy() for i in range(42)])
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(features)
```

**Heatmap (layer × head)** — доминирующая temporal distance:

```python
heatmap = np.array([[profiles[f"layer_{i}"][h].argmax().item()
                     for h in range(num_heads)]
                    for i in range(42)])
plt.imshow(heatmap, aspect='auto', cmap='viridis')
plt.xlabel("Head"); plt.ylabel("Layer")
plt.colorbar(label="Dominant temporal distance (frames)")
```

### Гипотезы по итогам фазы 1

| Layer range | Предполагаемый паттерн | Почему |
|-------------|------------------------|--------|
| 0–10        | Long-range temporal    | Глобальная структура движения |
| 10–30       | Local temporal + spatial | Движение объектов и детали |
| 30–42       | Short-range temporal   | Когерентность соседних кадров |

---

## Фаза 2: Targeted Ablation (4–5 дней)

Аблация **конкретных голов**, не целых слоёв.

### Хирургическое обнуление голов

```python
def zero_out_heads(model, layer_idx: int, head_indices: list[int]) -> dict:
    block = model.transformer_blocks[layer_idx]
    attn = block.temporal_attn
    head_dim = attn.inner_dim // attn.num_heads

    saved = {}
    for head_idx in head_indices:
        start = head_idx * head_dim
        end = start + head_dim
        saved[head_idx] = attn.to_out[0].weight.data[:, start:end].clone()
        attn.to_out[0].weight.data[:, start:end] = 0
    return saved

def restore_heads(model, layer_idx: int, saved: dict):
    block = model.transformer_blocks[layer_idx]
    attn = block.temporal_attn
    head_dim = attn.inner_dim // attn.num_heads
    for head_idx, weights in saved.items():
        start = head_idx * head_dim
        end = start + head_dim
        attn.to_out[0].weight.data[:, start:end] = weights
```

### Метрики качества

```python
import torch.nn.functional as F
from PIL import Image

def motion_score(video_frames: list) -> float:
    """Средний L2 optical flow между соседними кадрами."""
    import cv2
    scores = []
    for i in range(len(video_frames) - 1):
        f1 = cv2.cvtColor(np.array(video_frames[i]), cv2.COLOR_RGB2GRAY)
        f2 = cv2.cvtColor(np.array(video_frames[i + 1]), cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(f1, f2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        scores.append(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2).mean())
    return float(np.mean(scores))

def temporal_consistency(video_frames: list, clip_model, preprocess) -> float:
    frames_tensor = torch.stack([preprocess(Image.fromarray(np.array(f)))
                                 for f in video_frames])
    with torch.no_grad():
        features = clip_model.encode_image(frames_tensor.cuda())
    features = F.normalize(features, dim=-1)
    sims = (features[:-1] * features[1:]).sum(-1)
    return sims.mean().item()
```

### Протокол эксперимента

```python
def run_ablation(model, pipe, prompts, layer_idx, head_indices, n_videos=10):
    results = {"baseline": [], "ablated": []}

    for prompt in prompts[:n_videos]:
        video = pipe(prompt, num_frames=49).frames[0]
        results["baseline"].append({
            "motion_score": motion_score(video),
            "temporal_consistency": temporal_consistency(video, clip_model, preprocess),
        })

    saved = zero_out_heads(model, layer_idx, head_indices)
    for prompt in prompts[:n_videos]:
        video = pipe(prompt, num_frames=49).frames[0]
        results["ablated"].append({
            "motion_score": motion_score(video),
            "temporal_consistency": temporal_consistency(video, clip_model, preprocess),
        })
    restore_heads(model, layer_idx, saved)

    return results
```

---

## Фаза 3: Анализ и оформление (2–3 дня)

### Итоговая таблица (заполняется по результатам)

| Layer range | Dominant attention pattern | Ablation effect on motion_score | Ablation effect on consistency |
|-------------|---------------------------|--------------------------------|-------------------------------|
| 0–10        | long-range temporal       | −X%                            | −Y%                           |
| 10–30       | local temporal + spatial  | −X%                            | −Y%                           |
| 30–42       | short-range temporal      | −X%                            | −Y%                           |

### Графики

1. **Violin plot** — temporal distance по слоям  
2. **Scatter** — `layer_idx` vs `delta_motion_score` после аблации  
3. **Видео** — baseline vs ablated (GIF/MP4)

---

## Технические нюансы

**Flash Attention не возвращает веса.** Варианты: `attn_implementation="eager"`, патч `F.scaled_dot_product_attention`, кастомный processor.

**Память.** В eager-режиме полная матрица на head — порядка сотен GB. Уменьшайте T/H/W для визуализации или агрегируйте в хуке, не сохраняя полную матрицу.

**Почему `to_out`:** выход head'а — `softmax(QK^T/√d)V`, затем проекция `to_out`. Обнуление столбцов `to_out` для выбранных голов убирает их вклад без правки Q/K/V.

---

## Ожидаемый результат

- Карта специализации голов (ближние vs дальние кадры по слоям)  
- Количественное подтверждение: аблация long-range → motion coherence, short-range → frame-to-frame consistency  
- Heatmaps, violin plots, видео baseline vs ablated в `results/` и в этом README  

---

## Полезные команды (Makefile)

| Команда | Описание |
|---------|----------|
| `make install` | Установка зависимостей (Poetry) |
| `make docker` / `make docker-push` | Сборка и публикация Docker-образа |
| `make dvc-init` | Инициализация DVC |
| `make lint` | mypy, ruff, mdformat check |
| `make refactor` | black, mdformat, yamlfix, ruff --fix |
| `make test` | pytest |
| `make smoke-test` | pytest `-m fast` |
| `make clean` | Очистка кэшей и артефактов сборки |

Зависимости и конфиги линтеров — в `pyproject.toml`. Если Poetry не подтягивает нужный CUDA torch, задайте wheel явно, например:

```toml
torch = [
    { url = "https://download.pytorch.org/whl/cu118/torch-2.0.1%2Bcu118-cp39-cp39-linux_x86_64.whl", platform = "linux" },
    { url = "https://download.pytorch.org/whl/cpu/torch-2.0.1-cp39-none-macosx_11_0_arm64.whl", platform = "darwin" },
]
```

Обновите `name` и `authors` в `pyproject.toml` под этот репозиторий.
