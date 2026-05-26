# Temporal Attention Survey (CogVideoX)

> [Русская версия](README.md)

We study how **temporal attention** works inside CogVideoX — which heads and layers drive which kinds of motion. The workflow is visualize first, form hypotheses, then validate with surgical ablation of individual heads.

Portfolio narrative: *"I didn't just run the model — I understood what happens inside, then verified it quantitatively."*

**Estimated timeline:** 10–14 days (Phase 1: 3–4 days, Phase 2: 4–5 days, Phase 3: 2–3 days).

---

## Environment setup

Dependencies and linters live in `pyproject.toml`; quick commands in `Makefile`. Build the Docker image with `make docker`.

```bash
make prepare-environment   # venv + poetry (first time)
make install               # poetry install --no-root
```

**Model weights.** Download CogVideoX (e.g. `THUDM/CogVideoX-5b`) from Hugging Face and set the path in config or an environment variable (details in `configs/` and notebooks as they are added). For versioned data/weights with DVC: `make dvc-init`.

**Data.** Put reference (baseline) videos and experiment artifacts under `data/` and `results/` (see `.gitignore` / DVC).

Optional: `export.sh` at the repo root or under `scripts/` for `HF_HOME`, checkpoint paths, and GPU settings — same pattern as other pet-projects.

---

## Quick start

The main path for now is notebooks plus modules in `src/` (see [Repository layout](#repository-layout)):

```bash
make install
poetry run jupyter lab notebooks/
```

Suggested order:

1. `notebooks/01_baseline_generation.ipynb` — reference videos  
2. `notebooks/02_attention_extraction.ipynb` — hooks and visualization  
3. `notebooks/03_ablation_experiments.ipynb` — ablation and metrics  

Scripts under `scripts/` are for batch runs without notebooks (as they are added).

---

## Repository layout

```
temporal-attention-analysis/
├── notebooks/
│   ├── 01_baseline_generation.ipynb    # reference video generation
│   ├── 02_attention_extraction.ipynb   # hooks and visualization
│   └── 03_ablation_experiments.ipynb   # ablation and metrics
├── src/
│   ├── hooks.py        # register_attention_hooks, temporal_distance_profile
│   ├── ablation.py     # zero_out_heads, restore_heads, run_ablation
│   └── metrics.py      # motion_score, temporal_consistency
├── configs/            # model paths, prompts, generation params
├── scripts/            # CLI / batch runs
├── data/               # inputs, baseline videos
├── results/
│   ├── attention_heatmaps/
│   ├── ablation_videos/
│   └── metrics.json
├── tests/
├── Dockerfile
├── Makefile
└── pyproject.toml
```

| Path | Role |
|------|------|
| `src/hooks.py` | Forward hooks, temporal distance profile, heatmaps |
| `src/ablation.py` | Zero / restore heads, ablation protocol |
| `src/metrics.py` | Optical-flow motion score, CLIP temporal consistency |
| `results/` | Plots, metrics JSON, baseline vs ablated GIF/MP4 |

Experiment logging (ClearML, etc.) via `pyproject.toml` dependencies; configs in `configs/` as needed.

---

## Architecture context

### CogVideoX and temporal attention

CogVideoX is a DiT (Diffusion Transformer) with a single-stream design. Unlike image models (FLUX, SD3), the video latent is `(B, C, T, H, W)` instead of `(B, C, H, W)`.

CogVideoX-5B essentials:

- 42 transformer blocks (`transformer_blocks[0..41]`)
- Each block has **two attention modules**:
  - `attn1` — spatial (within a frame)
  - `temporal_attn` — temporal (across frames)
- Full 3D attention: all `T×H×W` tokens via flash attention
- Positions: 3D RoPE over `(t, x, y)`

### One block

```
CogVideoXBlock
├── norm1 + attn1 (spatial, inner_dim heads)
├── norm2 + temporal_attn (temporal, same head count)
└── norm3 + ff (feed-forward)
```

In temporal attention, token `(t1, h, w)` can attend to `(t2, h, w)` for any `t2` — this is where motion is encoded.

### 3D RoPE

- `dim//4` — time axis t  
- `dim//4` — spatial axis x  
- `dim//2` — spatial axis y  

RoPE is applied to Q and K before attention (not added to token embeddings). Attention patterns depend on position `(t, x, y)`.

### Sequence length

After the 3D VAE (4× temporal, 8× spatial), for 49 frames at 480×720:

```
T=13, H=60, W=90  →  13 × 60 × 90 = 70,200 tokens
```

Storing full attention weights is infeasible — use hooks with on-the-fly aggregation.

---

## Phase 1: Visualization (3–4 days)

### Step 1. Baseline — reference videos

```python
prompts = [
    "A person walking in a park",
    "Ocean waves crashing on shore",
    "A car driving on a highway",
    "Leaves falling from a tree",
    "A dancing figure in a spotlight",
]
# Generate N=50 videos and save as reference set
```

### Step 2. Extract attention maps via hooks

With ~70K tokens, `(B, heads, 70200, 70200)` does not fit in memory — aggregate on the fly or keep only the temporal slice.

```python
def register_attention_hooks(model):
    attention_maps = {}

    def hook_fn(name):
        def fn(module, input, output):
            # output[1] — attention weights (only if output_attentions=True)
            attn_weights = output[1].detach().cpu()  # (B, heads, T*H*W, T*H*W)
            attention_maps[name] = attn_weights.mean(0)  # (heads, seq, seq)
        return fn

    for i, block in enumerate(model.transformer_blocks):
        block.temporal_attn.register_forward_hook(hook_fn(f"layer_{i}"))
    return attention_maps
```

**Alternatives:** `attn_implementation="eager"`, patch `scaled_dot_product_attention`, or a custom attention processor (xformers).

### Step 3. What to visualize

**Temporal distance profile** — per head per layer: how many frames forward/backward attention looks on average.

```python
def temporal_distance_profile(attn_weights, T, H, W):
    """
    attn_weights: (heads, T*H*W, T*H*W)
    Returns: (heads, 2*T-1) — attention mass vs temporal distance
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

**Layer-wise clustering** (KMeans on mean profiles):

```python
from sklearn.cluster import KMeans

features = np.stack([profiles[f"layer_{i}"].mean(0).numpy() for i in range(42)])
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(features)
```

**Heatmap (layer × head)** — dominant temporal distance:

```python
heatmap = np.array([[profiles[f"layer_{i}"][h].argmax().item()
                     for h in range(num_heads)]
                    for i in range(42)])
plt.imshow(heatmap, aspect='auto', cmap='viridis')
plt.xlabel("Head"); plt.ylabel("Layer")
plt.colorbar(label="Dominant temporal distance (frames)")
```

### Hypotheses after phase 1

| Layer range | Expected pattern | Rationale |
|-------------|------------------|-----------|
| 0–10        | Long-range temporal | Global motion structure |
| 10–30       | Local temporal + spatial | Object motion and detail |
| 30–42       | Short-range temporal | Neighbor-frame coherence |

---

## Phase 2: Targeted ablation (4–5 days)

Ablate **specific heads**, not entire layers.

### Surgical head zeroing

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

### Quality metrics

```python
import torch.nn.functional as F
from PIL import Image

def motion_score(video_frames: list) -> float:
    """Mean L2 optical flow magnitude between consecutive frames."""
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

### Experiment protocol

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

## Phase 3: Analysis and write-up (2–3 days)

### Summary table (fill from results)

| Layer range | Dominant attention pattern | Ablation effect on motion_score | Ablation effect on consistency |
|-------------|---------------------------|--------------------------------|-------------------------------|
| 0–10        | long-range temporal       | −X%                            | −Y%                           |
| 10–30       | local temporal + spatial  | −X%                            | −Y%                           |
| 30–42       | short-range temporal      | −X%                            | −Y%                           |

### Figures

1. **Violin plot** — temporal distance by layer  
2. **Scatter** — `layer_idx` vs `delta_motion_score` after ablation  
3. **Videos** — baseline vs ablated (GIF/MP4)

---

## Technical notes

**Flash Attention does not return weights.** Options: `attn_implementation="eager"`, patch `F.scaled_dot_product_attention`, or a custom processor.

**Memory.** In eager mode, a full matrix per head can reach hundreds of GB. Reduce T/H/W for visualization or aggregate inside the hook instead of storing the full matrix.

**Why `to_out`:** each head output is `softmax(QK^T/√d)V`, then projected through `to_out`. Zeroing the corresponding columns of `to_out` removes those heads' contribution without editing Q/K/V.

---

## Expected deliverables

- Head specialization map (near vs far frames across layers)  
- Quantitative checks: long-range ablation → motion coherence drop; short-range → frame-to-frame consistency drop  
- Heatmaps, violin plots, and baseline vs ablated videos in `results/` and documented here  

---

## Makefile commands

| Command | Description |
|---------|-------------|
| `make install` | Install dependencies (Poetry) |
| `make docker` / `make docker-push` | Build and push Docker image |
| `make dvc-init` | Initialize DVC |
| `make lint` | mypy, ruff, mdformat check |
| `make refactor` | black, mdformat, yamlfix, ruff --fix |
| `make test` | pytest |
| `make smoke-test` | pytest `-m fast` |
| `make clean` | Remove caches and build artifacts |

Linter config lives in `pyproject.toml`. If Poetry cannot resolve the right CUDA torch build, pin wheels explicitly, e.g.:

```toml
torch = [
    { url = "https://download.pytorch.org/whl/cu118/torch-2.0.1%2Bcu118-cp39-cp39-linux_x86_64.whl", platform = "linux" },
    { url = "https://download.pytorch.org/whl/cpu/torch-2.0.1-cp39-none-macosx_11_0_arm64.whl", platform = "darwin" },
]
```

Update `name` and `authors` in `pyproject.toml` for this repository.
