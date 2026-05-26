# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A research project investigating the internal mechanics of **temporal attention in CogVideoX** — which heads and which layers encode which kind of motion. The methodology is two-staged: visualize attention patterns first, form hypotheses, then verify them quantitatively via surgical ablation of individual heads (not whole layers).

The repo follows a standard Poetry + Makefile + Docker layout (DVC/ClearML, Russian-language READMEs). What's specific to this repo is in the sections below.

## Current state of the codebase

This is a **fresh repo with the plan written but most code not yet implemented**. As of the last commit:

- `src/`, `scripts/`, `configs/`, `data/`, `assets/` contain **only stub `README.md` files** — none of the modules referenced in the main README (`src/hooks.py`, `src/ablation.py`, `src/metrics.py`) exist yet.
- `notebooks/` is empty apart from a README; the three planned notebooks (`01_baseline_generation.ipynb`, `02_attention_extraction.ipynb`, `03_ablation_experiments.ipynb`) have not been created.
- `tests/` contains a single placeholder (`simple_test.py`) marked `@pytest.mark.fast`.
- `pyproject.toml` still has the template's name (`research-template`) and minimal deps — `diffusers`, `transformers`, `accelerate`, `opencv`, CLIP, etc. will need to be added when implementing Phase 1.

When asked to implement Phase 1/2/3 work, treat `README.md` as the spec — it contains the concrete code sketches (hook registration, `temporal_distance_profile`, `zero_out_heads`/`restore_heads`, `motion_score`, `temporal_consistency`, `run_ablation`) intended to live in `src/`. There is also an English mirror in `README_en.md` — keep both in sync if you change the plan.

The README's "Структура репозитория" diagram shows a `results/` directory that does not exist on disk; create it when the first artifacts are produced and add it to DVC rather than git.

## Domain context that shapes the code

CogVideoX-5B is a **single-stream DiT** with 42 transformer blocks. Each block has two attention modules — `attn1` (spatial, intra-frame) and `temporal_attn` (inter-frame) — both with the same number of heads. 3D RoPE is split `dim//4` (t) + `dim//4` (x) + `dim//2` (y) and is applied to Q/K. After the 3D VAE (4× temporal, 8× spatial downsampling), a 49-frame 480×720 video has shape `T=13, H=60, W=90` → ~70 200 tokens, so a full `(B, heads, 70200, 70200)` attention matrix **cannot be materialized**. Two consequences:

1. Hooks must aggregate on the fly (e.g. accumulate into a `(heads, 2T-1)` temporal-distance profile) rather than save the full matrix. The README's `temporal_distance_profile` is the canonical reduction.
2. Flash Attention does not expose attention weights. To get them, switch to `attn_implementation="eager"`, monkey-patch `F.scaled_dot_product_attention`, or write a custom attention processor — pick one and stay consistent.

Ablation works on `to_out` (the output projection of `temporal_attn`), zeroing the columns corresponding to a head's slice (`[head_idx*head_dim : (head_idx+1)*head_dim]`). This kills the head's contribution without touching Q/K/V — important if you also want to reuse cached pre-attention activations. Always pair `zero_out_heads` with `restore_heads` (save weights, mutate, restore) so the same model instance can run baseline and ablated back-to-back. The README's `run_ablation` is the reference protocol.

Phase 2 metrics: `motion_score` is mean L2 of Farneback optical flow between adjacent frames (cv2); `temporal_consistency` is mean cosine similarity of adjacent-frame CLIP image embeddings. Both expect a list of PIL frames as produced by the diffusers pipeline (`pipe(...).frames[0]`).

## Standard commands

This repo follows the standard `make` targets. The ones that come up day-to-day:

| Command | Purpose |
| --- | --- |
| `make prepare-environment` | Create `.venv`, install Poetry into it (once). |
| `make install` | `poetry install --no-root` (note: `--no-root`, not the template default). |
| `make smoke-test` | `pytest -vv -m fast tests` — the only test currently exists and is marked `fast`. |
| `make test` | Full `pytest -vv tests`. |
| `make lint` | `mypy --strict` + `ruff check` + `mdformat --check` + `yamlfix --check`. |
| `make refactor` | `black` + `mdformat` + `yamlfix` + `ruff --fix`. |
| `make check` | `refactor` + `lint` + `test` + `clean`. |
| `make docker` | Builds `temporal-attn-survey:latest` (image name = directory name). |

Single test: `poetry run pytest -vv tests/simple_test.py::test_simple`.

`mypy --strict` runs over every `.py` file found by `find_all_py` in the Makefile — notebooks are excluded via the `[tool.mypy] exclude` regex in `pyproject.toml`, but anything under `src/`, `scripts/`, `tests/` is checked strictly. Plan for that when adding new modules (full annotations, no implicit Any).

The Dockerfile is based on `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime` and runs `poetry lock` inside the image — if you change `pyproject.toml`, the rebuild will regenerate the lock file from scratch (poetry.lock is checked in but `*.lock` is in `.gitignore`, which is inconsistent — flag this if it becomes a problem).

## Environment expectations

- Model weights: CogVideoX (`THUDM/CogVideoX-5b` or similar) via Hugging Face. Use the shared cache at `/devstorage/all_cache` (export `HF_HOME`, `HF_HUB_CACHE`, `TRANSFORMERS_CACHE`, `HF_DATASETS_CACHE` to point there) rather than re-downloading. The devcontainer also bind-mounts `${HOME}/model-weights` to `/model-weights` and runs with `--gpus=all --shm-size=10g`.
- DVC is initialized (`.dvc/` is present) but no remote has been configured yet in this repo. When the first artifact appears, set up the S3 remote per the workspace conventions before running `dvc push`.
- The `.gitignore` is minimal (`.github`, `.DS_Store`, `*.lock`) — when adding generated outputs (videos, heatmaps, `metrics.json`), extend `.gitignore` or move them under DVC. Don't commit `results/` blobs.

## Conventions worth knowing

- Notebooks are for throwaway exploration and finished demos with cleared outputs; anything load-bearing should live in `src/` as `.py`. This rule is loud in `notebooks/README.md` and is a team convention, not just a preference.
- The main README is in Russian; `README_en.md` is the English mirror. If you change one, change the other.
- The mypy config sets `disallow_any_generics = false` and `disallow_untyped_calls = false` — so generic `dict`/`list` without parameters and calls into untyped third-party libs are fine, but everything you write must have signatures.
