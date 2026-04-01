# EditReward-Bench Evaluation Guide
**(2-Pair, 3-Pair, 4-Pair Evaluation Using Our Framework)**

This guide explains how to evaluate multimodal LLM judges on EditReward-Bench using the scripts under `evaluate/`.

---

## 1. New Features

This evaluation folder now supports:

1. `Qwen3.5` via an OpenAI-compatible API wrapper under `genaibench/mllm_tools/`
2. loading the benchmark from either:
   - Hugging Face dataset name like `TIGER-Lab/EditReward-Bench`
   - a local dataset path passed to `--dataset_name`
3. one-command evaluation over all pair settings with `run_all_pairs_hf.py`

Supported registered model names:

- `qwen3.5`
- `qwen3.5-27b`

---

## 2. Overview

The evaluation pipeline:

1. Loads EditReward-Bench from Hugging Face or a local path
2. Builds a multimodal prompt with source image, edited candidates, and instruction
3. Calls your judge model through a wrapper under `genaibench/mllm_tools/`
4. Parses the final verdict like `A>B`, `B>A`, or `A=B`
5. Reports accuracy and saves detailed JSON results

---

## 3. Qwen3.5 Setup

The current `qwen3.5` wrapper expects an OpenAI-compatible endpoint.

Example environment:

```bash
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="EMPTY"
export QWEN3_5_MODEL_NAME="Qwen3.5-27B"
```

If your served model name differs, change `QWEN3_5_MODEL_NAME` accordingly.

---

## 4. Run Single Pair Evaluation

### 2-pair quick test

```bash
python inference_2pair_hf.py \
  --model_name qwen3.5 \
  --dataset_name TIGER-Lab/EditReward-Bench \
  --max_examples 50 \
  --results_dir results_test_2pair \
  --max_workers 8 \
  --overwrite
```

### 2-pair with a local dataset path

```bash
python inference_2pair_hf.py \
  --model_name qwen3.5 \
  --dataset_name /path/to/EditReward-Bench \
  --max_examples 50 \
  --results_dir results_local_2pair \
  --max_workers 8 \
  --overwrite
```

### 3-pair

```bash
python inference_3pair_hf.py \
  --model_name qwen3.5 \
  --dataset_name /path/to/EditReward-Bench \
  --max_examples 20 \
  --results_dir results_local_3pair \
  --max_workers 8 \
  --overwrite
```

### 4-pair

```bash
python inference_4pair_hf.py \
  --model_name qwen3.5 \
  --dataset_name /path/to/EditReward-Bench \
  --max_examples 20 \
  --results_dir results_local_4pair \
  --max_workers 8 \
  --overwrite
```

---

## 5. Run All Pairs

Use `run_all_pairs_hf.py` to run `2pair`, `3pair`, and `4pair` in one command.

### Quick test

```bash
python run_all_pairs_hf.py \
  --model_name qwen3.5 \
  --dataset_name /path/to/EditReward-Bench \
  --max_examples 20 \
  --max_workers 8 \
  --results_root results_qwen35_allpairs_test \
  --overwrite
```

### Full run

```bash
python run_all_pairs_hf.py \
  --model_name qwen3.5 \
  --dataset_name /path/to/EditReward-Bench \
  --max_workers 16 \
  --results_root outputs/results_qwen35_allpairs_full \
  --overwrite
```

You can also run only part of the benchmark:

```bash
python run_all_pairs_hf.py \
  --model_name qwen3.5 \
  --dataset_name /path/to/EditReward-Bench \
  --pairs 2pair,3pair \
  --max_examples 50 \
  --max_workers 8 \
  --results_root results_qwen35_partial \
  --overwrite
```

---

## 6. Output Files

Per-pair outputs are saved as:

```bash
<results_root>/<pair_type>/<model_name>_<template>_<pair_type>_hf.json
```

These JSON files include:

- total examples
- correct examples
- accuracy or group accuracy
- per-sample instruction
- human vote
- model response
- parsed model vote
- whether the prediction was correct

---

## 7. Notes

- `--dataset_name` now accepts both a remote HF dataset name and a local dataset directory.
- `run_all_pairs_hf.py` is intended for convenience and simply forwards shared arguments into the individual pair scripts.
- Generated result directories should not be committed; they are treated as local experiment artifacts.

