# PaddleOCR notebooks (Kaggle, single T4)

> **Layout:** `ocr_vl16_comparison.ipynb` (here) is the current combined notebook. The single-model
> notebooks below live in `legacy/`. Executed runs + results are under `../experiments/` — see
> `../experiments/README.md` for the run log, and `../README.md` for the repo overview.

## PaddleOCR model benchmark on Kaggle (single T4)

Five notebooks to compare PaddleOCR models on **your own images**, on a **single Kaggle T4 GPU**.
Each notebook reports **GPU peak memory**, **disk usage** (model cache + outputs),
**per-image latency**, **model info**, saves **JSON + annotated images (+ Markdown)**,
shows every output figure **inline in the cell**, and bundles everything into a **downloadable ZIP**.

## Notebooks

| Notebook | Model | Notes |
|---|---|---|
| `ocr_vl16_comparison.ipynb` | **small + medium + StructureV3 vs VL 1.6** | **One notebook.** Runs the three light models on `test-dataset-100`, uses PaddleOCR-VL 1.6 as a pseudo-reference (no ground truth), and reports quality-vs-VL (word F1 / char-sim / table recall), coverage, confidence, cross-model agreement, and speed/GPU/disk. Warning-free output, 5 sample images, figures, one ZIP. |
| `paddleocr_vl_1.6.ipynb` | **PaddleOCR-VL 1.6** (0.9B VLM) | SOTA doc parser. Native Paddle engine (not vLLM). |
| `pp_ocrv6_tiny.ipynb`   | **PP-OCRv6 tiny** (1.5M) | Detection + recognition OCR. |
| `pp_ocrv6_small.ipynb`  | **PP-OCRv6 small** (7.7M) | Detection + recognition OCR. |
| `pp_ocrv6_medium.ipynb` | **PP-OCRv6 medium** (34.5M) | Detection + recognition OCR. |
| `pp_structurev3.ipynb`  | **PP-StructureV3** | Full layout / table / formula / chart pipeline → Markdown. |

PP-OCRv6 is split into 3 notebooks (per your choice) so each size is measured in its own
isolated Kaggle session for the cleanest per-model GPU/disk numbers.

## Kaggle setup (per notebook)

1. **Accelerator** → `GPU T4 x2` (the code pins `device='gpu:0'`, i.e. one T4).
2. **Internet** → **On** (required for `pip install` and model download).
3. **Add Input** → attach the images dataset. The notebooks default to
   `INPUT_DIR = "/kaggle/input/datasets/amasikifthakerhamim/images"` and search it
   **recursively**, so it works whether the images sit directly there or in a nested
   `image/` subfolder. (If your attached path differs, edit `INPUT_DIR` in the CONFIG cell.)
   To use a URL host instead, set `BASE_URL` and leave the dataset unused.
4. **Run All**. If a paddle import errors right after install, use **Run → Restart & Run All** once.

## Why the native Paddle engine (not vLLM)?

The T4 is **compute capability 7.5**. PaddleOCR-VL's vLLM/SGLang backend requires **CC ≥ 8.0**
(and FlashAttention), and the docs explicitly flag T4/V100 as not recommended there (timeout/OOM).
The **native Paddle engine only needs CC ≥ 7.0**, so all notebooks use it — no vLLM, no
FlashAttention wheel needed. This is the correct, reliable path on a T4.

## Install (done inside each notebook)

```bash
pip install "paddleocr[doc-parser]"
pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
pip install nvidia-ml-py
```

The `cu126` PaddlePaddle wheel bundles its own CUDA libraries and runs on the T4.

## Comparing models

Each notebook writes `benchmark_summary.json` (inside its ZIP and shown as a table) with:
`gpu_peak_MB`, `gpu_after_load_MB`, `avg_latency_s`, `total_infer_s`, `model_disk_MB`,
`output_disk_MB`, `load_time_s`. Run all five, then line up those tables to decide which model
fits your accuracy vs. speed vs. footprint needs. PP-StructureV3 and PaddleOCR-VL do full
document parsing (tables/formulas/reading order); PP-OCRv6 is plain text OCR — compare like for like.

## Troubleshooting: install conflicts and the torch `ncclCommShrink` error

Installing `paddlepaddle-gpu` (cu126) downgrades the `nvidia-*-cu12` libraries to 12.6, which
Kaggle's preinstalled **torch** (built for cu128) depends on. You'll first see pip
"dependency conflict" lines, and then — critically — importing `paddleocr` fails with:

```
ImportError: .../torch/lib/libtorch_cuda.so: undefined symbol: ncclCommShrink
```

**Why:** `import paddleocr` → `paddlex` → `import modelscope`, and modelscope's logger imports
torch if it's present. The downgraded `nvidia-nccl-cu12` (2.25.1) lacks the newer `ncclCommShrink`
symbol torch needs, so the torch import crashes and takes paddleocr down with it.

**The fix (already built into the INSTALL cell):** these notebooks don't use torch, so the
install cell **uninstalls `torch torchvision torchaudio`** after installing paddle. modelscope
then sees torch is absent (`find_spec` → None) and skips it, and paddleocr imports cleanly.
The **VERIFY GPU** cell asserts torch is gone and imports paddleocr to confirm before any model load.

If you already hit the error in a running session, re-run the INSTALL cell (or **Run → Restart &
Run All**) so the torch uninstall takes effect.

(If you ever want PaddleOCR-VL's `engine="transformers"` path, that one *does* need torch — you'd
have to install a torch build matching paddle's CUDA libs. The default engine here is native
paddle, so torch is not needed.)

The occasional `ReadTimeoutError` is just the Paddle mirror being slow; `--timeout/--retries` handle it.

## Notes / caveats

- **First run downloads models** from the network, so `load_time_s` includes download time;
  the `model_disk_MB` figure is the on-disk size of what was fetched.
- **GPU memory** is read via NVML for the whole T4; since the Kaggle session owns the GPU,
  that closely tracks the model's usage.
- PaddleOCR-VL on a T4 works but is the heaviest of the set — expect higher latency than PP-OCRv6.
- Model APIs verified against PaddleOCR docs (PaddleOCR-VL 1.6, PP-OCRv6, PP-StructureV3), June 2026.
