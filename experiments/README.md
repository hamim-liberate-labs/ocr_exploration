# Experiment run log

All runs use the same notebook (`notebooks/ocr_vl16_comparison.ipynb`) on the 100-image
`test_dataset_100` set, with PaddleOCR-VL 1.6 as the pseudo-reference. The **only variable
across runs is the PP-StructureV3 config**; PP-OCRv6 small/medium and VL 1.6 are constant.

## Constant candidates (vs VL 1.6)

| Model | word F1 | word recall | table recall | latency | GPU peak | disk |
|---|---|---|---|---|---|---|
| PP-OCRv6 small  | 0.840 | 0.904 | 0 (text only) | 0.47 s | 2.1 GB | 30 MB |
| PP-OCRv6 medium | 0.849 | 0.915 | 0 (text only) | 0.81 s | 3.8 GB | 133 MB |
| PaddleOCR-VL 1.6 (reference) | — | — | — | 23.1 s | 15.0 GB | 1.97 GB |

## PP-StructureV3 configs (the experiment axis), vs VL 1.6

| Run | OCR | Modules | det tuning | word F1 | recall | table recall | latency | GPU peak | disk | load |
|---|---|---|---|---|---|---|---|---|---|---|
| **run1** baseline | OCRv5-server | all on | no  | 0.643 | 0.612 | **0.759** | 3.08 s | 13.0 GB | 3.9 GB | 549 s |
| **run2** lean     | OCRv5-server | lean  | yes | 0.617 | 0.581 | 0.745 | **2.03 s** | **4.5 GB** | **1.0 GB** | 297 s |
| **run3** ocrv6-all| OCRv6-medium | all on | yes | 0.700 | 0.672 | 0.742 | 3.93 s | 9.6 GB | 3.9 GB | 625 s |
| **run4** ocrv6-lean ✅ | OCRv6-medium | lean | yes | **0.709** | **0.693** | 0.745 | **2.09 s** | **3.5 GB** | **0.86 GB** | 233 s |

- **lean** = `use_formula_recognition / use_chart_recognition / use_seal_recognition = False`
  (this dataset has no formulas/charts/seals; they only cost load/GPU/disk).
- **det tuning** = `text_det_limit_side_len=1216`, `layout_threshold=0.3` (wider text/region coverage).

## What each run showed

- **run1 → run2** (trim modules): −66% GPU, −74% disk, −34% latency at ~same quality. The module trim is a big efficiency win; det tuning did **not** recover the missing text (the ~40% gap vs VL is layout-stage, not detection).
- **run1/run2 → run3** (OCRv5→OCRv6): OCRv6-medium **improved** text fidelity across the board (F1 +0.06, precision +0.08) — the OCR swap genuinely helps on these English docs.
- **run3 → run4** (all → lean): lean is **both better and cheaper** — F1 0.700 → **0.709**, latency 3.93 s → **2.09 s**, GPU 9.6 GB → **3.5 GB**. Turning off `doc_unwarping` (which distorted the flat templates) and the unused modules removed noise. **run4 is the winner and the current config in the source notebook.**

**Winner: run4 (StructureV3 lean + OCRv6-medium)** — the best light stand-in for VL 1.6 when you need
tables/layout: 0.709 F1 / 0.745 table recall at 2.1 s and 3.5 GB, vs VL's 23 s and 15 GB.

Note: run2 was stopped after the metrics cell, so it has no `results/` folder — numbers above are from its executed notebook's metric tables.

## The 11-image legacy experiment

`11img_legacy/` is the earlier study: 5 separate single-model notebooks (OCRv6 tiny/small/medium,
StructureV3, VL 1.6) run on 11 images, with `COMPARISON.md` as the writeup. Superseded by the
100-image runs above but kept for reference.

## The OvisOCR2 track (`ovisocr2_vs_vl/`)

A separate track from runs 1–4. Instead of a PaddleOCR config on Kaggle T4, it tests **OvisOCR2**
(a compact ~0.8B document-parsing VLM) as a light stand-in for VL 1.6, run on a **Modal L4** GPU,
and reuses the existing `run4` VL 1.6 outputs as the pseudo-reference. It keeps the same
divergence-vs-VL metrics **and** adds two things runs 1–4 don't have: operational/robustness flags
and a **Gemini 2.5 Flash LLM-judge** that picks the more faithful transcription per page (the
head-to-head "who's better" call that VL-similarity alone can't make). Self-contained in its own
`notebook.ipynb`; see `ovisocr2_vs_vl/README.md`.
