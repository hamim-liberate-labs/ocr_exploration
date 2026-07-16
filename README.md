# OCR Exploration — Finding a Lightweight Stand-in for PaddleOCR-VL 1.6

> A reproducible benchmarking study that searches for the **lightest OCR model** capable of
> reproducing **PaddleOCR-VL 1.6**'s document-to-HTML output on English business documents,
> trading a fraction of the quality for large gains in speed, GPU memory, and disk footprint.

---

## Table of contents

- [1. Abstract](#1-abstract)
- [2. Research question](#2-research-question)
- [3. Methodology](#3-methodology)
- [4. Dataset](#4-dataset)
- [5. Repository structure](#5-repository-structure)
- [6. Experiments](#6-experiments)
- [7. Key findings](#7-key-findings)
- [8. Reproducibility](#8-reproducibility)
- [9. Credentials & secrets](#9-credentials--secrets)
- [10. Limitations](#10-limitations)

---

## 1. Abstract

The task is to **extract all text and its layout as HTML from document images**. PaddleOCR-VL 1.6
(a ~15 GB vision-language OCR model) does this well but is expensive: ~23 s/image and 15 GB of GPU
memory. This study benchmarks a family of lighter PaddleOCR configurations — **PP-OCRv6** (small /
medium), **PP-StructureV3** (four module/OCR configurations), and the compact **OvisOCR2** (~0.8 B
VLM) — to find the best cost/quality stand-in.

Because the dataset has **no ground-truth transcription**, VL 1.6 is used as a **pseudo-reference**
and every candidate is scored on *divergence from VL* (proxy metrics), not absolute accuracy. A
separate **Gemini 2.5/3.5 Flash LLM-judge** provides the head-to-head "which transcription is more
faithful" verdict that similarity metrics alone cannot.

**Headline result:** for text-only needs, **PP-OCRv6 small** recovers ~90 % of VL's word recall at
**49× the speed**. For text + tables/layout, **PP-StructureV3 (lean modules + OCRv6-medium)** reaches
**F1 0.709 / table-recall 0.745 at 2.1 s and 3.5 GB** — versus VL's 23 s and 15 GB.

## 2. Research question

> On our task (document image → text + layout as HTML), what is the **lightest** PaddleOCR model
> that best reproduces **PaddleOCR-VL 1.6**, and what is the resulting **quality vs. cost** trade-off?

Sub-questions explored across the runs:

- How much does **trimming unused StructureV3 modules** (formula / chart / seal) save, and does it
  cost quality?
- Does swapping the OCR backbone **OCRv5-server → OCRv6-medium** improve text fidelity?
- Can a compact general VLM (**OvisOCR2, 0.8 B**) compete with VL 1.6 on faithfulness?

## 3. Methodology

**Pseudo-reference design.** There is no human-labelled ground truth, so PaddleOCR-VL 1.6 is treated
as the reference transcription. All metrics measure how far a candidate *diverges from VL*, which is a
proxy for quality — not an absolute score.

**Proxy metrics reported per model:**

| Category | Metrics |
|---|---|
| Text fidelity | word Precision / Recall / F1 and character similarity vs VL |
| Structure | table-capture recall |
| Model confidence | recognition confidence |
| Consistency | cross-model agreement |
| Cost | latency (s/img), peak GPU memory, disk footprint, model load time |
| Head-to-head | **Gemini LLM-judge** win-rate (faithfulness verdict per page) |

**Controlled comparison.** Across the four PaddleOCR runs, the *only* variable is the PP-StructureV3
configuration; PP-OCRv6 small/medium and VL 1.6 are held constant as anchors. The OvisOCR2 track is a
separate axis (a compact VLM on a Modal L4 GPU) that reuses VL 1.6's outputs as the pseudo-reference.

## 4. Dataset

`data/test_dataset_100/` — **100 English business-document images** plus `manifest.csv`, spanning five
categories: **agenda, form, invoice, report,** and **original**. There is no ground-truth text (see the
pseudo-reference design above). The 11-image legacy set that seeded the study is preserved under
`experiments/11img_legacy/`.

## 5. Repository structure

```
ocr_exploration/
├── data/
│   └── test_dataset_100/          100 images + manifest.csv (the benchmark set)
├── notebooks/
│   ├── ocr_vl16_comparison.ipynb  Source notebook: PP-OCRv6 + StructureV3 vs VL 1.6
│   ├── ovisocr2_image_to_html.ipynb  OvisOCR2 (0.8B VLM → HTML) source (Kaggle)
│   ├── legacy/                    Earlier single-model source notebooks (11-image era)
│   └── README.md                  Kaggle setup guide
├── experiments/                   One folder per run: executed notebook + results/
│   ├── 100img_run1_sv3_ocrv5_all/    StructureV3 baseline (OCRv5-server, all modules)
│   ├── 100img_run2_sv3_ocrv5_lean/   StructureV3 lean (OCRv5-server, modules off)
│   ├── 100img_run3_sv3_ocrv6_all/    StructureV3 all modules + OCRv6-medium
│   ├── 100img_run4_sv3_ocrv6_lean/   StructureV3 lean + OCRv6-medium (winning config)
│   ├── ovisocr2_vs_vl/               OvisOCR2 (0.8B VLM) vs VL 1.6 — Modal L4 + Gemini judge
│   ├── 11img_legacy/                 Earlier 11-image, 5-single-model experiment
│   └── README.md                     Full run-by-run log with configs + metrics
├── modal_deploy/                  Modal serving apps (OCR + OvisOCR2 endpoints)
│   └── README.md                  Deployment guide
├── .env.example                   Template for local secrets (copy → .env)
└── README.md                      ← you are here
```

Each `experiments/100img_run*/` holds the **executed** `notebook.ipynb` and its `results/` (metric
CSVs, figures, per-model JSON / annotated images / Markdown + HTML pages, and captured logs).

## 6. Experiments

All four PaddleOCR runs share `notebooks/ocr_vl16_comparison.ipynb` on the 100-image set (Kaggle T4),
with VL 1.6 as the pseudo-reference. The **only variable is the PP-StructureV3 config.**

**Constant anchors (vs VL 1.6):**

| Model | word F1 | word recall | table recall | latency | GPU peak | disk |
|---|---|---|---|---|---|---|
| PP-OCRv6 small  | 0.840 | 0.904 | 0 (text only) | 0.47 s | 2.1 GB | 30 MB |
| PP-OCRv6 medium | 0.849 | 0.915 | 0 (text only) | 0.81 s | 3.8 GB | 133 MB |
| PaddleOCR-VL 1.6 (reference) | — | — | — | 23.1 s | 15.0 GB | 1.97 GB |

**PP-StructureV3 configs (the experiment axis), vs VL 1.6:**

| Run | OCR | Modules | word F1 | recall | table recall | latency | GPU peak | disk |
|---|---|---|---|---|---|---|---|---|
| run1 baseline | OCRv5-server | all on | 0.643 | 0.612 | **0.759** | 3.08 s | 13.0 GB | 3.9 GB |
| run2 lean | OCRv5-server | lean | 0.617 | 0.581 | 0.745 | 2.03 s | 4.5 GB | 1.0 GB |
| run3 ocrv6-all | OCRv6-medium | all on | 0.700 | 0.672 | 0.742 | 3.93 s | 9.6 GB | 3.9 GB |
| **run4 ocrv6-lean ✅** | OCRv6-medium | lean | **0.709** | **0.693** | 0.745 | **2.09 s** | **3.5 GB** | **0.86 GB** |

- **lean** = `use_formula_recognition / use_chart_recognition / use_seal_recognition = False`
  (this dataset has none of these; they only cost load/GPU/disk).
- A separate **OvisOCR2 track** (`experiments/ovisocr2_vs_vl/`) tests a 0.8 B VLM on a Modal L4 GPU
  with a Gemini LLM-judge for the faithfulness verdict.

See **[`experiments/README.md`](experiments/README.md)** for the complete run-by-run analysis.

## 7. Key findings

- **Plain text only → PP-OCRv6 small.** ~90 % of VL's word recall at **0.47 s/img, 2.1 GB GPU, 30 MB**
  — 49× faster than VL. Medium adds only ~1 pp F1 for ~1.7× the cost.
- **Text + tables/layout → PP-StructureV3, and config matters a lot:**
  - Turning **off** unused modules cut StructureV3 from **13 GB → 4.5 GB GPU** and **3.9 GB → 1 GB
    disk** at roughly the same quality.
  - Swapping **OCRv5-server → OCRv6-medium** raised text fidelity (F1 0.64 → 0.70).
  - **Best config = OCRv6 + lean modules (run4):** F1 **0.709**, table recall **0.745** at **2.1 s /
    3.5 GB / 0.86 GB disk** — both more accurate *and* far cheaper than the baseline.
- **No light model fully replaces VL.** OCRv6 captures VL's *words* but not reading order; StructureV3
  captures *tables* but still misses ~33 % of VL's text. VL stays ahead when you need both at once.

## 8. Reproducibility

**Environment.** Runs 1–4 execute on a **Kaggle T4** GPU via `notebooks/ocr_vl16_comparison.ipynb`.
The OvisOCR2 track runs on a **Modal L4** GPU (see `experiments/ovisocr2_vs_vl/` and `modal_deploy/`).
Each experiment folder ships its executed notebook and full `results/`, so figures and CSVs can be
inspected without re-running.

**Quick start:**

```bash
git clone <your-repo-url> ocr_exploration && cd ocr_exploration
cp .env.example .env            # then fill in your own values (see §9)
```

- **PaddleOCR runs:** open `notebooks/ocr_vl16_comparison.ipynb` on Kaggle (T4) and run top-to-bottom.
- **OvisOCR2 track:** see `experiments/ovisocr2_vs_vl/README.md` (Modal L4 + Gemini judge).
- **Serving:** see `modal_deploy/README.md` to deploy the OCR endpoints.

## 9. Credentials & secrets

This repository is configured so that **no secret is ever committed** — every credential is listed in
`.gitignore`. To run the pieces that need external services, provide your own:

| Secret | Where it goes | Used by |
|---|---|---|
| `GOOGLE_API_KEY` | `.env` (copy from `.env.example`) | Gemini public-API mode (optional) |
| Vertex AI service account | `vertex-service-account.json` in the repo root | §8 Gemini LLM-judge in `experiments/ovisocr2_vs_vl/` |
| `OCR_URL` / `OCR_TOKEN` / `OVISOCR2_TOKEN` | `.env` (auto-loaded by the client) | `modal_deploy/test_client.py` |
| Modal deploy `AUTH_TOKEN` | Modal secret (source of truth); it is the value clients send as `OCR_TOKEN` / `OVISOCR2_TOKEN` | `modal_deploy/*.py` endpoints |

> **The Gemini LLM-judge authenticates via a Vertex AI service account, not an API key.** Download the
> service-account JSON from your Google Cloud project and save it as `vertex-service-account.json` in
> the repo root. The notebook points `GOOGLE_APPLICATION_CREDENTIALS` at that path automatically.

**If you ever committed a real key by mistake:** rotate/revoke it immediately (Google Cloud console for
the service account, and regenerate Modal tokens), because git history preserves it even after deletion.

## 10. Limitations

- **Pseudo-reference, not ground truth.** All quality numbers measure *agreement with VL 1.6*, so they
  inherit VL's own errors. The Gemini judge mitigates this for the OvisOCR2 head-to-head only.
- **Narrow domain.** 100 clean English business documents — no handwriting, non-Latin scripts, heavy
  noise, or formulas/charts/seals (which is *why* the lean configs win here).
- **Small sample.** 100 images is enough to rank configurations but not to claim tight confidence
  intervals.
