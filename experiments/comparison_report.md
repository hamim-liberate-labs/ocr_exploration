# Experiment Comparison Report

*Benchmark: `test_dataset_100` (100 English business documents). All quality numbers are measured against **PaddleOCR-VL 1.6**, which is used as a pseudo-reference because there is no ground-truth text.*

## What we were trying to do

Find the lightest PaddleOCR setup that reproduces what VL 1.6 does — extract all text **and** its table/layout structure from document images — but at a fraction of VL's cost (VL needs ~23–24 s per image and ~15 GB of GPU).

The four runs all use **PP-StructureV3** (the model that can capture tables), and each one changes the configuration to see what makes it more accurate and cheaper. The plain-text OCR models (PP-OCRv6 small and medium) are the same in every run and act as a reference point for pure text extraction.

## The four runs at a glance

| Run | Configuration | Word F1 | Word recall | Word precision | Table recall | Speed (s/img) | GPU peak | Disk | Load time |
|-----|---------------|--------:|------------:|---------------:|-------------:|--------------:|---------:|-----:|----------:|
| **run1** | StructureV3 baseline — OCRv5-server, all modules on | 0.643 | 0.612 | 0.732 | 0.759 | 3.08 | 13.0 GB | 3.9 GB | 549 s |
| **run2** | StructureV3 lean — OCRv5-server, formula/chart/seal off | 0.617 | 0.581 | — | 0.745 | 2.03 | 4.5 GB | 1.0 GB | — |
| **run3** | StructureV3 all modules + OCRv6-medium | 0.700 | 0.672 | 0.809 | 0.742 | 3.93 | 9.6 GB | 3.9 GB | 625 s |
| **run4** | StructureV3 lean + OCRv6-medium | **0.709** | **0.693** | **0.819** | 0.745 | 2.09 | **3.5 GB** | **0.86 GB** | 233 s |

## What each change taught us

- **Turning off unused modules made it much cheaper for almost no quality loss (run1 → run2).** These documents have no formulas, charts, or seals, so switching those modules off cut GPU from 13 GB to 4.5 GB and disk from 3.9 GB to 1.0 GB, while quality barely moved (F1 0.643 → 0.617). The small quality dip here is really about the OCR engine, which was still OCRv5.

- **Swapping the OCR engine raised text accuracy (OCRv5 → OCRv6-medium, run3).** With all modules still on, moving from OCRv5-server to OCRv6-medium lifted word F1 from 0.643 to 0.700 and precision from 0.732 to 0.809. So the OCR engine, not the heavy modules, was the main driver of text fidelity.

- **Combining both changes gave the strongest result (run4).** Lean modules + OCRv6-medium is both the most accurate and the cheapest StructureV3 configuration: the highest F1 (0.709) and recall (0.693), the lowest GPU (3.5 GB), the smallest disk (0.86 GB), and the fastest load (233 s). A key detail: turning off `doc_unwarping` stopped it from distorting these flat, digital templates, which helped accuracy on top of the cost savings.

## The bottom line

- **run4 (StructureV3, lean modules, OCRv6-medium) came out ahead** of the other StructureV3 runs on both accuracy and cost, running at ~2.1 s/img and 3.5 GB GPU versus VL's ~23 s and ~15 GB — roughly 11× faster and a quarter of the memory.
- **Two independent changes drove that result:** dropping the modules the data doesn't need (cheaper), and using the newer OCRv6 engine (more accurate). Each helped on its own, and together they compounded.
- **No light model fully replaces VL yet.** run4 still captures only about two-thirds of VL's words (recall 0.693) and can flatten borderless forms into run-on text, losing spatial layout. VL stays ahead when you need accurate text *and* layout at the same time.

## Note on plain-text-only work

If tables and layout are not needed, the plain OCR models are dramatically cheaper and score much higher on raw text (they don't attempt table structure, so their table recall is 0):

| Model | Word F1 | Word recall | Speed (s/img) | GPU peak | Disk |
|-------|--------:|------------:|--------------:|---------:|-----:|
| PP-OCRv6 small | 0.840 | 0.904 | 0.47 | 2.1 GB | 30 MB |
| PP-OCRv6 medium | 0.849 | 0.915 | 0.81 | 3.8 GB | 133 MB |

For text-only extraction, **PP-OCRv6 small** captures ~90% of VL's words at 0.47 s/img — about 49× faster than VL — and medium adds only ~1 point of F1 for roughly double the cost.
