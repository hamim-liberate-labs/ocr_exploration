# run1 — PP-StructureV3 baseline (OCRv5-server, all modules)

The first 100-image run. StructureV3 in its default form: OCRv5-server OCR, all modules on
(table/formula/chart/seal), no det tuning. This is the baseline the later runs optimize against.

Headline (vs VL 1.6): word F1 **0.643**, recall 0.612, table recall **0.759** · 3.08 s/img · 13.0 GB GPU · 3.9 GB disk · 549 s load.

`results/` (from the original `ocr_compare_results` zip) has metric CSVs, figures, logs, and per-model
json/img/markdown. Note: HTML output was added after this run, so StructureV3 here has markdown but no `.html`.
