# run2 — PP-StructureV3 lean (OCRv5-server)

StructureV3 with formula/chart/seal OFF, default OCRv5-server OCR, det tuning
(`text_det_limit_side_len=1216`, `layout_threshold=0.3`).

**No `results/` folder:** this run was stopped after the metrics cell (before the ZIP cell), so
only `notebook.ipynb` is here. The metric tables are in that notebook's output.

Headline (vs VL 1.6): word F1 **0.617**, recall 0.581, table recall 0.745 · 2.03 s/img · 4.5 GB GPU · 1.0 GB disk.
