# run3 — PP-StructureV3 all modules + OCRv6-medium

StructureV3 with all modules on but OCR swapped from OCRv5-server → **PP-OCRv6-medium**, plus det
tuning. This run proved the OCR swap helps: text fidelity rose vs the OCRv5 runs.

Headline (vs VL 1.6): word F1 **0.700**, recall **0.672**, precision 0.809, table recall 0.742 · 3.93 s/img · 9.6 GB GPU · 3.9 GB disk · 625 s load.

`results/` has metric CSVs, figures, logs, and per-model outputs including per-image `.html` pages
(StructureV3 and VL). All modules on = still heavy; run4 trims them.
