# run4 — PP-StructureV3 lean + OCRv6-medium  ✅ WINNER (quality-per-cost)

OCRv6-medium OCR + lean modules (formula/chart/seal off, no doc-unwarping) + det tuning.
Both **more accurate and cheaper** than every other StructureV3 config tried.

StructureV3 config (`STRUCT_KW` in `notebooks/ocr_vl16_comparison.ipynb`):
```python
text_detection_model_name="PP-OCRv6_medium_det",
text_recognition_model_name="PP-OCRv6_medium_rec",
use_table_recognition=True, use_region_detection=True,
use_formula_recognition=False, use_chart_recognition=False, use_seal_recognition=False,
use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False,
text_det_limit_type="max", text_det_limit_side_len=1216, layout_threshold=0.3,
```

Headline (vs VL 1.6): word F1 **0.709** (best), recall **0.693** (best), precision 0.819,
table recall 0.745 · **2.09 s/img · 3.5 GB GPU · 0.86 GB disk · 233 s load**.

Why it beat run3 (all-modules OCRv6): turning off `doc_unwarping` stopped it distorting these flat
digital templates, and dropping chart/formula/seal removed modules that mis-tagged a few regions.
GPU peak (3.5 GB) is the lowest of all StructureV3 configs — OCRv6 is lighter than OCRv5-server.

`results/` has metric CSVs, figures, logs, per-model json/img/markdown, and 100 `.html` pages
(StructureV3 + VL).

**Known limitation:** on borderless forms the layout stage can collapse fields into one run-on
paragraph in the HTML (e.g. `agenda_02.html`) — text is captured but spatial layout is lost. This is
the remaining gap vs VL and the ~31% of VL words StructureV3 still misses.
