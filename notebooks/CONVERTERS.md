# PDF → Markdown converters — shared standard

Four sibling notebooks convert the **same** `../pdf_resources/*.pdf` set to Markdown, each wrapping
a different engine, so their outputs compare **1:1**. This file documents the layout every one of
them follows — keep new converters to this shape.

| folder | engine | models? | OCR (scans) | speed | notes |
|---|---|---|---|---|---|
| `mineru/` | MinerU pipeline (layout+OCR+table+formula) | **yes**, ~1–2 GB | ✅ | ~15–240 s/doc | LaTeX formulas; native extras in `raw/` |
| `marker/` | Marker / Surya (layout+OCR+table+texify) | **yes**, ~3 GB | ✅ | ~sec–min/doc | LaTeX formulas; MPS/CUDA/CPU |
| `pymupdf4llm/` | PyMuPDF4LLM (text layer) | no | ❌ | ~0.1–2 s/doc | pure Python; good tables + image extraction |
| `markitdown/` | MarkItDown (pdfminer) | no | ❌ | ~0.05–1 s/doc | pure Python; broad format support |

The two model-based tools (mineru, marker) read the text layer on born-digital pages and only run
OCR where the layer is missing/bad; the two pure-Python tools can't OCR at all.

## Standard folder layout (every converter)

```
notebooks/<tool>/
├── README.md
├── <tool>_pdf_to_markdown.ipynb
├── out/                           # generated — one dir per source PDF
│   └── <safe_stem>/
│       ├── <safe_stem>.md         # the conversion
│       ├── <safe_stem>.<kind>.json  # sidecar (chunks|meta|content_list); omitted if none
│       ├── images/                # extracted figures; omitted if none / when embedded inline
│       └── raw/                   # engine-native extra files; MinerU only
└── html/                          # generated
    └── <original_stem>.html       # Markdown rendered with the shared house CSS
```

Only `out/` and `html/` are produced on a run — no manifest CSV, no zip bundle. The setup steps
live inline in each notebook's **section 0** (and its `README.md`).

**`safe_stem`** — the source filename stem, sanitized identically by every notebook:
`(` `)` `[` `]` → `-`, space → `_`, unicode dashes (‐‑‒–—―−) → `-`. (Origin: PyMuPDF4LLM's own
image-path sanitizer mangles spaces/parens, so folders are sanitized up front to match.)

**Per-PDF stats** — each notebook prints a stats table **inline** (in the collect/stats cell; no
file is written), with identical columns across all four so the runs compare directly:

```
pdf, ok, backend, latency_s, pages, chars, n_tables, n_images, n_equations, n_text
```

`chars` counts the Markdown text (base64 image blobs are stripped before counting, so embedding
images doesn't inflate it). `n_equations` is 0 for the two pure-Python tools.

## Per-tool specifics

| tool | sidecar `.json` | images | `raw/` |
|---|---|---|---|
| mineru | `.content_list.json` | `images/` (referenced from `content_list.json`) | native `_layout.pdf`, `_span.pdf`, `_model.json`, `_middle.json`, native `.md`/JSON |
| marker | `.meta.json` (`page_stats` block counts) | `images/` | — |
| pymupdf4llm | — (Markdown + HTML only) | `images/`, or **inline base64** when `EMBED_IMAGES=True` (default) | — |
| markitdown | — (single string) | — (pdfminer extracts no images) | — |
