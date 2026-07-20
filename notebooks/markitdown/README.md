# MarkItDown — PDF → Markdown (local, pure-Python)

Experiment with [**MarkItDown**](https://github.com/microsoft/markitdown) (Microsoft): a
lightweight utility that converts **PDF, Office (DOCX/PPTX/XLSX), HTML, images, audio, EPub, CSV,
JSON, ZIP…** into Markdown aimed at LLM consumption. It's the Swiss-army *format* converter — its
strength is breadth of input types, not deep PDF layout analysis.

For **PDFs specifically** (all this repo's `../../pdf_resources/` are PDFs) MarkItDown extracts the
text layer via **`pdfminer.six`** — fast, offline, no models. Like PyMuPDF4LLM it does **no OCR**
by default and does **not** extract embedded images, so a scanned page yields almost nothing.

Where it sits among the three notebooks in this repo:

| approach | needs | speed | tables | scanned PDFs |
|---|---|---|---|---|
| MinerU (`../mineru`) | layout+OCR+table/formula models, ~1–2 GB | ~15–240 s/doc | ✅ strong | ✅ OCRs |
| PyMuPDF4LLM (`../pymupdf4llm`) | pure Python | ~0.1–2 s/doc | ✅ good, +image extraction | ❌ text layer only |
| **MarkItDown (this notebook)** | **pure Python (pdfminer)** | **~0.05–1 s/doc** | ⚠️ weak/linear | ❌ text layer only |

**Optional cloud/LLM upgrades** (not wired here — keep the run offline):
- `MarkItDown(llm_client=..., llm_model=...)` → captions images with an LLM (Azure OpenAI, OpenAI,
  or a Claude client) — image conversion only, not PDF page OCR.
- `MarkItDown(docintel_endpoint=...)` → **Azure Document Intelligence** for real OCR + table structure.
- `MarkItDown(cu_endpoint=...)` → **Azure Content Understanding** (auto-selects analyzer per file type).

## Files

| file | what |
|---|---|
| `markitdown_pdf_to_markdown.ipynb` | the experiment: discover PDFs → convert → stats → render HTML |
| `out/<safe_stem>/` | per-PDF, [standard layout](../CONVERTERS.md): `<safe_stem>.md` (no sidecar/images — pdfminer extracts text only) — created on run |
| `html/` | Markdown rendered to standalone HTML pages — created on run |

## Setup (one time)

```bash
/opt/homebrew/bin/python3.12 -m venv .venv-markitdown
source .venv-markitdown/bin/activate
python -m pip install -U pip
pip install -U "markitdown[all]" pypdf
pip install jupyter ipykernel pandas markdown
python -m ipykernel install --user --name markitdown --display-name "Python (markitdown)"
```

Creates `.venv-markitdown/` at the repo root, installs `markitdown[all]` + `pypdf` + Jupyter, and
registers a **Python (markitdown)** kernel. **No model download.** Then:

```bash
source .venv-markitdown/bin/activate
jupyter lab   # open the notebook, select kernel "Python (markitdown)", Run All
```

## Knobs (Config cell)

- `MAX_PAGES` — MarkItDown has **no page-range option**, so to stay comparable with the mineru /
  pymupdf4llm runs the notebook first **slices** each PDF to the first N pages with `pypdf`, then
  converts (default 15; `None` → whole document — note the *Atlanta Code of Ordinances* is hundreds
  of pages).
- `MAX_FILES` — smoke-test the first N PDFs; `None` for all.

## One-liner equivalent

```python
from markitdown import MarkItDown
md = MarkItDown()
print(md.convert("pdf_resources/Marzano Rubrics with scales. potential evidence, and elements 2022.pdf").text_content)
```
