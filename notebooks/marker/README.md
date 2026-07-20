# Marker — PDF → Markdown (local, ML pipeline, Apple Silicon)

Experiment with [**Marker**](https://github.com/datalab-to/marker) (`marker-pdf`, by Datalab —
the [Surya](https://github.com/datalab-to/surya) authors): a deep-learning document pipeline that
converts **PDF / image / DOCX / PPTX / XLSX / EPub / HTML** into **Markdown / JSON / HTML**,
running **layout detection → OCR → table recognition → equation (texify) → reading order**. It
extracts figures, renders tables, and converts equations to **LaTeX**.

Marker is the **other heavyweight** alongside MinerU — a real OCR/layout stack (not a text-layer
reader), so it handles **scanned / image-only PDFs**. Its `pipeline` runs on CPU and
**Apple-Silicon (MPS)**, so this notebook runs **locally on your Mac**.

Where it sits among the four notebooks in this repo:

| approach | needs | speed | scanned PDFs | equations |
|---|---|---|---|---|
| **Marker (this notebook)** | Surya models, ~1–2 GB, MPS/CPU | ~sec–min/doc | ✅ OCRs | ✅ LaTeX (texify) |
| MinerU (`../mineru`) | layout+OCR+table+formula models, ~1–2 GB | ~15–240 s/doc | ✅ OCRs | ✅ LaTeX |
| PyMuPDF4LLM (`../pymupdf4llm`) | pure Python | ~0.1–2 s/doc | ❌ text layer only | ❌ |
| MarkItDown (`../markitdown`) | pure Python (pdfminer) | ~0.05–1 s/doc | ❌ text layer only | ❌ |

**Optional LLM boost** (not wired here — keep the run offline): `config={"use_llm": True}` plus an
LLM service (Gemini by default) improves tables, equations, and inline math. This notebook runs the
**base pipeline only**.

## Files

| file | what |
|---|---|
| `marker_pdf_to_markdown.ipynb` | the experiment: discover PDFs → convert → stats → render HTML |
| `out/<safe_stem>/` | per-PDF, [standard layout](../CONVERTERS.md): `<safe_stem>.md`, `<safe_stem>.meta.json`, `images/` — created on run |
| `html/` | Markdown rendered to standalone HTML pages — created on run |

## Setup (one time)

```bash
/opt/homebrew/bin/python3.12 -m venv .venv-marker
source .venv-marker/bin/activate
python -m pip install -U pip
pip install -U marker-pdf pypdf
pip install jupyter ipykernel pandas markdown
python -m ipykernel install --user --name marker --display-name "Python (marker)"
```

Creates `.venv-marker/` at the repo root, installs `marker-pdf` (+ PyTorch) + `pypdf` + Jupyter,
and registers a **Python (marker)** kernel. Then:

```bash
source .venv-marker/bin/activate
jupyter lab   # open the notebook, select kernel "Python (marker)", Run All
```

The **first conversion downloads the Surya models** (~1–2 GB: layout, recognition/OCR, table,
texify) into `~/.cache`. One time only; later runs are offline.

## Knobs (Config cell)

- `DEVICE` — auto-detected (`mps` on Apple Silicon, else `cuda`/`cpu`); set via the `TORCH_DEVICE`
  env var before models load.
- `MAX_PAGES` — cap pages per PDF via Marker's `page_range` (default 15, matching the other runs;
  the *Atlanta Code of Ordinances* is hundreds of pages). `None` → all pages.
- `MAX_FILES` — smoke-test the first N PDFs; `None` for all.
- `FORCE_OCR` — re-OCR every page even if it has a text layer (default `False`; Marker auto-OCRs
  pages whose embedded text looks bad).
- `USE_LLM` — `True` needs an LLM service + key; left `False` here.

## One-liner equivalent

```python
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

converter = PdfConverter(artifact_dict=create_model_dict())
text, ext, images = text_from_rendered(converter("pdf_resources/Bullseye Math Checklist  - Sheet1.pdf"))
```
