# PyMuPDF4LLM — convert any document to Markdown + HTML

A focused converter built on [**PyMuPDF4LLM**](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/):
point it at a **file or a folder** and it writes, for each document, a **Markdown** file and a
**standalone HTML** page. Nothing else — no manifest, no benchmark, no zip.

PyMuPDF4LLM is pure-Python (no models, no GPU) and reads the document's **text layer**, so it's
near-instant on born-digital PDFs but does **not** OCR scanned / image-only pages (those come back
nearly empty — use the `../mineru` or `../marker` notebooks for scans). PyMuPDF can open **PDF, XPS,
EPUB, MOBI, FB2, CBZ, SVG** (not DOCX/PPTX).

## Files

| file | what |
|---|---|
| `pymupdf4llm_pdf_to_markdown.ipynb` | the converter: config → discover → Markdown → HTML → preview |
| `out/<safe_stem>/<safe_stem>.md` | the full Markdown (images inlined as base64 by default) — created on run |
| `html/<original_stem>.html` | standalone HTML page, renders in any browser/preview — created on run |

## Setup (one time)

```bash
/opt/homebrew/bin/python3.12 -m venv .venv-pymupdf4llm
source .venv-pymupdf4llm/bin/activate
python -m pip install -U pip
pip install -U pymupdf4llm
pip install jupyter ipykernel markdown
python -m ipykernel install --user --name pymupdf4llm --display-name "Python (pymupdf4llm)"
jupyter lab   # open the notebook, pick the Python (pymupdf4llm) kernel, Run All
```

No model download — it runs fully offline from the first cell.

## Config (one cell)

- `INPUT` — a **single file** or a **folder** (a folder converts every supported document in it).
  Default: `../../pdf_resources/`.
- `PAGE_RANGE` — `None` = **every page** (full document, the default); or e.g. `list(range(15))` to cap.
- `EMBED_IMAGES` — `True` (default) inlines images as base64 so the `.md` **and** `.html` are
  self-contained and render in any previewer. `False` writes figures to `out/<stem>/images/` and
  references them (smaller `.md`, needs the folder alongside).
- `IMAGE_DPI` — rasterization DPI for images (default 150; lower → smaller files).
- `DROP_HEADERS_FOOTERS` — `True` strips repeating page headers/footers (useful for long documents).

## One-liner equivalent

```python
import pymupdf4llm
md = pymupdf4llm.to_markdown("pdf_resources/Atlanta, GA Code of Ordinances.pdf", embed_images=True)
open("out.md", "w").write(md)
```
