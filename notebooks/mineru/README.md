# MinerU — PDF → Markdown / HTML (local, Apple Silicon)

Experiment with [**MinerU**](https://github.com/opendatalab/mineru) (opendatalab, v3.4): a
one-stop open-source tool that converts **PDF / image / DOCX / PPTX / XLSX** into **Markdown +
JSON**, preserving reading order and extracting text, tables (as HTML), formulas (LaTeX), and
figures.

Unlike the PaddleOCR / OvisOCR2 notebooks in `../` (which need a Kaggle T4), **MinerU's
`pipeline` backend runs on CPU and Apple-Silicon (MPS)** — so this notebook runs **locally on
your Mac**. It converts every PDF in `../../pdf_resources/`.

## Files

| file | what |
|---|---|
| `mineru_pdf_to_markdown.ipynb` | the experiment: discover PDFs → convert → stats → render HTML |
| `out/<safe_stem>/` | per-PDF, [standard layout](../CONVERTERS.md): `<safe_stem>.md`, `<safe_stem>.content_list.json`, `images/`, and native MinerU extras under `raw/` — created on run |
| `html/` | Markdown rendered to standalone HTML pages — created on run |

## Setup (one time)

```bash
/opt/homebrew/bin/python3.12 -m venv .venv-mineru
source .venv-mineru/bin/activate
python -m pip install -U pip
pip install -U "mineru[core]"
pip install jupyter ipykernel pandas markdown
python -m ipykernel install --user --name mineru --display-name "Python (mineru)"
```

This installs Homebrew `python@3.12`, creates `.venv-mineru/` at the repo root, installs
`mineru[core]` + Jupyter, and registers a **Python (mineru)** kernel. Then:

```bash
source .venv-mineru/bin/activate
jupyter lab   # open the notebook, select kernel "Python (mineru)", Run All
```

The **first conversion downloads the pipeline models** (~1–2 GB: layout, OCR, table, formula)
into `~/.cache`. One time only; later runs are offline.

## Backends

| backend | needs | notes |
|---|---|---|
| `pipeline` (default) | CPU / MPS, ~4 GB | safe on a 16 GB M-series Mac; what this notebook uses |
| `hybrid` / `vlm` | ~8 GB free unified mem, GPU | higher accuracy (~95%), heavier — install `mineru[all]` |

## Knobs (Config cell)

- `BACKEND` — `pipeline` (default) / `vlm` / `hybrid`
- `DEVICE` — auto-detected (`mps` on Apple Silicon); override to `cpu` if MPS misbehaves
- `MAX_PAGES` — cap pages per PDF (default 15; the *Atlanta Code of Ordinances* is hundreds of
  pages). Set `None` for full documents.
- `MAX_FILES` — smoke-test the first N PDFs; `None` for all.

## CLI equivalent

The notebook shells out to the stable MinerU CLI; you can reproduce any single file with:

```bash
mineru -p "../../pdf_resources/Bullseye Math Checklist  - Sheet1.pdf" -o out/bullseye -b pipeline -l en -d mps
```

Flags: `-p` input · `-o` output dir · `-b` backend · `-l` OCR language · `-d` device ·
`-e` end page (0-indexed, inclusive).
