# PaddleOCR-VL 1.6 on Modal — image → HTML

A [Modal](https://modal.com) app that serves **PaddleOCR-VL 1.6** on a single **T4 GPU** behind
a FastAPI endpoint. POST document images, get parsed **HTML** back for each.

It reproduces the exact pipeline validated in the Kaggle benchmark
(`PaddleOCRVL(pipeline_version="v1.6", device="gpu:0")`, native Paddle engine — **not** vLLM,
since the T4 is compute-capability 7.5 < 8.0). Every optional sub-model (orientation, dewarping,
seal, chart) is disabled, so only layout + text/table recognition run for lowest latency.

## Files

| File | Purpose |
|---|---|
| `paddleocr_vl_app.py` | The Modal app: image build, GPU class, FastAPI endpoints. |
| `test_client.py` | Post local images to the deployed endpoint and save the HTML. |

## Prerequisites

```bash
pip install modal
modal token new          # one-time auth of the Modal CLI
```

## 1. Create the auth secret (once)

The endpoint requires a bearer token, stored as a Modal Secret named `paddleocr-vl-token`:

```bash
modal secret create paddleocr-vl-token \
    AUTH_TOKEN=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
```

Print it back when you need it for requests:

```bash
modal secret list      # confirms it exists (value stays hidden)
```

Keep the generated token — it's the `OCR_TOKEN` you send in the `Authorization` header.

## 2. Deploy

```bash
modal deploy modal_deploy/paddleocr_vl_app.py
```

The first deploy builds the image and **prefetches the ~2 GB of VL weights into the image layer**
(a short T4 build), so container cold starts don't re-download the model. Modal prints the public
URL, e.g.:

```
https://<workspace>--paddleocr-vl-document-ocr-paddlevl-web.modal.run
```

For iterative development with hot reload and a temporary URL:

```bash
modal serve modal_deploy/paddleocr_vl_app.py
```

## 3. Call it

Interactive OpenAPI docs are at `<url>/docs`.

```bash
export OCR_URL="https://<workspace>--paddleocr-vl-document-ocr-paddlevl-web.modal.run"
export OCR_TOKEN="<the AUTH_TOKEN from step 1>"

# health (no auth)
curl "$OCR_URL/health"

# one or more images (multipart field name is `files`, repeatable)
curl -X POST "$OCR_URL/v1/document-ocr" \
  -H "Authorization: Bearer $OCR_TOKEN" \
  -F "files=@image_001.jpg" \
  -F "files=@image_005.jpg"
```

Or use the client to save each result as an `.html` file. It auto-loads `OCR_URL` /
`OCR_TOKEN` / `OVISOCR2_TOKEN` from the repo-root `.env` (no `export` needed) and picks the
right token from the URL, so put your values there once:

```bash
python modal_deploy/test_client.py --out ./html_out image_001.jpg image_005.jpg
# or point it at a directory:
python modal_deploy/test_client.py --out ./html_out .
```

## API

### `GET /health`
Liveness/readiness. No auth. Returns `{"status":"ok","model":"PaddleOCR-VL-1.6","gpu":"T4"}`.

### `POST /v1/document-ocr`
Bearer-token auth. Multipart form, field **`files`** (one or more images).

**Response** (`application/json`):

```json
{
  "model": "PaddleOCR-VL-1.6",
  "count": 2,
  "results": [
    {"filename": "image_001.jpg", "status": "ok", "html": "<!doctype html>...", "blocks": 4, "seconds": 17.63},
    {"filename": "bad.gif",       "status": "error", "error": "unsupported image format: gif"}
  ]
}
```

Per-image failures are isolated: one bad image is reported as `status:"error"` without failing
the rest of the batch. Null fields are omitted from the response.

Error codes: `401` (bad/missing token), `413` (more than 16 files), `422` (malformed multipart /
missing `files` field), `500` (unexpected server error).

**Limits** (edit the constants at the top of `paddleocr_vl_app.py`): up to `16` images/request,
`25 MB`/image, formats JPEG/PNG/BMP/WebP/TIFF (validated by decoding, not just the content-type).

> **Swagger `/docs`**: the `files` field renders as a real file-picker (the schema is
> post-processed to `format: binary`, since FastAPI's OpenAPI 3.1 `contentMediaType` output
> otherwise shows up as a text box in the bundled Swagger UI).

## Notes on performance & cost

- **Scale to zero — no idle cost.** No `min_containers` is set, so when there's no traffic
  there are no running containers (and no GPU billing). Containers spin up on demand.
- **Cold start ≈ 25 s, never a re-download.** The VL weights are **baked into the image** at
  build time, so a cold container only *loads* them from the local image layer (~25 s) — it does
  **not** download the model. `scaledown_window=300` keeps a container warm for 5 min after the
  last request, so bursts stay warm.
- **Concurrency is 1 image per container** (`@modal.concurrent(max_inputs=1)`): VL peaks
  ~14.8 GB, so two at once would OOM a 16 GB T4. Modal autoscales to more containers for
  concurrent requests.
- **Latency** ~12–18 s per page, up to ~40 s on very dense small-font pages (from the benchmark).
- **The 150 s synchronous limit.** Modal web endpoints answer synchronously for up to ~150 s;
  beyond that they switch to an async redirect that a plain `POST` can't follow. A cold start
  (~25 s load) plus the first inference — whose CUDA autotune makes the *first* page unusually
  slow — can occasionally approach this. Recommended caller pattern (both are in `test_client.py`):
    1. **Warm first:** `GET /health` blocks until a container is up with the model loaded, so the
       following `POST /v1/document-ocr` hits a warm container.
    2. **Retry once** on a timeout/5xx — the failed call warms the container; the retry is fast.
  Also send a **few images per request** (Modal parallelizes across containers for throughput).
  For guaranteed no-cold-start latency, set `min_containers=1` (keeps one T4 warm, ~$0.60/hr).
  For genuinely large batches, use an async job pattern instead.
- **Faster/real-time option:** a ≥ Ampere GPU (`gpu="A10G"`/`"L4"`) unlocks the vLLM backend,
  but that changes the install and is outside this T4-pinned setup.

---

# Second endpoint: OvisOCR2 (`ovisocr2_app.py`)

A separate deployment that serves **OvisOCR2** — a compact ~0.8B document-parsing VLM — on an **L4**
via **vLLM**, behind the **same FastAPI contract** as the VL app above. The request/response shape is
identical, so the **same `test_client.py` works** — just point `OCR_URL` at this endpoint.

The only per-image difference is the `blocks` field: for OvisOCR2 it reports the number of **HTML
tables** the model emitted (VL reports layout blocks). Everything else — `filename`, `status`,
`html`, `seconds`, `error`, the auth, the formats (JPEG/PNG/BMP/WebP/TIFF), 25 MB/image — matches.
The per-request file cap is **4** (vs VL's 16): OvisOCR2 runs ~10–50 s/page sequentially, so 4 keeps
a single request under Modal's ~150 s synchronous web limit. Send more pages across several requests.

## Deploy

```bash
# 1. Auth token secret (once) — note the DIFFERENT secret name
modal secret create ovisocr2-token \
    AUTH_TOKEN=$(python -c "import secrets; print(secrets.token_urlsafe(32))")

# 2. Pre-download the weights into the volume + prove inference works (optional but recommended)
modal run modal_deploy/ovisocr2_app.py          # -> prints a warmup result

# 3. Deploy
modal deploy modal_deploy/ovisocr2_app.py
```

Modal prints the public URL, e.g. `https://<workspace>--ovisocr2-document-ocr-ovisocr2-web.modal.run`.

## Call it (same client, same shape)

Set `OCR_URL` to this endpoint (in `.env` or `--url`); the client auto-selects `OVISOCR2_TOKEN`
because the URL contains `ovis`.

```bash
export OCR_URL="https://<workspace>--ovisocr2-document-ocr-ovisocr2-web.modal.run"

curl "$OCR_URL/health"                            # {"status":"ok","model":"OvisOCR2","gpu":"L4"}
# token is read from .env (OVISOCR2_TOKEN); or pass --url/--token explicitly:
python modal_deploy/test_client.py --url "$OCR_URL" --out ./ovis_html image_001.jpg image_005.jpg
```

## How it differs from the VL app

| | VL 1.6 (`paddleocr_vl_app.py`) | OvisOCR2 (`ovisocr2_app.py`) |
|---|---|---|
| Model | PaddleOCR-VL 1.6 (layout model + VLM) | OvisOCR2 (single ~0.8B VLM) |
| Engine | PaddleOCR pipeline + vLLM genai **subprocess** | **in-process vLLM** `LLM(...)` |
| Secret | `paddleocr-vl-token` | `ovisocr2-token` |
| HF volume | `paddleocr-vl-hf` | `ovisocr2-hf` |
| `blocks` field | layout blocks detected | HTML tables emitted |
| Cold start | vLLM server boot + pipeline | vLLM weight load (FlashInfer sampler disabled) |

OvisOCR2 keeps **one L4 always warm** (`min_containers=1`) so there are no cold starts — it bills
continuously (~$0.80/hr) even when idle; drop that line to scale to zero like the VL app. Extra
containers autoscaled for bursts still stay warm 5 min after their last request
(`scaledown_window=300`). Both apps process **one image per container**
(`@modal.concurrent(max_inputs=1)`; Modal autoscales to more containers for concurrent requests).

> **Log hygiene:** the deprecation lines you may have seen on first load (flashinfer `tcgen05`,
> `torch.jit.script_method`, transformers `use_fast` / `Qwen2VLImageProcessorFast`) are all internal
> to the pinned `vllm==0.22.1` stack — none are from this app's code, and they don't affect output.
> They're emitted by vLLM's engine **subprocess**, so they're silenced via env the subprocess inherits
> (`PYTHONWARNINGS`, `TRANSFORMERS_VERBOSITY` in `QUIET_ENV`), not an in-process warnings filter.
> The only actionable one is HuggingFace's "unauthenticated requests" note: add an `HF_TOKEN` key to
> the `ovisocr2-token` secret to silence it and lift anonymous download rate limits (optional).
