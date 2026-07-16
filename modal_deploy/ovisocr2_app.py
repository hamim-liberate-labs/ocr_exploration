"""
OvisOCR2 document parser on Modal (L4 GPU, vLLM) behind a FastAPI API.

Same request/response contract as paddleocr_vl_app.py (PaddleOCR-VL 1.6), so the same client works against
either endpoint: POST images as multipart/form-data (field `files`, repeatable) -> JSON with a
self-contained HTML page per image. The only per-image difference is that `blocks` reports the number
of HTML tables OvisOCR2 emitted (VL reports layout blocks) — see the field description.

OvisOCR2 (ATH-MaaS/OvisOCR2) is a compact ~0.8B document-parsing VLM; it reads a page image and emits
a single Markdown document with HTML tables + LaTeX, which we render to HTML here.

Deploy: modal deploy modal_deploy/ovisocr2_app.py
Secret: modal secret create ovisocr2-token \\
            AUTH_TOKEN=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
"""

from __future__ import annotations

import os

import modal

APP_NAME = "ovisocr2-document-ocr"
MODEL_NAME = "OvisOCR2"
GPU = "L4"
MODEL_ID = "ATH-MaaS/OvisOCR2"

MAX_FILES_PER_REQUEST = 4   # batched into one vLLM call; 4 keeps a request under Modal's ~150s sync web limit
MAX_FILE_BYTES = 25 * 1024 * 1024
VALID_IMAGE_FORMATS = {"jpeg", "jpg", "png", "bmp", "webp", "tiff", "mpo"}

SECRET = modal.Secret.from_name("ovisocr2-token")
HF_SECRET = modal.Secret.from_name("huggingface-secret")   # shared workspace secret; provides HF_TOKEN

# HF weights cache persists across cold starts (scale-to-zero), so weights download only once.
HF_VOL = modal.Volume.from_name("ovisocr2-hf", create_if_missing=True)
VOLUMES = {"/root/.cache/huggingface": HF_VOL}

# FLASHINFER_SAMPLER=0: use the native Torch sampler (FlashInfer's JIT-compiles a kernel needing nvcc).
# PYTHONWARNINGS/TRANSFORMERS_VERBOSITY silence upstream deprecation noise from vLLM's engine
# subprocess (flashinfer/torch.jit/transformers) — env is the only channel that reaches that subprocess.
# HF_TOKEN (from the secret) is what removes the HF "unauthenticated requests" note.
QUIET_ENV = {
    "HF_HOME": "/root/.cache/huggingface",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "VLLM_LOGGING_LEVEL": "WARNING",
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
    "PYTHONWARNINGS": "ignore::DeprecationWarning,ignore::FutureWarning",
    "TRANSFORMERS_VERBOSITY": "error",
}


class ImageClientError(Exception):
    """Expected per-image failure (empty / too large / not a supported image)."""


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm==0.22.1",          # model card pins vLLM 0.22.1 (bundles torch + cu wheels)
        "pillow",
        "markdown",
        "fastapi[standard]",
        "python-multipart",
    )
    .env(QUIET_ENV)
)

app = modal.App(APP_NAME)

with image.imports():
    import io
    import logging
    import secrets as _secrets
    import time
    import warnings
    from typing import Literal

    # Main-process only (the subprocess is handled by PYTHONWARNINGS/TRANSFORMERS_VERBOSITY in
    # QUIET_ENV). Scope to deprecation-style categories so genuine warnings still surface.
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger("ovisocr2")
    for _noisy in ("vllm", "PIL", "transformers"):
        logging.getLogger(_noisy).setLevel(logging.ERROR)

    import markdown as _markdown
    from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
    from fastapi.openapi.utils import get_openapi
    from fastapi.responses import JSONResponse
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    from PIL import Image as PILImage
    from pydantic import BaseModel, Field
    from vllm import LLM, SamplingParams

    class OCRResult(BaseModel):
        """Parse outcome for one uploaded image; a failure here never fails the batch."""

        filename: str = Field(..., description="Original uploaded filename.")
        status: Literal["ok", "error"] = Field(..., description="Per-image outcome.")
        html: str | None = Field(None, description="Self-contained HTML page (present iff ok).")
        blocks: int | None = Field(
            None, ge=0, description="HTML tables OvisOCR2 emitted for this page (present iff ok)."
        )
        seconds: float | None = Field(
            None, ge=0,
            description="Inference wall-time for this image (batch wall-time amortized per image "
                        "when several are sent in one request).",
        )
        error: str | None = Field(None, description="Failure reason (present iff status='error').")

    class OCRResponse(BaseModel):
        """One OCRResult per uploaded image, in upload order."""

        model: str = Field(MODEL_NAME, description="Model that produced these results.")
        count: int = Field(..., ge=0, description="Number of results.")
        results: list[OCRResult] = Field(..., description="Per-image results, in upload order.")

    class HealthResponse(BaseModel):
        status: Literal["ok"] = Field(..., description="'ok' when serving.")
        model: str = Field(..., description="Model served by this deployment.")
        gpu: str = Field(..., description="GPU type the container runs on.")


# The fixed OCR prompt from the model card (the "<" + "img" split only avoids a literal <img> token
# appearing in this source file).
OCR_PROMPT = (
    "\nExtract all readable content from the image in natural human reading order and output the "
    "result as a single Markdown document. For charts or images, represent them using an HTML image "
    'tag: <' + 'img src="images/bbox_{left}_{top}_{right}_{bottom}.jpg" />, where left, top, right, '
    "bottom are bounding box coordinates scaled to [0, 1000). Format formulas as LaTeX. Format "
    "tables as HTML: <table>...</table>. Transcribe all other text as standard Markdown. Preserve "
    "the original text without translation or paraphrasing."
)

HOUSE_CSS = (
    "body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:900px;"
    "margin:2rem auto;padding:0 1rem;line-height:1.5;color:#1a1a1a}"
    "table{border-collapse:collapse;width:100%;margin:1rem 0}"
    "table,th,td{border:1px solid #bbb}th,td{padding:6px 10px;text-align:left;vertical-align:top;"
    "word-wrap:break-word}h1,h2,h3{line-height:1.25}img{max-width:100%;height:auto}"
)

MATHJAX = (
    "<script>window.MathJax={tex:{inlineMath:[['$','$'],['\\\\(','\\\\)']],"
    "displayMath:[['$$','$$'],['\\\\[','\\\\]']]}};</script>"
    "<script async src='https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'></script>"
)

_HTML_TEMPLATE = (
    "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
    "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
    "<title>{title}</title><style>{css}</style>{mathjax}</head>\n<body>\n{body}\n</body></html>\n"
)


@app.cls(
    image=image,
    gpu=GPU,
    volumes=VOLUMES,
    min_containers=1,       # keep ONE L4 always warm (no cold starts; bills continuously ~$0.80/hr)
    scaledown_window=300,   # extra autoscaled containers stay warm 5 min after their last request
    timeout=1800,           # room for a slow cold start (weight download + vLLM load)
    secrets=[SECRET, HF_SECRET],
)
@modal.concurrent(max_inputs=1)   # vLLM generate isn't concurrency-safe; scale out across containers
class OvisOCR2:
    @modal.enter()
    def load(self) -> None:
        self.model = LLM(
            model=MODEL_ID,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.85,
            gdn_prefill_backend="triton",   # model architecture requirement (linear attention)
            # CUDA graphs ~3.8-4x decode on this decode-bound 0.8B model (measured on L4); captured once
            # at load, kept warm by min_containers=1. Revert to True if capture ever fails on this arch.
            enforce_eager=False,
        )
        self.prompt = self.model.get_tokenizer().apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": OCR_PROMPT}]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        # 4096 (not the card's 16384): real pages top out ~2600 tokens, so this never truncates but
        # caps the ~9% degenerate repeat loops that otherwise run to 16384 tokens (~500s).
        self.sampling_params = SamplingParams(max_tokens=4096, temperature=0.0)
        try:
            self._parse_one(self._synth_page())   # warm JIT/compile paths at a realistic page size
        except Exception as e:
            logger.warning("warmup skipped: %s", e)
        logger.info("%s ready on %s (vLLM).", MODEL_NAME, GPU)

    @staticmethod
    def _synth_page():
        """A dataset-sized synthetic page so the enter-time warmup JITs the shapes real pages hit."""
        from PIL import ImageDraw
        img = PILImage.new("RGB", (1240, 1600), "white")
        draw = ImageDraw.Draw(img)
        for i in range(24):
            draw.text((60, 50 + i * 60),
                      f"Warmup line {i}: 0123456789 ABCDEFG the quick brown fox jumps", fill="black")
        return img

    def _clean_truncated_repeats(self, text, min_text_len=8000, max_period=200,
                                 min_period=1, min_repeat_chars=100, min_repeat_times=5):
        n = len(text)
        if n < min_text_len:
            return text
        max_period = min(max_period, n - 1)
        for unit_len in range(min_period, max_period + 1):
            if text[n - 1] != text[n - 1 - unit_len]:
                continue
            match_len, idx = 1, n - 2
            while idx >= unit_len and text[idx] == text[idx - unit_len]:
                match_len += 1
                idx -= 1
            total_len = match_len + unit_len
            if total_len // unit_len >= min_repeat_times and total_len >= min_repeat_chars:
                return text[: n - total_len + unit_len] + text[n - (total_len % unit_len):]
        return text

    def _req_dict(self, pil_image) -> dict:
        """The vLLM request dict for one page image (shared by the single + batched paths)."""
        return {
            "prompt": self.prompt,
            "multi_modal_data": {"image": pil_image},
            "mm_processor_kwargs": {"images_kwargs": {
                "min_pixels": 448 * 448, "max_pixels": 2880 * 2880}},
        }

    def _postprocess(self, text: str, filter_imgtags: bool = True) -> str:
        text = text.strip()
        if filter_imgtags:
            text = "\n\n".join(
                b for b in text.split("\n\n")
                if not b.strip().startswith('<img src="images/bbox_'))
        return self._clean_truncated_repeats(text)

    def _parse_one(self, pil_image, filter_imgtags: bool = True) -> str:
        outputs = self.model.generate([self._req_dict(pil_image)], self.sampling_params)
        return self._postprocess(outputs[0].outputs[0].text, filter_imgtags)

    def _md_to_html(self, md_text: str, title: str) -> str:
        body = _markdown.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists"])
        return _HTML_TEMPLATE.format(title=title, css=HOUSE_CSS, mathjax=MATHJAX, body=body)

    @staticmethod
    def _validate_image(data: bytes) -> None:
        if not data:
            raise ImageClientError("empty file")
        if len(data) > MAX_FILE_BYTES:
            raise ImageClientError(f"file too large: {len(data)} bytes > {MAX_FILE_BYTES}")
        try:
            with PILImage.open(io.BytesIO(data)) as im:
                fmt = (im.format or "").lower()
        except Exception:
            raise ImageClientError("not a decodable image")
        if fmt not in VALID_IMAGE_FORMATS:
            raise ImageClientError(f"unsupported image format: {fmt or 'unknown'}")

    def _predict_to_html(self, data: bytes, filename: str) -> tuple[str, float, int]:
        with PILImage.open(io.BytesIO(data)) as im:
            pil = im.convert("RGB")
        t0 = time.time()
        md_text = self._parse_one(pil)
        seconds = time.time() - t0
        n_tables = md_text.lower().count("<table")
        return self._md_to_html(md_text, filename), seconds, n_tables

    def _result_from_text(self, filename: str, raw_text: str, seconds: float) -> "OCRResult":
        """Turn one model output into an ok OCRResult (post-process -> HTML -> count tables)."""
        md_text = self._postprocess(raw_text)
        n_tables = md_text.lower().count("<table")
        return OCRResult(filename=filename, status="ok", html=self._md_to_html(md_text, filename),
                         blocks=n_tables, seconds=round(seconds, 3))

    def _process_one_pil(self, filename: str, pil_image) -> "OCRResult":
        """Single-image generate — used only as the batch fallback path."""
        try:
            t0 = time.time()
            outputs = self.model.generate([self._req_dict(pil_image)], self.sampling_params)
            return self._result_from_text(filename, outputs[0].outputs[0].text, time.time() - t0)
        except Exception as e:
            logger.exception("failed to parse %s", filename)
            return OCRResult(filename=filename, status="error", error=f"internal error: {e}")

    def _process_batch(self, items: "list[tuple[str, object]]") -> "list[OCRResult]":
        """One batched generate over all valid pages (interleaved decode >> one-by-one on the L4).
        ``items`` is [(filename, RGB image)], returned in input order; ``seconds`` is the batch
        wall-time amortized per image. Falls back to per-image if the batch call itself errors."""
        if not items:
            return []
        t0 = time.time()
        try:
            outputs = self.model.generate([self._req_dict(pil) for _, pil in items], self.sampling_params)
        except Exception:
            logger.exception("batch generate failed; falling back to per-image")
            return [self._process_one_pil(fn, pil) for fn, pil in items]
        per_image = (time.time() - t0) / len(items)
        results: list[OCRResult] = []
        for (filename, _), out in zip(items, outputs):
            try:
                res = self._result_from_text(filename, out.outputs[0].text, per_image)
                logger.info("parsed %s (batch of %d, ~%.2fs/img, %s tables)",
                            filename, len(items), per_image, res.blocks)
                results.append(res)
            except Exception as e:
                logger.exception("post-process failed for %s", filename)
                results.append(OCRResult(filename=filename, status="error", error=f"internal error: {e}"))
        return results

    @modal.method()
    def warmup(self) -> dict:
        """`modal run` entrypoint target: pre-populate the volume and prove inference works."""
        buf = io.BytesIO(); self._synth_page().save(buf, format="PNG")
        html, seconds, n_tables = self._predict_to_html(buf.getvalue(), "warmup.png")
        return {"seconds": round(seconds, 3), "tables": n_tables, "html_len": len(html)}

    @modal.asgi_app()
    def web(self):
        api = FastAPI(
            title="OvisOCR2 Document OCR",
            version="1.0.0",
            description="Convert document images to structured HTML using OvisOCR2 (vLLM).",
        )

        # Swagger renders `files` as a file-picker only when the item schema is `format: binary`;
        # FastAPI's OpenAPI 3.1 emits `contentMediaType` instead, so rewrite it.
        def _custom_openapi() -> dict:
            if api.openapi_schema:
                return api.openapi_schema
            schema = get_openapi(
                title=api.title, version=api.version, description=api.description, routes=api.routes
            )

            def _fix_binary(node: object) -> None:
                if isinstance(node, dict):
                    if node.get("type") == "string" and node.get("contentMediaType") == "application/octet-stream":
                        node.pop("contentMediaType", None)
                        node["format"] = "binary"
                    for value in node.values():
                        _fix_binary(value)
                elif isinstance(node, list):
                    for value in node:
                        _fix_binary(value)

            _fix_binary(schema)
            api.openapi_schema = schema
            return schema

        api.openapi = _custom_openapi
        bearer = HTTPBearer(auto_error=False)

        def verify_token(creds: HTTPAuthorizationCredentials | None = Depends(bearer)) -> None:
            expected = os.environ.get("AUTH_TOKEN")
            if not expected:
                raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "server auth token not configured")
            if creds is None or not _secrets.compare_digest(creds.credentials, expected):
                raise HTTPException(
                    status.HTTP_401_UNAUTHORIZED,
                    "missing or invalid bearer token",
                    headers={"WWW-Authenticate": "Bearer"},
                )

        @api.exception_handler(Exception)
        async def _unhandled(request: Request, exc: Exception):
            logger.exception("unhandled error on %s", request.url.path)
            return JSONResponse(status_code=500, content={"detail": "internal server error"})

        @api.post(
            "/v1/document-ocr",
            tags=["Document OCR"],
            summary="Parse Document Images to HTML",
            operation_id="parse_document_images",
            response_model=OCRResponse,
            response_model_exclude_none=True,
            dependencies=[Depends(verify_token)],
            responses={
                401: {"description": "Missing or invalid bearer token."},
                413: {"description": f"More than {MAX_FILES_PER_REQUEST} files in one request."},
                422: {"description": "Malformed multipart body / missing `files` field."},
                500: {"description": "Unexpected server error."},
            },
        )
        async def parse_document_images(
            files: list[UploadFile] = File(..., description="One or more document images (multipart).")
        ) -> OCRResponse:
            """Parse each uploaded image and return HTML per image.

            Bearer-token protected. Up to MAX_FILES_PER_REQUEST images/request, ≤ MAX_FILE_BYTES
            each (JPEG/PNG/BMP/WebP/TIFF). Per-image problems are reported as ``status="error"``
            on that item and never fail the batch; request-level problems raise 4xx/5xx.
            """
            if not files:
                raise HTTPException(status.HTTP_400_BAD_REQUEST, "no files uploaded")
            if len(files) > MAX_FILES_PER_REQUEST:
                raise HTTPException(
                    status.HTTP_413_CONTENT_TOO_LARGE,
                    f"too many files: {len(files)} > {MAX_FILES_PER_REQUEST}",
                )

            # Validate/decode all uploads, then batch the good pages into one generate() (order
            # preserved; validation failures never enter the batch).
            results: list[OCRResult | None] = [None] * len(files)
            batch: list[tuple[int, str, object]] = []
            for i, f in enumerate(files):
                filename = f.filename or "upload"
                data = await f.read()
                try:
                    self._validate_image(data)
                    with PILImage.open(io.BytesIO(data)) as im:
                        pil = im.convert("RGB")
                except ImageClientError as e:
                    logger.warning("skip %s: %s", filename, e)
                    results[i] = OCRResult(filename=filename, status="error", error=str(e))
                except Exception as e:
                    logger.exception("failed to read %s", filename)
                    results[i] = OCRResult(filename=filename, status="error", error=f"internal error: {e}")
                else:
                    batch.append((i, filename, pil))

            if batch:
                processed = self._process_batch([(fn, pil) for _, fn, pil in batch])
                for (i, _, _), res in zip(batch, processed):
                    results[i] = res

            return OCRResponse(count=len(files), results=[r for r in results if r is not None])

        @api.get("/health", tags=["System"], summary="Liveness/Readiness Probe", response_model=HealthResponse)
        def health() -> HealthResponse:
            """Unauthenticated readiness probe; 200 means the vLLM model is loaded."""
            return HealthResponse(status="ok", model=MODEL_NAME, gpu=GPU)

        return api


@app.local_entrypoint()
def main():
    """`modal run modal_deploy/ovisocr2_app.py` -> pre-download weights into the volume + validate."""
    print("warmup result:", OvisOCR2().warmup.remote())
