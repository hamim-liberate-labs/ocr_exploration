"""
PaddleOCR-VL 1.6 document parser on Modal (L4 GPU, vLLM backend) behind a FastAPI API.

POST images as multipart/form-data (field `files`, repeatable) -> JSON with parsed HTML each.
Layout model runs on CPU-Paddle; the VLM runs in a local vLLM genai server on the L4.

Deploy: modal deploy modal_deploy/paddleocr_vl_app.py
Secret: modal secret create paddleocr-vl-token \\
            AUTH_TOKEN=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
"""

from __future__ import annotations

import base64
import glob
import mimetypes
import os
import re
import tempfile
import time

import modal

APP_NAME = "paddleocr-vl-document-ocr"
MODEL_NAME = "PaddleOCR-VL-1.6"
GPU = "L4"

GENAI_MODEL = "PaddleOCR-VL-1.6-0.9B"
GENAI_PORT = 8118
GENAI_URL = f"http://127.0.0.1:{GENAI_PORT}/v1"
GENAI_READY_TIMEOUT_S = 600

MAX_FILES_PER_REQUEST = 16
MAX_FILE_BYTES = 25 * 1024 * 1024
VALID_IMAGE_FORMATS = {"jpeg", "jpg", "png", "bmp", "webp", "tiff", "mpo"}

SECRET = modal.Secret.from_name("paddleocr-vl-token")
HF_SECRET = modal.Secret.from_name("huggingface-secret")   # shared workspace secret; provides HF_TOKEN

# Model caches persist across cold starts (scale-to-zero) so models download only once.
PADDLEX_VOL = modal.Volume.from_name("paddleocr-vl-paddlex", create_if_missing=True)
HF_VOL = modal.Volume.from_name("paddleocr-vl-hf", create_if_missing=True)
VOLUMES = {"/root/.paddlex": PADDLEX_VOL, "/root/.cache/huggingface": HF_VOL}

QUIET_ENV = {
    "GLOG_minloglevel": "3",
    "GLOG_v": "0",
    "FLAGS_call_stack_level": "0",
    "HF_HOME": "/root/.cache/huggingface",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    # Silence upstream deprecation noise from the vLLM subprocess (env is the only channel that reaches it).
    "PYTHONWARNINGS": "ignore::DeprecationWarning,ignore::FutureWarning",
    "TRANSFORMERS_VERBOSITY": "error",
}


class ImageClientError(Exception):
    """Expected per-image failure (empty / too large / not a supported image)."""


image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04", add_python="3.11")
    .apt_install("libgl1", "libglib2.0-0", "libgomp1", "git")
    .pip_install("paddleocr[doc-parser]")
    .run_commands("paddleocr install_genai_server_deps vllm")   # vLLM + torch (cu126)
    # CPU Paddle for the layout model — the -gpu wheel downgrades nccl and breaks vLLM's torch. VLM runs on GPU.
    .pip_install("paddlepaddle==3.2.1")
    .pip_install("markdown", "fastapi[standard]", "python-multipart", "pillow", "httpx")
    .apt_install("build-essential")   # vLLM's Triton JIT needs a host C compiler at runtime
    .env(QUIET_ENV)
    .run_commands("rm -rf /root/.paddlex /root/.cache/huggingface")   # empty the volume mount paths
)

app = modal.App(APP_NAME)

with image.imports():
    import contextlib
    import io
    import logging
    import secrets as _secrets
    import subprocess
    import sys
    import warnings
    from typing import Literal

    import httpx

    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger("paddleocr_vl")
    for _noisy in ("paddleformers", "paddlex", "paddleocr", "ppocr", "PIL", "vllm", "httpx"):
        logging.getLogger(_noisy).setLevel(logging.ERROR)

    import markdown as _markdown
    from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
    from fastapi.openapi.utils import get_openapi
    from fastapi.responses import JSONResponse
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    from paddleocr import PaddleOCRVL
    from PIL import Image as PILImage
    from pydantic import BaseModel, Field

    class OCRResult(BaseModel):
        """Parse outcome for one uploaded image; a failure here never fails the batch."""

        filename: str = Field(..., description="Original uploaded filename.")
        status: Literal["ok", "error"] = Field(..., description="Per-image outcome.")
        html: str | None = Field(None, description="Self-contained HTML page (present iff ok).")
        blocks: int | None = Field(None, ge=0, description="Layout blocks detected (present iff ok).")
        seconds: float | None = Field(None, ge=0, description="Inference wall-time for this image.")
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

    @contextlib.contextmanager
    def _quiet():
        """Silence Paddle's glog/banner output at the fd level (redirect_stdout is not enough)."""
        devnull = os.open(os.devnull, os.O_WRONLY)
        sys.stdout.flush()
        sys.stderr.flush()
        old_out, old_err = os.dup(1), os.dup(2)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(old_out, 1)
            os.dup2(old_err, 2)
            os.close(devnull)
            os.close(old_out)
            os.close(old_err)


_IMG_MD_RE = re.compile(r"!\[(.*?)\]\((.*?)\)")

_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
          max-width: 900px; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; color: #1a1a1a; }}
  h1, h2, h3 {{ line-height: 1.25; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
  table, th, td {{ border: 1px solid #bbb; }}
  th, td {{ padding: 6px 10px; text-align: left; vertical-align: top; word-wrap: break-word; }}
  img {{ max-width: 100%; height: auto; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def _inline_images(md_text: str, base_dir: str) -> str:
    """Inline local markdown image refs as base64 data URIs so the HTML is self-contained."""

    def repl(m: "re.Match[str]") -> str:
        alt, src = m.group(1), m.group(2)
        if src.startswith(("http://", "https://", "data:")):
            return m.group(0)
        path = os.path.join(base_dir, src)
        if os.path.exists(path):
            mime = mimetypes.guess_type(path)[0] or "image/jpeg"
            with open(path, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode()
            return f"![{alt}](data:{mime};base64,{b64})"
        return m.group(0)

    return _IMG_MD_RE.sub(repl, md_text)


@app.cls(
    image=image,
    gpu=GPU,
    volumes=VOLUMES,
    min_containers=1,       # keep one container always warm -> no cold starts (bills the L4 24/7)
    scaledown_window=600,   # keep a warm container 10 min after the last request
    timeout=1800,           # room for a slow cold start (vLLM model load)
    secrets=[SECRET, HF_SECRET],
)
@modal.concurrent(max_inputs=1)   # pipeline isn't concurrency-safe; scale out across containers
class PaddleVL:
    @modal.enter()
    def load(self) -> None:
        cfg_path = os.path.join(tempfile.gettempdir(), "vllm_config.yml")
        # max-num-seqs=64: each page region decodes as its own sequence, so a high cap lets a dense
        # page's regions batch (~10s) instead of serializing (~25s). CUDA graphs stay on by default.
        with open(cfg_path, "w") as fh:
            fh.write("gpu-memory-utilization: 0.55\nmax-num-seqs: 64\n")

        # vLLM genai server (VLM) as a subprocess on the L4; the pipeline talks to it over HTTP.
        self._server_log = os.path.join(tempfile.gettempdir(), "genai_server.log")
        self._server_logf = open(self._server_log, "w")
        self._server = subprocess.Popen(
            [
                "paddleocr", "genai_server",
                "--model_name", GENAI_MODEL,
                "--backend", "vllm",
                "--host", "127.0.0.1",
                "--port", str(GENAI_PORT),
                "--backend_config", cfg_path,
            ],
            stdout=self._server_logf,
            stderr=subprocess.STDOUT,
        )
        self._await_server_ready()

        with _quiet():
            self.predictor = PaddleOCRVL(
                vl_rec_backend="vllm-server",
                vl_rec_server_url=GENAI_URL,
                vl_rec_api_model_name=GENAI_MODEL,
                device="cpu",
            )
        logger.info("%s ready on %s (vLLM backend).", MODEL_NAME, GPU)

    def _server_log_tail(self, n: int = 200) -> str:
        try:
            with open(self._server_log, encoding="utf-8", errors="replace") as fh:
                return "".join(fh.readlines()[-n:])
        except Exception:
            return "(no server log)"

    def _await_server_ready(self) -> None:
        deadline = time.monotonic() + GENAI_READY_TIMEOUT_S
        while time.monotonic() < deadline:
            if self._server.poll() is not None:
                raise RuntimeError(
                    f"vLLM genai server exited early (code {self._server.returncode}).\n"
                    f"--- server log tail ---\n{self._server_log_tail()}"
                )
            try:
                if httpx.get(f"{GENAI_URL}/models", timeout=5).status_code == 200:
                    logger.info("vLLM genai server is ready.")
                    return
            except httpx.HTTPError:
                pass
            time.sleep(3)
        raise RuntimeError(
            f"vLLM genai server did not become ready in time.\n"
            f"--- server log tail ---\n{self._server_log_tail()}"
        )

    @modal.exit()
    def shutdown(self) -> None:
        server = getattr(self, "_server", None)
        if server and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except Exception:
                server.kill()

    @modal.method()
    def warmup(self) -> dict:
        """`modal run` entrypoint target: pre-populate the volumes and prove inference works."""
        from PIL import Image as _Img
        from PIL import ImageDraw as _Draw

        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "warmup.png")
            img = _Img.new("RGB", (960, 640), "white")
            _Draw.Draw(img).text((60, 280), "PaddleOCR VL warmup 1234567890 ABCDEFG", fill="black")
            img.save(p)
            with open(p, "rb") as fh:
                data = fh.read()
        html, seconds, blocks = self._predict_to_html(data, ".png")
        return {"seconds": round(seconds, 3), "blocks": blocks, "html_len": len(html)}

    @staticmethod
    def _validate_image(data: bytes) -> str:
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
        return f".{'jpg' if fmt in ('jpeg', 'mpo') else fmt}"

    def _predict_to_html(self, data: bytes, suffix: str) -> tuple[str, float, int]:
        with tempfile.TemporaryDirectory() as din, tempfile.TemporaryDirectory() as dout:
            ipath = os.path.join(din, f"page{suffix}")
            with open(ipath, "wb") as fh:
                fh.write(data)

            t0 = time.time()
            with _quiet():
                results = list(
                    self.predictor.predict(
                        ipath,
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_seal_recognition=False,
                        use_chart_recognition=False,
                    )
                )
                for res in results:
                    res.save_to_markdown(save_path=dout)   # VL markdown embeds tables as raw HTML
            seconds = time.time() - t0

            md_files = sorted(glob.glob(os.path.join(dout, "**", "*.md"), recursive=True))
            md_text = "\n\n".join(open(m, encoding="utf-8").read() for m in md_files)
            md_text = _inline_images(md_text, dout)

        body = _markdown.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists"])
        return _HTML_TEMPLATE.format(title="document", body=body), seconds, len(results)

    def _process_one(self, filename: str, data: bytes) -> "OCRResult":
        try:
            suffix = self._validate_image(data)
            html, seconds, blocks = self._predict_to_html(data, suffix)
            logger.info("parsed %s in %.2fs (%d blocks)", filename, seconds, blocks)
            return OCRResult(
                filename=filename, status="ok", html=html, blocks=blocks, seconds=round(seconds, 3)
            )
        except ImageClientError as e:
            logger.warning("skip %s: %s", filename, e)
            return OCRResult(filename=filename, status="error", error=str(e))
        except Exception as e:
            logger.exception("failed to parse %s", filename)
            return OCRResult(filename=filename, status="error", error=f"internal error: {e}")

    @modal.asgi_app()
    def web(self):
        api = FastAPI(
            title="PaddleOCR-VL 1.6 Document OCR",
            version="1.0.0",
            description="Convert document images to structured HTML using PaddleOCR-VL 1.6 (vLLM).",
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

            results: list[OCRResult] = []
            for f in files:
                data = await f.read()
                results.append(self._process_one(f.filename or "upload", data))
            return OCRResponse(count=len(results), results=results)

        @api.get("/health", tags=["System"], summary="Liveness/Readiness Probe", response_model=HealthResponse)
        def health() -> HealthResponse:
            """Unauthenticated readiness probe; 200 means the vLLM backend + pipeline are loaded."""
            return HealthResponse(status="ok", model=MODEL_NAME, gpu=GPU)

        return api


@app.local_entrypoint()
def main():
    """`modal run modal_deploy/paddleocr_vl_app.py` -> pre-download models into the volumes + validate."""
    print("warmup result:", PaddleVL().warmup.remote())
