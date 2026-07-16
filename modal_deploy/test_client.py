#!/usr/bin/env python3
"""
Send images to the deployed PaddleOCR-VL 1.6 endpoint and save the returned HTML.

Credentials are read from the repo-root `.env` (OCR_URL, OCR_TOKEN, OVISOCR2_TOKEN) —
no manual `export` needed. Anything already set in the shell environment wins over `.env`.
The token is picked automatically from the URL: an OvisOCR2 endpoint uses OVISOCR2_TOKEN,
otherwise OCR_TOKEN. Override either with --url/--token.

Usage:
    # one or more image files (URL/token come from .env):
    python test_client.py --out ./html_out image_001.jpg image_005.jpg
    # or a whole directory of images:
    python test_client.py --out ./html_out ../

The endpoint scales to zero, so the FIRST call after an idle period cold-starts a container
(~25 s to load the model). This client retries once on a cold-start timeout/5xx — the same
pattern a calling backend should use.
"""
from __future__ import annotations

import argparse
import glob
import mimetypes
import os
import sys
import time

import requests

EXTS = ("jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff")
COLD_START_RETRIES = 1        # retry once; the failed call warms the container
REQUEST_TIMEOUT_S = 300       # generous: cold start + inference for a small batch

# Repo-root .env (this file lives in modal_deploy/).
DOTENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, ".env")


def load_dotenv(path: str = DOTENV_PATH) -> None:
    """Populate os.environ from a .env file without overwriting already-set vars."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    except FileNotFoundError:
        pass


def resolve_token(url: str | None, explicit: str | None) -> str | None:
    """Pick the bearer token: explicit --token wins, else by endpoint (OvisOCR2 vs VL)."""
    if explicit:
        return explicit
    if url and "ovis" in url.lower() and os.environ.get("OVISOCR2_TOKEN"):
        return os.environ["OVISOCR2_TOKEN"]
    return os.environ.get("OCR_TOKEN")


def collect(paths: list[str]) -> list[str]:
    """Expand the given files/directories into a sorted, de-duplicated list of image paths."""
    files: list[str] = []
    for p in paths:
        if os.path.isdir(p):
            for e in EXTS:
                files += glob.glob(os.path.join(p, f"*.{e}"))
                files += glob.glob(os.path.join(p, f"*.{e.upper()}"))
        elif os.path.isfile(p):
            files.append(p)
        else:
            print(f"skip (not found): {p}", file=sys.stderr)
    return sorted(set(files))


def post_images(endpoint: str, token: str, images: list[str]) -> dict:
    """POST all images as multipart/form-data, retrying once on a cold-start failure."""
    headers = {"Authorization": f"Bearer {token}"}
    # Read bytes up front so we can safely retry without reopening file handles.
    payload = [
        ("files", (os.path.basename(p), open(p, "rb").read(),
                   mimetypes.guess_type(p)[0] or "application/octet-stream"))
        for p in images
    ]

    last_exc: Exception | None = None
    for attempt in range(COLD_START_RETRIES + 1):
        try:
            resp = requests.post(endpoint, headers=headers, files=payload, timeout=REQUEST_TIMEOUT_S)
            if resp.status_code >= 500 and attempt < COLD_START_RETRIES:
                print(f"  {resp.status_code} from server, retrying (cold start?)…", file=sys.stderr)
                time.sleep(2)
                continue
            resp.raise_for_status()
            return resp.json()
        except (requests.Timeout, requests.ConnectionError) as e:
            last_exc = e
            if attempt < COLD_START_RETRIES:
                print("  request timed out, retrying (cold start?)…", file=sys.stderr)
                time.sleep(2)
                continue
            raise
    raise last_exc  # pragma: no cover


def main() -> int:
    load_dotenv()  # pull OCR_URL / OCR_TOKEN / OVISOCR2_TOKEN from repo-root .env

    ap = argparse.ArgumentParser(description="PaddleOCR-VL / OvisOCR2 test client")
    ap.add_argument("paths", nargs="+", help="image files and/or directories")
    ap.add_argument("--url", default=os.environ.get("OCR_URL"), help="endpoint base URL")
    ap.add_argument("--token", default=None,
                    help="bearer token (default: auto from .env by endpoint)")
    ap.add_argument("--out", default="./html_out", help="directory to write .html into")
    args = ap.parse_args()

    token = resolve_token(args.url, args.token)
    if not args.url or not token:
        ap.error("set --url/--token or OCR_URL + OCR_TOKEN/OVISOCR2_TOKEN (in .env or env vars)")

    images = collect(args.paths)
    if not images:
        ap.error("no images found")

    os.makedirs(args.out, exist_ok=True)
    base = args.url.rstrip("/")
    endpoint = base + "/v1/document-ocr"

    # Warm the container first: GET /health blocks until a container is up and the model is
    # loaded, so the subsequent POST hits a warm container and stays under the 150 s web limit.
    print("warming container via /health…")
    try:
        requests.get(base + "/health", timeout=180).raise_for_status()
    except requests.RequestException as e:
        print(f"  warm-up call failed ({e}); continuing anyway", file=sys.stderr)

    print(f"POST {len(images)} image(s) -> {endpoint}")
    payload = post_images(endpoint, token, images)

    for r in payload["results"]:
        if r["status"] == "ok":
            stem = os.path.splitext(r["filename"])[0]
            dst = os.path.join(args.out, f"{stem}.html")
            with open(dst, "w", encoding="utf-8") as fh:
                fh.write(r["html"])
            print(f"  ok    {r['filename']:<24} {r['seconds']:>6.2f}s  {r['blocks']:>3} blocks -> {dst}")
        else:
            print(f"  ERROR {r['filename']:<24} {r['error']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
