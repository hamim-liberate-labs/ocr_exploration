"""
Modal app for OvisOCR2 (image -> Markdown) on an L4 GPU, used by
`ovisocr2_modal_l4.ipynb`.

This lives in a real module (not inline in the notebook) on purpose: a class defined
inside a notebook is sent to Modal via cloudpickle ("serialized" function), which
requires the local Python to match the container's Python (3.11). Importing the app
from a module instead skips that path, so the notebook can run under any local Python.
"""

from __future__ import annotations

import modal

MODEL_ID = "ATH-MaaS/OvisOCR2"
GPU = "L4"   # Ampere/Ada -> vLLM + FlashInfer work (unlike Kaggle's T4)

app = modal.App("ovisocr2-l4-image-to-html")

HF_VOL = modal.Volume.from_name("ovisocr2-hf-cache", create_if_missing=True)

vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.22.1", "pillow", "markdown")
    .env({
        "HF_HOME": "/root/.cache/huggingface",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "VLLM_LOGGING_LEVEL": "WARNING",
        # Use vLLM's native Torch top-k/top-p sampler instead of FlashInfer's. FlashInfer's
        # sampler JIT-compiles a CUDA kernel on first use, which needs nvcc — the slim image
        # has no CUDA toolkit, so init would crash with "Could not find nvcc". Attention runs
        # on a precompiled backend and is unaffected. (Same env the Kaggle T4 build needed,
        # for a different reason.)
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
    })
)

with vllm_image.imports():
    import io
    import time

    from PIL import Image as PILImage
    from vllm import LLM, SamplingParams

# The fixed OCR prompt from the model card (kept verbatim; the split on "<" + "img" only
# avoids a literal <img ...> token appearing in this source file).
OCR_PROMPT = (
    "\nExtract all readable content from the image in natural human reading order and output the "
    "result as a single Markdown document. For charts or images, represent them using an HTML image "
    'tag: <' + 'img src="images/bbox_{left}_{top}_{right}_{bottom}.jpg" />, where left, top, right, '
    "bottom are bounding box coordinates scaled to [0, 1000). Format formulas as LaTeX. Format "
    "tables as HTML: <table>...</table>. Transcribe all other text as standard Markdown. Preserve "
    "the original text without translation or paraphrasing."
)


@app.cls(
    image=vllm_image,
    gpu=GPU,
    volumes={"/root/.cache/huggingface": HF_VOL},
    scaledown_window=300,   # keep the L4 warm 5 min after the last call
    timeout=1800,           # room for cold start (weight download + model load)
)
class OvisOCR2:
    @modal.enter()
    def load(self):
        self.model = LLM(
            model=MODEL_ID,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.85,
            gdn_prefill_backend="triton",   # model architecture requirement (linear attention)
            enforce_eager=True,             # fast, reliable load; set False for more decode speed
        )
        self.prompt = self.model.get_tokenizer().apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": OCR_PROMPT}]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        self.sampling_params = SamplingParams(max_tokens=16384, temperature=0.0)

        # Warmup so the first real image isn't slowed by vLLM's one-time compile/autotune.
        try:
            self._parse_one(PILImage.new("RGB", (448, 448), "white"))
            print("warmup ok")
        except Exception as e:
            print("warmup skipped:", e)

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
            repeat_times = total_len // unit_len
            tail_len = total_len % unit_len
            if repeat_times >= min_repeat_times and total_len >= min_repeat_chars:
                return text[: n - total_len + unit_len] + text[n - tail_len:]
        return text

    def _parse_one(self, image, filter_imgtags: bool = True) -> str:
        vllm_inputs = [{
            "prompt": self.prompt,
            "multi_modal_data": {"image": image},
            "mm_processor_kwargs": {"images_kwargs": {
                "min_pixels": 448 * 448, "max_pixels": 2880 * 2880}},
        }]
        outputs = self.model.generate(vllm_inputs, self.sampling_params)
        text = outputs[0].outputs[0].text.strip()
        if filter_imgtags:
            text = "\n\n".join(
                b for b in text.split("\n\n")
                if not b.strip().startswith('<img src="images/bbox_'))
        return self._clean_truncated_repeats(text)

    @modal.method()
    def parse_batch(self, items: list) -> list:
        """items: list of (filename, image_bytes). Returns per-image dicts with real latency."""
        out = []
        for filename, data in items:
            img = PILImage.open(io.BytesIO(data)).convert("RGB")
            t = time.time()
            md = self._parse_one(img)
            lat = time.time() - t
            out.append({
                "image": filename,
                "latency": round(lat, 3),
                "chars": len(md),
                "n_tables": md.lower().count("<table"),
                "markdown": md,
            })
            print(f"  parsed {filename}  {lat:.2f}s  tables={md.lower().count('<table')}")
        return out
