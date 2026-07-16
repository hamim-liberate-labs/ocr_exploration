# OCR Model Comparison — PaddleOCR on a single Kaggle T4

Detailed comparison of **5 PaddleOCR models** run over the **same 11 document images**
(`image_001` … `image_011`), each in its own isolated Kaggle session on **one T4 GPU**
(native Paddle engine — no vLLM, since the T4 is compute-capability 7.5 < 8.0).

The 11 images are all **English business/school documents** — meeting agendas, schedules,
board agendas, planning tables. Most contain **tables**; several mix headings, bullet
lists, and small dense text. This makes the set a good stress test of *layout + table
reconstruction*, not just raw text recognition.

| # | Model | Class | Params | What it produces |
|---|-------|-------|--------|------------------|
| 1 | **PP-OCRv6 tiny**   | Plain text OCR (det + rec) | 1.5 M | JSON: text boxes + text |
| 2 | **PP-OCRv6 small**  | Plain text OCR (det + rec) | 7.7 M | JSON: text boxes + text |
| 3 | **PP-OCRv6 medium** | Plain text OCR (det + rec) | 34.5 M | JSON: text boxes + text |
| 4 | **PP-StructureV3**  | Full document-parsing pipeline (7 modules) | pipeline | Markdown + JSON + HTML tables + visualizations |
| 5 | **PaddleOCR-VL 1.6**| Vision-Language document parser | ~0.9 B | Markdown + JSON + HTML tables |

> **Not apples-to-apples.** PP-OCRv6 (tiny/small/medium) only *reads text* — it returns a
> flat list of text lines with no reading order or table structure. PP-StructureV3 and
> PaddleOCR-VL do *document understanding* — layout analysis, reading order, and table
> reconstruction into Markdown/HTML. Compare PP-OCRv6 sizes against each other, and
> PP-StructureV3 against PaddleOCR-VL, then weigh the two groups by your need.

---

## 1. Resource & speed summary (from `benchmark_summary.json`)

| Metric | PP-OCRv6 tiny | PP-OCRv6 small | PP-OCRv6 medium | PP-StructureV3 | PaddleOCR-VL 1.6 |
|---|---|---|---|---|---|
| **Avg latency / image (s)** | **0.279** | 0.513 | 0.763 | 2.288 | 17.956 |
| **Total inference, 11 imgs (s)** | **3.07** | 5.64 | 8.40 | 25.16 | 197.52 |
| **GPU peak (MB)** | **879** | 1 243 | 2 281 | 8 033 | 14 803 |
| **GPU after load (MB)** | **573** | 595 | 711 | 7 541 | 10 505 |
| **Model on disk (MB)** | **6.3** | 30.1 | 132.8 | 3 889 | 1 967 |
| **Output on disk (MB)** | 2.5 | 2.4 | 2.4 | 20.3 | **2.2** |
| **Load / warm-up (s)** | 8.9 | 8.4 | 9.6 | 59.5 | 23.9 |
| **Engine** | paddle | paddle | paddle | paddle | paddle (native, no vLLM) |

**Reading the numbers**

- **Speed spread is ~64×.** Tiny does the whole set in ~3 s (0.28 s/img); PaddleOCR-VL
  takes ~198 s (18 s/img). PP-StructureV3 sits in between at ~2.3 s/img.
- **GPU footprint.** PP-OCRv6 fits comfortably on any small GPU (<2.3 GB peak). PP-StructureV3
  needs ~8 GB; PaddleOCR-VL peaks at ~14.8 GB — nearly the full 16 GB T4.
- **Disk.** PP-OCRv6 tiny is a **6 MB** model; PaddleOCR-VL (~2 GB) and PP-StructureV3
  (~3.9 GB, it downloads 7 sub-models) are heavyweights.
- **Load time.** PP-StructureV3's 59.5 s reflects downloading/initializing 7 modules;
  PaddleOCR-VL's 23.9 s is the VLM weights. These are largely one-time per session.
- **Output size.** PP-StructureV3 writes 20 MB because it saves 6–8 visualization JPGs per
  image (layout, order, region, OCR overlay, table-cell, preprocessed). The others save one
  overlay (OCR models) or none (VL saves only md+json).

### Per-image inference latency (seconds)

| Image | tiny | small | medium | StructureV3 | VL 1.6 |
|-------|-----:|------:|-------:|------------:|-------:|
| image_001 | 0.906 | 1.383 | 1.920 | 12.736 | 17.630 |
| image_002 | 0.249 | 0.455 | 0.670 | 1.401 | 11.356 |
| image_003 | 0.171 | 0.273 | 0.342 | 0.627 | 13.433 |
| image_004 | 0.167 | 0.322 | 0.476 | 0.743 | 12.731 |
| image_005 | 0.247 | 0.639 | 1.023 | 1.625 | 37.298 |
| image_006 | 0.449 | 0.897 | 1.329 | 2.452 | 39.394 |
| image_007 | 0.223 | 0.425 | 0.654 | 1.599 | 16.506 |
| image_008 | 0.159 | 0.287 | 0.444 | 1.113 | 12.761 |
| image_009 | 0.178 | 0.371 | 0.605 | 0.978 | 14.096 |
| image_010 | 0.205 | 0.371 | 0.605 | 1.227 | 17.311 |
| image_011 | 0.114 | 0.217 | 0.328 | 0.662 | 5.002 |

*Notes:* The **first image is always the slowest** (kernel autotune / lazy graph warm-up) —
most visible for StructureV3 (12.7 s vs ~1 s steady) and VL. VL latency **tracks text
density**: the two dense, small-font pages (`image_005`, `image_006`) cost ~37–39 s each
because the VLM decodes far more tokens; the sparse `image_011` is only 5 s. PP-OCRv6 latency
scales gently with size (tiny→medium ≈ 2–3× slower) and with the amount of text on the page.

---

## 2. Text-recognition quality — PP-OCRv6 tiny vs small vs medium

All three return a flat list of text lines. Reported **average recognition confidence** and
**region counts** across four representative images:

| Image | tiny (regions / conf) | small | medium |
|-------|----|----|----|
| image_003 (schedule) | 33 / 0.971 | 33 / **0.990** | 34 / 0.980 |
| image_005 (dense board agenda) | 64 / 0.928 | 68 / 0.921 | 75 / **0.947** |
| image_008 (weekly planner) | 46 / 0.975 | 47 / 0.983 | 47 / **0.990** |
| image_011 (simple table) | 24 / 0.984 | 24 / 0.989 | 24 / **0.995** |

**Qualitative findings** (same lines, compared word-for-word):

- **Clean, simple pages** (`image_011`, `image_003`, `image_008`): all three sizes are
  near-perfect. Differences are tiny — e.g. tiny wrote `Bulding`, `Classrcom`, `2ºgrade`;
  small/medium got `Building`, `Classroom`, `2grade`. Bullet glyphs (`•`) that tiny drops
  or mangles (`·`) are handled better by small/medium.
- **Small/medium fix common tiny errors:** `Arriva/Donuts`→`Arrival/Donuts`,
  `Keys to Classrcom`→`Keys to Classroom`, `5*`/`5h grade`→cleaner. Confidence rises
  monotonically tiny→small→medium on the easy pages.
- **Dense, small-font page** (`image_005`) is where they diverge sharply:
  - **tiny** garbles many lines: `ScHocL.B0ABn MrNnEBs`, `Jay Whesler`, `Distrid 1`,
    `Vioe Chair`, `OPEN FOR PURLIC COMMENT`, `SMARTHINIKING`, drops several phone/address rows.
  - **small** recovers a lot: correct `Jay Wheeler`, `District 1 – Kissimmee`, `Vice Chair`,
    `OPEN FOR PUBLIC COMMENT`, `SMARTHINKING`, and picks up 68 regions — but still misreads
    the tiny grey address block (`Tarmmy`, `Cooe-Otterson`, `Thomias A. Phe(ps`).
  - **medium** finds the **most regions (75)** — it recovers extra address/phone lines the
    others skipped (`1200 Varmest Avera`, `809 Beck ldevaad`, `PHONE: 407-818-2964`) — though
    those micro-font lines are themselves partly wrong. Names like `Wiliam C. Collins`,
    `SMARTHINKING` are correct.

**Verdict for the OCR trio:**

| | Accuracy | Speed | Footprint | Best for |
|---|---|---|---|---|
| **tiny**   | Good on clean text, degrades on small/dense fonts | Fastest (0.28 s) | 6 MB / 0.9 GB GPU | High-throughput, clean printed text, edge/CPU |
| **small**  | Clearly better than tiny; best accuracy/cost balance | 0.51 s | 30 MB / 1.2 GB GPU | **Default choice** for plain-text OCR |
| **medium** | Best raw recall, esp. tiny/faint text; highest confidence on clean pages | 0.76 s | 133 MB / 2.3 GB GPU | Max text recall when speed is secondary |

Going tiny→small is a real accuracy jump for ~2× the time. small→medium is a **smaller**
gain (mostly on the hardest micro-text) for another ~1.5× time — often not worth it unless
you specifically need faint-text recall.

---

## 3. Document parsing quality — PP-StructureV3 vs PaddleOCR-VL 1.6

These two produce **Markdown with reconstructed tables**. Both benefit from an accuracy
reference: PaddleOCR-VL reports **OmniDocBench v1.6 = 96.33** overall; PP-StructureV3 is the
older pipeline approach. Observed on the sample set:

### image_001 — meeting agenda with a clean 3-column table
Both reconstruct the 7-row agenda table well. Differences:
- **PaddleOCR-VL** keeps bullet structure inside cells (`• Opening remarks\n• Brief
  introductions…`) and header text cleanly (`DATE: 24 JANUARY 2026`, `TIME: 8.30 AM - 1:20 PM`).
- **PP-StructureV3** flattens bullets into a run-on (`Opening remarks Brief introductions…`),
  drops the `TIME` line, and makes casing slips (`QaA` for `Q&A`, `CLaudia`). It reordered the
  header block (`DATE:` above `ATTENDEES:`).
- **Both** repeat the same real source typo `Juliana SIlva`.

### image_005 — dense board agenda, small grey address blocks (the hard case)
This is the clearest quality gap.
- **PaddleOCR-VL** produces clean, readable, correctly-ordered Markdown: proper headings,
  correct names/phones (`Jay Wheeler`, `407-462-6598`), well-formed action-item list. A few
  faint lines wobble (`52, Cloud`, `Kissimons`) but the document is faithful and usable.
- **PP-StructureV3** badly garbles the small-font regions:
  `ScHCCLBOAD MEDERS`, `AOMINISTRATINE COMPLEXORRICES`, `ExoeptioralSucdlentBdueetion`,
  `407-348713`, collapsed address blocks with lost spacing
  (`437-870-488*FAC 487-870-4029THESCHOOLDISTRICT…`). Reading order and word spacing break down.

### image_006 — very large, wide planning table (~30 rows × 7 cols)
- **PaddleOCR-VL** reconstructs the table cleanly with correct column alignment, `colspan`
  section rows (`Sweetwater County School District…`), and correctly-placed dates/times/locations.
- **PP-StructureV3** misaligns the header columns — it splits the title across rows and
  **shifts fields into the wrong columns** (e.g. Location/Purpose/Date-completed land under the
  wrong headers), and introduces spelling slips (`Communiy`, `StaffForum`). The data is mostly
  present but the table structure is less trustworthy.

### Speed / cost of the two parsers
- **PP-StructureV3** is **~8× faster** (2.3 s vs 18 s per image) and uses **half the GPU**
  (8 GB vs 14.8 GB). It also emits rich visualizations (layout, reading-order, region, table-cell
  overlays) useful for debugging — at the cost of 20 MB output and a 3.9 GB model download.
- **PaddleOCR-VL** is slow and memory-hungry on a T4, and latency balloons on dense pages,
  but its **output quality — text fidelity, reading order, and table structure — is clearly
  higher**, especially on small fonts and complex layouts.

**Verdict for the parsers:**

| | Output quality | Speed | GPU | Best for |
|---|---|---|---|---|
| **PP-StructureV3** | Good on clean layouts; degrades on dense/small text and wide tables; occasional column misalignment | **2.3 s/img** | 8 GB | High-volume doc parsing where layout is fairly clean; when you want layout/table visualizations |
| **PaddleOCR-VL 1.6** | **Best** — faithful text, correct reading order, robust tables even on hard pages | 18 s/img (up to ~39 s on dense pages) | ~14.8 GB (near T4 limit) | Highest-accuracy document understanding when latency/GPU budget allows |

---

## 4. Overall recommendation

```
Need                                   →  Pick
─────────────────────────────────────────────────────────────
Plain text, max speed / tiny GPU / CPU →  PP-OCRv6 tiny
Plain text, best balance (default)     →  PP-OCRv6 small
Plain text, max recall on faint text   →  PP-OCRv6 medium
Structured docs, high volume, clean    →  PP-StructureV3
Structured docs, best accuracy         →  PaddleOCR-VL 1.6
```

- If you only need **text strings** (search, indexing, keyword extraction) → use **PP-OCRv6**,
  and prefer **small** as the default; drop to **tiny** for throughput/edge, step up to
  **medium** only if you're missing faint text.
- If you need **structure** (tables → HTML/Markdown, reading order, formulas) → choose between
  **PP-StructureV3** (fast, cheaper, good enough on clean layouts) and **PaddleOCR-VL** (slow,
  heavy, but the most accurate on dense and complex documents).
- On a **single T4**, PaddleOCR-VL is usable but near the memory ceiling and slow on dense
  pages — fine for batch/offline, not for interactive/real-time. PP-StructureV3 is the
  pragmatic structured-parsing choice on this hardware.

### One-line takeaways
- **Fastest:** PP-OCRv6 tiny (0.28 s/img, 6 MB, 0.9 GB GPU).
- **Best value plain OCR:** PP-OCRv6 small.
- **Best structured parsing on a T4:** PP-StructureV3 (speed) or PaddleOCR-VL (accuracy).
- **Highest accuracy overall:** PaddleOCR-VL 1.6 — at ~64× the latency and ~17× the GPU of tiny.

---

*Source: `benchmark_summary.json`, `*_res.json`, and `*.md` outputs under
`output/{pp_ocrv6_tiny,pp_ocrv6_small,pp_ocrv6_medium,pp_structurev3_results,paddleocr_vl_1.6}_results/`.
All models run natively on Paddle on one Kaggle T4 GPU (CC 7.5), 11 images.*
