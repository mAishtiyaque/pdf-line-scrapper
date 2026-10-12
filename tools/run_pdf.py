"""CLI: strip ruled notebook lines from a scanned PDF, page by page.

    python run_pdf.py "in.pdf" -o "out.pdf" [--quality 90] [--pages 1-3]
                       [--debug-dir DIR] [--no-preserve-vlines]

Page order, page size and page geometry are preserved: every output page is
rebuilt with the original MediaBox and the cleaned scan placed exactly over it.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time

import numpy as np
from PIL import Image

import pymupdf as fitz  # PyMuPDF (the bare `fitz` alias is deprecated)

from lineclean import Config, clean_page


def parse_pages(spec: str, n: int) -> list[int]:
    if not spec:
        return list(range(n))
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a) - 1, int(b)))
        else:
            out.append(int(part) - 1)
    return [i for i in dict.fromkeys(out) if 0 <= i < n]


def decode(data: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.uint8)


def encode_jpeg(arr: np.ndarray, quality: int) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(
        buf, format="JPEG", quality=quality, subsampling=0, optimize=True, dpi=(273, 273)
    )
    return buf.getvalue()


def page_image(doc, page) -> tuple[np.ndarray, str, fitz.Rect]:
    """Best available source pixels for one page, plus the rect they occupy.

    The rect matters.  This PDF places 68 of its 78 scans in a *sub-rect* of the
    MediaBox - a narrow strip of content inside an A4 page (300x842pt, not
    595x842) - with the scan's own aspect preserved.  Inserting the cleaned scan
    into the full MediaBox instead would stretch it ~2x horizontally and move
    every stroke, so the original placement rect is what has to be reproduced.
    """
    imgs = page.get_images(full=True)
    rect = page.rect
    if len(imgs) == 1:
        info = doc.extract_image(imgs[0][0])
        if info["width"] >= 600:  # ignore tiny logo/watermark stamps
            rects = page.get_image_rects(imgs[0][0])
            if rects:
                rect = rects[0]
            return (decode(info["image"]), f"embedded {info['width']}x{info['height']}",
                    rect)
    z = max(1.5, min(4.0, 2400.0 / max(page.rect.width, 1.0)))
    pm = page.get_pixmap(matrix=fitz.Matrix(z, z), alpha=False)
    return decode(pm.tobytes("png")), f"render {pm.width}x{pm.height}", rect


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--pages", default="", help="e.g. 1-3,7 (default: all)")
    ap.add_argument("--only", action="store_true",
                    help="write only the selected pages (short preview PDF)")
    ap.add_argument("--debug-dir", default="", help="write before/after PNGs here")
    ap.add_argument("--no-preserve-vlines", action="store_true")
    ap.add_argument("--keep-showthrough", action="store_true",
                    help="keep faint background bands (show-through, blush, ringing)")
    ap.add_argument("--line-radius-frac", type=float, default=Config.line_radius_frac)
    ap.add_argument("--vdeep-radius", type=int, default=Config.vdeep_radius,
                    help="vertical run radius that marks a real pen stroke")
    ap.add_argument("--white-span", type=float, default=Config.white_span)
    ap.add_argument("--bg-white", type=float, default=Config.bg_white,
                    help="whiten everything lighter than paper - this (tone curve)")
    args = ap.parse_args()

    cfg = Config()
    cfg.preserve_vlines = not args.no_preserve_vlines
    cfg.clear_showthrough = not args.keep_showthrough
    cfg.line_radius_frac = args.line_radius_frac
    cfg.vdeep_radius = args.vdeep_radius
    cfg.white_span = args.white_span
    cfg.bg_white = args.bg_white

    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)

    src = fitz.open(args.src)
    n = src.page_count
    todo = parse_pages(args.pages, n)
    print(f"source: {args.src}  pages={n}  processing={len(todo)}", flush=True)

    out = fitz.open()
    stats_all: list[dict] = []
    t0 = time.time()

    for idx in (todo if args.only else range(n)):
        page = src[idx]
        rect = page.rect
        newpage = out.new_page(width=rect.width, height=rect.height)
        stats: dict = {"page": idx + 1}

        if idx not in todo:
            arr, how, place = page_image(src, page)
            newpage.insert_image(place, stream=encode_jpeg(arr, args.quality))
            stats["skipped"] = True
            stats_all.append(stats)
            continue

        t1 = time.time()
        arr, how, place = page_image(src, page)
        before = arr
        cleaned = clean_page(arr, cfg, stats)
        raw = encode_jpeg(cleaned, args.quality)
        newpage.insert_image(place, stream=raw)
        stats["src"] = how
        stats["placement"] = [round(v, 1) for v in place]
        stats["secs"] = round(time.time() - t1, 2)
        stats["kb"] = round(len(raw) / 1024)
        stats_all.append(stats)

        if args.debug_dir:
            Image.fromarray(before).save(os.path.join(args.debug_dir, f"p{idx+1:03d}_before.png"))
            Image.fromarray(cleaned).save(os.path.join(args.debug_dir, f"p{idx+1:03d}_after.png"))

        if (idx + 1) % 5 == 0 or idx == todo[-1]:
            el = time.time() - t0
            print(f"  page {idx+1:>3}/{n}  paper={stats.get('paper_level')} "
                  f"skew={stats.get('skew_deg')}deg  bands={stats.get('bands')}  "
                  f"cleared_dark={stats.get('cleared_dark_px')} "
                  f"cleared_faint={stats.get('cleared_faint_px')}  "
                  f"{stats['secs']}s  [{el:.0f}s elapsed]", flush=True)

    out.set_metadata({k: v for k, v in src.metadata.items() if v})
    out.save(args.out, garbage=4, deflate=True, clean=True)
    size_mb = os.path.getsize(args.out) / 1048576
    print(f"\nwrote {args.out}  ({size_mb:.1f} MB)  {time.time()-t0:.0f}s")

    import json
    with open(os.path.splitext(args.out)[0] + "_stats.json", "w", encoding="utf-8") as fh:
        json.dump(stats_all, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
