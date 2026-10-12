"""Ground-truth probe for a scanned PDF page.

    python probe_pdf.py <pdf> [page_index] [out_dir]

Answers three questions at once:
  A. are the rules raster or vector?      -> get_drawings() / image bboxes
  B. is the scan rotated (skewed)?        -> row profiles of two x-strips, so a
                                             tilted rule shows up as a dip that
                                             is offset between them
  C. what do the pixels actually look like? -> 1:1 crops written to out_dir

Row profiles print mean and min luminance per row, to 4 decimals; a rule is a
dip of a few thousandths, which is why it survives every coarser test.
"""
from __future__ import annotations

import io
import os
import sys

import numpy as np
import pymupdf
from PIL import Image


def lum(a: np.ndarray) -> np.ndarray:
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def probe(pdf_path: str, page_no: int = 0, out_dir: str | None = None, y0: int = 300,
          y1: int = 900) -> None:
    doc = pymupdf.open(pdf_path)
    page = doc[page_no]
    print(f"=== {pdf_path}  page {page_no + 1}/{doc.page_count} ===")
    print(f"rect {page.rect.width:.1f}x{page.rect.height:.1f}  rotation {page.rotation}")

    imgs = page.get_images(full=True)
    print(f"images on page: {len(imgs)}")
    for im in imgs:
        d = doc.extract_image(im[0])
        print(f"   xref={im[0]} {d['width']}x{d['height']} ext={d['ext']}")
    for info in page.get_image_info():
        print(f"   placed bbox {[round(v, 1) for v in info['bbox']]}")

    draws = page.get_drawings()
    print(f"vector drawings: {len(draws)}")
    for item in draws[:8]:
        print(f"   {item['type']} rect={[round(v, 1) for v in item['rect']]} "
              f"width={item.get('width')} items={len(item['items'])}")

    txt = page.get_text("blocks")
    print(f"text blocks: {len(txt)}")

    if not imgs:
        return
    im = Image.open(io.BytesIO(doc.extract_image(imgs[0][0])["image"])).convert("RGB")
    a = np.asarray(im).astype(np.float32) / 255.0
    L = lum(a)
    h, w = L.shape
    print(f"\nimage {w}x{h}  paper(L95)={np.percentile(L, 95):.4f}  "
          f"darkest={L.min():.4f}")

    # ---- B. skew: compare where the dips fall in a left and a right strip ----
    xl = slice(int(w * 0.05), int(w * 0.30))
    xr = slice(int(w * 0.75), int(w * 1.00))
    prof_l = L[:, xl].mean(axis=1)
    prof_r = L[:, xr].mean(axis=1)
    print(f"\nrow profiles, y={y0}..{y1}  (left strip {xl.start}-{xl.stop}, "
          f"right strip {xr.start}-{xr.stop})")
    print("      y   meanL_all      minL   meanL_left  meanL_right")
    for y in range(y0, min(y1, h)):
        print(f" {y:6d} {L[y].mean():11.4f} {L[y].min():9.4f} "
              f"{prof_l[y]:12.4f} {prof_r[y]:12.4f}")

    # best-effort skew estimate: shift that best aligns the two strips' dips
    k = 400
    best, bestlag = -1.0, 0
    for lag in range(-k, k + 1, 2):
        if 0 <= y0 + lag and y1 + lag < h:
            seg_a = prof_l[y0:y1] - prof_l[y0:y1].mean()
            seg_b = prof_r[y0 + lag:y1 + lag] - prof_r[y0 + lag:y1 + lag].mean()
            c = float((seg_a * seg_b).sum())
            if c > best:
                best, bestlag = c, lag
    span = (xr.start + xr.stop) / 2 - (xl.start + xl.stop) / 2
    print(f"\nbest vertical shift between strips: {bestlag} px over {span:.0f} px "
          f"-> skew ~= {np.degrees(np.arctan2(bestlag, span)):.3f} deg")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        for name, box in (("crop_left", (100, 350, 900, 800)),
                          ("crop_bottom", (100, 2650, 900, 3100))):
            c = im.crop(box)
            c.save(os.path.join(out_dir, f"{name}.png"))
            c.resize((c.width * 2, c.height * 2), Image.NEAREST).save(
                os.path.join(out_dir, f"{name}_2x.png"))
        print(f"wrote crops to {out_dir}")


if __name__ == "__main__":
    probe(sys.argv[1],
          int(sys.argv[2]) if len(sys.argv) > 2 else 0,
          sys.argv[3] if len(sys.argv) > 3 else None)
