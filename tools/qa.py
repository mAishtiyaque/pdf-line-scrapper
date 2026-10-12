"""QA: verify that cleaning removed the rules *without* damaging the handwriting.

    python qa.py original.pdf cleaned.pdf [--pages 1,10,40] [--out-dir qa]

For each inspected page it reports, in original-scan pixels:

  ink_kept     - dark pen pixels that stayed dark          (want: nearly all)
  ink_lighten  - mean luminance rise over pen pixels       (want: ~0)
  ink_lost     - pen pixels that ended up near-white       (want: 0)
  rule_cleared - rule-coloured pixels removed              (want: high)
  rule_left    - rule-coloured pixels still present        (want: ~0)

and writes <page>_compare.png (original | cleaned, same scale) for eyeballing.
"""
from __future__ import annotations

import argparse
import io
import os

import numpy as np
from PIL import Image

import fitz

from lineclean import lum

TH_INK = 0.60      # "pen" darkness on a flattened page
TH_PAPER = 0.92    # "paper" on a flattened page
TH_LEFT = 0.80     # still this bright after cleaning == lost


def flatten(rgb: np.ndarray) -> np.ndarray:
    """Same white-balance as the cleaner, so thresholds mean the same thing."""
    a = rgb.astype(np.float32) / 255.0
    for c in range(3):
        lvl = float(np.percentile(a[::4, ::4, c], 92.0))
        a[..., c] = np.clip(a[..., c] / max(lvl, 1e-6), 0, 1)
    return lum(a)


def get_page_array(doc, idx: int) -> np.ndarray:
    page = doc[idx]
    imgs = page.get_images(full=True)
    if len(imgs) == 1:
        return np.asarray(Image.open(io.BytesIO(doc.extract_image(imgs[0][0])["image"]))
                          .convert("RGB"), dtype=np.uint8)
    z = max(1.5, min(4.0, 2400.0 / max(page.rect.width, 1.0)))
    pm = page.get_pixmap(matrix=fitz.Matrix(z, z), alpha=False)
    return np.asarray(Image.open(io.BytesIO(pm.tobytes("png"))).convert("RGB"), dtype=np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("original")
    ap.add_argument("cleaned")
    ap.add_argument("--pages", default="1")
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    a_doc, b_doc = fitz.open(args.original), fitz.open(args.cleaned)
    pages = []
    for part in args.pages.split(","):
        if "-" in part:
            x, y = part.split("-")
            pages.extend(range(int(x), int(y) + 1))
        else:
            pages.append(int(part))
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    print(f"{'page':>5} {'ink_kept':>9} {'ink_lighten':>12} {'ink_lost':>9} "
          f"{'rule_cleared':>13} {'rule_left':>10}")
    for p in pages:
        orig = get_page_array(a_doc, p - 1)
        new = get_page_array(b_doc, p - 1)
        if orig.shape != new.shape:
            print(f"{p:>5}  size mismatch {orig.shape} vs {new.shape}")
            continue

        go, gn = flatten(orig), flatten(new)
        rgb_o = orig.astype(np.float32) / 255.0
        reddish = ((rgb_o[..., 0] - rgb_o[..., 2]) > 0.10) & ((rgb_o[..., 0] - rgb_o[..., 1]) > 0.05)

        pen = (go < TH_INK) & ~reddish
        rule = (go >= TH_INK) & (go < 0.90) & ~reddish

        n_pen = int(pen.sum())
        kept = int((pen & (gn < TH_INK)).sum())
        lighten = float(np.mean(gn[pen] - go[pen])) if n_pen else 0.0
        lost = int((pen & (gn > TH_LEFT)).sum())
        altered = int((pen & (np.abs(gn - go) > 0.15)).sum())

        n_rule = int(rule.sum())
        cleared = int((rule & (gn > TH_PAPER)).sum())
        left = int((rule & (gn < 0.90)).sum())

        print(f"{p:>5} {kept:>9} {lighten:>12.4f} {lost:>9} "
              f"{cleared:>13} {left:>10}   (orig pen px {n_pen}, rule px {n_rule})"
              f"{'' if not altered else f', ALTERED {altered}'}")

        if args.out_dir:
            h, w = orig.shape[:2]
            s = 1100.0 / w
            small = (int(w * s), int(h * s))
            left_im = Image.fromarray(orig).resize(small, Image.LANCZOS)
            right_im = Image.fromarray(new).resize(small, Image.LANCZOS)
            sheet = Image.new("RGB", (small[0] * 2 + 12, small[1]), (255, 0, 0))
            sheet.paste(left_im, (0, 0))
            sheet.paste(right_im, (small[0] + 12, 0))
            sheet.save(os.path.join(args.out_dir, f"p{p:03d}_compare.png"))


if __name__ == "__main__":
    main()
