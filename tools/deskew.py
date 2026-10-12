"""Deskew a page and report its row structure - rules vs handwriting, by the numbers.

    python tools/deskew.py preview/p001_before.png

Recomputes the skew exactly as the cleaner does, shears the *luminance*, and then
lists every group of consecutive rows that is darker than the paper.  After
deskewing, a ruled line collapses into a clean row run spanning the full width,
so its true darkness and thickness are finally measurable.

Interpretation:
  * runs of thickness ~3-9px with low mean luminance  -> the rules are DARK, and
    any darkness-based ink guard will protect them (they must be found by
    geometry alone)
  * runs with mean luminance ~0.93-0.99               -> the rules are FAINT
  * runs of thickness >= ~10px                        -> pen strokes
"""
from __future__ import annotations

import sys

import numpy as np
from PIL import Image

from lineclean import Config, estimate_skew, lum, shear, shear_offsets


def main() -> None:
    path = sys.argv[1]
    cfg = Config()
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    g = lum(a)
    paper = float(np.percentile(g[::4, ::4], cfg.white_pct))
    faint = g < (paper - cfg.ink_delta)
    medium = g < (paper - cfg.medium_delta)
    slope = estimate_skew(faint & ~medium, cfg)
    print(f"paper={paper:.4f}  skew={np.degrees(np.arctan(slope)):.3f} deg "
          f"({slope:+.5f} px/px)")

    Ls = shear(g, shear_offsets(g.shape[1], slope))
    for tag, m in (("dark  (L<0.78)", Ls < paper - 0.22),
                   ("faint (L>=0.78)", (Ls >= paper - 0.22) & (Ls < paper - 0.004))):
        cov = m.mean(axis=1)
        idx = np.where(cov > 0.5)[0]
        runs = []
        if len(idx):
            start, prev = idx[0], idx[0]
            for y in idx[1:]:
                if y != prev + 1:
                    runs.append((start, prev))
                    start = y
                prev = y
            runs.append((start, prev))
        print(f"\n{tag}: rows with >50% coverage = {len(idx)}  "
              f"grouped into {len(runs)} runs")
        if runs:
            print("     y0    y1  thick   meanL    minL   coverage   spacing")
            prev_c = None
            for r0, r1 in runs[:45]:
                seg = Ls[r0:r1 + 1]
                c = (r0 + r1) / 2
                sp = "" if prev_c is None else f"{c - prev_c:8.1f}"
                print(f"  {int(r0):5d} {int(r1):5d} {int(r1-r0+1):6d} "
                      f"{seg.mean():8.4f} {seg.min():8.4f} {seg.mean(axis=1).mean():9.3f} {sp}")
                prev_c = c


if __name__ == "__main__":
    main()
