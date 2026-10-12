"""List every horizontal band the cleaner detects on one page, with its shape.

    python tools/bands.py preview/p001_before.png

Printed rules are a *family*: same thickness, same horizontal extent, regular
spacing, and - decisively - the same positions on all 78 pages.  A hand-drawn
horizontal line (a diagram bar, a long underline) is a one-off: arbitrary
thickness, arbitrary extent, arbitrary position.

This prints the shape of every detection, plus the median thickness and extent a
family filter would test against, so the family can be identified from numbers
rather than guessed at.  The bands listed are exactly the rows pass 4 clears
(before the vertical-extent guard).

`detect()` is the shared entry point: `tools/families.py` imports it to pool
band positions across every page of the PDF.
"""
from __future__ import annotations

import sys

import numpy as np
from PIL import Image

from lineclean import (Config, dilate, erode, estimate_skew, lum, shear,
                       shear_offsets)


def groups(idx: np.ndarray) -> list[tuple[int, int]]:
    """Group sorted integer indices into runs of consecutive values."""
    out: list[tuple[int, int]] = []
    if not len(idx):
        return out
    start = prev = int(idx[0])
    for v in idx[1:]:
        v = int(v)
        if v != prev + 1:
            out.append((start, prev))
            start = v
        prev = v
    out.append((start, prev))
    return out


def detect(rgb: np.ndarray, cfg: Config | None = None) -> tuple[list[dict], dict]:
    """The bands this page would clear, in the *deskewed* frame.

    Runs exactly the detection half of `lineclean.clean_page` (masks, skew,
    shear, opening), before the vertical-extent guard is applied.  Each row is
    {y, t, x0, x1, ext, dark, dfrac}; `y` is comparable across pages because it
    is measured after the shear has straightened the page.
    """
    cfg = cfg or Config()
    a = np.asarray(rgb, dtype=np.float32) / 255.0
    g = lum(a)
    h, w = g.shape
    paper = float(np.percentile(g[::4, ::4], cfg.white_pct))
    faint = g < (paper - cfg.ink_delta)
    medium = g < (paper - cfg.medium_delta)
    skew_mask = faint & ~medium
    if float(skew_mask.mean()) < 0.002:      # as clean_page: fall back to thin dark
        skew_mask = medium & ~erode(medium, cfg.vdeep_radius, 0)
    slope = estimate_skew(skew_mask, cfg)
    off = shear_offsets(w, slope)
    radius = max(cfg.line_min_radius, int(round(cfg.line_radius_frac * w)))

    thick = dilate(shear(faint, off), cfg.vpad, 0)
    opened = dilate(erode(thick, radius, 1), radius, 1)
    Ls = shear(g, off)

    rows = []
    for r0, r1 in groups(np.where(opened.any(axis=1))[0]):
        cols = np.where(opened[r0:r1 + 1].any(axis=0))[0]
        if not len(cols):
            continue
        seg = Ls[r0:r1 + 1]
        solid = seg[seg < paper - cfg.medium_delta]
        rows.append({
            "y": (r0 + r1) // 2,
            "t": int(r1 - r0 + 1),
            "x0": int(cols[0]), "x1": int(cols[-1]),
            "ext": int(cols[-1] - cols[0] + 1),
            "dark": float(solid.mean()) if solid.size else float("nan"),
            "dfrac": float(solid.size) / seg.size,
        })
    info = {"paper": paper, "slope": slope, "radius": radius, "h": h, "w": w}
    return rows, info


def main() -> None:
    path = sys.argv[1]
    cfg = Config()
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    rows, info = detect(a, cfg)
    h, w = info["h"], info["w"]
    radius = info["radius"]

    print(f"paper={info['paper']:.4f}  skew={np.degrees(np.arctan(info['slope'])):.3f}deg  "
          f"needs a {2 * radius + 1}px run  vdilate={cfg.vpad}")

    if not rows:
        print("no bands detected")
        return

    tm = float(np.median([r["t"] for r in rows]))
    em = float(np.median([r["ext"] for r in rows]))
    print(f"\n{len(rows)} bands   median thickness {tm:.0f}px   median extent {em:.0f}px")
    print(f"{'y':>6} {'thick':>5} {'x0':>5} {'x1':>5} {'extent':>6} "
          f"{'ink L':>6} {'inkfrac':>7}  family test")
    for r in rows:
        bad = []
        if r["t"] > 2.0 * tm or r["t"] < 0.5 * tm:
            bad.append("thickness")
        if abs(r["ext"] - em) > 0.25 * w:
            bad.append("extent")
        print(f"{r['y']:6d} {r['t']:5d} {r['x0']:5d} {r['x1']:5d} {r['ext']:6d} "
              f"{r['dark']:6.3f} {r['dfrac']:7.3f}  "
              + ("OFF-FAMILY: " + ",".join(bad) if bad else "rule-like"))
    sp = np.diff([r["y"] for r in rows])
    if len(sp):
        print(f"\nspacing: median {np.median(sp):.1f}px  min {sp.min():.0f}  "
              f"max {sp.max():.0f}")


if __name__ == "__main__":
    main()
