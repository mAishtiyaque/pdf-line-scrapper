"""Numeric verdict on a clean-up run: what went, and was it a loss?

    python tools/verify.py before.png after.png [--slope -0.0164] [--near 2]

A ruled line and a pen stroke in this scan are both dark, so "how dark was it"
cannot tell them apart - only shape can.  So every ink pixel is tagged with the
length of the horizontal and vertical run it belongs to, measured in the
*deskewed* frame (a tilted rule has no long horizontal run at all, which is why
this has to be measured after straightening).

  rule-shaped   long horizontal run, far longer than it is thick
  ink-shaped    short, or about as thick as it is long

Five numbers carry the verdict.  `LOSS` (should be 0) counts pixels the cleaner
removed without licence.  `rule bands ... still visible after` (should be 0) asks
the same question the other way round, as contrast - it exists because every count
in this file once reported success while a crop showed a ruled line still standing
(radius 0.35: `rule-shaped after 55,302` and `rows >60% after 0`, both reading
success while a 1:1 crop of the box showed the line there).  `background` answers
instruction 2.
`rows >60% ink across the page` is kept for continuity and is the weakest of them:
it needs a rule to cover 60% of the *full* width, so a faded rule can be visible
while it reads 0.

A removed dark pixel is only counted as LOSS when *both* of these hold:

  * it did not sit on a row the cleaner was *licensed* to touch.  That licence is
    a length: the opening keeps a structure only if it is at least 2*radius + 1
    long, so a row carrying such a run is a row the cleaner may clear, and rows
    within `--row-pad` px of one go with it - the `vpad` dilation merges a rule
    with its halo before the opening, so fringe rows leave with the core.  This
    is a per-*row* test, not a per-pixel one, because a rule is thin enough that
    its fringe rows contain it only over part of the width: a per-pixel test
    calls those fringes "ink" and reports the rule's own removal as damage.  An
    earlier version of this tool did exactly that and produced 28k pixels of
    false alarm - and a later one scored the licence against half the page width
    instead of against the cleaner's own radius, which called the removal of
    every short rule segment a loss.
  * no dark ink survived within `--near` px of it, so it is not the anti-aliased
    edge or halo of a stroke that is still there, one pixel thinner.  Thinning an
    edge by a pixel is invisible and is not a loss - but a stroke that vanished
    outright leaves removed pixels with nothing dark beside them, and every one
    of those is reported with a crop box.

LOSS is therefore a count of pixels the cleaner removed *without licence*, which
is a conservative thing to count: a rule band the licence covers is not scored at
all.  The two questions a count cannot answer are asked separately, as contrast:

  rule bands   each rule detected in the before page, re-measured after: how much
               darker than paper is its darkest pixel still?  A rule can survive
               as fragments (thousands of ink pixels, a broken line) or vanish
               into a grey haze (thousands of ink pixels, invisible) - only
               contrast says which, and a pixel count says neither.
  background   how much of the page's non-ink area is still below paper, i.e. is
               instruction 2 ("clean plain white") actually satisfied?


Because `LOSS` is still a 2-condition test, it comes with its own geometry dump:
how far each LOSS pixel is, in the deskewed frame, from the nearest row that
really carried a rule.  Small distances mean rule halo and a pad still too tight;
large ones mean a separate structure - a hand-drawn line, which is content.
"""
from __future__ import annotations

import sys

import numpy as np
from PIL import Image

from lineclean import (Config, _groups, dilate, estimate_skew, lum, shear,
                       shear_offsets, unshear)


def runlen_along(m: np.ndarray, axis: int) -> np.ndarray:
    """Length of the run of True covering each True pixel, along one axis."""
    if axis == 1:
        return runlen_along(np.ascontiguousarray(m.T), 0).T
    h, w = m.shape
    fwd = np.zeros((h, w), np.int32)
    cur = np.zeros(w, np.int32)
    for y in range(h):
        cur = np.where(m[y], cur + 1, 0)
        fwd[y] = cur
    bwd = np.zeros((h, w), np.int32)
    cur = np.zeros(w, np.int32)
    for y in range(h - 1, -1, -1):
        cur = np.where(m[y], cur + 1, 0)
        bwd[y] = cur
    return np.where(m, fwd + bwd - 1, 0)


def rows_with_coverage(ink: np.ndarray, off: np.ndarray, frac: float) -> list[tuple[int, float]]:
    """Deskewed rows that are `frac` ink across the page - i.e. the rules.

    A rule fills its whole row; a long pen stroke or an arrow covers a fraction
    of one, so this counts rules specifically and ignores stray long ink.
    """
    s = shear(ink, off)
    cov = s.mean(axis=1)
    out = []
    for y, c in _groups(np.where(cov > frac)[0]):
        yc = (y + c) // 2
        out.append((int(yc), float(cov[yc])))
    return out


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    slope = None
    long_frac, ratio, cov_frac, near = 0.25, 40.0, 0.60, 2
    rule_frac, rule_min_radius, row_pad = 0.15, 40, 20
    cfg = Config()
    pad = cfg.vpad
    for i, a in enumerate(sys.argv):
        if a == "--slope" and i + 1 < len(sys.argv):
            slope = float(sys.argv[i + 1])
        elif a == "--long" and i + 1 < len(sys.argv):
            long_frac = float(sys.argv[i + 1])
        elif a == "--ratio" and i + 1 < len(sys.argv):
            ratio = float(sys.argv[i + 1])
        elif a == "--coverage" and i + 1 < len(sys.argv):
            cov_frac = float(sys.argv[i + 1])
        elif a == "--dilate" and i + 1 < len(sys.argv):
            pad = int(sys.argv[i + 1])
        elif a == "--near" and i + 1 < len(sys.argv):
            near = int(sys.argv[i + 1])
        elif a == "--rule-frac" and i + 1 < len(sys.argv):
            rule_frac = float(sys.argv[i + 1])
        elif a == "--rule-min-radius" and i + 1 < len(sys.argv):
            rule_min_radius = int(sys.argv[i + 1])
        elif a == "--row-pad" and i + 1 < len(sys.argv):
            row_pad = int(sys.argv[i + 1])
    if len(args) < 2:
        print(__doc__)
        return

    Lb = lum(np.asarray(Image.open(args[0]).convert("RGB"), dtype=np.float32) / 255.0)
    La = lum(np.asarray(Image.open(args[1]).convert("RGB"), dtype=np.float32) / 255.0)
    if Lb.shape != La.shape:
        print(f"shape mismatch {Lb.shape} vs {La.shape}")
        return
    h, w = Lb.shape
    pb, pa = float(np.percentile(Lb, 95)), float(np.percentile(La, 95))

    ink_b = Lb < (pb - cfg.ink_delta)
    ink_a = La < (pa - cfg.ink_delta)
    if slope is None:
        slope = estimate_skew(ink_b & ~(Lb < pb - cfg.medium_delta), cfg)
    off = shear_offsets(w, slope)
    rad = max(rule_min_radius, int(round(rule_frac * w)))
    run_need = 2 * rad + 1
    print(f"paper before={pb:.4f} after={pa:.4f}   "
          f"skew={np.degrees(np.arctan(slope)):.3f} deg   long>={long_frac * w:.0f}px "
          f"ratio>={ratio:.0f}x   vdilate={pad}   near={near}px")
    print(f"removal licence: a row carrying a run >= {run_need}px "
          f"(radius {rad} = {rule_frac:.2f} of {w}px, floor {rule_min_radius}) "
          f"+/- {row_pad}px")

    def shape(ink: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # the same vertical dilation the cleaner applies before it looks for long
        # runs.  Without it a 3px rule that straddles two pixel rows fragments
        # into ~300px pieces within one row, and gets misread as ink, not a rule.
        s = dilate(shear(ink, off), pad, 0)
        return unshear(runlen_along(s, 1), off), unshear(runlen_along(s, 0), off)

    hb, vb = shape(ink_b)
    ha, va = shape(ink_a)
    rule_b = (hb >= long_frac * w) & (hb >= ratio * vb)
    rule_a = (ha >= long_frac * w) & (ha >= ratio * va)

    print(f"\nink pixels      before {int(ink_b.sum()):9d}   after {int(ink_a.sum()):9d}")
    print(f"rule-shaped     before {int(rule_b.sum()):9d}   after {int(rule_a.sum()):9d}"
          f"   ({100.0 * rule_a.sum() / max(1, rule_b.sum()):.1f}% of the original)"
          f"   <- the rules; this is the number that must fall")

    rb = rows_with_coverage(ink_b, off, cov_frac)
    ra = rows_with_coverage(ink_a, off, cov_frac)
    print(f"\nrows >{cov_frac:.0%} ink across the page (rules, measured after deskew):"
          f"\n  before {len(rb)}   after {len(ra)}   <- the headline: must fall to ~0")
    if len(rb) > 2:
        sp = np.diff([y for y, _ in rb])
        print(f"  spacing before: median {np.median(sp):.0f}px over {len(rb)} rules")
    if ra:
        print("  survivors (deskewed y:coverage) "
              + ", ".join(f"{y}:{c:.2f}" for y, c in ra[:14]))

    # ---- what was removed, and was any of it a loss? ----------------------- #
    gone = ink_b & (La > pa - 0.004)
    dark_b = Lb < (pb - cfg.medium_delta)
    dark_a = La < (pa - cfg.medium_delta)
    gone_dark = gone & dark_b

    # Rows the cleaner was licensed to clear - those carrying a run at least as
    # long as its opening survives - widened by row_pad, because the removal band
    # is far taller than the rule's dark core: a rule's fringe rows, where it
    # covers only part of the width, leave with the core.
    s_b = dilate(shear(ink_b, off), pad, 0)
    rowmax = runlen_along(s_b, 1).max(axis=1)          # longest horizontal run per row
    licensed = rowmax >= run_need
    rule_row = dilate(licensed[:, None], row_pad, 0)[:, 0]
    ctx = unshear(np.repeat(rule_row[:, None], w, axis=1), off)

    survive = dilate(dilate(dark_b & dark_a, near, 0), near, 1)
    on_rule = gone_dark & ctx
    off_rule = gone_dark & ~ctx
    thin = off_rule & survive
    loss = off_rule & ~survive

    print(f"\nremoved total   {int(gone.sum()):9d} px  "
          f"(dark {int(gone_dark.sum())}, faint {int((gone & ~dark_b).sum())})")
    print("  dark, on a rule row (the rules leaving)     "
          f"{int(on_rule.sum()):9d}")
    print(f"  dark, off-rule, dark ink within {near}px      {int(thin.sum()):9d}"
          f"   <- a stroke's edge/halo, 1px thinner: not a loss")
    print("  dark, off-rule, nothing survives            "
          f"{int(loss.sum()):9d}   <- LOSS")

    if not loss.any():
        print(f"\nno LOSS: every dark pixel removed was either on a rule row or the "
              f"edge of ink still present within {near}px.")
    else:
        rr = np.where(loss.any(axis=1))[0]
        runs = sorted(_groups(rr), key=lambda g: -int(loss[g[0]:g[1] + 1].sum()))
        print(f"\nLOSS: {int(loss.sum())} px of dark ink removed with nothing dark "
              f"within {near}px, in {len(runs)} row-runs.  Largest first "
              f"(crop --box x0,y0,x1,y1):")
        for r0, r1 in runs[:12]:
            sub = loss[r0:r1 + 1]
            cols = np.where(sub.any(axis=0))[0]
            x0, x1 = int(cols[0]), int(cols[-1])
            print(f"    y {r0:4d}-{r1:4d}  x {x0:4d}-{x1:4d}  {int(sub.sum()):6d} px   "
                  f"--box {max(0, x0 - 40)},{max(0, r0 - 40)},{x1 + 41},{r1 + 41}")
        if len(runs) > 12:
            print(f"    ... and {len(runs) - 12} smaller runs")
        db = Lb[loss]
        print(f"    darkness: min {db.min():.3f}  mean {db.mean():.3f}   "
              f"(rules measure 0.15-0.50 here, so this tells you whether the "
              f"removed material is rule-dark or pen-dark)")

        # where is this LOSS in the deskewed frame?  near a licensed row means it
        # is that rule's own halo and either the pad or the licence is still too
        # tight; far from one means the cleaner removed something it had no
        # licence for, i.e. a structure of its own - content.
        long_idx = np.where(licensed)[0]
        ys_l, xs_l = np.where(loss)
        d = np.clip(ys_l - off[xs_l], 0, h - 1)
        rm = rowmax[d]
        print(f"    deskewed rows carrying a >= {run_need}px run: {len(long_idx)}")
        print(f"    LOSS row's own longest run: median {np.median(rm):.0f}px  "
              f"p10 {np.percentile(rm, 10):.0f}px  (licence {run_need}px)")
        if len(long_idx):
            lb = long_idx.astype(np.int32)
            dist = np.abs(lb[None, :] - d[:, None].astype(np.int32)).min(axis=1)
            print(f"    LOSS distance to nearest such row: median {np.median(dist):.0f}px  "
                  f"p90 {np.percentile(dist, 90):.0f}px  max {int(dist.max())}px")
            print("    small distance = that rule's halo, raise --row-pad; large "
                  "distance = a separate structure: content")

    # ---- did the rules actually go?  contrast, not pixels ------------------ #
    # Neither a pixel count nor a coverage count can answer this.  A rule that
    # survives as a thinner, broken line, and one removed down to a haze below
    # the ink threshold, both leave thousands of "rule-shaped" pixels: 55,302 of
    # them at radius 0.35 on this page, where a crop shows the line plainly still
    # there.  What a reader sees is contrast, so each rule band found in the
    # *before* page gets one number - how much darker than paper its darkest
    # pixel still is, in that band's own columns.  0.06 is the visibility cut.
    sr_b, sr_a = shear(Lb, off), shear(La, off)
    sr_med = sr_b < (pb - cfg.medium_delta)
    bands: list[tuple[int, int, float, float, int, int]] = []
    n_thick = 0
    for r0, r1 in _groups(np.where(licensed)[0]):
        lo_r, hi_r = max(0, r0 - 3), min(h - 1, r1 + 3)
        sub_b, sub_a = sr_b[lo_r:hi_r + 1], sr_a[lo_r:hi_r + 1]
        med = sr_med[lo_r:hi_r + 1]
        cols = np.where(med.any(axis=0))[0]
        if len(cols) < 60:            # a flicker, not a rule band
            continue
        if float(np.median(med[:, cols].sum(axis=0))) > 6:
            n_thick += 1              # ink-like: thick enough that the cleaner's
            continue                  # vdeep guard shields it by design
        cb = float(np.median(pb - sub_b[:, cols].min(axis=0)))
        ca = float(np.median(pa - sub_a[:, cols].min(axis=0)))
        bands.append((int(r0), int(r1), cb, ca, int(cols[0]), int(cols[-1])))

    if bands:
        res = [b[3] for b in bands]
        still = sorted([b for b in bands if b[3] > 0.06], key=lambda b: -b[3])
        print(f"\nrule bands under the licence: {len(bands)}   "
              f"still visible after (contrast > 0.06): {len(still)}   <- must be 0"
              + (f"   [+{n_thick} bands skipped: too thick to be a rule]" if n_thick else ""))
        print(f"  contrast before: median {np.median([b[2] for b in bands]):.3f}   "
              f"after: median {np.median(res):.3f}  max {max(res):.3f}   "
              f"(0 = plain paper)")
        for r0, r1, cb, ca, c0, c1 in still[:8]:
            print(f"    rows {r0}-{r1}  x {c0}-{c1}  contrast {cb:.2f} -> {ca:.2f}   "
                  f"--box {max(0, c0 - 40)},{max(0, r0 - 60)},{c1 + 41},{min(h, r1 + 61)}")

    # ---- is the paper plain white?  (instruction 2) ------------------------- #
    # Background is everything that is neither ink nor a preserved vertical margin
    # line (instruction 5 keeps those).  The definition is built from the before
    # page and applied to both, so the cleaner's own success cannot shrink the
    # area it is judged on.
    vrun = runlen_along(dark_b, 0) >= 0.35 * h
    protect = (dilate(dark_b, 3, 0) | dilate(dark_b, 3, 1)
               | dilate(vrun, 8, 0) | dilate(vrun, 8, 1))
    bg = ~protect
    if bg.any():
        print(f"\nbackground (not ink, not a vertical margin line): "
              f"{100.0 * bg.mean():.1f}% of the page")
        for tag, L, p in (("before", Lb, pb), ("after ", La, pa)):
            n = int((L[bg] < p - 0.03).sum())
            print(f"  {tag}: {n:8d} px more than 0.03 below paper "
                  f"({100.0 * (L[bg] < p - 0.03).mean():.2f}% of the background), "
                  f"mean darkness {float((p - L[bg]).mean()):.3f}")
        print("  clean white wants the after row near zero and its darkness at "
              "~0.00-0.01; a two-thirds drop is the background being whitened, "
              "not cleaned")

    dark_after = int(dark_a.sum())
    print(f"\ndark ink (L<{pb - cfg.medium_delta:.2f})   before {int(dark_b.sum()):9d}   "
          f"after {dark_after:9d}")
    print("  the rules are dark too, so a large drop here is the rules leaving; "
          "the number that matters is LOSS above.")


if __name__ == "__main__":
    main()
