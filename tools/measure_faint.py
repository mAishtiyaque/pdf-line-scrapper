"""What is left in the background, and how dark is it?

    python tools/measure_faint.py IMAGE [IMAGE ...] [--rows A B] [--ink 0.35]

Answers one question, for choosing `--bg-white`: everything that is not solid
ink - show-through, blush, JPEG ringing, a pale highlighter - lives in a band of
darkness, and the whitening cut has to sit *above* that band's darkest member
while staying *below* the lightest real stroke.  So measure the band instead of
guessing at it.

Prints, for the non-ink pixels only:
  * the darkness percentiles (paper - luminance), so the band's top edge is a
    number rather than an impression;
  * the mean RGB of the darkest 10% of them, which distinguishes a neutral grey
    haze (R~G~B) from a coloured one, e.g. the reverse page's purple ink
    (R and B above G) - the two want different cuts;
  * what the current cut would do to them.

`--rows A B` restricts to a horizontal band, for when a crop pointed at the
material in question and you want its numbers rather than the page's.
"""
from __future__ import annotations

import sys

import numpy as np
from PIL import Image

from lineclean import dilate, lum


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    rows: tuple[int, int] | None = None
    ink_delta = 0.35
    for i, a in enumerate(sys.argv):
        if a == "--rows" and i + 2 < len(sys.argv):
            rows = (int(sys.argv[i + 1]), int(sys.argv[i + 2]))
        elif a == "--ink" and i + 1 < len(sys.argv):
            ink_delta = float(sys.argv[i + 1])
    if not args:
        print(__doc__)
        return

    for path in args:
        rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        L = lum(rgb)
        paper = float(np.percentile(L, 95))
        h, w = L.shape
        if rows:
            lo_r, hi_r = max(0, rows[0]), min(h, rows[1])
        else:
            lo_r, hi_r = 0, h
        sub_rgb, sub_L = rgb[lo_r:hi_r], L[lo_r:hi_r]

        ink = L < (paper - ink_delta)
        # every pixel within `halo` px of solid ink belongs to a stroke, not to
        # the background, and is excluded - otherwise a stroke's own anti-aliased
        # skirt swamps the faint population and inflates its darkness.
        halo = dilate(dilate(ink, 6, 0), 6, 1)
        faint = (L < paper - 0.02) & ~halo
        if rows:
            faint = faint[lo_r:hi_r]

        print(f"\n{path}"
              + (f"  rows {lo_r}-{hi_r}" if rows else "  (whole page)")
              + f"\n  paper level {paper:.4f}   solid ink (L < {paper - ink_delta:.2f}): "
              f"{100.0 * ink[lo_r:hi_r].mean():.2f}% of the region")
        if not faint.any():
            print("  no faint material at all - nothing to whiten")
            continue

        d = paper - sub_L[faint]
        qs = [1, 10, 25, 50, 75, 90, 99]
        print(f"  faint pixels {int(faint.sum()):8d} "
              f"({100.0 * faint.mean():.2f}% of the region)")
        print("  darkness below paper (0 = paper, 1 = black):")
        print("    " + "  ".join(f"p{q}={np.percentile(d, q):.3f}" for q in qs))
        print(f"    min {d.min():.3f}  max {d.max():.3f}  mean {d.mean():.3f}")

        # Decile by darkness, with the colour of each decile.  A neutral haze
        # (JPEG ringing, scanner blush) sits at R~G~B; the reverse page's ink
        # shows its own hue, so where the colour appears in this table *is* the
        # depth the show-through occupies - and that is what the cut has to
        # clear, without reaching down into the deciles where real pale strokes
        # would live.
        print("  darkness deciles (10% of the faint pixels each), mean RGB:")
        edges = np.percentile(d, np.arange(0, 101, 10))
        for k in range(10):
            sel = (d >= edges[k]) & (d <= edges[k + 1] if k == 9 else d < edges[k + 1])
            if not sel.any():
                continue
            m = sub_rgb[faint][sel].mean(axis=0)
            rs = int(round(100 * abs(float(m[0] - m[2]))))
            print(f"    {edges[k]:.3f}-{edges[k + 1]:.3f}  {int(sel.sum()):8d} px  "
                  f"RGB {m[0]:.3f} {m[1]:.3f} {m[2]:.3f}  "
                  f"{'coloured (R-B ' + str(rs) + '%)' if max(abs(float(m[0] - m[1])), abs(float(m[2] - m[1]))) > 0.02 else 'neutral'}")

        # The purple is the thing being chased, so count it by colour directly:
        # a pixel whose channels disagree is ink with a hue, not grey paper or
        # grey JPEG haze.  Reports both populations for every candidate cut, so
        # the choice is "how much purple goes" against "how much else goes".
        chroma = sub_rgb.max(axis=2) - sub_rgb.min(axis=2)
        col = faint & (chroma > 0.03)
        print(f"  coloured faint (chroma > 0.03): {int(col.sum()):8d} px"
              + (f"  ({100.0 * col.sum() / max(1, faint.sum()):.0f}% of the faint "
                 f"population)" if col.any() else ""))
        if col.any():
            m = sub_rgb[col].mean(axis=0)
            hue = ("purple/violet - R and B above G"
                   if m[0] > m[1] + 0.02 and m[2] > m[1] + 0.02 else
                   "warm - R above G and B" if m[0] > m[2] + 0.02 else
                   "cool - B above R and G" if m[2] > m[0] + 0.02 else "mixed")
            dc = paper - sub_L[col]
            print(f"    mean RGB {m[0]:.3f} {m[1]:.3f} {m[2]:.3f}  ({hue})")
            print("    its darkness: " + "  ".join(
                f"p{q}={np.percentile(dc, q):.3f}" for q in (10, 50, 90, 99)))

        print("  candidate cuts (--bg-white): a pixel reaches plain paper when "
              "paper - L <= cut; the ramp white_span wide *below* the cut only "
              "fades it, so the numbers given are 'reaches paper' and the "
              "darkest material still below the cut afterwards")
        col_d = paper - sub_L[col] if col.any() else np.zeros(0)
        for c in (0.18, 0.22, 0.26, 0.30, 0.34, 0.40, 0.50):
            to_paper = d <= c
            surv = d[~to_paper]
            top = (f"darkest left {surv.max():.3f}" if len(surv)
                   else "nothing left - the background is paper")
            cg = int((col_d <= c).sum()) if col.any() else 0
            print(f"    --bg-white {c:.2f}  (cut at {paper - c:.2f}): "
                  f"{100.0 * to_paper.mean():5.1f}% of the faint reaches paper, "
                  f"{cg:5d} of {int(col.sum()):5d} coloured px do"
                  f"   {top}")


if __name__ == "__main__":
    main()
