"""Compare faint content down single columns, before vs after cleaning.

    python tools/runs.py before.png after.png [--x 150,700,1400,2100]

For each probed column it lists every run of non-paper pixels in the BEFORE
image and reports what the AFTER image has at the same rows, so a surviving
rule is named explicitly instead of being inferred from a rendered picture.

It also prints global pixel accounting, which is the check that matters most:
how many faint pixels went to white (rules removed) and how many dark pixels
did (that number must be zero - dark pixels are the handwriting).
"""
from __future__ import annotations

import sys

import numpy as np
from PIL import Image


def lum(a):
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def load(p):
    return lum(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0)


def runs(col, thr, min_len=1):
    out, y = [], 0
    n = len(col)
    while y < n:
        if col[y] < thr:
            y0 = y
            while y < n and col[y] < thr:
                y += 1
            if y - y0 >= min_len:
                out.append((y0, y - 1, y - y0, float(col[y0:y].mean())))
        else:
            y += 1
    return out


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    xs = [150, 700, 1400, 2100]
    for i, a in enumerate(sys.argv):
        if a == "--x" and i + 1 < len(sys.argv):
            xs = [int(v) for v in sys.argv[i + 1].split(",")]
    ymax = 1200
    for i, a in enumerate(sys.argv):
        if a == "--ymax" and i + 1 < len(sys.argv):
            ymax = int(sys.argv[i + 1])

    Lb, La = load(args[0]), load(args[1])
    if Lb.shape != La.shape:
        print(f"shape mismatch {Lb.shape} vs {La.shape}")
        return
    pb, pa = float(np.percentile(Lb, 95)), float(np.percentile(La, 95))
    tb, ta = pb - 0.005, pa - 0.005
    print(f"before: paper={pb:.4f} thr={tb:.4f}    after: paper={pa:.4f} thr={ta:.4f}")

    faint_b = (Lb >= pb - 0.22) & (Lb < tb)
    faint_a = (La >= pa - 0.22) & (La < ta)
    dark_b = Lb < pb - 0.22
    dark_a = La < pa - 0.22
    print("\npixel accounting")
    print(f"  faint (rules+show-through): before {faint_b.sum():9d}  after {faint_a.sum():9d}"
          f"   removed {int((faint_b & ~faint_a).sum()):9d}")
    print(f"  dark  (handwriting):        before {dark_b.sum():9d}  after {dark_a.sum():9d}"
          f"   LOST    {int((dark_b & (La > pa - 0.05)).sum()):9d}")
    print(f"  faint-before that are still faint after: {int((faint_b & faint_a).sum()):9d}"
          f"  ({100.0 * (faint_b & faint_a).sum() / max(1, faint_b.sum()):.1f}% of them)")

    for x in xs:
        print(f"\n=== column x={x} (y<{ymax}) ===")
        bs = [r for r in runs(Lb[:ymax, x], tb) if r[0] >= 0]
        if not bs:
            print("  no runs in before")
            continue
        print("   y0    y1  thick  meanL_before   ->  after")
        for y0, y1, th, mb in bs[:60]:
            seg = La[y0:y1 + 1, x]
            ma = float(seg.mean())
            verdict = "REMOVED" if ma > ta else f"still {ma:.3f}"
            print(f"  {y0:5d} {y1:5d} {th:6d}   {mb:10.3f}   ->  {verdict}")


if __name__ == "__main__":
    main()
