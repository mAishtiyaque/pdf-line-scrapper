"""Analyze scan images: paper/ink colour, ruled-line rows, vertical margin lines.

Usage: python analyze.py <image> [image ...]
"""
import sys
import numpy as np
from PIL import Image


def load(path):
    im = Image.open(path).convert("RGB")
    return np.asarray(im).astype(np.float32) / 255.0


def lum(a):
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def sat(a):
    mx = a.max(axis=2)
    mn = a.min(axis=2)
    return np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)


def report(name, a):
    L, S = lum(a), sat(a)
    h, w = L.shape
    print(f"\n=== {name}  {w}x{h} ===")
    print("L pct  ", [round(float(np.percentile(L, p)), 3)
                      for p in (1, 5, 25, 50, 75, 90, 95, 99)])
    hist, _ = np.histogram(L, bins=10, range=(0, 1))
    print("L hist%", (hist / L.size * 100).round(1).tolist())

    for t in (0.30, 0.45, 0.60, 0.75, 0.85, 0.92):
        print(f"  frac L<{t}: {float((L < t).mean()):.4f}")

    for lo, hi, tag in ((0.0, 0.45, "dark"), (0.55, 0.90, "mid/line")):
        m = (L >= lo) & (L < hi)
        if m.sum():
            px = a[m]
            print(f"  {tag:9s} n={m.sum():9d} rgb={px.mean(axis=0).round(3).tolist()}"
                  f" sat={S[m].mean():.3f}")

    # ---- horizontal ruled lines: row-wise ink fraction -------------------
    ink = L < 0.90
    rowf = ink.mean(axis=1)
    print("  rows frac>0.50:", int((rowf > 0.50).sum()),
          " >0.35:", int((rowf > 0.35).sum()))
    cand = np.where(rowf > 0.50)[0]
    if len(cand):
        # group consecutive rows
        groups, start = [], cand[0]
        for i in range(1, len(cand)):
            if cand[i] != cand[i - 1] + 1:
                groups.append((start, cand[i - 1]))
                start = cand[i]
        groups.append((start, cand[-1]))
        print("  line bands (y0,y1,centre):",
              [(int(a0), int(b0), int((a0 + b0) / 2)) for a0, b0 in groups[:25]])
        if len(groups) > 2:
            c = [ (g[0]+g[1])/2 for g in groups ]
            gaps = np.diff(c)
            print("  spacing median %.1f  min %.1f  max %.1f"
                  % (np.median(gaps), gaps.min(), gaps.max()))
        # colour of the line pixels
        linemask = np.zeros_like(ink)
        for a0, b0 in groups:
            linemask[a0:b0 + 1] = True
        if linemask.sum():
            px = a[linemask]
            Lm = L[linemask]
            print("  line px rgb=%s sat=%.3f L=%.3f"
                  % (px.mean(axis=0).round(3).tolist(), S[linemask].mean(), Lm.mean()))

    # ---- vertical margin lines: column-wise profile ----------------------
    colf = ink.mean(axis=0)
    top = np.argsort(colf)[-10:]
    print("  top cols ink-frac:", sorted((int(c), round(float(colf[c]), 3)) for c in top))
    reddish = ((a[..., 0] - a[..., 2]) > 0.06)
    rf = reddish.mean(axis=0)
    top = np.argsort(rf)[-10:]
    print("  top cols reddish :", sorted((int(c), round(float(rf[c]), 3)) for c in top))
    # vertical line = column that is ink for a large fraction of its height
    vcand = np.where(colf > 0.30)[0]
    print("  cols frac>0.30:", vcand[:40].tolist(), "count", len(vcand))


def row_detail(a, deltas=(0.002, 0.005, 0.01, 0.02, 0.04, 0.08)):
    """Pin down how dark the ruled lines actually are, relative to the paper.

    A rule row covers most of the page width with few ink/no-ink transitions;
    a text row covers less and flickers many times.  Printing both together
    separates rules from handwriting without guessing a threshold.
    """
    L = lum(a)
    paper = float(np.percentile(L, 95))
    print(f"  paper(L95) = {paper:.4f}   darkest = {float(L.min()):.4f}")

    print("  coverage sweep - rows with >50% of pixels below (paper - delta):")
    for d in deltas:
        thr = paper - d
        cov = (L < thr).mean(axis=1)
        print(f"    paper-{d:<6g} thr={thr:.4f}   rows cov>0.5 = {int((cov > 0.5).sum()):4d}"
              f"   cov>0.8 = {int((cov > 0.8).sum()):4d}   max cov = {float(cov.max()):.3f}")

    print("  which rows come closest to full-width at paper-0.005:")
    thr = paper - 0.005
    m = L < thr
    cov = m.mean(axis=1)
    edges = np.abs(np.diff(m.astype(np.int8), axis=1)).sum(axis=1)
    idx = np.argsort(cov)[-25:]
    print("      y     cov  edges   minL  meanL_of_ink")
    for y in sorted(int(i) for i in idx):
        row = m[y]
        meanl = float(L[y][row].mean()) if row.any() else 1.0
        print(f"   {y:7d} {cov[y]:7.3f} {int(edges[y]):6d} "
              f"{float(L[y].min()):6.3f} {meanl:13.3f}")

    print("  column probe - runs below paper-0.005 along a single column:")
    for x in (150, 700, 1400, 2100):
        col = L[:, x]
        mm = col < (paper - 0.005)
        runs, y = [], 0
        while y < len(col):
            if mm[y]:
                y0 = y
                while y < len(col) and mm[y]:
                    y += 1
                runs.append((y0, y - 1, round(float(col[y0:y].mean()), 4)))
            else:
                y += 1
        print(f"    x={x:5d}  {len(runs)} runs"
              + (f"  first: {runs[:14]}" if runs else "  (column is pure paper)"))


paths = [a for a in sys.argv[1:] if not a.startswith("--")]
for p in paths:
    name = p.split("\\")[-1].split("/")[-1]
    arr = load(p)
    report(name, arr)
    if "--rows" in sys.argv:
        print(f"--- row detail: {name} ---")
        row_detail(arr)
