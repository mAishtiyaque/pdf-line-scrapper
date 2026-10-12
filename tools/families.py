"""Which horizontal bands recur across the whole PDF?  Only printed rules do.

    python tools/families.py "DATA structure 1.pdf" [--step 3] [--probe 0.08]

** The recurrence half of this tool DOES NOT WORK on "DATA structure 1.pdf". **
Its method is position: a printed rule puts a band at the same y page after page,
so those y values pile up, while a hand-drawn horizontal line is an arbitrary y on
one page and stays a stray.  That assumes the pages are the same size and framing.
They are not: this PDF's scans range from 2261x3200 (page 1, in a full A4 rect) to
1148x3200 (page 13, placed in a 302pt-wide strip of the still-A4 page), and 68 of
the 78 sit in a sub-rect of their MediaBox.  One rule therefore does not land at
one y twice, nothing accumulates, and the lattice prints empty - a misleading
"nothing recurs, so everything is content" result rather than an error.

The valid form of the same idea is *interval*, not position: the pitch between
neighbouring bands is invariant to crop and scale, and a printed ruling is a
lattice of near-equal pitches while hand-drawn lines are not.  Not implemented
here.  Until it is, the per-page half (the band table: y, extent, thickness,
darkness of every structure the cleaner would consider) is correct and useful, and
`--list` on a single page is the honest way to use this file.

What it was for: the discriminator shape cannot provide.  A hand-drawn line *can*
be long, thin and horizontal - `tools/bands.py` finds five partial-width bands on
page 1 alone (extents 599-1364px against the rules' full 2261px) - so no test on
one band in isolation can tell it from a rule.

`--probe` sets the opening radius used for *detection*, deliberately lower than
the cleaner's, so the listing also shows the short structures (down to
2*0.08*width) that the cleaner would never touch.  It is a probe, not a setting.

Output: the recurrent lattice - one line per rule with how many pages carry it -
then every band that is *not* on it, then a per-page match count.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import fitz

from bands import detect
from lineclean import Config
from run_pdf import page_image, parse_pages


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--step", type=int, default=2, help="use every Nth page")
    ap.add_argument("--pages", default="")
    ap.add_argument("--tol", type=int, default=4, help="+/- px that counts as the same y")
    ap.add_argument("--support", type=float, default=0.5,
                    help="fraction of sampled pages a rule must appear on")
    ap.add_argument("--probe", type=float, default=0.08,
                    help="opening radius fraction for detection (cleaner uses 0.15+)")
    args = ap.parse_args()

    cfg = Config()
    cfg.line_radius_frac = args.probe
    cfg.line_min_radius = 8

    doc = fitz.open(args.src)
    todo = parse_pages(args.pages, doc.page_count)[::max(1, args.step)]
    need = args.support * len(todo)
    print(f"source: {args.src}  pages={doc.page_count}  sampled={len(todo)}  "
          f"probe_radius={cfg.line_radius_frac} x width  vpad={cfg.vpad}", flush=True)

    per_page: list[tuple[int, list[dict]]] = []
    ys_all: list[int] = []
    w_px = 0
    for i in todo:
        arr, how, _place = page_image(doc, doc[i])
        rows, info = detect(arr, cfg)
        w_px = info["w"]
        per_page.append((i + 1, rows))
        ys_all.extend(r["y"] for r in rows)
        print(f"  p{i + 1:>3}  bands={len(rows):3d}  "
              f"skew={np.degrees(np.arctan(info['slope'])):+.3f}deg  {how}", flush=True)
    print(f"scan is {w_px}px wide, so this listing shows runs >= "
          f"{2 * int(round(cfg.line_radius_frac * w_px)) + 1}px "
          f"(a rule spans the full {w_px})", flush=True)

    if not ys_all:
        print("no bands found on any sampled page")
        return 1

    ys = np.asarray(ys_all, dtype=np.int64)
    H = int(ys.max()) + 2 * args.tol + 2
    hist = np.bincount(ys, minlength=H).astype(np.float64)
    k = args.tol
    dens = np.convolve(hist, np.ones(2 * k + 1), mode="same")

    def support(y: int) -> int:
        lo, hi = max(0, y - k), min(H, y + k + 1)
        return int(dens[lo:hi].max())

    # y values carrying enough support, one entry per rule
    rules: list[int] = []
    hot = np.where(dens >= need)[0]
    if len(hot):
        start = prev = int(hot[0])
        for v in [int(x) for x in hot[1:]] + [10 ** 9]:
            if v != prev + 1:
                seg = np.arange(start, prev + 1)
                rules.append(int(seg[np.argmax(dens[seg])]))
                start = v
            prev = v

    print(f"\nrecurrent bands (carried by >={need:.0f} of {len(todo)} sampled pages): "
          f"{len(rules)}")
    print(f"{'y':>6} {'pages':>6} {'thick':>6} {'extent':>7} {'inkL':>6}")
    lattice_ext: list[float] = []
    for y in rules:
        m = [r for _, rows in per_page for r in rows if abs(r["y"] - y) <= k]
        e = float(np.median([r["ext"] for r in m]))
        lattice_ext.append(e)
        print(f"{y:6d} {support(y):6d} {np.median([r['t'] for r in m]):6.0f} "
              f"{e:7.0f} {np.median([r['dark'] for r in m]):6.3f}")
    rule_ext = float(np.median(lattice_ext)) if lattice_ext else float(w_px)

    offs = [(pno, r) for pno, rows in per_page for r in rows if support(r["y"]) < need]
    print(f"\nbands NOT on that lattice ({len(offs)} of {len(ys_all)} band detections) - "
          f"candidate hand-drawn lines, or page-specific artefacts:")
    if not offs:
        print("  none: every band on every sampled page is a recurrent rule")
    else:
        print(f"{'page':>5} {'y':>6} {'thick':>6} {'extent':>7} {'inkL':>6} "
              f"{'to rule':>8} {'ext/rule':>9}   verdict")
        ranked = sorted(offs, key=lambda t: -t[1]["ext"])
        for pno, r in ranked[:40]:
            near = min((abs(r["y"] - y) for y in rules), default=-1)
            frac = r["ext"] / max(1.0, rule_ext)
            if frac >= 0.9:
                verdict = ("LONG - as long as a rule: if it recurs it is a rule, "
                           "if not it is content the 0.15 opening will eat. CHECK")
            else:
                verdict = "short: the cleaner's 0.15 opening already spares it"
            print(f"{pno:5d} {r['y']:6d} {r['t']:6d} {r['ext']:7d} {r['dark']:6.3f} "
                  f"{near:8d} {frac:9.2f}   {verdict}")
        if len(ranked) > 40:
            print(f"  ... and {len(ranked) - 40} shorter ones (all < "
                  f"{ranked[40][1]['ext']}px by construction of this sort)")

    print("\nper page: bands / on-lattice / off-lattice")
    for pno, rows in per_page:
        on = sum(1 for r in rows if support(r["y"]) >= need)
        print(f"  p{pno:>3}  {len(rows):3d} / {on:3d} / {len(rows) - on:3d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
