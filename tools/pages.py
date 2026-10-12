"""Per-page geometry of the source PDF: page size, embedded scan, placement.

    python tools/pages.py "DATA structure 1.pdf"

Needed because this PDF's pages are NOT uniform.  Their embedded scans differ
from page to page - 2261x3200 on page 1, 1148x3200 on page 13, 2396x3392 on
page 7 - so nothing may be assumed about "the page".

For each page this prints the MediaBox, the largest embedded scan's pixel size,
and the rect that scan is placed in.  That placement rect is the one the cleaned
scan must be inserted into: inserting into the full MediaBox instead would
stretch the scan whenever the two differ, which is a layout change, not a
clean-up.

Two flags matter:

  STRETCHED     the placement rect's aspect does not match the scan's, so the
                PDF itself already stretches it - reproduce that, don't fix it.
  SUB-RECT      the scan occupies less than the full MediaBox (a margin, a
                corner, a watermark-less border) - insert into that rect, or the
                content moves and changes size.
"""
from __future__ import annotations

import sys

import pymupdf


def main() -> int:
    doc = pymupdf.open(sys.argv[1])
    print(f"{'pg':>3} {'mediabox pt':>13} {'asp':>5} {'n':>2} {'scan px':>13} "
          f"{'asp':>5} {'placement pt':>13} {'asp':>5}  flags")
    boxes: dict[tuple[int, int], int] = {}
    scans: dict[tuple[int, int], int] = {}
    rot: dict[int, int] = {}
    n_stretch = n_sub = 0
    for i, page in enumerate(doc):
        r = page.rect
        imgs = page.get_images(full=True)
        big = None
        for im in imgs:
            d = doc.extract_image(im[0])
            if big is None or d["width"] * d["height"] > big[1] * big[2]:
                big = (im[0], int(d["width"]), int(d["height"]))
        pw = ph = 0.0
        if big is not None:
            rects = page.get_image_rects(big[0])
            if rects:
                pw, ph = rects[0].width, rects[0].height
        asp_page = r.width / max(r.height, 1e-6)
        asp_scan = (big[1] / big[2]) if big else float("nan")
        asp_place = (pw / ph) if ph else float("nan")

        flags = []
        if ph:
            if abs(asp_place - asp_scan) / max(asp_scan, 1e-6) > 0.01:
                flags.append("STRETCHED")
                n_stretch += 1
            if abs(pw - r.width) > 1.0 or abs(ph - r.height) > 1.0:
                flags.append("SUB-RECT")
                n_sub += 1
        scan_s = f"{big[1]}x{big[2]}" if big else "-"
        place_s = f"{pw:.0f}x{ph:.0f}" if ph else "-"
        print(f"{i + 1:>3} {r.width:6.0f}x{r.height:<6.0f} {asp_page:5.3f} {len(imgs):>2} "
              f"{scan_s:>13} {asp_scan:5.3f} {place_s:>13} {asp_place:5.3f}  "
              f"{', '.join(flags)}")

        b = (round(r.width), round(r.height))
        boxes[b] = boxes.get(b, 0) + 1
        if big:
            s = (big[1], big[2])
            scans[s] = scans.get(s, 0) + 1
        rot[page.rotation] = rot.get(page.rotation, 0) + 1

    print("\nmediabox (w x h pt) histogram:", boxes)
    print("scan (px) histogram:", scans)
    print("page rotations:", rot)
    print(f"\npages the PDF itself stretches (placement aspect != scan aspect): {n_stretch}")
    print(f"pages whose scan sits in a sub-rect of the MediaBox:                {n_sub}")
    print(f"total pages: {doc.page_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
