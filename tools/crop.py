"""Write a 1:1 side-by-side crop of two page images, for visual comparison.

    python tools/crop.py before.png after.png --box 100,350,900,800 --out cmp_left
                       [--box x0,y0,x1,y1 ...]

Writes <out>.png (1:1) and <out>_2x.png (nearest-neighbour upscale, so the
actual pixel structure is visible rather than a resampled impression).
"""
from __future__ import annotations

import sys

from PIL import Image


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    boxes, out = [], None
    for i, a in enumerate(sys.argv):
        if a == "--box" and i + 1 < len(sys.argv):
            boxes.append(tuple(int(v) for v in sys.argv[i + 1].split(",")))
        if a == "--out" and i + 1 < len(sys.argv):
            out = sys.argv[i + 1]
    if len(args) < 2 or not boxes:
        print(__doc__)
        return
    out = out or "cmp"
    A = Image.open(args[0]).convert("RGB")
    B = Image.open(args[1]).convert("RGB")

    for idx, box in enumerate(boxes):
        ca, cb = A.crop(box), B.crop(box)
        gap = 10
        sheet = Image.new("RGB", (ca.width * 2 + gap, ca.height), (255, 0, 0))
        sheet.paste(ca, (0, 0))
        sheet.paste(cb, (ca.width + gap, 0))
        name = f"{out}_{idx}.png" if len(boxes) > 1 else f"{out}.png"
        sheet.save(name)
        sheet.resize((sheet.width * 2, sheet.height * 2), Image.NEAREST).save(
            name.replace(".png", "_2x.png"))
        print(f"wrote {name}  ({sheet.width}x{sheet.height})  box={box}  "
              f"left=before right=after")


if __name__ == "__main__":
    main()
