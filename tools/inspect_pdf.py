"""Report PDF structure: pages, sizes, embedded images.

Usage: python inspect_pdf.py <pdf>
"""
import sys
import fitz

doc = fitz.open(sys.argv[1])
print("pages:", doc.page_count, "meta:", {k: v for k, v in doc.metadata.items() if v})
counts = {}
for i, page in enumerate(doc):
    imgs = page.get_images(full=True)
    counts[len(imgs)] = counts.get(len(imgs), 0) + 1
    if i < 4 or i == doc.page_count - 1:
        r = page.rect
        print(f"\npage {i}: mediabox={r.width:.1f}x{r.height:.1f} rot={page.rotation}"
              f" images={len(imgs)}")
        for im in imgs:
            xref = im[0]
            d = doc.extract_image(xref)
            print(f"   xref={xref} {d['width']}x{d['height']} ext={d['ext']}"
                  f" cs={d.get('colorspace')} bytes={len(d['image'])}")
        for b in page.get_image_info():
            print("   bbox", [round(v, 1) for v in b["bbox"]])
print("\nimages-per-page histogram:", counts)
