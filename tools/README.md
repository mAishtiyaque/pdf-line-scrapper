# Ruled-line removal for scanned PDFs

Strips the printed horizontal notebook rules from a scanned PDF and flattens the
paper to white, **without redrawing or regenerating any handwriting** — the
original scan pixels are kept, only the rule pixels are turned white.

## Files

| file | purpose |
|---|---|
| `tools/lineclean.py` | core algorithm (numpy + Pillow only) |
| `tools/run_pdf.py`   | CLI: process a whole PDF, rebuild it page for page |
| `tools/verify.py`    | **the check that matters**: shape-based before/after accounting |
| `tools/measure_faint.py` | what the whitening cut leaves behind: the faint population's darkness and hue, decile by decile, and what each candidate `--bg-white` would do to it |
| `tools/bands.py`     | every band one page would clear, with its shape and family test |
| `tools/families.py`  | per-page band table (valid); its cross-page recurrence test is invalid on this PDF — see Verifying |
| `tools/deskew.py`    | row structure of one page — rules vs handwriting, by the numbers |
| `tools/inspect_pdf.py` | dump page count / page size / embedded image info |
| `tools/probe_pdf.py` | ground truth from the PDF itself (drawings, images, skew) |
| `tools/runs.py`      | per-column before/after run listing |
| `tools/crop.py`      | 1:1 side-by-side crops (**always 1:1** — a scaled render lies) |
| `tools/analyze.py`, `tools/qa.py` | older colour/metric helpers |

Dependencies: `numpy`, `Pillow`, `PyMuPDF` (`pip install pymupdf pillow numpy`).

## Usage

```bash
# 0. probe the PDF
python tools/inspect_pdf.py "DATA structure 1.pdf"

# 1. clean one page and keep the before/after PNGs
python tools/run_pdf.py "DATA structure 1.pdf" -o preview.pdf --pages 1 --only --debug-dir preview

# 2. verify it: rules must go, ink must stay
python tools/verify.py preview/p001_before.png preview/p001_after.png

# 3. look at it, 1:1
python tools/crop.py preview/p001_before.png preview/p001_after.png --box 100,700,900,1050

# 3b. the band table for a page or two: shape, extent, darkness, family test
#     (its cross-page recurrence half is invalid on this PDF - see Verifying)
python tools/families.py "DATA structure 1.pdf" --step 3

# 4. full run
python tools/run_pdf.py "DATA structure 1.pdf" -o "DATA structure 1 - clean.pdf" --quality 90
```

## What this scan actually looks like (measured, not assumed)

| | luminance |
|---|---|
| paper | ~1.00 |
| **ruled lines** | **~0.15–0.50 — dark, essentially as dark as the pen** |
| show-through (reverse page, JPEG ringing) | ~0.89–0.99 |
| pen ink | ~0.11 |

Two facts follow, and between them they dictate the whole design:

1. **The rules are not faint.** They are as dark as the handwriting, so *no*
   darkness threshold can separate them. An earlier version of this tool guarded
   removal with "never clear anything darker than paper − 0.22" — which matches
   the rules perfectly and therefore protected every single one of them. The
   lines are still in that version's output, pixel for pixel.
2. **The scan is rotated ~0.94°.** A rule drifts ~37px across an A4 page, so no
   pixel row contains a whole rule: a tilted 3px line covers only ~15% of any
   row. Every non-deskewed test — per-row coverage, coverage sweeps down to
   paper − 0.002, an un-deskewed horizontal opening — fails completely on a rule
   that looks perfectly horizontal to the eye.

Rule and pen differ **only in shape**, so shape has to carry the whole
discrimination: a rule is long, thin and horizontal; a pen stroke is none of
those.

## Why the output won't look like the WhatsApp sample

The two WhatsApp images are `604x704` and `1364x1153` — aspect ratios 0.86 and
1.18. The same page cannot move between those by having its lines removed, so the
"after" is not a cleaned version of the "before": it was **re-rendered**. The
letterforms change from the writer's own hand to a uniform italic, the ink becomes
flat with no pressure variation, the hand-drawn cloud outline becomes a perfectly
smooth uniform curve, and content is lost or corrupted — the underline under
`ababa` disappears, `𝕃` becomes `Ị`, `α` becomes `a`. That is precisely what
instructions 3 and 4 forbid ("preserve the original handwriting… do not redraw or
regenerate"). Note also that its notebook has *pale blue* rules under a blue pen,
so colour alone would separate them there; this PDF has black rules under a black
pen, which is why shape is the only usable discriminator.

This tool keeps the original pixels instead: in any `crop.py` before/after pair the
handwriting is identical stroke for stroke — same strokes, same pressure, same
colour — because no pixel of it is ever rewritten.

## What the algorithm does

1. **Illumination flattening** — estimates the local paper white level
   (max-filter then blur, at reduced resolution) and divides it out, removing
   shadows, uneven lighting and colour cast.
2. **Background whitening** — a smoothstep whose cut sits **below** the measured
   paper level (`paper − 0.18`, ramp 0.22 wide) maps show-through and the reverse
   page's writing to exactly 255 and leaves anything darker than `paper − 0.40`
   untouched. Ink keeps its density and anti-aliasing, and being a tone curve it
   cannot move geometry at all — only a stroke's faintest edge can thin. The cut
   *has* to be below paper: at `paper + 0.02` nothing below paper was ever
   *fully* whitened, so faint material came back only about half faded — still
   visible in print, which is what "there is a vertical line and back page text
   also seems come" was. The measured populations leave a wide empty slot for the
   cut: faintest rule 0.50, darkest show-through 0.89, so 0.78 separates them with
   margin on both sides.
3. **Deskew** — the slope is estimated by shearing a mask and scoring how
   concentrated the row profile becomes (coarse sweep, then an 8× refinement).
   Only the *mask* is sheared, never the image, so handwriting is never resampled;
   the inverse shear is exact, so masks round-trip.
4. **Rule removal by geometry alone** — in the deskewed frame, a long horizontal
   opening (`erode` then `dilate`, kernel ≈ 0.15 · page width) keeps only
   structures unbroken for ~30% of the page width, at their full extent. That is
   the rules. The radius is a **floor on run length**: at 0.15 a structure must
   stay unbroken for 679px, which a full-width rule does and a hand-drawn
   underline (~560px under `2k12 … 2k18` in this scan) does not. Lowering it
   starts eating hand-drawn lines. Raising it is safer for content but it does
   **not** stay safe for the rules: at 0.35 the floor becomes 1583px, and a rule
   that is faded or broken anywhere along its length fails that for its *whole*
   extent, so it is left standing. Measured, not assumed — the same page at 0.35
   reports `rule-shaped after 55,302` where 0.15 reports `0`, and both headline
   counts read success while a 1:1 crop of the box showed the line still standing
   at 0.35 and gone at 0.15. Reproduce it in a minute: `run_pdf.py "…" -o r35.pdf
   --pages 1 --only --debug-dir r35 --line-radius-frac 0.35`, then `verify.py
   r35/p001_before.png r35/p001_after.png --rule-frac 0.35`, then `crop.py` on
   whatever box it reports. Ink is protected by **vertical
   extent**: a pen stroke is far taller than the rule wherever it crosses one,
   while the rule's dark core is only 6–8px. Crucially this is measured on the
   *solid* mask (`g < paper − 0.22`), never on the inclusive "non-paper" mask —
   that one carries a grey halo several px wide around every dark line, so a 3px
   rule measures 8–12px tall there and guards itself from removal.
5. **Faint show-through, second pass** — what step 2 left of it, and again by
   darkness rather than by shape: it is what the ink mask has left once solid ink
   is excluded. Removed only where it also forms a long horizontal run, and
   additionally guarded by darkness, so blush and ringing go while dark ink of
   any shape is never touched by this pass.
6. **Vertical margin lines** — preserved by construction: a vertical line has no
   long horizontal run, so it is in neither pass, and the vertical-extent guard
   shields it at every crossing. `--no-preserve-vlines` reverses the geometry to
   remove it (never with a darkness guard — it is dark too).

## Knobs (`lineclean.Config` / CLI)

| flag | default | effect |
|---|---|---|
| `--vdeep-radius` | 4 | vertical run ≥ `2r+1` px counts as a real pen stroke. On this scan the rules measure 2–5px by the >50%-coverage test, so their dark core runs 6–8px: a guard of 3 (≥7px) lets the thicker parts of every rule shield themselves and leaves them behind as dashes. Raise to 5 if fragments survive, lower to 3 if thin ink is being nicked |
| `--line-radius-frac` | 0.15 | floor on how long a run must stay unbroken before it counts as a rule (= 2·frac = 30% of page width at the default). **Lowering** it starts eating hand-drawn lines. **Raising** it protects content but leaves the rules standing: at 0.35 the floor is 1583px, so any rule faded or broken along its length fails it and stays on the page, visibly. Decide with `verify.py --rule-frac` plus a 1:1 crop of a rule box, never by count alone |
| `--no-preserve-vlines` | off | also clear vertical margin lines (instruction 5 keeps them) |
| `--keep-showthrough` | off | keep faint background bands (show-through, blush, ringing) |
| `--bg-white` | 0.18 | whiten everything lighter than `paper − this` — the knob for "apply contrast/light". Raise toward 0.30 if background text is still visible, lower if faint strokes are thinning |
| `--white-span` | 0.22 | width of the ramp below that cut over which paper fades out |
| `--quality` | 90 | JPEG quality (4:4:4, so coloured ink keeps its chroma) |

## Verifying (do this, do not eyeball it)

`tools/verify.py` reports, in the deskewed frame, for every ink pixel: the
horizontal and vertical run it belongs to — measured after the same ±3px
vertical dilation the cleaner applies (`--dilate`), because a 3px rule straddling
two pixel rows fragments into ~300px pieces within a single raw row and would be
misread as ink. It then prints four numbers, and they answer different questions:

- **`rule bands under the licence … still visible after (contrast > 0.06): 0`.**
  The direct test of instructions 1 and 2, and the one to trust. Every rule band
  detected in the before page is re-measured after: how much darker than paper is
  its darkest pixel still, in that band's own columns. Contrast and not a pixel
  count, because a rule can survive two opposite-looking ways and *both* leave
  thousands of "rule-shaped" pixels — as a broken thinner line (visible) or as a
  haze below the ink threshold (invisible), and only contrast says which. Bands
  too thick to be a rule (median dark thickness > 6px) are excluded: the
  cleaner's `vdeep` guard shields those by design, so they are not expected to go.
- **`background … more than 0.03 below paper`: before N → after near 0.** How
  much of the non-ink, non-margin-line area is still not plain paper. The
  definition is taken from the before page and applied to both, so the cleaner's
  own success cannot shrink the area it is judged on.
- **`LOSS ≈ 0`.** A removed dark pixel counts as a loss only when *both* hold: it
  did not sit on a row the cleaner was **licensed** to clear, **and** no dark ink
  survived within 2px of it. The licence is a length, because the opening only
  keeps runs of `2·radius+1` — so `--rule-frac` must be given the radius actually
  used, or the score is against the wrong rule. Both conditions are needed. A rule
  is thin enough that its **fringe rows** carry it only over part of the width, so
  a per-pixel shape test calls those fringes "ink" and reports the rule's own
  removal as damage — an earlier version did exactly that and produced 28,688
  pixels of false alarm. Scoring the licence against *half the page width* instead
  produced a second wave of the same mistake: clearing an 847px rule segment that
  sits on the notebook's own 104px pitch is not a loss, and a count that calls it
  one is measuring the wrong thing. Every surviving loss is listed with a
  ready-to-paste `crop.py --box`.
- **`rows >60% ink across the page`: before N → after ~0.** Kept, but do not lean
  on it: it needs a rule to cover 60% of the *full* width, so a partly faded rule
  can be plainly visible while this reads 0.

Full-page renders are unreliable for this — the viewer contrast-normalises, so a
`0.94` line on white paper displays as solid black. Crop 1:1 with `tools/crop.py`.
But note what the counts are for and what they are not: they are the evidence for
ink loss and for background purity, a 1:1 crop is the evidence for whether a line
is *visible*, and neither substitutes for the other — the 0.35 experiment is the
proof, since every count looked acceptable while a line stood there in the crop.

`tools/families.py` was the attempt to answer what shape alone cannot: a
hand-drawn line *can* be long, thin and horizontal exactly like a rule — page 1
carries partial-width bands of 599–1364px against the rules' full 2261px — but it
cannot recur at the same y page after page. **Its recurrence test does not work on
this document.** The pages are cropped to different widths and heights (page 1's
scan is 2261×3200 in a full A4 rect; page 13's is 1148×3200 in a 302pt-wide
strip), so one rule does not land at one y twice and nothing recurs. The valid
form is *interval*, not position — the pitch between neighbouring bands is
invariant to crop and scale — and that is not implemented. Its per-page band table
(shape, extent, darkness) is still correct and useful.

## Known limits

- Dark ink that is genuinely long, thin and horizontal and spans ≥30% of the page
  width (≥679px on this scan) is indistinguishable from a rule by shape and will
  be removed. Raising `--line-radius-frac` spares it, but puts visibly surviving
  rules back on the page, and no setting separates the two by shape alone. The
  usable discriminator is the ruling's *pitch*, not its shape — see
  `tools/families.py`.
- Very light ink (light pencil, pale highlighter, L > 0.78) inside a rule band is
  treated as show-through and cleared. Black/blue pen is unaffected.
- Pages that are not rotated need no deskew, but the estimator simply returns
  ≈0 — there is no separate code path for them.

## Guarantees

- Page count, page order and page size (A4 595×842 pt) are preserved.
- Each output page places the cleaned scan exactly where the original was.
- No OCR, no text layer, no regeneration — pixels in, pixels out.
