"""Remove printed ruled lines from scanned notebook pages while keeping handwriting.

Pure numpy + Pillow (no OpenCV needed).

Why this is not just "threshold the grey" - measured on a real 78-page scan:

  paper            ~1.00
  ruled lines      ~0.15-0.50  (dark!  essentially as dark as the pen)
  show-through     ~0.89-0.99  (the reverse page's ruling + JPEG ringing)
  pen ink          ~0.11       (near black)

Two consequences drive the whole design:

  1. The rules are *not* faint here, so darkness cannot separate them from the
     pen - the two differ only in *shape*.  A rule is a long, thin, horizontal
     structure; a pen stroke is neither long, nor thin, nor horizontal.  So the
     rules are found with a long horizontal opening of the ink mask, and ink is
     protected only by its *vertical extent* - a pen stroke has it, a 2-5px rule
     does not.  A darkness guard must NOT be used here: it matches the rules
     (they are dark) and silently protects every one of them.
  2. Scans are *rotated*.  A 0.94 deg skew drifts a rule ~37px across the page,
     so no single pixel row ever contains a whole rule: a tilted dark line covers
     only ~15% of any row, and both a per-row coverage test and a non-deskewed
     horizontal opening fail completely on it.  The page is therefore deskewed
     *before* detection, by shearing the mask only (never the image, so the
     handwriting is never resampled).

The faint 0.89-0.99 material is handled twice, both times by *darkness* rather
than by shape.  The white point is cut below the paper level, which sends that
whole family to plain white as a tone curve - no geometry is touched, so nothing
can be moved or thinned but its faintest edge.  What is left of it is then swept
up by a separate pass where it also forms a long horizontal run, guarded by
darkness, so dark ink of any shape is never touched by that pass.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

EPS = 1e-6


# --------------------------------------------------------------------------- #
# numeric helpers
# --------------------------------------------------------------------------- #
def lum(a: np.ndarray) -> np.ndarray:
    """Luminance of an HxWx3 float array in [0,1]."""
    return (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).astype(np.float32)


def _shift_or(m: np.ndarray, s: int, axis: int) -> np.ndarray:
    """Binary dilation by `s` along one axis (edge replicated)."""
    out = m.copy()
    if axis == 0:
        if s < m.shape[0]:
            out[s:, :] |= m[:-s, :]
            out[:-s, :] |= m[s:, :]
    else:
        if s < m.shape[1]:
            out[:, s:] |= m[:, :-s]
            out[:, :-s] |= m[:, s:]
    return out


def dilate(m: np.ndarray, radius: int, axis: int) -> np.ndarray:
    """Dilate a bool array by `radius` along `axis` using binary decomposition."""
    if radius <= 0:
        return m
    r, s = int(radius), 1
    while r > 0:
        if r & 1:
            m = _shift_or(m, s, axis)
        r >>= 1
        s <<= 1
    return m


def erode(m: np.ndarray, radius: int, axis: int) -> np.ndarray:
    """Erode a bool array by `radius` along `axis` (dual of dilate)."""
    if radius <= 0:
        return m
    return ~dilate(~m, radius, axis)


def shear_offsets(w: int, slope: float) -> np.ndarray:
    """Per-column vertical shift (px) that straightens a line of slope `slope`."""
    xs = np.arange(w, dtype=np.float64)
    return np.round(slope * (xs - w / 2.0)).astype(np.int64)


def _shear(m: np.ndarray, off: np.ndarray, sign: int) -> np.ndarray:
    h = m.shape[0]
    idx = np.clip(np.arange(h, dtype=np.int64)[:, None] + sign * off[None, :], 0, h - 1)
    return np.take_along_axis(m, idx, axis=0)


def shear(m: np.ndarray, off: np.ndarray) -> np.ndarray:
    """Deskew: column x is sampled `off[x]` rows further down."""
    return _shear(m, off, 1)


def unshear(m: np.ndarray, off: np.ndarray) -> np.ndarray:
    """Inverse of `shear` (exact, so masks round-trip)."""
    return _shear(m, off, -1)


def paper_field(gray: np.ndarray, max_size: int = 9, blur: float = 6.0) -> np.ndarray:
    """Local paper-white estimate, removing shadows and uneven lighting.

    A max filter drops dark ink (paper is the brightest thing locally); the blur
    smooths the blocky result into a smooth illumination field.  Computed at
    reduced resolution and upsampled - the field is smooth by construction, so
    this is a pure speed-up.
    """
    h, w = gray.shape
    f = max(1, int(round(min(h, w) / 700.0)))
    small = Image.fromarray(np.clip(gray * 255.0, 0, 255).astype(np.uint8))
    small = small.resize((max(8, w // f), max(8, h // f)), Image.BOX)
    small = small.filter(ImageFilter.MaxFilter(max_size)).filter(ImageFilter.GaussianBlur(blur))
    return np.asarray(small.resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0


def _smoothstep(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    t = np.clip((x - lo) / max(hi - lo, EPS), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _groups(idx: np.ndarray) -> list[tuple[int, int]]:
    """Group sorted integer indices into runs of consecutive values."""
    if len(idx) == 0:
        return []
    out, start, prev = [], int(idx[0]), int(idx[0])
    for v in idx[1:]:
        v = int(v)
        if v != prev + 1:
            out.append((start, prev))
            start = v
        prev = v
    out.append((start, prev))
    return out


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
class Config:
    # thresholds, all relative to the measured paper luminance
    ink_delta: float = 0.012      # "not paper" = this far below the paper level
    medium_delta: float = 0.22    # darker than this = real ink (pen; rules too!)

    # background whitening - the "make the paper plain white" step.  Everything
    # lighter than (paper - bg_white) is pushed to paper-white, over a ramp
    # `white_span` wide below that cut.  The measured populations leave a wide
    # empty gap: show-through measures 0.89-0.99 and pen ink ~0.11, so a cut at
    # paper - 0.18 takes all of the former and none of the latter, and the
    # 0.55-0.78 ramp band holds almost nothing for it to thin.  Being a tone
    # curve it cannot move a stroke's geometry, only its faintest edge.
    bg_white: float = 0.18

    # ruled-line detection
    line_radius_frac: float = 0.15  # horizontal erosion radius, fraction of width
    line_min_radius: int = 40
    vpad: int = 3                   # vertical dilation that absorbs residual skew
    line_pad_v: int = 3             # extra rows above/below a show-through band
    vdeep_radius: int = 4           # vertical run >= 2r+1 px == a real stroke
    band_pad_h: int = 40            # horizontal over-fill when bridging a band

    # faint (>= paper - medium_delta) long horizontal bands: show-through, blush,
    # JPEG ringing.  Removed in a separate, darkness-guarded pass.
    clear_showthrough: bool = True

    # white point
    paper_pct: float = 92.0         # per-channel paper level percentile
    white_pct: float = 95.0         # percentile that defines "this is paper"
    white_span: float = 0.22        # luminance band over which paper fades out

    # skew search
    skew_downsample: int = 4
    skew_range: float = 0.05        # +/- slope searched
    skew_step: float = 0.004        # coarse step, refined by 8x after

    # vertical margin lines: preserved by construction, unless switched off
    preserve_vlines: bool = True
    vline_radius: int = 60


# --------------------------------------------------------------------------- #
# skew estimation
# --------------------------------------------------------------------------- #
def estimate_skew(faint: np.ndarray, cfg: Config) -> float:
    """Slope (vertical drift per pixel of width) that best straightens the rules.

    Scored by how concentrated the row profile becomes: a horizontal rule piles
    all of its ink into one row, a tilted one smears it over ~26.  Run on faint
    non-pen pixels only, so the (much darker, much larger) handwriting cannot
    swamp the signal.
    """
    h, w = faint.shape
    d = max(1, cfg.skew_downsample)
    H, W = h // d, w // d
    if H < 8 or W < 8:
        return 0.0
    small = faint[:H * d, :W * d].reshape(H, d, W, d).mean(axis=(1, 3)).astype(np.float32)
    x_full = np.arange(W, dtype=np.float64) * d + (d - 1) / 2.0

    def score(slope: float) -> float:
        off = np.round(slope * (x_full - w / 2.0) / d).astype(np.int64)
        idx = np.clip(np.arange(H, dtype=np.int64)[:, None] + off[None, :], 0, H - 1)
        rows = np.take_along_axis(small, idx, axis=0).sum(axis=1)
        return float(((rows - rows.mean()) ** 2).mean())

    coarse = np.arange(-cfg.skew_range, cfg.skew_range + 1e-9, cfg.skew_step)
    best = float(max(coarse, key=score))
    fine = np.arange(best - cfg.skew_step, best + cfg.skew_step + 1e-9, cfg.skew_step / 8)
    return float(max(fine, key=score))


# --------------------------------------------------------------------------- #
# main routine
# --------------------------------------------------------------------------- #
def clean_page(rgb: np.ndarray, cfg: Config | None = None, stats: dict | None = None) -> np.ndarray:
    """Clean one page. `rgb` is HxWx3 uint8; returns HxWx3 uint8."""
    cfg = cfg or Config()
    a = np.asarray(rgb, dtype=np.float32) / 255.0
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    if a.shape[2] == 4:
        a = a[..., :3]
    h, w = a.shape[:2]

    # ---- 1. illumination flattening: kills shadows and uneven lighting ----- #
    field = paper_field(lum(a))
    field = np.clip(field / max(float(np.median(field)), EPS), 0.45, 1.30)
    a = np.clip(a / field[..., None], 0.0, 1.0)

    # ---- 2. per-channel white balance + measured white point -------------- #
    for c in range(3):
        lvl = float(np.percentile(a[::4, ::4, c], cfg.paper_pct))
        a[..., c] = np.clip(a[..., c] / max(lvl, EPS), 0.0, 1.0)

    g = lum(a)
    paper = float(np.percentile(g[::4, ::4], cfg.white_pct))
    # The white point sits *below* the paper level, not at it.  A cut at
    # paper - bg_white is what turns show-through and the reverse page's writing
    # into plain white; at paper + 0.02 nothing below paper was ever fully
    # whitened, so faint material came back only about half faded and stayed
    # visible in print - which is exactly what "back page text" was.
    hi = min(0.985, paper - cfg.bg_white)
    lo = max(0.25, hi - cfg.white_span)
    k = _smoothstep(g, lo, hi)[..., None]
    a = a * (1.0 - k) + k

    # ---- 3. masks --------------------------------------------------------- #
    g = lum(a)
    faint = g < (paper - cfg.ink_delta)          # everything that is not paper
    medium = g < (paper - cfg.medium_delta)      # solid ink - and the dark rules too
    # A stroke is measured on the *solid* mask, never on `faint`: `faint` carries
    # a grey halo several px wide around every dark line, so a 3px rule measures
    # 8-12px tall there and guards itself from removal.  On `medium` a rule is
    # its true thin self and a pen stroke is its true tall self.
    #
    # The radius is deliberately just above the rules' measured footprint: on
    # this scan the rules are 2-5px by the >50%-coverage measure, so their dark
    # core runs 6-8px.  A guard of 7px therefore protects the thicker parts of
    # every rule and leaves them behind as dashes; 9px clears them while still
    # shielding any real stroke, which is far taller wherever it crosses a rule.
    vdeep = erode(medium, cfg.vdeep_radius, 0)   # strokes >= 9px tall (a rule is not)

    # ---- 4. deskew, then find long horizontal runs ------------------------- #
    # The rules are as dark as the pen, so no darkness test can tell them apart -
    # only shape can.  A long horizontal opening keeps whatever is at least
    # 2*radius wide, which a rule satisfies and a pen stroke does not.  Pixels
    # outside any such run are never touched, so ink survives wherever it is not
    # sitting on a rule - and there it is protected by its vertical extent alone
    # (a stroke has it, a 2-5px rule does not).
    #
    # Skew is measured on the faint non-ink material (the reverse page's ruling,
    # which is skewed by exactly the same amount); if a page has none, fall back
    # to thin dark structures - the rules themselves - and still exclude blobs.
    skew_mask = faint & ~medium
    if float(skew_mask.mean()) < 0.002:
        skew_mask = medium & ~erode(medium, cfg.vdeep_radius, 0)
    slope = estimate_skew(skew_mask, cfg)
    off = shear_offsets(w, slope)
    radius = max(cfg.line_min_radius, int(round(cfg.line_radius_frac * w)))

    # the vertical dilation merges the rule into a solid band first, so the long
    # horizontal erosion tolerates the residual tilt left by the shear quantisation.
    # erode+dilate is an opening: it keeps structures at least 2*radius wide at
    # their full extent, and the erosion alone would leave a 339px stub of every
    # rule at each margin.
    thick = dilate(shear(faint, off), cfg.vpad, 0)
    remove = dilate(erode(thick, radius, 1), radius, 1)
    remove = unshear(remove, off) & ~vdeep

    # ---- 5. faint show-through, blush and ringing -------------------------- #
    # A separate pass, because this material can only be recognised by darkness:
    # it is what is left of the ink mask once real ink is excluded.
    bands = 0
    if cfg.clear_showthrough:
        band = erode(dilate(shear(faint & ~medium, off), cfg.vpad, 0), radius, 1)
        band = dilate(dilate(band, radius, 1), cfg.line_pad_v, 0)
        # bridge faded stretches, in the deskewed frame where the band really is
        # a straight horizontal run
        fill = band.copy()
        groups = _groups(np.where(band.any(axis=1))[0])
        bands = len(groups)
        for r0, r1 in groups:
            cols = np.where(band[r0:r1 + 1].any(axis=0))[0]
            if not len(cols):
                continue
            c0 = max(0, int(cols[0]) - cfg.band_pad_h)
            c1 = min(w, int(cols[-1]) + 1 + cfg.band_pad_h)
            fill[r0:r1 + 1, c0:c1] = True
        remove |= unshear(fill, off) & ~medium

    # ---- 6. the vertical margin lines (instruction 5) ---------------------- #
    # Preserved by construction: a vertical line has no long horizontal run, so
    # it is in neither pass 4 nor pass 5, and `vdeep` shields it at every
    # crossing.  Removing it is opt-in, so it needs the geometry guard run the
    # other way, and never the darkness guard - the line is dark too.
    if not cfg.preserve_vlines:
        vr = max(8, cfg.vline_radius)
        vband = dilate(erode(dilate(faint, cfg.vpad, 1), vr, 0), vr, 0)
        vband = dilate(vband, cfg.line_pad_v, 1)
        hdeep = erode(medium, cfg.vdeep_radius, 1)
        remove |= vband & ~hdeep

    a[remove] = 1.0

    if stats is not None:
        is_dark = g < (paper - cfg.medium_delta)
        stats.update(
            paper_level=round(paper, 4),
            ink_thresh=round(paper - cfg.ink_delta, 4),
            white_lo=round(lo, 4),
            white_hi=round(hi, 4),
            ink_frac=round(float(faint.mean()) * 100, 3),
            skew_deg=round(float(np.degrees(np.arctan(slope))), 3),
            bands=bands,
            cleared_dark_px=int((remove & is_dark).sum()),
            cleared_faint_px=int((remove & ~is_dark).sum()),
            cleared_px=int(remove.sum()),
            cleared_pct=round(float(remove.mean()) * 100, 4),
        )

    return (np.clip(a, 0.0, 1.0) * 255.0).round().astype(np.uint8)
