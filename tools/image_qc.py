"""
Image QC tool.

Visually inspects a photo of a screen-printed electrode (SPE) sensor for
manufacturing defects, before the sensor is run electrochemically. Uses the
same Gemini model as the main chat agent (natively multimodal, so no
separate vision model is needed) to look at the image and return
structured flags.

Electrodes are laid out on a lettered/numbered grid (e.g. "E5"), and not
every position has a photo. If the exact electrode isn't available, this
tool can fall back to the nearest photographed neighbor on the same plate
as a visual proxy -- electrodes from the same fabrication batch/plate are
expected to be similar, so an adjacent print is informative even if it
isn't the exact one tested. The result is always labeled as a proxy when
this happens so it's never mistaken for the real electrode's photo.

Optionally also renders a 3D surface plot of the image (ImageJ-style: pixel
luminance interpreted as height) and reports quantitative roughness metrics
from it -- a screen-printed electrode with even ink coverage should read as
a fairly flat surface; texture, streaking, or isolated spikes (contamination,
missed print) show up as luminance variance. This is a deterministic,
un-calibrated heuristic (see analyze_surface_topology docstring) meant to
complement, not replace, the vision model's defect read.
"""

import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless -- this runs as a tool, not an interactive session
import matplotlib.pyplot as plt
import numpy as np
from google.genai import types
from matplotlib.colors import LightSource
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.stats import kurtosis, skew

from tools._gemini import client, MODEL
from tools.electrode_notes import append_qc_result, extract_batch_date, extract_grid_code

_IMAGE_EXTS = (".bmp", ".jpg", ".jpeg", ".png")
_MIME_TYPES = {".bmp": "image/bmp", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
_GRID_CODE_RE = re.compile(r"^([A-Za-z])(\d+)$")
SURFACE_PLOTS_DIR = Path("surface_plots")

IMAGE_QC_SCHEMA = {
    "type": "function",
    "function": {
        "name": "image_qc",
        "description": (
            "Visually QC a photo of a screen-printed electrode (SPE) sensor. "
            "Uses a vision model to flag visible defects -- incomplete/smudged "
            "printing, electrode misalignment, contamination, cracks, or "
            "delamination -- before the sensor is run electrochemically."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": (
                        "Path to a specific electrode photo (jpg/png/bmp). Use this "
                        "when you already know the exact file. Omit if using "
                        "electrode_code + image_dir instead."
                    ),
                },
                "electrode_code": {
                    "type": "string",
                    "description": (
                        "Grid position of the electrode to check, e.g. 'E5'. Use with "
                        "image_dir when you don't have a direct file path -- if no "
                        "photo exists for this exact position, the nearest "
                        "photographed neighbor on the same plate is used as a proxy "
                        "and the result is labeled as such."
                    ),
                },
                "image_dir": {
                    "type": "string",
                    "description": "Directory of grid-labeled electrode photos to search (used with electrode_code). Or pass batch instead.",
                },
                "batch": {
                    "type": "string",
                    "description": "Batch date, e.g. '20260707' -- shorthand for image_dir='reference_images/<batch>'.",
                },
                "context": {
                    "type": "string",
                    "description": (
                        "Optional context -- expected electrode design, batch id, "
                        "or specific things to look for."
                    ),
                },
                "include_surface_analysis": {
                    "type": "boolean",
                    "description": (
                        "Also render a 3D luminance-as-height surface plot and report ISO-4287-style "
                        "roughness parameters (Ra, Rq, Rz, Rt, Rsk, Rku) computed on luminance rather "
                        "than physical height -- a relative photo-only proxy, not calibrated to nm/um "
                        "(see analyze_surface_topology's docstring for the caveat). Default false."
                    ),
                },
                "surface_grid_size": {
                    "type": "integer",
                    "description": "Downsample width (px) for the surface plot grid. Default 100.",
                },
                "max_luminance_cv": {
                    "type": "number",
                    "description": (
                        "Flag the surface as uneven if luminance coefficient-of-variation "
                        "exceeds this. Weak signal -- confirmed to flag ~100% of a real batch "
                        "with no discriminating power (whole-frame aggregate, can't localize a "
                        "defect). Prefer compare_to_batch_reference for actually finding outliers."
                    ),
                },
                "surface_enhance_contrast": {
                    "type": "boolean",
                    "description": (
                        "ImageJ-style contrast stretch (percentile clip + normalize to 0-255) "
                        "before building the height map -- on by default. Without it the "
                        "electrode surface's own texture gets visually compressed by "
                        "background pixels sharing the same height axis."
                    ),
                },
                "surface_contrast_saturate_pct": {
                    "type": "number",
                    "description": "Percent of extreme pixels clipped at each end before the contrast stretch. Default 1.0.",
                },
                "surface_crop_box": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": (
                        "[left, top, right, bottom] as fractions 0-1 of the image, cropping "
                        "before the height map is built. Defaults to WORKING_ELECTRODE_CROP_BOX, "
                        "which isolates just the working electrode's central disc (roughness "
                        "should describe that surface specifically, not the counter-electrode "
                        "ring/leads around it) -- tuned for the current single-electrode-per-photo "
                        "framing (reference_images/20260804 on). For a 20260707-style 2x2-cluster "
                        "photo, don't pass this -- use predict_electrode_performance's sub_position "
                        "instead, which selects the right sub-electrode's disc automatically. Pass "
                        "[0,0,1,1] for the full, un-cropped frame, or your own box for a differently "
                        "framed batch."
                    ),
                },
            },
        },
    },
}

_PROMPT_TEMPLATE = """You are inspecting a photo of a screen-printed electrode (SPE) \
biosensor for visible manufacturing defects, prior to electrochemical testing.

Look for:
- Incomplete, smudged, or uneven ink printing
- Misalignment between working, reference, and counter electrodes
- Visible contamination, debris, or fingerprints on the electrode surface
- Cracks, scratches, or delamination of the conductive traces
- Exposed substrate where ink should be
{context_line}
Respond ONLY with JSON matching this schema:
{{"status": "pass" | "warn" | "fail", "defects": ["short description", ...], "notes": "one or two sentence overall assessment"}}

Use "fail" only for defects likely to affect electrochemical performance (broken \
traces, missing ink over the electrode area). Use "warn" for cosmetic issues worth \
a second look. Use "pass" if the print looks clean and complete."""


def _parse_grid_code(name: str):
    m = _GRID_CODE_RE.match(name)
    if not m:
        return None
    return m.group(1).upper(), int(m.group(2))


def _grid_distance(a, b):
    row_a, col_a = a
    row_b, col_b = b
    row_dist = abs(ord(row_a) - ord(row_b))
    col_dist = abs(col_a - col_b)
    return max(row_dist, col_dist)  # Chebyshev: diagonal neighbors count as adjacent


# Filenames always carry the batch as a prefix, sheet is optional in between:
#   "20260707_A1"       -> sheet=None, code="A1"   (single-sheet batch)
#   "20260804-S3-A1"    -> sheet="S3",  code="A1"   (multi-sheet batch)
# Sheet must be folded into the index key ("S3-A1") wherever it's present --
# batch 20260805 has two sheets' worth of photos in one directory, and S1's
# A1 / S3's A1 are different physical electrodes. Keying by bare "A1" would
# silently merge them (caught before any of that data was processed).
_FILENAME_ID_RE = re.compile(r"^\d{8}[-_](?:([A-Za-z0-9]+)[-_])?([A-Za-z]\d+)$")


def _parse_filename_identity(stem: str):
    """Returns (sheet_or_None, code) from a filename stem, or None if it
    doesn't match the "<batch>[-_][<sheet>-_]<code>" pattern.
    """
    m = _FILENAME_ID_RE.match(stem)
    if not m:
        return None
    sheet, code = m.group(1), m.group(2)
    return (sheet.upper() if sheet else None), code.upper()


def _split_key(key: str):
    """Inverse of building an index key: "S3-A1" -> ("S3", "A1"); "A1" -> (None, "A1")."""
    if "-" in key:
        sheet, code = key.split("-", 1)
        return sheet, code
    return None, key


def _index_grid_images(image_dir: str):
    """Map electrode key -> photo path(s). Key is "A1" for single-sheet
    batches, "S3-A1" for multi-sheet batches (see module note above on why
    sheet has to be part of the key). Some older single-sheet photos have a
    duplicate shot suffixed "_L" (second angle/lighting); both index under
    the same key, non-"_L" preferred when picking.
    """
    index = {}
    for path in sorted(Path(image_dir).glob("*")):
        if path.suffix.lower() not in _IMAGE_EXTS:
            continue
        stem = path.stem
        base = stem[:-2] if stem.upper().endswith("_L") else stem
        parsed = _parse_filename_identity(base)
        if parsed is None:
            continue
        sheet, code = parsed
        key = f"{sheet}-{code}" if sheet else code
        index.setdefault(key, []).append(path)
    return index


def _resolve_electrode_image(electrode_code: str, image_dir: str):
    """Returns (image_path, used_code, grid_distance) or (None, None, None)
    if electrode_code doesn't parse or image_dir has no grid-labeled photos.
    grid_distance is 0 for an exact match, >0 when a neighbor was substituted.
    electrode_code may be a bare code ("A1") or, for multi-sheet batches, a
    "<sheet>-<code>" composite ("S3-A1"); proxy search stays within the same
    sheet only -- a different sheet is a different fabrication run, not a
    nearby position.
    """
    requested = electrode_code.strip().upper()
    target_sheet, target_code = _split_key(requested)
    target = _parse_grid_code(target_code)
    if target is None:
        return None, None, None

    index = _index_grid_images(image_dir)
    if not index:
        return None, None, None

    if requested in index:
        nearest_key, distance = requested, 0
    else:
        same_sheet = [k for k in index if _split_key(k)[0] == target_sheet]
        if not same_sheet:
            return None, None, None

        # Tie-break same Chebyshev distance in favor of same-row/same-column
        # neighbors over diagonal ones (smaller Manhattan distance).
        def _sort_key(k):
            pos = _parse_grid_code(_split_key(k)[1])
            row_dist = abs(ord(target[0]) - ord(pos[0]))
            col_dist = abs(target[1] - pos[1])
            return (max(row_dist, col_dist), row_dist + col_dist)

        nearest_key = min(same_sheet, key=_sort_key)
        nearest_pos = _parse_grid_code(_split_key(nearest_key)[1])
        distance = _grid_distance(target, nearest_pos)

    candidates = sorted(index[nearest_key], key=lambda p: p.stem.upper().endswith("_L"))
    return str(candidates[0]), nearest_key, distance


def _enhance_contrast(arr: np.ndarray, saturate_pct: float) -> np.ndarray:
    """ImageJ Enhance-Contrast-style stretch: clip the extreme saturate_pct%
    of pixels at each end, then linearly rescale the rest to fill 0-255.

    Matters here because the raw luminance already technically spans 0-255
    (a dark background/border) while the electrode surface itself sits in a
    much narrower band (empirically ~159-255 in these photos) -- without
    this, that background eats a big share of the height axis and the
    actual print texture gets visually compressed into a thin slice of it.
    """
    lo, hi = np.percentile(arr, [saturate_pct, 100 - saturate_pct])
    if hi <= lo:
        return arr
    return np.clip((arr - lo) * (255.0 / (hi - lo)), 0, 255)


# Fractional (left, top, right, bottom) crop isolating just the printed
# electrode pads/traces, tuned by eye against reference_images/20260707/20260707_A1.bmp
# -- excludes the corner contact pads, background margins, and (importantly)
# a glare streak off the glossy substrate that otherwise reads as a fake
# height peak unrelated to print quality. Specific to this camera rig/framing;
# override crop_box if a future batch is shot at a different zoom/position.
DEFAULT_CROP_BOX = (0.32, 0.20, 0.78, 0.82)

# Working-electrode-only crop (the central carbon disc, excluding the
# surrounding counter-electrode ring, reference pad, and lead traces) --
# roughness should describe the working electrode specifically, since that's
# the surface that actually does the electrochemistry; the ring/leads are
# print quality of a different, less relevant feature. Located by circle
# detection (connected-component/Hough, not eyeballed) against real photos
# from each format:
#   - New format (20260804 on, single electrode filling ~the whole frame):
#     WORKING_ELECTRODE_CROP_BOX, confirmed centered on the disc in
#     20260804-S3-A1/A5 and 20260805-S1-A1.
#   - Old format (20260707, 2x2 cluster of 4 sub-electrodes per photo): see
#     sub_pad_working_electrode_crop_box, which insets further into whichever
#     quadrant sub_pad_crop_box already isolates.
# Same camera-rig caveat as DEFAULT_CROP_BOX -- re-tune if a future batch is
# shot at a different zoom/position.
WORKING_ELECTRODE_CROP_BOX = (0.31, 0.34, 0.69, 0.77)

# A single 20260707-style photo shows a 2x2 cluster of (up to) 4 distinct
# physical electrodes, not one -- confirmed against real photos: positions
# 1/2/3 (matching the CV filename suffix, e.g. "707-A5-1.csv") are
# top-left/top-right/bottom-left (corrected -- originally had 2 and 3
# swapped); position 4 (bottom-right) is always the unfilled Ag/AgCl
# reference pad, never carbon-inked, hence no CV file for it. Treating a
# whole photo as "one electrode" would average together three physically
# different prints. Newer batches (20260804 on) are one electrode per
# photo and don't need this.
_SUB_PAD_POSITIONS = ("1", "2", "3")


def sub_pad_crop_box(position: str, base_crop_box=DEFAULT_CROP_BOX):
    if position not in _SUB_PAD_POSITIONS:
        raise ValueError(f"Unknown sub-electrode position '{position}', expected one of {_SUB_PAD_POSITIONS}.")
    left, top, right, bottom = base_crop_box
    mx, my = (left + right) / 2, (top + bottom) / 2
    return {
        "1": (left, top, mx, my),
        "2": (mx, top, right, my),
        "3": (left, my, mx, bottom),
    }[position]


# Relative inset (as a fraction of a sub-pad quadrant, i.e. sub_pad_crop_box's
# output) isolating the working-electrode disc within it -- located via Hough
# circle detection against 20260707_A1.bmp's position "1" quadrant, then
# shrunk further inward (radius x0.85) to stay clear of the surrounding
# counter-electrode ring, which sits close enough to touch the disc in this
# format (unlike the newer format's photos, which have a visible gap).
_SUB_PAD_WORKING_ELECTRODE_CENTER = (0.51, 0.53)
_SUB_PAD_WORKING_ELECTRODE_RADIUS = (0.15, 0.145)


def sub_pad_working_electrode_crop_box(position: str, base_crop_box=DEFAULT_CROP_BOX):
    """Working-electrode-only crop for a 20260707-style photo: sub_pad_crop_box's
    quadrant, inset further to just the central disc. See the module-level
    WORKING_ELECTRODE_CROP_BOX note for why roughness should isolate this
    rather than the whole sub-electrode print (ring + disc + leads).
    """
    left, top, right, bottom = sub_pad_crop_box(position, base_crop_box)
    sw, sh = right - left, bottom - top
    cx, cy = _SUB_PAD_WORKING_ELECTRODE_CENTER
    rx, ry = _SUB_PAD_WORKING_ELECTRODE_RADIUS
    return (left + (cx - rx) * sw, top + (cy - ry) * sh, left + (cx + rx) * sw, top + (cy + ry) * sh)


def _luminance_grid(
    image_path: str,
    grid_size: int,
    enhance_contrast: bool = True,
    saturate_pct: float = 1.0,
    crop_box=None,
) -> np.ndarray:
    img = Image.open(image_path).convert("L")  # 8-bit, ITU-R 601-2 luminance (~ImageJ's weighted conversion)
    if crop_box is not None:
        w0, h0 = img.size
        left, top, right, bottom = crop_box
        img = img.crop((int(left * w0), int(top * h0), int(right * w0), int(bottom * h0)))
    arr = np.asarray(img, dtype=float)
    if enhance_contrast:
        arr = _enhance_contrast(arr, saturate_pct)
        img = Image.fromarray(arr.astype(np.uint8))
    w, h = img.size
    target_w = grid_size
    target_h = max(1, round(grid_size * h / w))
    img = img.resize((target_w, target_h), Image.LANCZOS)  # antialiased downsample, avoids aliasing spikes
    return np.asarray(img, dtype=float)


def _render_surface_plot(luminance: np.ndarray, out_path: Path, title: str, render_smooth_sigma: float = 2.5):
    # Smoothing + directional-light shading is for this render only -- QC
    # metrics (computed by the caller) use the raw, unsmoothed luminance so
    # a prettier picture never masks real per-pixel variance. Plain
    # plot_surface renders sharp ink/background edges as jagged flat-shaded
    # spikes rather than clean cliffs; a light Gaussian smooth plus a
    # LightSource hillshade (matching ImageJ's shaded-surface look) fixes that.
    smoothed = gaussian_filter(luminance, sigma=render_smooth_sigma)

    # Ink is dark (low luminance) but physically the deposited, raised
    # material; bare substrate is bright (high luminance) with nothing on
    # it. Height must track "amount of ink", not raw luminance, or ink
    # prints render as valleys and bare substrate as peaks -- backwards
    # from the physical reality. Inverting here (not upstream) keeps every
    # numeric roughness metric unchanged, since Ra/Rq/Rz/Rt are all based
    # on deviation from the mean, which a uniform inversion doesn't alter.
    height = smoothed.max() - smoothed

    ys, xs = np.mgrid[0:height.shape[0], 0:height.shape[1]]

    light = LightSource(azdeg=315, altdeg=45)
    rgb = light.shade(height, cmap=plt.get_cmap("viridis"), vert_exag=1.5, blend_mode="soft")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(xs, ys, height, facecolors=rgb, linewidth=0, antialiased=True, shade=False)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_zlabel("ink height (inverted luminance)")
    ax.set_title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _roughness_params(grid: np.ndarray, n_tiles: int = 5) -> dict:
    """Standard ISO-4287-style roughness parameters, computed on the
    luminance height map (see analyze_surface_topology's caveat on why
    these are luminance units, not calibrated physical roughness).
    """
    z = grid.astype(float)
    deviations = z - np.mean(z)

    ra = float(np.mean(np.abs(deviations)))        # arithmetic mean roughness
    rq = float(np.sqrt(np.mean(deviations ** 2)))   # RMS roughness (== population std dev)
    rt = float(np.max(z) - np.min(z))               # total peak-to-valley

    # Rz (simplified): split into an n_tiles x n_tiles grid, take peak-to-valley
    # per tile, average -- the standard ISO definition averages peak-to-valley
    # across multiple sampling lengths of a 1D profile; this is the natural 2D
    # analogue, and is more robust to a single outlier pixel than Rt alone.
    h, w = z.shape
    tile_h, tile_w = max(1, h // n_tiles), max(1, w // n_tiles)
    tile_ranges = [
        float(np.max(z[i:i + tile_h, j:j + tile_w]) - np.min(z[i:i + tile_h, j:j + tile_w]))
        for i in range(0, h - tile_h + 1, tile_h)
        for j in range(0, w - tile_w + 1, tile_w)
    ]
    rz = float(np.mean(tile_ranges)) if tile_ranges else rt

    return {
        "Ra": ra,
        "Rq": rq,
        "Rz": rz,
        "Rt": rt,
        "Rsk": float(skew(z.ravel())),      # skewness: asymmetry of the height distribution
        "Rku": float(kurtosis(z.ravel())),  # excess kurtosis: peakedness (0 == normal distribution)
    }


def analyze_surface_topology(
    image_path: str,
    grid_size: int = 100,
    max_luminance_cv: float = 0.25,
    enhance_contrast: bool = True,
    contrast_saturate_pct: float = 1.0,
    crop_box=WORKING_ELECTRODE_CROP_BOX,
) -> dict:
    """ImageJ-style surface plot: pixel luminance (8-bit, contrast-enhanced,
    cropped to just the working electrode's disc -- not the counter-electrode
    ring, reference pad, or lead traces) treated as height. Renders the 3D
    surface and reports roughness metrics -- an evenly-inked print should
    read as fairly flat; texture/streaking/isolated spikes (missed ink,
    contamination, glare) inflate the variance.

    Caveats, since this is a first pass and not a validated QC method:
      - Luminance responds to lighting/glare as much as to the print itself,
        so uneven lighting across shots will read as "roughness" too --
        this is only meaningful for photos taken under consistent lighting.
        crop_box (default WORKING_ELECTRODE_CROP_BOX) isolates the working
        electrode's disc, but won't help with glare that falls inside that
        cropped region on a given shot.
      - max_luminance_cv is a weak signal, confirmed empirically, not just
        suspected: across a real 48-electrode batch, luminance_cv ranged
        only 0.609-0.660 (essentially no spread) while every single
        electrode sat above the old default (0.25), flagging 100% of the
        batch with zero discriminating power. It's a whole-frame aggregate
        (mean spread of dark vs. bright pixels), so it can't tell "evenly
        lit frame" from "one specific localized defect" the way pixel-by-
        pixel comparison can. Prefer compare_to_batch_reference (SSIM
        against the batch's own average pattern) for actually finding
        which electrodes are unusual -- on that same batch it flagged a
        differentiated 8/48 (17%) and correctly caught the electrodes
        independently confirmed to have real electrical failures. Treat
        this flag as a rough, secondary descriptive note at best.
      - WORKING_ELECTRODE_CROP_BOX is tuned for the current single-electrode-
        per-photo framing (reference_images/20260804 on). For a 20260707-style
        2x2-cluster photo, pass sub_pad_working_electrode_crop_box(position)
        instead to isolate one sub-electrode's disc. Pass crop_box=None for
        the full, un-cropped frame, or your own (left, top, right, bottom)
        fractions, if a future batch is shot differently.
      - The "roughness" metrics (Ra/Rq/Rz/Rt/Rsk/Rku) are the standard
        ISO-4287 formulas, but computed on luminance (0-255), not physical
        height (nm/um) -- there's no calibration from pixel brightness to
        real surface height without actual profilometry/AFM data. Treat
        these as a relative, photo-only proxy for comparing electrodes
        against each other under matched lighting, not as trustworthy
        absolute roughness values for a spec sheet or paper.
    """
    if not os.path.exists(image_path):
        return {"status": "error", "message": f"Image not found: {image_path}"}

    luminance = _luminance_grid(image_path, grid_size, enhance_contrast, contrast_saturate_pct, crop_box)
    mean_l = float(np.mean(luminance))
    std_l = float(np.std(luminance))
    cv = (std_l / mean_l) if mean_l > 0 else None
    roughness = _roughness_params(luminance)

    out_path = SURFACE_PLOTS_DIR / Path(image_path).parent.name / f"{Path(image_path).stem}_surface.png"
    _render_surface_plot(luminance, out_path, title=Path(image_path).name)

    flags = []
    if cv is not None and cv > max_luminance_cv:
        flags.append(
            f"luminance CV = {cv:.3f} exceeds {max_luminance_cv:.3f} -- weak, low-specificity signal "
            f"(confirmed to flag ~100% of a real batch in testing); use compare_to_batch_reference "
            f"for an actually discriminating outlier check, don't treat this alone as meaningful."
        )

    return {
        "status": "warn" if flags else "pass",
        "plot_path": str(out_path),
        "metrics": {
            "mean_luminance": mean_l,
            "luminance_cv": cv,
        },
        # ISO-4287-style roughness parameters (Ra/Rq/Rz/Rt/Rsk/Rku), in
        # luminance units -- Rq is the same value as the old std_luminance
        # field, Rt the same as the old luminance_range field, now with
        # proper names plus the additional Ra/Rz/Rsk/Rku parameters.
        "roughness_luminance_units": roughness,
        "flags": flags,
    }


def qc_sensor_image(
    image_path: str = "",
    electrode_code: str = "",
    image_dir: str = "",
    batch: str = "",
    context: str = "",
    include_surface_analysis: bool = False,
    surface_grid_size: int = 100,
    max_luminance_cv: float = 0.25,
    surface_enhance_contrast: bool = True,
    surface_contrast_saturate_pct: float = 1.0,
    surface_crop_box: list = None,
) -> dict:
    # batch is a tolerated alias for image_dir -- see compare_to_batch_reference
    # for why: the model has repeatedly passed a bare batch string instead
    # of a real directory path, which would otherwise crash with an
    # unexpected-keyword-argument error rather than silently doing nothing.
    image_dir = image_dir or (f"reference_images/{batch}" if batch else "")
    proxy_info = {}

    if not image_path:
        if not electrode_code or not image_dir:
            return {
                "status": "error",
                "message": "Provide either image_path, or both electrode_code and image_dir.",
            }
        resolved_path, used_code, distance = _resolve_electrode_image(electrode_code, image_dir)
        if resolved_path is None:
            return {
                "status": "error",
                "message": f"No grid-labeled photos found for '{electrode_code}' in {image_dir}.",
            }
        image_path = resolved_path
        if distance > 0:
            proxy_info = {
                "proxy_image": True,
                "requested_electrode": electrode_code.strip().upper(),
                "used_electrode": used_code,
                "grid_distance": distance,
            }

    if not os.path.exists(image_path):
        return {"status": "error", "message": f"Image not found: {image_path}"}

    context_line = f"\nAdditional context: {context}\n" if context else ""
    prompt = _PROMPT_TEMPLATE.format(context_line=context_line)

    mime_type = _MIME_TYPES.get(Path(image_path).suffix.lower(), "image/jpeg")
    response = client().models.generate_content(
        model=MODEL,
        contents=[prompt, types.Part.from_bytes(data=Path(image_path).read_bytes(), mime_type=mime_type)],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )

    content = response.text
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return {
            "status": "error",
            "message": "Vision model did not return valid JSON.",
            "raw": content,
        }

    parsed.setdefault("defects", [])
    parsed.setdefault("notes", "")
    parsed["image_path"] = image_path

    if proxy_info:
        parsed.update(proxy_info)
        parsed["notes"] = (
            f"PROXY IMAGE: no photo exists for electrode {proxy_info['requested_electrode']}; "
            f"showing nearest photographed neighbor {proxy_info['used_electrode']} "
            f"(grid distance {distance}) instead. " + parsed["notes"]
        )

    if include_surface_analysis:
        surface = analyze_surface_topology(
            image_path, surface_grid_size, max_luminance_cv,
            surface_enhance_contrast, surface_contrast_saturate_pct,
            tuple(surface_crop_box) if surface_crop_box else WORKING_ELECTRODE_CROP_BOX,
        )
        parsed["surface_analysis"] = surface
        rank = {"pass": 0, "warn": 1, "fail": 2}
        if surface.get("status") in rank and rank[surface["status"]] > rank.get(parsed["status"], 0):
            parsed["status"] = surface["status"]

    try:
        if electrode_code:
            note_code, note_batch = electrode_code.strip().upper(), extract_batch_date(Path(image_dir).name)
        else:
            note_code, note_batch = extract_grid_code(Path(image_path).stem), extract_batch_date(Path(image_path).parent.name)
        if note_code:
            note_metrics = {"defects": parsed.get("defects", []), "proxy_image": parsed.get("proxy_image", False)}
            if include_surface_analysis:
                # Persist roughness too, not just the vision defect flags --
                # this is the actual data a visual-vs-CV-performance
                # correlation needs, and it was silently going uncaptured.
                note_metrics["roughness"] = parsed.get("surface_analysis", {}).get("roughness_luminance_units")
            append_qc_result(
                note_batch, note_code, tool="image_qc", status=parsed["status"],
                metrics=note_metrics,
                file_ref=image_path, file_kind="image_paths",
            )
    except Exception:
        pass  # note-linking is best-effort, never blocks the QC result

    return parsed
