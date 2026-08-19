"""
Reference-diff tool: unsupervised "does this look like the rest of the
batch" QC for electrode photos.

Luminance-as-height (tools/image_qc.py's analyze_surface_topology) treats a
single photo in isolation, which conflates real print texture with
reflectivity, lighting, and glare -- there's no ground truth to compare
against. This tool instead builds a per-pixel average (and std-dev) image
across every electrode in a batch, on the assumption that most of a batch is
representative, and flags electrodes whose print deviates most from that
average as outliers worth a manual look. This is how real automated optical
inspection (AOI) systems approach the problem, just without a labeled
"golden" reference -- swap in a real one (build_batch_reference against a
directory containing only known-good photos) once you've hand-verified some.

Reuses tools/image_qc.py's crop/contrast pipeline so the two tools are
directly comparable -- same ROI, same preprocessing.

Chip placement isn't pixel-identical shot to shot, so a fixed crop alone
doesn't guarantee alignment -- confirmed empirically (see conversation): a
first pass without registration gave every electrode a uniformly low,
narrow-spread SSIM (~0.17-0.33) against the batch average, which is the
signature of registration noise dominating, not real print variance.
Each image is now registered (translation-only phase cross-correlation)
against the batch average before comparison, in two passes -- align to an
arbitrary first image, re-align to the resulting preliminary average -- and
a margin is trimmed post-alignment to drop shift-induced edge artifacts.
Translation-only alignment used to leave actual rotation (not just a shift)
between shots uncorrected -- confirmed to matter a lot in practice (an
uncorrected 20-degree tilt on a real photo dropped its SSIM from a genuine
0.618 to 0.197, i.e. a false outlier flag purely from orientation). The
target image's rotation is now auto-detected against the reference before
alignment (_estimate_rotation: a coarse-to-fine angle search on the
downsampled luminance grid, scored by how well each candidate rotation
aligns -- cheap, since it operates on a ~150px grid, not the full photo).
rotation_degrees still exists as an optional manual seed for gross
correction outside the auto-search's +-20 degree range; the auto-detected
angle is reported in the result either way so a correction is never silent.

Which batch/sheet(s) a photo compares against, by default: if the caller
gives image_dir/batch explicitly, the average is built from that one
batch+sheet only, same as always. If image_dir/batch is omitted, the batch
identity itself matters less than which electrodes were actually printed
the same way -- so the reference pools together every batch/sheet on record
(via electrode_notes' set_batch_metadata) that shares the target's own
carbon ink formula, giving a bigger, more representative "average electrode"
than any one small batch alone. Requires that ink formula actually be
recorded for the target's batch/sheet; if it isn't, compare_to_batch_reference
says so plainly rather than silently falling back to a single-batch average
or guessing a formula.
"""

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import rotate as nd_rotate
from scipy.ndimage import shift as nd_shift
from skimage.metrics import structural_similarity
from skimage.registration import phase_cross_correlation
from skimage.transform import resize as sk_resize

from tools.electrode_notes import (
    BATCH_INFO_FILENAME, NOTES_DIR, _load_note, append_qc_result, extract_batch_date,
    extract_grid_code, get_batch_metadata,
)
from tools.image_qc import (
    _MIME_TYPES, DEFAULT_CROP_BOX, _index_grid_images, _luminance_grid, _parse_filename_identity,
    _resolve_electrode_image, _split_key, _validate_image_file,
)
from tools.framing_qc import check_photo_framing

REFERENCE_DIFF_DIR = Path("reference_diff")
GRID_SIZE = 150  # finer than image_qc's 3D-plot grid; no mesh-rendering cost to worry about here
ALIGN_MARGIN_FRAC = 0.10  # trimmed off all sides post-alignment to drop shift-induced edge artifacts


def _match_shape(grid: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Resize grid to target_shape if it doesn't already match.

    _luminance_grid derives each image's grid height from its own aspect
    ratio (grid_size fixed, height computed from w/h), so two photos shot on
    a different camera/rig -- e.g. a researcher's own uploaded photo being
    compared against a batch shot differently, an explicitly supported use
    case -- can produce differently-shaped grids. phase_cross_correlation
    and np.mean/structural_similarity all require matching shapes, so
    without this a mismatched pair crashes instead of comparing.
    """
    if grid.shape == target_shape:
        return grid
    return sk_resize(grid, target_shape, preserve_range=True, anti_aliasing=True)


ROTATION_SEARCH_RANGE_DEG = 20.0
ROTATION_COARSE_STEP_DEG = 2.0
ROTATION_FINE_STEP_DEG = 0.25


def _estimate_rotation(raw_grid: np.ndarray, reference: np.ndarray) -> float:
    """Find the rotation (degrees) that best aligns raw_grid to reference,
    so a photo shot slightly askew in the jig doesn't score poorly on SSIM
    for orientation reasons that have nothing to do with print quality.

    Operates on the already-downsampled luminance grid (~150px), not the
    full-resolution photo, so a several-dozen-angle search costs
    milliseconds rather than seconds. Coarse-to-fine: a wide sweep at
    ROTATION_COARSE_STEP_DEG first, then a narrow refinement around the
    best coarse angle at ROTATION_FINE_STEP_DEG -- cheaper than one dense
    sweep and still finds the optimum between coarse steps.

    +-ROTATION_SEARCH_RANGE_DEG covers a shot tilted in the jig, not an
    arbitrary orientation (e.g. someone photographing the electrode
    upside down) -- that failure mode hasn't been observed in practice,
    and a full 0-360 search would cost more and risk false positives from
    the grid's own approximate symmetry.
    """
    matched = _match_shape(raw_grid, reference.shape)

    def score(angle: float) -> float:
        rotated = matched if angle == 0 else nd_rotate(matched, angle, reshape=False, mode="nearest", order=1)
        aligned = _align(rotated, reference)
        return float(structural_similarity(aligned, reference, data_range=255))

    best_angle, best_score = 0.0, score(0.0)
    coarse_angles = np.arange(-ROTATION_SEARCH_RANGE_DEG, ROTATION_SEARCH_RANGE_DEG + ROTATION_COARSE_STEP_DEG, ROTATION_COARSE_STEP_DEG)
    for angle in coarse_angles:
        angle = float(angle)
        if angle == 0.0:
            continue
        s = score(angle)
        if s > best_score:
            best_angle, best_score = angle, s

    fine_angles = np.arange(best_angle - ROTATION_COARSE_STEP_DEG, best_angle + ROTATION_COARSE_STEP_DEG + ROTATION_FINE_STEP_DEG, ROTATION_FINE_STEP_DEG)
    for angle in fine_angles:
        angle = float(angle)
        s = score(angle)
        if s > best_score:
            best_angle, best_score = angle, s

    return best_angle


def _align(grid: np.ndarray, reference: np.ndarray) -> np.ndarray:
    grid = _match_shape(grid, reference.shape)
    shift_yx = phase_cross_correlation(reference, grid, upsample_factor=10)[0]
    return nd_shift(grid, shift_yx, mode="nearest")


def _trim_margin(grid: np.ndarray, frac: float = ALIGN_MARGIN_FRAC) -> np.ndarray:
    my, mx = int(grid.shape[0] * frac), int(grid.shape[1] * frac)
    return grid[my:grid.shape[0] - my, mx:grid.shape[1] - mx]

REFERENCE_DIFF_SCHEMA = {
    "type": "function",
    "function": {
        "name": "compare_to_batch_reference",
        "description": (
            "Compare one electrode photo against the average of other electrodes (unsupervised "
            "-- no labeled 'good' example needed) and report how much it deviates (SSIM "
            "similarity score + percentile rank within the reference pool). Large deviation "
            "flags a print worth a manual look. Omit image_dir/batch to auto-pool the reference "
            "from every batch/sheet on record sharing the target's own carbon ink formula "
            "(bigger, more representative average than one batch alone) -- requires that ink "
            "formula be recorded via set_batch_metadata first, or it errors rather than "
            "guessing. Pass image_dir/batch to instead compare against just that one batch's "
            "own average. Builds and caches the reference average on first use; pass "
            "force_rebuild=true after adding new photos or updating ink-formula metadata."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "image_dir": {
                    "type": "string",
                    "description": (
                        "Directory of grid-labeled electrode photos to scope the reference average "
                        "to, e.g. reference_images/20260707. Or pass batch instead. Omit both to "
                        "auto-pool by the target's carbon ink formula instead (see tool description) "
                        "-- requires image_path (a photo whose own batch/sheet can be read from its "
                        "filename) in that case, not electrode_code."
                    ),
                },
                "batch": {
                    "type": "string",
                    "description": "Batch date, e.g. '20260707' -- shorthand for image_dir='reference_images/<batch>'. Use this or image_dir.",
                },
                "sheet": {
                    "type": "string",
                    "description": (
                        "Which sheet's average to compare against, e.g. 'S3', for a batch with "
                        "multiple sheets. Normally inferred from the filename, but a researcher's "
                        "own uploaded photo won't have a filename that encodes one -- pass this "
                        "explicitly in that case if the batch has more than one sheet (auto-resolved "
                        "if it only has one)."
                    ),
                },
                "electrode_code": {
                    "type": "string",
                    "description": (
                        "Grid position to check, e.g. 'E5'. For a batch with multiple sheets in "
                        "one directory, use the '<sheet>-<code>' form, e.g. 'S3-A1' -- a bare "
                        "code is ambiguous there and will error. Use this or image_path."
                    ),
                },
                "image_path": {
                    "type": "string",
                    "description": "Direct path to the photo to check, if not using electrode_code.",
                },
                "crop_box": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": (
                        "[left, top, right, bottom] as fractions 0-1, same as image_qc's "
                        "surface_crop_box. Defaults to the box tuned for the original 20260707 "
                        "batch's framing -- pass [0,0,1,1] (or your own box) for a differently "
                        "framed batch, e.g. one already tightly cropped to a single electrode."
                    ),
                },
                "force_rebuild": {
                    "type": "boolean",
                    "description": "Recompute the reference average instead of using the cached one. Default false.",
                },
                "rotation_degrees": {
                    "type": "number",
                    "description": (
                        "Usually not needed -- the tool auto-detects and corrects a tilted target "
                        "photo's orientation against the reference before comparing (search range "
                        "+-20 degrees). Only pass this for a manual seed on a rotation larger than "
                        "that, or a known-exact angle you don't want auto-detection to second-guess. "
                        "Counterclockwise positive, only applies to the photo being checked, not the "
                        "batch reference average. Default 0."
                    ),
                },
            },
        },
    },
}


def _batch_cache_dir(image_dir: str, sheet: str = None) -> Path:
    return REFERENCE_DIFF_DIR / Path(image_dir).name / (sheet or "_all")


def _ink_cache_dir(carbon_ink_formula: str) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "_", carbon_ink_formula.strip().lower()).strip("_") or "unknown"
    return REFERENCE_DIFF_DIR / "_by_ink" / slug


def _find_sheets_by_carbon_ink(carbon_ink_formula: str) -> list:
    """Every (batch, sheet_number) on record (via set_batch_metadata) whose
    carbon_ink_formula matches, case/whitespace-insensitive. Scans
    electrode_notes/*/_batch_info.md directly rather than going through
    get_batch_metadata per-batch, since there's no index of "every batch
    that exists" to iterate otherwise.
    """
    target = carbon_ink_formula.strip().lower()
    matches = []
    if not NOTES_DIR.exists():
        return matches
    for path in sorted(NOTES_DIR.glob(f"*/{BATCH_INFO_FILENAME}")):
        try:
            meta, _ = _load_note(path)
        except Exception:
            continue
        batch = meta.get("batch", path.parent.name)
        for sheet_number, fields in meta.get("sheets", {}).items():
            formula = (fields.get("carbon_ink_formula") or "").strip().lower()
            if formula and formula == target:
                matches.append((batch, sheet_number))
    return matches


def build_batch_reference(sources: list, cache_dir: Path, crop_box=DEFAULT_CROP_BOX, force_rebuild: bool = False) -> dict:
    """sources is a list of (image_dir, sheet) pairs pooled together into one
    average -- normally just one pair (compare against a single batch/sheet),
    or several when compare_to_batch_reference auto-pools every batch/sheet
    sharing a target's carbon ink formula (see module docstring). sheet
    restricts each source dir to one sheet's photos -- required for a
    directory with multiple sheets (e.g. batch 20260805 has S1 and S3), since
    different sheets can use different ink formulas; averaging across them
    within one source would produce a meaningless "typical" reference and
    corrupt every SSIM score. Leave a source's sheet=None for a single-sheet
    batch (all entries share the same, sheet-less key already).
    """
    mean_path = cache_dir / "mean.npy"
    std_path = cache_dir / "std.npy"
    align_ref_path = cache_dir / "align_reference.npy"
    rotation_ref_path = cache_dir / "rotation_reference.npy"
    scores_path = cache_dir / "scores.json"

    if not force_rebuild and all(p.exists() for p in (mean_path, std_path, align_ref_path, rotation_ref_path, scores_path)):
        return {
            "mean": np.load(mean_path),
            "std": np.load(std_path),
            "align_reference": np.load(align_ref_path),
            "rotation_reference": np.load(rotation_ref_path),
            "scores": json.loads(scores_path.read_text()),
        }

    codes, raw_grids = [], []
    missing = []
    for image_dir, sheet in sources:
        full_index = _index_grid_images(image_dir)  # {"A1" or "S3-A1": [paths]}
        index = {k: v for k, v in full_index.items() if _split_key(k)[0] == sheet}
        if not index:
            missing.append((image_dir, sheet))
            continue
        # Prefix with the source dir so pooling multiple batches can't collide
        # two different physical electrodes that happen to share a grid code.
        prefix = Path(image_dir).name
        for key, paths in sorted(index.items()):
            primary = sorted(paths, key=lambda p: p.stem.upper().endswith("_L"))[0]  # skip "_L" duplicate shots
            codes.append(f"{prefix}:{key}")
            raw_grids.append(_luminance_grid(str(primary), GRID_SIZE, True, 1.0, crop_box))

    if not raw_grids:
        return {"error": f"No grid-labeled photos found for any of the given source(s): {sources}."}

    # Two-pass registration (see module docstring for why this is necessary):
    # align everything to an arbitrary first image, then re-align to the
    # resulting preliminary average -- more stable than aligning to one shot.
    # This bootstraps align_reference only. Every image's *actual* aligned
    # version (used for scoring, here and for every future single-image
    # query) then goes through one identical alignment pass against that
    # frozen reference -- phase correlation isn't perfectly stable across
    # slightly different reference frames, so build-time and query-time
    # scoring must share the exact same reference and pass count, or the
    # same file can silently score differently depending on which code
    # path touched it (this was a real, confirmed bug, not a hypothetical).
    pass1_aligned = [_align(g, raw_grids[0]) for g in raw_grids]
    align_reference = np.mean(pass1_aligned, axis=0)

    aligned_grids = [_align(g, align_reference) for g in raw_grids]
    trimmed = [_trim_margin(g) for g in aligned_grids]
    mean_img = np.mean(trimmed, axis=0)
    std_img = np.std(trimmed, axis=0)

    # raw_grids[0], kept unaveraged and in the same frame as align_reference
    # (see above), specifically for rotation estimation -- confirmed
    # empirically that averaging across many photos, each with its own small
    # uncorrected rotational jitter, smears out exactly the oriented texture
    # a rotation search needs: searching against align_reference/mean_img
    # picked a spurious angle nowhere near a known-injected 15-degree tilt's
    # true correction, while searching against this single sharp photo
    # recovered it cleanly (SSIM 0.92 at the true angle vs <0.4 five degrees
    # off either side). Never used for the actual SSIM score, only to find
    # the rotation angle before the real translation-alignment pass.
    rotation_reference = raw_grids[0]

    scores = {
        code: float(structural_similarity(grid, mean_img, data_range=255))
        for code, grid in zip(codes, trimmed)
    }

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(mean_path, mean_img)
    np.save(std_path, std_img)
    np.save(align_ref_path, align_reference)
    np.save(rotation_ref_path, rotation_reference)
    scores_path.write_text(json.dumps(scores, indent=2))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(mean_img, cmap="viridis")
    title = f"reference average, n={len(codes)} across {len(sources) - len(missing)} source(s) (registered)"
    ax.set_title(title)
    fig.savefig(cache_dir / "mean.png", dpi=120)
    plt.close(fig)

    result = {
        "mean": mean_img, "std": std_img, "align_reference": align_reference,
        "rotation_reference": rotation_reference, "scores": scores,
    }
    if missing:
        result["missing_sources"] = [
            {"image_dir": d, "sheet": s} for d, s in missing
        ]  # recorded in the note-linking caller's flags, not persisted to cache
    return result


def compare_to_batch_reference(
    image_dir: str = "",
    electrode_code: str = "",
    image_path: str = "",
    batch: str = "",
    sheet: str = None,
    crop_box=DEFAULT_CROP_BOX,
    force_rebuild: bool = False,
    rotation_degrees: float = 0.0,
) -> dict:
    # batch is a tolerated alias for image_dir -- the model has repeatedly
    # passed a bare batch string (matching the convention every OTHER tool
    # in this suite uses) instead of the actual directory path this
    # function needs, which silently errored instead of comparing anything.
    # Resolving it here is more reliable than depending on prompting alone.
    image_dir = image_dir or (f"reference_images/{batch}" if batch else "")
    # Captured before the electrode_code+image_dir catalog lookup below can
    # overwrite image_path -- True only for a direct upload (expected to show
    # one electrode), same distinction qc_sensor_image's gate makes.
    expect_single_electrode = bool(image_path)

    if not image_path:
        if not electrode_code or not image_dir:
            return {
                "status": "error",
                "message": (
                    "Provide either image_path, or both electrode_code and image_dir/batch. "
                    "image_dir/batch may be omitted only when using image_path directly -- the "
                    "comparison then auto-pools every batch/sheet sharing that photo's recorded "
                    "carbon ink formula, instead of just one batch's average."
                ),
            }
        image_path, used_code, _ = _resolve_electrode_image(electrode_code, image_dir)
        if image_path is None:
            return {"status": "error", "message": f"No grid-labeled photos found for '{electrode_code}' in {image_dir}."}
        electrode_code = used_code
        if sheet is None:
            sheet = _split_key(used_code)[0]
    elif sheet is None:
        parsed = _parse_filename_identity(Path(image_path).stem)
        sheet = parsed[0] if parsed else None

    pooled_by_ink = not image_dir
    reference_scope = {}

    if pooled_by_ink:
        target_batch = extract_batch_date(Path(image_path).stem)
        meta = get_batch_metadata(target_batch, sheet)
        carbon_ink = (meta.get("metadata") or {}).get("carbon_ink_formula") if meta.get("status") == "ok" else None
        if not carbon_ink:
            return {
                "status": "error",
                "message": (
                    f"No image_dir/batch given, and no carbon ink formula is recorded for batch "
                    f"'{target_batch}'" + (f" sheet '{sheet}'" if sheet else "") + " (via "
                    "set_batch_metadata) -- can't auto-pool a reference by ink formula. Record "
                    "the formula first, or pass image_dir/batch to compare against one batch's "
                    "own average instead."
                ),
            }
        matches = _find_sheets_by_carbon_ink(carbon_ink)
        if not matches:
            return {
                "status": "error",
                "message": f"No batch/sheet on record uses carbon ink formula '{carbon_ink}' -- not even {target_batch}'s own (check set_batch_metadata).",
            }
        sources = [(f"reference_images/{b}", s) for b, s in matches]
        cache_dir = _ink_cache_dir(carbon_ink)
        reference_scope = {"mode": "carbon_ink_formula", "carbon_ink_formula": carbon_ink, "batches_sheets": matches}
    else:
        if sheet is None:
            # Filename didn't encode a sheet -- e.g. a researcher's own upload
            # with a generic name, not one of the batch's own cataloged files.
            # If the batch only has one sheet there's no real ambiguity, so use
            # it rather than erroring; only ask for sheet explicitly when the
            # batch genuinely has more than one and we can't tell which.
            available_sheets = sorted({s for s in (_split_key(k)[0] for k in _index_grid_images(image_dir)) if s is not None})
            if len(available_sheets) == 1:
                sheet = available_sheets[0]
            elif len(available_sheets) > 1:
                return {
                    "status": "error",
                    "message": (
                        f"Can't tell which sheet to compare against in {image_dir} (found {available_sheets}) "
                        "-- the image's filename doesn't encode one. Pass sheet explicitly."
                    ),
                }
        sources = [(image_dir, sheet)]
        cache_dir = _batch_cache_dir(image_dir, sheet)
        reference_scope = {"mode": "single_batch", "image_dir": image_dir, "sheet": sheet}

    error = _validate_image_file(image_path)
    if error:
        return {"status": "error", "message": error}

    # Framing-QC gate, same as qc_sensor_image's -- refuses to build/compare
    # against a photo that's unusable to begin with (blurred, wrong subject,
    # multiple electrodes when one was expected, etc.) rather than quietly
    # computing an SSIM score against garbage input. This tool is reachable
    # directly by the agent (not only via qc_sensor_image), so it needs its
    # own gate rather than relying on qc_sensor_image's -- confirmed this was
    # a real gap, not a hypothetical one: an early version of this feature
    # only gated qc_sensor_image, and a bad photo could still reach a full
    # SSIM comparison through this tool untouched.
    mime_type = _MIME_TYPES.get(Path(image_path).suffix.lower(), "image/jpeg")
    framing = check_photo_framing(
        image_bytes=Path(image_path).read_bytes(), mime_type=mime_type,
        expect_single_electrode=expect_single_electrode,
    )
    if not framing["proceed"]:
        return {
            "status": "framing_rejected",
            "framing_ok": False,
            "issues": framing["issues"],
            "user_message": framing["user_message"],
            "electrode_code": electrode_code or None,
            "measurements": framing.get("measurements"),
        }

    try:
        reference = build_batch_reference(sources, cache_dir, crop_box, force_rebuild)
    except ValueError as e:
        return {"status": "error", "message": str(e)}
    if "error" in reference:
        return {"status": "error", "message": reference["error"]}
    mean_img, align_reference, batch_scores = reference["mean"], reference["align_reference"], reference["scores"]
    rotation_reference = reference["rotation_reference"]

    try:
        # rotation_degrees applies only to this target image, not the
        # reference pool built above -- it corrects one photo (e.g. a
        # researcher's own tilted upload), not a uniform correction across a
        # whole batch of otherwise-consistent cataloged photos.
        target_raw = _luminance_grid(image_path, GRID_SIZE, True, 1.0, crop_box, rotation_degrees)
    except ValueError as e:
        return {"status": "error", "message": str(e)}

    # Auto-detect any residual tilt against the reference orientation --
    # runs regardless of whether rotation_degrees was given, since a manual
    # value is a coarse seed, not guaranteed exact. Comes out ~0 when the
    # target is already well-aligned, so this is a no-op in the common case.
    auto_rotation = _estimate_rotation(target_raw, rotation_reference)
    if auto_rotation:
        target_raw = nd_rotate(_match_shape(target_raw, align_reference.shape), auto_rotation, reshape=False, mode="nearest", order=1)

    target = _trim_margin(_align(target_raw, align_reference))  # same registration as the reference average
    ssim_score, ssim_map = structural_similarity(target, mean_img, data_range=255, full=True)

    ranked = sorted(batch_scores.values())
    percentile = float(np.searchsorted(ranked, ssim_score) / len(ranked) * 100)

    diff_path = cache_dir / f"{Path(image_path).stem}_diff.png"
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    axes[0].imshow(target, cmap="viridis"); axes[0].set_title("this electrode")
    axes[1].imshow(mean_img, cmap="viridis"); axes[1].set_title(f"reference average (n={len(batch_scores)})")
    im = axes[2].imshow(1 - ssim_map, cmap="inferno", vmin=0, vmax=1)
    axes[2].set_title("dissimilarity (bright = deviates)")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{Path(image_path).name} -- SSIM={ssim_score:.3f} (percentile {percentile:.0f} within reference pool)")
    diff_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(diff_path, dpi=120)
    plt.close(fig)

    flags = []
    if percentile < 15:
        flags.append(
            f"OUTLIER: SSIM={ssim_score:.3f} puts this electrode in the bottom {percentile:.0f}% of the "
            f"reference pool by similarity to the average print -- worth a manual look."
        )
    if reference.get("missing_sources"):
        flags.append(
            f"{len(reference['missing_sources'])} recorded batch/sheet(s) sharing this ink formula had no "
            f"photos on disk and were skipped: {reference['missing_sources']}."
        )

    status = "warn" if flags else "pass"

    try:
        note_code = (electrode_code or extract_grid_code(Path(image_path).stem) or "").upper()
        note_batch = extract_batch_date(Path(image_path).stem) if pooled_by_ink else extract_batch_date(Path(image_dir).name)
        if note_code:
            append_qc_result(
                note_batch, note_code, tool="compare_to_batch_reference", status=status,
                metrics={"ssim_vs_batch_average": ssim_score, "batch_percentile": percentile},
                file_ref=image_path, file_kind="image_paths",
            )
    except Exception:
        pass  # note-linking is best-effort, never blocks the QC result

    return {
        "status": status,
        "electrode_code": electrode_code or None,
        "reference_scope": reference_scope,
        "source_image_path": str(image_path),
        "diff_plot_path": str(diff_path),
        "rotation_degrees_requested": rotation_degrees,
        "rotation_degrees_auto_detected": auto_rotation,
        "metrics": {
            "ssim_vs_batch_average": ssim_score,
            "batch_percentile": percentile,
            "batch_n": len(batch_scores),
        },
        "flags": flags,
    }
