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
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import shift as nd_shift
from skimage.metrics import structural_similarity
from skimage.registration import phase_cross_correlation

from tools.electrode_notes import append_qc_result, extract_batch_date, extract_grid_code
from tools.image_qc import (
    DEFAULT_CROP_BOX, _index_grid_images, _luminance_grid, _parse_filename_identity,
    _resolve_electrode_image, _split_key,
)

REFERENCE_DIFF_DIR = Path("reference_diff")
GRID_SIZE = 150  # finer than image_qc's 3D-plot grid; no mesh-rendering cost to worry about here
ALIGN_MARGIN_FRAC = 0.10  # trimmed off all sides post-alignment to drop shift-induced edge artifacts


def _align(grid: np.ndarray, reference: np.ndarray) -> np.ndarray:
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
            "Compare one electrode photo against the average of every other electrode in "
            "its batch (unsupervised -- no labeled 'good' example needed) and report how much "
            "it deviates (SSIM similarity score + percentile rank within the batch). Large "
            "deviation flags a print worth a manual look. Builds and caches the batch average "
            "on first use for a given image_dir; pass force_rebuild=true after adding new "
            "photos to that directory."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "image_dir": {
                    "type": "string",
                    "description": "Directory of grid-labeled electrode photos forming the batch, e.g. reference_images/20260707. Or pass batch instead.",
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
                    "description": "Recompute the batch average instead of using the cached one. Default false.",
                },
            },
            "required": ["image_dir"],
        },
    },
}


def _batch_cache_dir(image_dir: str, sheet: str = None) -> Path:
    return REFERENCE_DIFF_DIR / Path(image_dir).name / (sheet or "_all")


def build_batch_reference(image_dir: str, sheet: str = None, crop_box=DEFAULT_CROP_BOX, force_rebuild: bool = False) -> dict:
    """sheet restricts to one sheet's photos -- required for a directory
    with multiple sheets (e.g. batch 20260805 has S1 and S3), since
    different sheets can use different ink formulas; averaging across them
    would produce a meaningless "typical" reference and corrupt every SSIM
    score. Leave sheet=None for a single-sheet batch (all entries share the
    same, sheet-less key already).
    """
    cache_dir = _batch_cache_dir(image_dir, sheet)
    mean_path = cache_dir / "mean.npy"
    std_path = cache_dir / "std.npy"
    align_ref_path = cache_dir / "align_reference.npy"
    scores_path = cache_dir / "scores.json"

    if not force_rebuild and all(p.exists() for p in (mean_path, std_path, align_ref_path, scores_path)):
        return {
            "mean": np.load(mean_path),
            "std": np.load(std_path),
            "align_reference": np.load(align_ref_path),
            "scores": json.loads(scores_path.read_text()),
        }

    full_index = _index_grid_images(image_dir)  # {"A1" or "S3-A1": [paths]}
    index = {k: v for k, v in full_index.items() if _split_key(k)[0] == sheet}
    if not index:
        available = sorted({_split_key(k)[0] for k in full_index}) if full_index else []
        return {"error": (
            f"No grid-labeled photos for sheet '{sheet}' in {image_dir}."
            + (f" Sheets found there: {available}." if available else "")
        )}

    codes, raw_grids = [], []
    for key, paths in sorted(index.items()):
        primary = sorted(paths, key=lambda p: p.stem.upper().endswith("_L"))[0]  # skip "_L" duplicate shots
        codes.append(key)
        raw_grids.append(_luminance_grid(str(primary), GRID_SIZE, True, 1.0, crop_box))

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

    scores = {
        code: float(structural_similarity(grid, mean_img, data_range=255))
        for code, grid in zip(codes, trimmed)
    }

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(mean_path, mean_img)
    np.save(std_path, std_img)
    np.save(align_ref_path, align_reference)
    scores_path.write_text(json.dumps(scores, indent=2))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(mean_img, cmap="viridis")
    ax.set_title(f"batch average, n={len(codes)} (registered)")
    fig.savefig(cache_dir / "mean.png", dpi=120)
    plt.close(fig)

    return {"mean": mean_img, "std": std_img, "align_reference": align_reference, "scores": scores}


def compare_to_batch_reference(
    image_dir: str = "",
    electrode_code: str = "",
    image_path: str = "",
    batch: str = "",
    sheet: str = None,
    crop_box=DEFAULT_CROP_BOX,
    force_rebuild: bool = False,
) -> dict:
    # batch is a tolerated alias for image_dir -- the model has repeatedly
    # passed a bare batch string (matching the convention every OTHER tool
    # in this suite uses) instead of the actual directory path this
    # function needs, which silently errored instead of comparing anything.
    # Resolving it here is more reliable than depending on prompting alone.
    image_dir = image_dir or (f"reference_images/{batch}" if batch else "")
    if not image_dir:
        return {"status": "error", "message": "Provide image_dir (e.g. 'reference_images/20260707') or batch (e.g. '20260707')."}

    if not image_path:
        if not electrode_code:
            return {"status": "error", "message": "Provide either image_path or electrode_code."}
        image_path, used_code, _ = _resolve_electrode_image(electrode_code, image_dir)
        if image_path is None:
            return {"status": "error", "message": f"No grid-labeled photos found for '{electrode_code}' in {image_dir}."}
        electrode_code = used_code
        if sheet is None:
            sheet = _split_key(used_code)[0]
    elif sheet is None:
        parsed = _parse_filename_identity(Path(image_path).stem)
        sheet = parsed[0] if parsed else None

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

    reference = build_batch_reference(image_dir, sheet, crop_box, force_rebuild)
    if "error" in reference:
        return {"status": "error", "message": reference["error"]}
    mean_img, align_reference, batch_scores = reference["mean"], reference["align_reference"], reference["scores"]

    target_raw = _luminance_grid(image_path, GRID_SIZE, True, 1.0, crop_box)
    target = _trim_margin(_align(target_raw, align_reference))  # same registration as the batch average
    ssim_score, ssim_map = structural_similarity(target, mean_img, data_range=255, full=True)

    ranked = sorted(batch_scores.values())
    percentile = float(np.searchsorted(ranked, ssim_score) / len(ranked) * 100)

    diff_path = _batch_cache_dir(image_dir, sheet) / f"{Path(image_path).stem}_diff.png"
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    axes[0].imshow(target, cmap="viridis"); axes[0].set_title("this electrode")
    axes[1].imshow(mean_img, cmap="viridis"); axes[1].set_title(f"batch average (n={len(batch_scores)})")
    im = axes[2].imshow(1 - ssim_map, cmap="inferno", vmin=0, vmax=1)
    axes[2].set_title("dissimilarity (bright = deviates)")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{Path(image_path).name} -- SSIM={ssim_score:.3f} (percentile {percentile:.0f} within batch)")
    diff_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(diff_path, dpi=120)
    plt.close(fig)

    flags = []
    if percentile < 15:
        flags.append(
            f"OUTLIER: SSIM={ssim_score:.3f} puts this electrode in the bottom {percentile:.0f}% of the "
            f"batch by similarity to the average print -- worth a manual look."
        )

    status = "warn" if flags else "pass"

    try:
        note_code = (electrode_code or extract_grid_code(Path(image_path).stem) or "").upper()
        note_batch = extract_batch_date(Path(image_dir).name)
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
        "diff_plot_path": str(diff_path),
        "metrics": {
            "ssim_vs_batch_average": ssim_score,
            "batch_percentile": percentile,
            "batch_n": len(batch_scores),
        },
        "flags": flags,
    }
