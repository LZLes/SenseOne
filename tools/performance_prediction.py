"""
Visual-vs-electrical-performance correlation and prediction.

Core question: can visual characteristics of an electrode (surface
roughness, print texture) predict its CV performance (delta Ep, peak
current, ipa/ipc ratio) before running the actual electrochemical test?

Answer as of the current data (see conversation): not yet. n=9 paired
sub-electrodes (20260707 electrodes A1/A3/A5, each split into its 3 real
physical sub-electrodes -- see image_qc.sub_pad_crop_box's docstring for
why a whole 20260707-style photo is 3 electrodes, not 1, and why averaging
them as "replicates" was wrong and has been corrected). Every correlation
tested came back weak and non-significant (|r| < 0.31, p > 0.4 for all
pairs), and the CV outcomes barely varied across the sample (all "good" by
sensor_qc's own thresholds) -- there's no good/bad split yet for a visual
feature to explain.

This module re-tests that automatically as more paired data accumulates in
electrode_notes (both image-side roughness and CV-side sensor_qc results
are already logged there per sub-electrode via their auto-append). It only
ever reports a "prediction" grounded in a correlation that's statistically
real *given the current data* -- otherwise it says so plainly instead of
guessing. As more CV data comes in (per-electrode, not averaged), rerun
correlate_visual_cv_performance; once real signal shows up,
predict_electrode_performance starts using it automatically.
"""

from scipy.stats import pearsonr

from tools.electrode_notes import NOTES_DIR, DIGEST_FILENAME, BATCH_INFO_FILENAME, _load_note
from tools.image_qc import (
    WORKING_ELECTRODE_CROP_BOX, analyze_surface_topology, sub_pad_working_electrode_crop_box,
    _resolve_electrode_image,
)
from tools.framing_qc import check_photo_framing

_VISUAL_FIELDS = ("Ra", "Rq", "Rz", "Rt", "Rsk", "Rku")
# From sensor_qc (single-scan snapshot) and analyze_cv_stability (scan-to-scan
# behavior) respectively -- per the researcher, stability is the more
# important signal (a good electrode should stay roughly constant scan to
# scan), so both are correlated against, not just the single-scan values.
_CV_FIELDS = ("delta_ep_v", "peak_current_a", "ipa_ipc_ratio", "peak_current_cv_pct", "delta_ep_cv_pct")
# Both floors are deliberately explicit rather than left implicit: below
# MIN_N, correlation coefficients from this few points are noisy/unstable
# even before significance is considered; SIGNIFICANCE_P is the standard
# p<0.05 convention, not tuned to this data.
MIN_N_FOR_CORRELATION = 15
SIGNIFICANCE_P = 0.05

CORRELATE_VISUAL_CV_SCHEMA = {
    "type": "function",
    "function": {
        "name": "correlate_visual_cv_performance",
        "description": (
            "Test whether any visual/roughness feature actually correlates with CV electrical "
            "performance (delta Ep, peak current, ipa/ipc ratio), using every electrode that has "
            "both logged in its electrode_notes history. Reports real Pearson r/p per pair, and "
            "is explicit when there isn't enough data or no significant relationship exists yet "
            "-- never asserts a relationship that isn't statistically supported by what's on record."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

PREDICT_ELECTRODE_PERFORMANCE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "predict_electrode_performance",
        "description": (
            "Predict whether an electrode will perform well electrically (CV), from its photo, "
            "before running the actual electrochemical test. Extracts the same roughness "
            "features image_qc uses, then checks them against whatever statistically "
            "significant visual-vs-CV relationship currently exists (see "
            "correlate_visual_cv_performance). As of the current (small) paired dataset, no "
            "such relationship is established, so by default this reports the visual features "
            "plainly rather than a confident-sounding guess -- it will start giving grounded, "
            "hedged predictions automatically once enough real CV data accumulates to support "
            "one. Only pass preliminary=true if the researcher explicitly asks for a best-effort "
            "read despite that -- it then uses the strongest available correlation even if not "
            "statistically significant, but the result must always be reported as preliminary "
            "with its actual r/p/n shown, never stated as if it were validated."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Direct path to the electrode photo. Use this or electrode_code+image_dir -- don't guess a filename.",
                },
                "electrode_code": {
                    "type": "string",
                    "description": "Grid position, e.g. 'A1' or 'S3-A1' for a multi-sheet batch. Use with image_dir instead of image_path.",
                },
                "image_dir": {
                    "type": "string",
                    "description": "Directory of grid-labeled photos, used with electrode_code, e.g. reference_images/20260707. Or pass batch instead.",
                },
                "batch": {
                    "type": "string",
                    "description": "Batch date, e.g. '20260707' -- shorthand for image_dir='reference_images/<batch>'. Not needed when image_path is given directly.",
                },
                "sub_position": {
                    "type": "string",
                    "enum": ["1", "2", "3"],
                    "description": (
                        "For a 20260707-style photo showing a 2x2 cluster of electrodes, which "
                        "one to assess (1=top-left, 2=top-right, 3=bottom-left; 4/bottom-right "
                        "is always the unfilled reference pad, never has CV data). Omit for "
                        "newer batches (20260804 on), which are one electrode per photo."
                    ),
                },
                "crop_box": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Override crop, same convention as image_qc's surface_crop_box. Ignored if sub_position is given.",
                },
                "preliminary": {
                    "type": "boolean",
                    "description": (
                        "Only set true if the researcher explicitly asks for a best-effort/"
                        "preliminary read despite insufficient data. Uses the strongest available "
                        "correlation(s) regardless of statistical significance -- result is "
                        "clearly marked 'preliminary', with the real r/p/n included so the "
                        "researcher can judge reliability themselves. Default false."
                    ),
                },
            },
        },
    },
}


def _iter_notes():
    if not NOTES_DIR.exists():
        return
    for path in sorted(NOTES_DIR.glob("*/*.md")):
        if path.name in (DIGEST_FILENAME, BATCH_INFO_FILENAME):
            continue
        try:
            meta, _ = _load_note(path)
        except Exception:
            continue
        yield meta


def gather_paired_data() -> list:
    """One row per electrode note with BOTH a roughness reading (from
    image_qc's include_surface_analysis) and a CV reading -- from sensor_qc
    (single-scan delta_ep_v/peak_current_a/ipa_ipc_ratio), analyze_cv_stability
    (scan-to-scan peak_current_cv_pct/delta_ep_cv_pct), or both merged
    together if an electrode has both logged. Most recent entry per tool
    wins if it ran more than once.
    """
    rows = []
    for meta in _iter_notes():
        visual, cv = None, {}
        for entry in meta.get("qc_history", []):
            m = entry.get("metrics") or {}
            if entry.get("tool") == "image_qc" and m.get("roughness"):
                visual = m["roughness"]
            if entry.get("tool") == "sensor_qc" and "delta_ep_v" in m:
                cv.update(m)
            if entry.get("tool") == "analyze_cv_stability" and "peak_current_cv_pct" in m:
                cv.update(m)
        if visual and cv:
            rows.append({"electrode_id": meta.get("electrode_id"), "batch": meta.get("batch"), **visual, **cv})
    return rows


def correlate_visual_cv_performance() -> dict:
    rows = gather_paired_data()
    if len(rows) < 3:
        return {
            "status": "insufficient_data",
            "n_paired_electrodes": len(rows),
            "message": (
                f"Only {len(rows)} electrode(s) have both visual and CV data logged -- need "
                f"far more (at least {MIN_N_FOR_CORRELATION}) before correlation means anything."
            ),
        }

    results = []
    for visual in _VISUAL_FIELDS:
        for cv in _CV_FIELDS:
            paired = [(r[visual], r[cv]) for r in rows if r.get(visual) is not None and r.get(cv) is not None]
            if len(paired) < 3:
                continue
            x, y = zip(*paired)
            if len(set(x)) < 2 or len(set(y)) < 2:
                continue  # zero variance -- correlation undefined, not just weak (e.g. Rt saturates to 255 for every high-contrast electrode so far)
            r_val, p_val = pearsonr(x, y)
            results.append({"visual_feature": visual, "cv_metric": cv, "r": float(r_val), "p": float(p_val), "n": len(paired)})

    significant = [r for r in results if r["p"] < SIGNIFICANCE_P and r["n"] >= MIN_N_FOR_CORRELATION]

    if len(rows) < MIN_N_FOR_CORRELATION:
        message = (
            f"n={len(rows)} paired electrodes -- below the {MIN_N_FOR_CORRELATION} needed for a "
            "reliable correlation regardless of the r/p values below; treat everything here as "
            "exploratory, not a validated relationship."
        )
    elif significant:
        message = f"{len(significant)} statistically significant correlation(s) found (p<{SIGNIFICANCE_P})."
    else:
        message = "No statistically significant correlation found in the current data."

    return {
        "status": "ok",
        "n_paired_electrodes": len(rows),
        "min_n_for_reliable_correlation": MIN_N_FOR_CORRELATION,
        "all_correlations": results,
        "significant_correlations": significant,
        "message": message,
    }


def _build_predictions(visual: dict, relationships: list) -> list:
    predictions = []
    for rel in relationships:
        value = visual.get(rel["visual_feature"])
        if value is None:
            continue
        # For r > 0 the two move together (higher-with-higher or
        # lower-with-lower); for r < 0 they move oppositely (higher-with-
        # lower). Describing both sides with the same word regardless of
        # sign (as an earlier version of this did) states the literal
        # opposite of a negative correlation's actual meaning.
        cv_direction = "higher" if rel["r"] > 0 else "lower"
        predictions.append({
            "cv_metric": rel["cv_metric"],
            "based_on": f"{rel['visual_feature']}={value:.2f}",
            "relationship": (
                f"r={rel['r']:+.2f} (n={rel['n']}, p={rel['p']:.3f}): higher "
                f"{rel['visual_feature']} associates with {cv_direction} {rel['cv_metric']}"
            ),
        })
    return predictions


def predict_electrode_performance(
    image_path: str = "",
    electrode_code: str = "",
    image_dir: str = "",
    batch: str = "",
    sub_position: str = None,
    crop_box=None,
    preliminary: bool = False,
) -> dict:
    # batch is a tolerated alias for image_dir, same reasoning as
    # compare_to_batch_reference -- accepted here mainly so passing it
    # alongside an already-given image_path doesn't crash with an
    # unexpected-keyword-argument error (observed happening in practice).
    image_dir = image_dir or (f"reference_images/{batch}" if batch else "")
    expect_single_electrode = bool(image_path)  # captured before catalog resolution can overwrite image_path

    if not image_path:
        if not electrode_code or not image_dir:
            return {"status": "error", "message": "Provide either image_path, or both electrode_code and image_dir (or batch)."}
        image_path, _, _ = _resolve_electrode_image(electrode_code, image_dir)
        if image_path is None:
            return {"status": "error", "message": f"No grid-labeled photos found for '{electrode_code}' in {image_dir}."}

    # Same framing-QC gate as image_qc/compare_to_batch_reference -- a
    # prediction built on a bad photo's roughness reading (blurred, clipped,
    # cut off) would carry the same false confidence as a defect read from
    # one, just laundered through a correlation instead of stated directly.
    framing = check_photo_framing(image_path=image_path, expect_single_electrode=expect_single_electrode)
    if not framing["proceed"]:
        return {
            "status": "framing_rejected",
            "framing_ok": False,
            "issues": framing["issues"],
            "user_message": framing["user_message"],
            "measurements": framing.get("measurements"),
        }

    effective_crop = (
        sub_pad_working_electrode_crop_box(sub_position) if sub_position
        else (tuple(crop_box) if crop_box else WORKING_ELECTRODE_CROP_BOX)
    )

    surface = analyze_surface_topology(image_path, crop_box=effective_crop)
    if surface.get("status") == "error":
        return surface
    visual = surface["roughness_luminance_units"]

    correlation = correlate_visual_cv_performance()
    significant = correlation.get("significant_correlations", [])

    if significant:
        return {
            "status": "ok",
            "visual_features": visual,
            "predictions": _build_predictions(visual, significant),
            "correlation_basis": correlation,
        }

    if not preliminary:
        return {
            "status": "no_prediction",
            "message": (
                f"No statistically significant visual-vs-CV relationship is established yet "
                f"({correlation.get('message', '')}) -- reporting visual features only, not a "
                "performance prediction. Run sensor_qc on this electrode's own CV file (once "
                "available) to add a real data point and strengthen this over time. Pass "
                "preliminary=true for a best-effort read using the strongest available "
                "(non-significant) correlation instead."
            ),
            "visual_features": visual,
            "correlation_basis": correlation,
        }

    # Explicitly requested despite insufficient data: use the strongest
    # available correlations (lowest p) regardless of the significance gate,
    # but every field below keeps saying "preliminary"/"not validated" so
    # this can never be mistaken for the grounded "ok" case above.
    candidates = sorted(correlation.get("all_correlations", []), key=lambda c: c["p"])[:3]
    if not candidates:
        return {
            "status": "no_prediction",
            "message": "Not enough paired data exists to compute even a preliminary read yet.",
            "visual_features": visual,
            "correlation_basis": correlation,
        }

    return {
        "status": "preliminary",
        "message": (
            "PRELIMINARY -- NOT statistically validated "
            f"(n={correlation.get('n_paired_electrodes')} paired electrodes, below the "
            f"{MIN_N_FOR_CORRELATION}-point floor and/or p>={SIGNIFICANCE_P} for these "
            "relationships). Treat this as a rough, unreliable hint, not a real prediction -- "
            "shown only because it was explicitly requested despite insufficient data."
        ),
        "visual_features": visual,
        "preliminary_predictions": _build_predictions(visual, candidates),
        "correlation_basis": correlation,
    }
