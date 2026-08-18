"""
Electrochemical batch-outlier detection: is this electrode's CV/CA
performance a statistical outlier relative to the rest of its own batch?

The pipeline already does this for photos (tools/reference_diff.py's
compare_to_batch_reference, SSIM against a pixel-average reference) -- but
nothing does the equivalent check on the actual electrical measurements
themselves, even though those are the ground-truth signal the photo-based
checks are ultimately trying to predict. This closes that gap using the CV
metrics sensor_qc/analyze_cv_stability, and the CA metrics ca_calibration,
already auto-log into electrode_notes -- no new measurement or file format
needed, just a QC pass over data that's already being collected.

Method: a modified z-score on median + median absolute deviation (MAD),
not mean/stddev -- standard practice for outlier detection on real QC data
(Iglewicz & Hoaglin, 1993). A couple of genuine outliers inflate a
mean/stddev calculation (the very spread being used to judge them), which
can mask them ("masking"); the median and MAD are robust to exactly that.
Flags |modified z| > 3.5, the threshold from the same reference -- an
established convention, not tuned to this project's data.

What this deliberately does NOT attempt, and why: scan-rate-dependent
kinetics (Randles-Sevcik, Laviron, Nicholson's k0 method) would need CVs at
multiple scan rates per electrode, which this lab's protocol doesn't
collect (repeated scans at one rate, not swept rates) -- implementing those
here would produce numbers with no real data behind them. Same reasoning
ruled out a time-series drift/SPC check across a batch: qc_history's "date"
field is per-day, not a real print-order sequence, so there's no genuine
ordering to test a trend against.
"""

import numpy as np

from tools._paths import safe_path_component
from tools.electrode_notes import (
    BATCH_INFO_FILENAME, DIGEST_FILENAME, NOTES_DIR, _load_note, append_qc_result,
)

MODIFIED_Z_THRESHOLD = 3.5  # Iglewicz & Hoaglin (1993) standard robust-outlier cutoff
MIN_BATCH_N = 5  # below this, median/MAD over so few points isn't meaningful

# tool name -> metric fields it contributes, exactly as sensor_qc.py /
# cv_stability.py / ca_calibration.py already persist them into qc_history
# (confirmed against real notes, not assumed -- sensor_qc does NOT persist
# noise_ratio, for instance, even though it computes it).
_METRIC_SOURCES = {
    "sensor_qc": ("peak_current_a", "delta_ep_v", "ipa_ipc_ratio"),
    "analyze_cv_stability": ("peak_current_cv_pct", "delta_ep_cv_pct"),
    "ca_calibration": ("sensitivity_ua_per_mM", "r_squared", "lod_mM"),
}
_ALL_METRICS = tuple(m for fields in _METRIC_SOURCES.values() for m in fields)

COMPARE_CV_TO_BATCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "compare_cv_to_batch_reference",
        "description": (
            "Check whether an electrode's CV/CA metrics (peak current, delta Ep, ipa/ipc ratio, "
            "scan-to-scan stability, CA sensitivity/R^2/LOD -- whichever are on record) are "
            "statistical outliers relative to the rest of its own batch. Unsupervised, no "
            "labeled good/bad examples needed -- uses every electrode in the batch that already "
            "has sensor_qc/analyze_cv_stability/ca_calibration data logged in electrode_notes. "
            "This is the electrical-data counterpart to compare_to_batch_reference (which does "
            "the same thing for photos) -- reach for this when the question is about electrical "
            "performance specifically ('is this electrode's CV normal for this batch'), not "
            "visual appearance. Omit electrode_code for a batch-wide sweep flagging every "
            "outlier electrode at once instead of checking just one."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "batch": {"type": "string", "description": "Fabrication/imaging batch date, e.g. '20260707'."},
                "electrode_code": {
                    "type": "string",
                    "description": (
                        "Optional: check just this one electrode, e.g. 'A1-1' (CV sub-electrode "
                        "form) or 'S3-A1'. Omit for a batch-wide sweep."
                    ),
                },
            },
            "required": ["batch"],
        },
    },
}


def _iter_batch_notes(batch: str):
    batch_dir = NOTES_DIR / safe_path_component(batch)
    if not batch_dir.exists():
        return
    for path in sorted(batch_dir.glob("*.md")):
        if path.name in (DIGEST_FILENAME, BATCH_INFO_FILENAME) or ".corrupt-" in path.name:
            continue
        try:
            meta, _ = _load_note(path)
        except Exception:
            continue
        yield meta


def _latest_electrochem_metrics(meta: dict) -> dict:
    """Most recent value per metric across this electrode's qc_history --
    later entries overwrite earlier ones for the same field, same "most
    recent wins" convention as performance_prediction.gather_paired_data.
    """
    values = {}
    for entry in meta.get("qc_history", []):
        fields = _METRIC_SOURCES.get(entry.get("tool"))
        if not fields:
            continue
        m = entry.get("metrics") or {}
        values.update({k: v for k, v in m.items() if k in fields and v is not None})
    return values


def _modified_z_scores(values: np.ndarray) -> np.ndarray:
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad == 0:
        return np.zeros_like(values)  # every value identical (or MAD-degenerate) -- nothing reads as an outlier
    return 0.6745 * (values - median) / mad


def _batch_outlier_report(batch: str):
    """Returns (per_electrode_metric_report, flagged_codes, raw_rows)."""
    rows = {}
    for meta in _iter_batch_notes(batch):
        code = meta.get("electrode_id")
        values = _latest_electrochem_metrics(meta)
        if code and values:
            rows[code] = values

    report = {}
    flagged_codes = set()
    for metric in _ALL_METRICS:
        codes = [c for c, m in rows.items() if metric in m]
        if len(codes) < MIN_BATCH_N:
            continue
        raw = np.array([rows[c][metric] for c in codes], dtype=float)
        z_scores = _modified_z_scores(raw)
        for code, value, z in zip(codes, raw, z_scores):
            is_outlier = bool(abs(z) > MODIFIED_Z_THRESHOLD)
            report.setdefault(code, {})[metric] = {"value": float(value), "modified_z": float(z), "is_outlier": is_outlier}
            if is_outlier:
                flagged_codes.add(code)
    return report, flagged_codes, rows


def compare_cv_to_batch_reference(batch: str, electrode_code: str = None) -> dict:
    report, flagged_codes, rows = _batch_outlier_report(batch)

    if not rows:
        return {
            "status": "error",
            "message": f"No electrode in batch '{batch}' has sensor_qc/analyze_cv_stability/ca_calibration data logged yet.",
        }
    if not report:
        return {
            "status": "insufficient_data",
            "message": (
                f"Only {len(rows)} electrode(s) in batch '{batch}' have electrochemical data logged -- "
                f"need at least {MIN_BATCH_N} sharing the same metric before a robust outlier check "
                "means anything. Treat any single-electrode read as exploratory, not validated."
            ),
            "n_electrodes_with_data": len(rows),
        }

    if electrode_code:
        code = electrode_code.strip().upper()
        if code not in rows:
            return {
                "status": "error",
                "message": f"No sensor_qc/analyze_cv_stability/ca_calibration data logged for '{code}' in batch '{batch}'.",
            }
        metrics = report.get(code, {})
        flags = [
            f"OUTLIER: {m} = {d['value']:.3e} is a robust outlier vs. the rest of the batch "
            f"(modified z-score {d['modified_z']:+.1f}, threshold ±{MODIFIED_Z_THRESHOLD})."
            for m, d in metrics.items() if d["is_outlier"]
        ]
        result = {
            "status": "warn" if flags else "pass",
            "batch": batch,
            "electrode_code": code,
            "metrics_checked": metrics,
            "flags": flags,
            "batch_n": len(rows),
        }
        try:
            append_qc_result(
                batch, code, tool="compare_cv_to_batch_reference", status=result["status"],
                metrics={"flagged_metrics": [m for m, d in metrics.items() if d["is_outlier"]]},
            )
        except Exception:
            pass  # note-linking is best-effort, never blocks the QC result
        return result

    return {
        "status": "warn" if flagged_codes else "pass",
        "batch": batch,
        "batch_n": len(rows),
        "metrics_checked": sorted({m for metrics in report.values() for m in metrics}),
        "outlier_electrodes": sorted(flagged_codes),
        "full_report": report,
    }
