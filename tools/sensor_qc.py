"""
Sensor QC tool.

Reads a CSV of electrochemical data (CV, CA, or SWV) and checks it against
configurable pass/fail thresholds -- the kind of first-pass screening you'd
otherwise do by eye in Origin/R before deciding whether an SPE batch is worth
running further characterization on.

Handles two kinds of input:
  1. Clean CSVs with named columns (case-insensitive, flexible naming):
       potential column: one of "potential", "potential_v", "e", "voltage"
       current column:   one of "current", "current_a", "i"
  2. Raw instrument exports (e.g. PSTrace-style), which are messier:
       - UTF-16 or legacy single-byte encoded, with a multi-line metadata
         header ("Date and time:", "Notes:", ...) before the real data
       - current reported in microamps, not amps
       - CV files lay multiple scans out side by side as repeated
         potential/current column pairs (use `scan_index` to pick one,
         default is the last/most stable scan)
     These are detected structurally -- the first non-numeric row
     immediately followed by an all-numeric row of the same width marks
     the header/data boundary -- rather than by matching exact column
     label text, since encoding quirks mangle "µA" unpredictably.

This is deliberately simple (scipy.signal.find_peaks + basic thresholds) --
swap in your own peak-fitting / baseline-subtraction pipeline from your R
scripts if you want it to match your existing analysis exactly.
"""

import re
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.signal import find_peaks

from tools._io import load_instrument_csv
from tools.electrode_notes import append_qc_result, extract_batch_date

# Two CV filename conventions are in use:
#   "707-A5-1"           (20260707 batch): 3-digit date fragment, code,
#      sub-electrode position -- the trailing number is NOT a replicate/scan
#      index, it's which of the (up to 3) distinct physical electrodes
#      within that photo's cluster this file measures (confirmed: 1/2/3 =
#      top-left/top-right/bottom-left; bottom-right is always the unfilled
#      Ag/AgCl reference, hence never a "-4" file). Dropping this suffix and
#      keying notes by "A5" alone would silently merge three different
#      electrodes' CV results into one note -- caught before it did.
#   "20260804-S3-A2-CV"  (20260804 on): full 8-digit batch, sheet, code,
#      literal "-CV" suffix -- one electrode per photo already, so this
#      maps directly onto the same "<sheet>-<code>" identity the images use.
_CV_FILENAME_RE = re.compile(r"^\d+-([A-Za-z]\d+)-(\d+)$")
_CV_FILENAME_NEW_RE = re.compile(r"^(\d{8})-([A-Za-z0-9]+)-([A-Za-z]\d+)-CV$")


def parse_cv_filename(stem: str):
    """Returns (note_code, batch_override) or None. batch_override is set
    only for the new format, which encodes the real 8-digit batch itself;
    the old format's 3-digit date fragment can't be expanded reliably and
    relies on the parent directory name (via extract_batch_date) instead.
    """
    m = _CV_FILENAME_NEW_RE.match(stem)
    if m:
        batch, sheet, code = m.groups()
        return f"{sheet}-{code}", batch
    m = _CV_FILENAME_RE.match(stem)
    if m:
        code, position = m.groups()
        return f"{code}-{position}", None
    return None

SENSOR_QC_SCHEMA = {
    "type": "function",
    "function": {
        "name": "sensor_qc",
        "description": (
            "QC an electrochemical sensor dataset (CV, CA, or SWV) from a CSV file. "
            "Computes peak current and baseline noise, and for CV also peak "
            "potentials (E_pa/E_pc), delta Ep, formal potential (E_half), and the "
            "ipa/ipc ratio, then flags the run as pass/fail/warn against thresholds. "
            "For CA calibration-curve analysis (sensitivity, LOD, saturation), use "
            "ca_calibration instead."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "csv_path": {
                    "type": "string",
                    "description": "Path to the CSV file containing the sensor run.",
                },
                "technique": {
                    "type": "string",
                    "enum": ["CV", "CA", "SWV"],
                    "description": "Electrochemical technique used to record the data.",
                },
                "min_peak_current_a": {
                    "type": "number",
                    "description": "Minimum acceptable |peak current| in amps. Default 1e-7 (100 nA).",
                },
                "max_delta_ep_v": {
                    "type": "number",
                    "description": "Maximum acceptable peak-to-peak separation (CV only), in volts. Default 0.15.",
                },
                "max_noise_ratio": {
                    "type": "number",
                    "description": "Max acceptable baseline noise / peak current ratio. Default 0.1 (10%).",
                },
                "scan_index": {
                    "type": "integer",
                    "description": (
                        "For raw CV exports with multiple scans laid out side by side: "
                        "which scan to QC (0-based, negative counts from the end). "
                        "Default -1 (the last/most stable scan). Ignored for single-scan files."
                    ),
                },
            },
            "required": ["csv_path", "technique"],
        },
    },
}


def _find_column(columns, candidates):
    lowered = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    return None


def qc_electrochemical_data(
    csv_path: str,
    technique: str,
    min_peak_current_a: float = 1e-7,
    max_delta_ep_v: float = 0.15,
    max_noise_ratio: float = 0.10,
    scan_index: int = -1,
) -> dict:
    if not Path(csv_path).is_file():
        return {"status": "error", "message": f"File not found: {csv_path}"}

    potential = current = None

    # Attempt 1: clean CSV with named potential/current columns.
    try:
        df = pd.read_csv(csv_path)
        e_col = _find_column(df.columns, ["potential", "potential_v", "e", "voltage"])
        i_col = _find_column(df.columns, ["current", "current_a", "i"])
        if e_col is not None and i_col is not None:
            potential = df[e_col].to_numpy(dtype=float)
            current = df[i_col].to_numpy(dtype=float)
    except Exception:
        pass

    # Attempt 2: raw instrument export -- current is in microamps, and CV
    # files may lay out several scans side by side as repeated column pairs.
    if potential is None:
        data = load_instrument_csv(csv_path)
        if data is None or data.ndim != 2 or data.shape[1] < 2 or data.shape[1] % 2 != 0:
            return {
                "status": "error",
                "message": f"Could not find potential/current data in {csv_path}.",
            }
        n_scans = data.shape[1] // 2
        idx = scan_index if scan_index >= 0 else n_scans + scan_index
        idx = max(0, min(idx, n_scans - 1))
        potential = data[:, 2 * idx]
        current = data[:, 2 * idx + 1] * 1e-6  # microamps -> amps

    if potential is None or current is None or len(current) == 0:
        return {"status": "error", "message": f"No usable data rows found in {csv_path}."}

    flags = []
    metrics = {}

    # Baseline noise: std dev of current over the first 10% of points
    # (assumes the run starts before the analyte response kicks in -- adjust
    # for your protocol if that's not true).
    n_baseline = max(5, int(0.1 * len(current)))
    baseline_noise = float(np.std(current[:n_baseline]))
    metrics["baseline_noise_a"] = baseline_noise

    # Peak detection (both polarities, since redox peaks can go either way)
    pos_peaks, _ = find_peaks(current, prominence=(np.max(current) - np.min(current)) * 0.05)
    neg_peaks, _ = find_peaks(-current, prominence=(np.max(current) - np.min(current)) * 0.05)

    peak_current = float(np.max(np.abs(current)))
    metrics["peak_current_a"] = peak_current

    if peak_current < min_peak_current_a:
        flags.append(
            f"LOW SIGNAL: peak current {peak_current:.3e} A is below threshold "
            f"{min_peak_current_a:.3e} A -- check electrode connection, enzyme loading, or fouling."
        )

    noise_ratio = baseline_noise / peak_current if peak_current > 0 else float("inf")
    metrics["noise_ratio"] = noise_ratio
    if noise_ratio > max_noise_ratio:
        flags.append(
            f"HIGH NOISE: baseline noise is {noise_ratio:.1%} of peak current "
            f"(threshold {max_noise_ratio:.0%}) -- check grounding, membrane defects, or contamination."
        )

    if technique == "CV":
        if len(pos_peaks) > 0 and len(neg_peaks) > 0:
            i_pa = float(np.max(current[pos_peaks]))
            i_pc = float(np.min(current[neg_peaks]))
            e_anodic = potential[pos_peaks[np.argmax(current[pos_peaks])]]
            e_cathodic = potential[neg_peaks[np.argmin(current[neg_peaks])]]
            delta_ep = float(abs(e_anodic - e_cathodic))
            metrics["delta_ep_v"] = delta_ep
            metrics["e_anodic_v"] = float(e_anodic)
            metrics["e_cathodic_v"] = float(e_cathodic)
            metrics["e_half_v"] = float((e_anodic + e_cathodic) / 2)  # formal potential estimate
            metrics["i_pa_a"] = i_pa
            metrics["i_pc_a"] = i_pc
            metrics["ipa_ipc_ratio"] = float(abs(i_pa / i_pc)) if i_pc != 0 else None

            if delta_ep > max_delta_ep_v:
                flags.append(
                    f"POOR REVERSIBILITY: delta Ep = {delta_ep:.3f} V exceeds threshold "
                    f"{max_delta_ep_v:.3f} V -- possible surface fouling, slow kinetics, "
                    f"or membrane resistance."
                )
        else:
            flags.append("Could not identify both anodic and cathodic peaks for delta Ep calculation.")

    # "Could not identify..." is fail-level here to match analyze_cv_stability's
    # severity for the identical condition -- previously it only ranked as
    # "warn" in this tool, so the same underlying failure (no CV peaks found)
    # got two different verdicts depending on which tool was called.
    status = "fail" if any(
        "LOW SIGNAL" in f or "POOR REVERSIBILITY" in f or f.startswith("Could not identify") for f in flags
    ) else ("warn" if flags else "pass")

    # CA files don't carry a grid code in their filename (only ca_calibration.py's
    # sampleinfo lookup can resolve electrode identity for those) -- only CV
    # filenames are parseable here, via parse_cv_filename (handles both
    # naming conventions currently in use).
    if technique == "CV":
        try:
            parsed = parse_cv_filename(Path(csv_path).stem)
            if parsed:
                note_code, batch_override = parsed
                batch = batch_override or extract_batch_date(Path(csv_path).parent.name)
                append_qc_result(
                    batch, note_code, tool="sensor_qc",
                    status=status, metrics={k: v for k, v in metrics.items() if k in ("delta_ep_v", "peak_current_a", "ipa_ipc_ratio")},
                    file_ref=csv_path, file_kind="cv_files",
                )
        except Exception:
            pass  # note-linking is best-effort, never blocks the QC result

    return {
        "status": status,
        "technique": technique,
        "n_points": len(potential),
        "metrics": metrics,
        "flags": flags,
    }
