"""
CA calibration analysis tool.

The lab's CA protocol is a multi-point standard-addition calibration: analyte
standards are spiked into the cell at known times during a single continuous
i-vs-t run, at increasing concentration. This tool locates each point's
steady-state response just before the next spike, fits a calibration curve
across points, and reports sensitivity, limit of detection (LOD), and
whether the response is saturating at the higher concentrations.

Concentrations/timings are normally looked up from reference_data/sampleinfo_ca.txt
by matching the CSV filename (YYYYMMDD_sampleN_chC.csv) against that file's
"Experiment date" / "Experiment No." / channel blocks -- see _parse_sampleinfo.
They can also be passed explicitly (e.g. if a run isn't logged there yet).
"""

import re
from pathlib import Path

import numpy as np

from tools._io import load_instrument_csv
from tools.electrode_notes import append_qc_result, extract_batch_date, extract_grid_code

CA_CALIBRATION_SCHEMA = {
    "type": "function",
    "function": {
        "name": "ca_calibration",
        "description": (
            "Analyze a chronoamperometry (CA) multi-point calibration run: extracts "
            "the steady-state response at each standard-addition point, fits a "
            "calibration curve, and reports sensitivity (slope), limit of detection "
            "(LOD), linearity (R^2), and whether the response saturates at higher "
            "concentrations. By default looks up the concentration/timing protocol "
            "for the file from reference_data/sampleinfo_ca.txt; pass concentrations_mM "
            "and timings_s explicitly to override or if the run isn't logged there."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "csv_path": {
                    "type": "string",
                    "description": "Path to the CA CSV file (e.g. reference_data/ca/20260703_sample1_ch1.csv).",
                },
                "sampleinfo_path": {
                    "type": "string",
                    "description": "Path to the sampleinfo text log. Default reference_data/sampleinfo_ca.txt.",
                },
                "concentrations_mM": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Override: known standard concentrations (mM), one per calibration point, in order.",
                },
                "timings_s": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Override: elapsed time (s) of each spike, same order/length as concentrations_mM.",
                },
                "stop_s": {
                    "type": "number",
                    "description": "Override: elapsed time (s) the run ended -- bounds the window for the last point.",
                },
                "saturation_threshold": {
                    "type": "number",
                    "description": (
                        "Flag saturation if the last point-to-point slope drops below this "
                        "fraction of the average of the earlier slopes. Default 0.5 (50%)."
                    ),
                },
            },
            "required": ["csv_path"],
        },
    },
}

_FILENAME_RE = re.compile(r"(\d{8})_sample(\d+)_ch(\d+)", re.IGNORECASE)
_CHANNEL_MAP_RE = re.compile(r"Chn(\d):\s*(\w+)", re.IGNORECASE)
_DATE_RE = re.compile(r"Experiment date:\s*(\d{2})/(\d{2})/(\d{2})")
_SAMPLE_RE = re.compile(r"Experiment No\.:\s*Sample\s*(\d+)", re.IGNORECASE)
_TYPE_RE = re.compile(r"Experiment Type:\s*(.+)")
_ELECTRODE_RE = re.compile(r"Electrode ID:\s*(\S+)")
_PT_RE = re.compile(
    # Closing paren is optional -- the "pt 4" line is missing it in every
    # block of the source log (a consistent transcription typo, not a
    # one-off), so a strict ")" would silently drop point 4 everywhere.
    r"pt\s*(\d+)\s*\(([\d.]+)\s*mM\s*/\s*([\d.]+)\s*mM\s*/\s*([\d.]+)\s*mM\)?\s*"
    r"([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)",
    re.IGNORECASE,
)
_STOP_RE = re.compile(r"stop\s+([\d.]+)", re.IGNORECASE)


def _parse_sampleinfo(path):
    if not Path(path).is_file():
        raise FileNotFoundError(f"Sampleinfo file not found: {path}")
    text = open(path, "r", encoding="utf-8", errors="replace").read()
    blocks = re.split(r"\n=+\n", text)

    channel_analyte = {int(m.group(1)): m.group(2).lower() for m in _CHANNEL_MAP_RE.finditer(blocks[0])}

    experiments = []
    for block in blocks[1:]:
        date_m, sample_m = _DATE_RE.search(block), _SAMPLE_RE.search(block)
        if not date_m or not sample_m:
            continue
        dd, mm, yy = date_m.groups()
        points = sorted(
            (
                {
                    "point": int(pm.group(1)),
                    "concentrations_mM": [float(pm.group(2)), float(pm.group(3)), float(pm.group(4))],
                    "timings_s": [float(pm.group(5)), float(pm.group(6)), float(pm.group(7))],
                }
                for pm in _PT_RE.finditer(block)
            ),
            key=lambda p: p["point"],
        )
        stop_m = _STOP_RE.search(block)
        type_m, electrode_m = _TYPE_RE.search(block), _ELECTRODE_RE.search(block)
        experiments.append({
            "date": f"20{yy}-{mm}-{dd}",
            "sample_no": int(sample_m.group(1)),
            "experiment_type": type_m.group(1).strip() if type_m else None,
            "electrode_id": electrode_m.group(1) if electrode_m else None,
            "points": points,
            "stop_s": float(stop_m.group(1)) if stop_m else None,
        })
    return channel_analyte, experiments


def _lookup_protocol(csv_path, sampleinfo_path):
    m = _FILENAME_RE.search(Path(csv_path).stem)
    if not m:
        return None
    date_str = f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:]}"
    sample_no, channel = int(m.group(2)), int(m.group(3))

    channel_analyte, experiments = _parse_sampleinfo(sampleinfo_path)
    analyte = channel_analyte.get(channel)
    ch_idx = channel - 1

    for exp in experiments:
        if exp["date"] == date_str and exp["sample_no"] == sample_no:
            if not exp["points"]:
                return None
            return {
                "analyte": analyte,
                "channel": channel,
                "electrode_id": exp["electrode_id"],
                "experiment_type": exp["experiment_type"],
                "concentrations_mM": [p["concentrations_mM"][ch_idx] for p in exp["points"]],
                "timings_s": [p["timings_s"][ch_idx] for p in exp["points"]],
                "stop_s": exp["stop_s"],
            }
    return None


def analyze_ca_calibration(
    csv_path: str,
    sampleinfo_path: str = "reference_data/sampleinfo_ca.txt",
    concentrations_mM=None,
    timings_s=None,
    stop_s: float = None,
    saturation_threshold: float = 0.5,
) -> dict:
    if not Path(csv_path).is_file():
        return {"status": "error", "message": f"File not found: {csv_path}"}

    protocol_meta = {}
    if concentrations_mM is None or timings_s is None:
        try:
            protocol = _lookup_protocol(csv_path, sampleinfo_path)
        except FileNotFoundError as e:
            return {"status": "error", "message": str(e)}
        if protocol is None:
            return {
                "status": "error",
                "message": (
                    f"No calibration protocol found for {csv_path} in {sampleinfo_path}. "
                    "Pass concentrations_mM and timings_s explicitly."
                ),
            }
        concentrations_mM = protocol["concentrations_mM"]
        timings_s = protocol["timings_s"]
        stop_s = stop_s if stop_s is not None else protocol["stop_s"]
        protocol_meta = {
            "analyte": protocol["analyte"],
            "channel": protocol["channel"],
            "electrode_id": protocol["electrode_id"],
            "experiment_type": protocol["experiment_type"],
        }

    if len(concentrations_mM) != len(timings_s):
        return {"status": "error", "message": "concentrations_mM and timings_s must be the same length."}

    data = load_instrument_csv(csv_path)
    if data is None or data.shape[1] < 2:
        return {"status": "error", "message": f"Could not parse time/current data from {csv_path}."}
    time_s, current_a = data[:, 0], data[:, 1] * 1e-6  # µA -> A

    order = np.argsort(timings_s)
    concentrations_mM = [concentrations_mM[i] for i in order]
    timings_s = [timings_s[i] for i in order]
    window_ends = timings_s[1:] + [stop_s if stop_s is not None else time_s[-1]]

    # Baseline: current just before the first spike (used as the "blank" for LOD).
    baseline_mask = time_s < timings_s[0]
    baseline_current = current_a[baseline_mask] if baseline_mask.any() else current_a[:5]
    baseline_a = float(np.mean(baseline_current))
    baseline_noise_a = float(np.std(baseline_current))

    calibration_points = []
    for conc, t_start, t_end in zip(concentrations_mM, timings_s, window_ends):
        window = (time_s >= t_start) & (time_s < t_end)
        if not window.any():
            continue
        # Steady-state response: average over the last 20% of the window
        # (right before the next spike, once the signal has settled).
        idx = np.where(window)[0]
        tail = idx[int(len(idx) * 0.8):] if len(idx) >= 5 else idx
        response_a = float(np.mean(current_a[tail])) - baseline_a
        calibration_points.append({
            "concentration_mM": conc,
            "response_a": response_a,
            "response_ua": response_a * 1e6,
        })

    if len(calibration_points) < 2:
        return {
            "status": "error",
            "message": "Fewer than 2 calibration points could be extracted from the trace.",
        }

    x = np.array([p["concentration_mM"] for p in calibration_points])
    y = np.array([p["response_a"] for p in calibration_points])

    if len(set(x.tolist())) < 2:
        return {
            "status": "error",
            "message": "All calibration points share the same concentration -- can't fit a calibration "
                       "curve (need at least 2 distinct concentrations).",
        }

    slope, intercept = np.polyfit(x, y, 1)
    y_fit = slope * x + intercept
    ss_res = float(np.sum((y - y_fit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else None

    lod_mM = (3.3 * baseline_noise_a / slope) if slope > 0 else None
    loq_mM = (10 * baseline_noise_a / slope) if slope > 0 else None

    # Saturation: compare the last point-to-point slope against the average
    # of the earlier ones -- a steep drop signals the response is plateauing.
    # The protocol sometimes repeats a concentration across two points (a
    # hold step), so dedupe by concentration (averaging response) first --
    # otherwise a repeated x gives a zero-width, divide-by-zero step.
    x_unique, inverse = np.unique(x, return_inverse=True)
    y_unique = np.array([y[inverse == k].mean() for k in range(len(x_unique))])
    local_slopes = np.diff(y_unique) / np.diff(x_unique)
    if len(local_slopes) < 2:
        saturation = {"detected": False, "note": "Not enough distinct concentration levels to assess."}
    else:
        earlier_avg = float(np.mean(local_slopes[:-1]))
        last = float(local_slopes[-1])
        if earlier_avg <= 0:
            saturation = {"detected": False, "note": "Response isn't consistently increasing, so saturation isn't assessed."}
        else:
            ratio = last / earlier_avg
            saturation = {
                "detected": ratio < saturation_threshold,
                "last_slope_vs_earlier_avg": ratio,
                "note": (
                    f"Response at the highest concentration is only {ratio:.0%} of the "
                    f"earlier average slope -- possible saturation."
                    if ratio < saturation_threshold
                    else "Response stays roughly linear across the calibration range."
                ),
            }

    flags = []
    if slope <= 0:
        flags.append("NON-RESPONSIVE OR INVERTED: fitted slope is not positive -- check electrode/analyte pairing.")
    if r_squared is not None and r_squared < 0.98:
        flags.append(f"POOR LINEARITY: R^2 = {r_squared:.3f} is below 0.98 -- calibration may not be reliable.")
    if saturation["detected"]:
        flags.append("SATURATION: " + saturation["note"])

    status = "fail" if any(f.startswith("NON-RESPONSIVE") for f in flags) else ("warn" if flags else "pass")

    try:
        electrode_id = protocol_meta.get("electrode_id")
        if electrode_id:
            note_code, note_batch = extract_grid_code(electrode_id), extract_batch_date(electrode_id)
            if note_code:
                append_qc_result(
                    note_batch, note_code, tool="ca_calibration", status=status,
                    metrics={
                        "analyte": protocol_meta.get("analyte"),
                        "sensitivity_ua_per_mM": float(slope * 1e6),
                        "r_squared": r_squared,
                        "lod_mM": lod_mM,
                    },
                    file_ref=csv_path, file_kind="ca_files",
                )
    except Exception:
        pass  # note-linking is best-effort, never blocks the QC result

    return {
        "status": status,
        **protocol_meta,
        "calibration_points": calibration_points,
        "sensitivity_a_per_mM": float(slope),
        "sensitivity_ua_per_mM": float(slope * 1e6),
        "intercept_a": float(intercept),
        "r_squared": r_squared,
        "baseline_noise_a": baseline_noise_a,
        "lod_mM": lod_mM,
        "loq_mM": loq_mM,
        "saturation": saturation,
        "flags": flags,
    }
