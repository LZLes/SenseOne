"""
CV multi-scan stability analysis.

sensor_qc.py's qc_electrochemical_data looks at one scan (by default, the
last of however many a CV file has). That misses what actually matters for
judging a screen-printed electrode from its CV response: every file in this
lab's protocol records several scans -- checked directly against the raw
files rather than assumed, and it varies by batch (20260707: 5 scans;
20260804/20260805: 3) -- and a good electrode should behave *consistently*
scan to scan, not just look fine in isolation on any one of them. This
module adapts to however many scans a given file actually has rather than
hardcoding either count.

Per the researcher's guidance: disregard scan 1 (conditioning/break-in),
then check that peak current and delta Ep stay roughly constant across the
remaining scans -- drift there is a sign of fouling/degradation under
repeated cycling, not something a single-scan snapshot would ever catch.
Also flags per-scan baseline noise and isolated single-point spikes/drops
in the current trace (contact glitches, bubbles) that are structurally
different from a real, smooth redox peak.
"""

from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from tools._io import load_instrument_csv
from tools.electrode_notes import append_qc_result, extract_batch_date
from tools.sensor_qc import parse_cv_filename

CV_STABILITY_SCHEMA = {
    "type": "function",
    "function": {
        "name": "analyze_cv_stability",
        "description": (
            "Analyze a multi-scan CV file's scan-to-scan stability, not just one scan's snapshot. "
            "Scan 1 is disregarded (conditioning/break-in); a good electrode should show peak "
            "current and delta Ep staying roughly constant across the remaining scans -- drift "
            "signals fouling or degradation under cycling. Also flags per-scan baseline noise and "
            "isolated single-point spikes/drops in the current trace (contact glitches, bubbles) "
            "distinct from real, smooth redox peak shape. Use this instead of sensor_qc when "
            "scan-to-scan behavior matters, not just a single scan."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "csv_path": {
                    "type": "string",
                    "description": "Direct path to the CV CSV file. Use this or electrode_code+csv_dir -- don't guess a filename.",
                },
                "electrode_code": {
                    "type": "string",
                    "description": "Sub-electrode id in '<code>-<position>' form, e.g. 'A1-1'. Use with csv_dir instead of csv_path.",
                },
                "csv_dir": {
                    "type": "string",
                    "description": "Directory of CV CSVs to search, used with electrode_code, e.g. reference_data/20260707_cv.",
                },
                "disregard_first_n_scans": {
                    "type": "integer",
                    "description": "Leading scans to exclude as conditioning/break-in. Default 1.",
                },
                "max_delta_ep_v": {"type": "number", "description": "Same convention as sensor_qc. Default 0.15."},
                "max_noise_ratio": {"type": "number", "description": "Same convention as sensor_qc. Default 0.10."},
                "max_stability_cv_pct": {
                    "type": "number",
                    "description": (
                        "Flag as unstable if peak current or delta Ep varies (std/mean) by more "
                        "than this fraction across the used scans. Default 0.15 -- an unvalidated "
                        "starting point, tune once you have known-stable/known-drifting examples."
                    ),
                },
                "artifact_threshold_factor": {
                    "type": "number",
                    "description": (
                        "Flag a single point as an artifact if it deviates from its immediate "
                        "neighbors by more than this many multiples of that scan's own baseline "
                        "noise. Default 10.0."
                    ),
                },
            },
        },
    },
}


def _detect_artifacts(current: np.ndarray, baseline_noise: float, threshold_factor: float) -> int:
    """Count isolated single-point spikes/drops. A real redox peak is smooth
    over several samples; a point that jumps away from the straight-line
    average of its immediate neighbors by many multiples of the scan's own
    noise floor is more likely a contact glitch or bubble than chemistry.
    """
    if baseline_noise <= 0 or len(current) < 3:
        return 0
    neighbor_avg = (current[:-2] + current[2:]) / 2
    deviation = np.abs(current[1:-1] - neighbor_avg)
    return int(np.sum(deviation > threshold_factor * baseline_noise))


def _scan_metrics(potential, current, max_delta_ep_v, max_noise_ratio, artifact_threshold_factor):
    n_baseline = max(5, int(0.1 * len(current)))
    baseline_noise = float(np.std(current[:n_baseline]))
    peak_current = float(np.max(np.abs(current)))
    noise_ratio = baseline_noise / peak_current if peak_current > 0 else float("inf")
    n_artifacts = _detect_artifacts(current, baseline_noise, artifact_threshold_factor)

    pos_peaks, _ = find_peaks(current, prominence=(np.max(current) - np.min(current)) * 0.05)
    neg_peaks, _ = find_peaks(-current, prominence=(np.max(current) - np.min(current)) * 0.05)

    result = {
        "baseline_noise_a": baseline_noise,
        "peak_current_a": peak_current,
        "noise_ratio": noise_ratio,
        "n_artifacts": n_artifacts,
        "delta_ep_v": None,
        "flags": [],
    }
    if noise_ratio > max_noise_ratio:
        result["flags"].append(f"HIGH NOISE: {noise_ratio:.1%} of peak current (threshold {max_noise_ratio:.0%}).")
    if n_artifacts > 0:
        result["flags"].append(f"ARTIFACT: {n_artifacts} isolated spike(s)/drop(s) inconsistent with smooth peak shape.")

    if len(pos_peaks) > 0 and len(neg_peaks) > 0:
        e_anodic = potential[pos_peaks[np.argmax(current[pos_peaks])]]
        e_cathodic = potential[neg_peaks[np.argmin(current[neg_peaks])]]
        delta_ep = float(abs(e_anodic - e_cathodic))
        result["delta_ep_v"] = delta_ep
        if delta_ep > max_delta_ep_v:
            result["flags"].append(f"POOR REVERSIBILITY: delta Ep = {delta_ep:.3f} V exceeds {max_delta_ep_v:.3f} V.")
    else:
        result["flags"].append("Could not identify both anodic and cathodic peaks.")

    return result


def _coefficient_of_variation(values):
    if len(values) < 2 or np.mean(values) == 0:
        return None
    return float(np.std(values) / abs(np.mean(values)))


def _resolve_cv_path(electrode_code: str, csv_dir: str):
    # parse_cv_filename normalizes both naming conventions to the same
    # "<code>-<position>" / "<sheet>-<code>" note_code shape, so the caller's
    # electrode_code can be matched against it directly either way.
    target = electrode_code.strip().upper()
    for path in sorted(Path(csv_dir).glob("*.csv")):
        parsed = parse_cv_filename(path.stem)
        if parsed and parsed[0].upper() == target:
            return str(path)
    return None


def analyze_cv_stability(
    csv_path: str = "",
    electrode_code: str = "",
    csv_dir: str = "",
    disregard_first_n_scans: int = 1,
    max_delta_ep_v: float = 0.15,
    max_noise_ratio: float = 0.10,
    max_stability_cv_pct: float = 0.15,
    artifact_threshold_factor: float = 10.0,
) -> dict:
    if not csv_path:
        if not electrode_code or not csv_dir:
            return {"status": "error", "message": "Provide either csv_path, or both electrode_code and csv_dir."}
        csv_path = _resolve_cv_path(electrode_code, csv_dir)
        if csv_path is None:
            return {"status": "error", "message": f"No CV file found for '{electrode_code}' in {csv_dir}."}

    if not Path(csv_path).is_file():
        return {"status": "error", "message": f"File not found: {csv_path}"}

    data = load_instrument_csv(csv_path)
    if data is None or data.ndim != 2 or data.shape[1] < 2 or data.shape[1] % 2 != 0:
        return {"status": "error", "message": f"Could not find multi-scan CV data in {csv_path}."}

    n_scans = data.shape[1] // 2
    if disregard_first_n_scans < 0:
        # range(negative, n_scans) doesn't error -- it produces negative
        # indices that silently wrap around via Python's negative-index
        # semantics (data[:, 2*idx] aliases columns from the end), reporting
        # duplicate/garbage "scans" under nonsensical negative scan numbers
        # instead of failing. Reject it explicitly rather than let that happen.
        return {
            "status": "error",
            "message": f"disregard_first_n_scans must be >= 0, got {disregard_first_n_scans}.",
        }
    used_indices = list(range(disregard_first_n_scans, n_scans))
    if not used_indices:
        return {
            "status": "error",
            "message": f"Only {n_scans} scan(s) in {csv_path}; disregard_first_n_scans={disregard_first_n_scans} leaves none to analyze.",
        }

    per_scan = []
    for idx in used_indices:
        potential = data[:, 2 * idx]
        current = data[:, 2 * idx + 1] * 1e-6  # microamps -> amps
        m = _scan_metrics(potential, current, max_delta_ep_v, max_noise_ratio, artifact_threshold_factor)
        m["scan"] = idx + 1  # 1-based, matches the file's own "Scan N" labeling
        per_scan.append(m)

    peak_currents = [s["peak_current_a"] for s in per_scan]
    delta_eps = [s["delta_ep_v"] for s in per_scan if s["delta_ep_v"] is not None]
    peak_current_cv_pct = _coefficient_of_variation(peak_currents)
    delta_ep_cv_pct = _coefficient_of_variation(delta_eps)

    fail_flags, warn_flags = [], []
    for s in per_scan:
        for f in s["flags"]:
            (fail_flags if f.startswith(("POOR REVERSIBILITY", "Could not identify")) else warn_flags).append(f"Scan {s['scan']}: {f}")

    if peak_current_cv_pct is not None and peak_current_cv_pct > max_stability_cv_pct:
        fail_flags.append(
            f"UNSTABLE PEAK CURRENT: varies {peak_current_cv_pct:.1%} (std/mean) across scans "
            f"{[s['scan'] for s in per_scan]} -- threshold {max_stability_cv_pct:.0%}. A good "
            "electrode should stay roughly constant scan to scan; drift suggests fouling or "
            "degradation under repeated cycling."
        )
    if delta_ep_cv_pct is not None and delta_ep_cv_pct > max_stability_cv_pct:
        fail_flags.append(
            f"UNSTABLE DELTA EP: varies {delta_ep_cv_pct:.1%} (std/mean) across scans "
            f"{[s['scan'] for s in per_scan]} -- threshold {max_stability_cv_pct:.0%}."
        )

    status = "fail" if fail_flags else ("warn" if warn_flags else "pass")
    flags = fail_flags + warn_flags

    try:
        parsed = parse_cv_filename(Path(csv_path).stem)
        if parsed:
            note_code, batch_override = parsed
            batch = batch_override or extract_batch_date(Path(csv_path).parent.name)
            append_qc_result(
                batch, note_code, tool="analyze_cv_stability",
                status=status,
                metrics={
                    "peak_current_cv_pct": peak_current_cv_pct,
                    "delta_ep_cv_pct": delta_ep_cv_pct,
                    "n_scans_used": len(per_scan),
                },
                file_ref=csv_path, file_kind="cv_files",
            )
    except Exception:
        pass  # note-linking is best-effort, never blocks the QC result

    return {
        "status": status,
        "n_scans_total": n_scans,
        "scans_used": [s["scan"] for s in per_scan],
        "per_scan": per_scan,
        "stability": {
            "peak_current_cv_pct": peak_current_cv_pct,
            "delta_ep_cv_pct": delta_ep_cv_pct,
        },
        "flags": flags,
    }
