"""
Dashboard-style rendering for the Streamlit GUI (app.py).

Two things live here:
  1. Per-tool result renderers (DASHBOARD_RENDERERS) -- turn a QC tool's
     raw JSON result into status badges / metric cards, used inline in the
     chat's tool-call panel instead of (in addition to, via a "raw result"
     expander) a plain st.json() dump. Only registered for the tools that
     actually return structured numeric QC metrics worth visualizing --
     literature/note/digest tools stay as their existing JSON/text view,
     since there's nothing meaningfully "dashboard" about a list of papers.
  2. render_batch_dashboard_page() -- a standalone page giving an at-a-
     glance view of one batch: pass/fail counts, a per-electrode status
     grid, fabrication metadata, and an electrochemical outlier sweep.
     Calls the underlying tool functions directly (agent.AVAILABLE_FUNCTIONS),
     not through the LLM -- this is a deterministic view over data already
     on disk, so there's no reason to spend a model call (or the shared
     API quota) rendering it.
"""

from pathlib import Path

import streamlit as st

import agent

_STATUS_STYLE = {
    "pass": ("✅", "green"),
    "ok": ("✅", "green"),
    "warn": ("⚠️", "orange"),
    "preliminary": ("\U0001f7e1", "orange"),
    "fail": ("❌", "red"),
    "error": ("\U0001f534", "red"),
    "insufficient_data": ("ℹ️", "gray"),
    "no_prediction": ("ℹ️", "gray"),
    "framing_rejected": ("\U0001f4f7", "orange"),
}


def status_badge(status: str) -> None:
    icon, color = _STATUS_STYLE.get(status, ("❓", "gray"))
    st.markdown(f":{color}[{icon} **{(status or 'unknown').upper()}**]")


def _fmt(value, kind: str = "num") -> str:
    """Best-effort human formatting -- QC metrics span picoamps to
    percentages to ratios, and a single "%.4g" everywhere reads badly
    (e.g. 0.0001606 A instead of 160.6 uA). Falls back to str() for
    anything that isn't numeric rather than raising mid-render.
    """
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if kind == "amps":
        return f"{v * 1e6:,.2f} µA" if abs(v) < 1 else f"{v:.3g} A"
    if kind == "pct":
        return f"{v:.2%}"
    if kind == "volts":
        return f"{v:.4f} V"
    return f"{v:.4g}"


def _flags(flags) -> None:
    for f in flags or []:
        (st.error if f.upper().startswith(("FAIL", "OUTLIER", "NON-RESPONSIVE")) else st.warning)(f)


def render_sensor_qc(result: dict) -> None:
    status_badge(result.get("status"))
    m = result.get("metrics", {})
    cols = st.columns(4)
    cols[0].metric("Peak Current", _fmt(m.get("peak_current_a"), "amps"), border=True)
    cols[1].metric("ΔEp", _fmt(m.get("delta_ep_v"), "volts") if "delta_ep_v" in m else "—", border=True)
    cols[2].metric("ipa/ipc Ratio", _fmt(m.get("ipa_ipc_ratio")), border=True)
    cols[3].metric("Noise Ratio", _fmt(m.get("noise_ratio"), "pct"), border=True)
    _flags(result.get("flags"))


def render_cv_stability(result: dict) -> None:
    status_badge(result.get("status"))
    s = result.get("stability", {})
    cols = st.columns(3)
    cols[0].metric("Scans Used", f"{len(result.get('scans_used', []))} / {result.get('n_scans_total', '—')}", border=True)
    cols[1].metric("Peak Current CV", _fmt(s.get("peak_current_cv_pct"), "pct"), border=True)
    cols[2].metric("ΔEp CV", _fmt(s.get("delta_ep_cv_pct"), "pct"), border=True)
    per_scan = result.get("per_scan", [])
    if per_scan:
        st.caption("per-scan peak current")
        st.line_chart({"peak_current_a": [s.get("peak_current_a") for s in per_scan]})
    _flags(result.get("flags"))


def render_ca_calibration(result: dict) -> None:
    status_badge(result.get("status"))
    cols = st.columns(4)
    cols[0].metric("Sensitivity", f"{_fmt(result.get('sensitivity_ua_per_mM'))} µA/mM", border=True)
    cols[1].metric("R²", _fmt(result.get("r_squared")), border=True)
    cols[2].metric("LOD", f"{_fmt(result.get('lod_mM'))} mM", border=True)
    cols[3].metric("LOQ", f"{_fmt(result.get('loq_mM'))} mM", border=True)
    points = result.get("calibration_points", [])
    if points:
        st.caption("calibration curve")
        st.scatter_chart({"concentration_mM": [p["concentration_mM"] for p in points], "response_ua": [p["response_ua"] for p in points]}, x="concentration_mM", y="response_ua")
    sat = result.get("saturation", {})
    if sat.get("detected"):
        st.warning(sat.get("note", "Saturation detected."))
    _flags(result.get("flags"))


def render_image_qc(result: dict) -> None:
    status_badge(result.get("status"))
    if result.get("status") == "framing_rejected":
        # No defect analysis was attempted -- the framing gate refused to
        # guess on an unusable photo, so this stops here rather than
        # falling through to the defects/surface-analysis sections below,
        # which would just render as empty and read like a clean pass.
        st.warning(result.get("user_message", "Photo framing failed the pre-analysis check."))
        issues = result.get("issues", [])
        if issues:
            st.caption("issues detected: " + ", ".join(i.replace("_", " ") for i in issues))
        return
    if result.get("proxy_image"):
        st.info(f"Proxy image -- showing {result.get('used_electrode')} (grid distance {result.get('grid_distance')}) instead of the requested {result.get('requested_electrode')}.")
    if result.get("rotation_degrees_applied"):
        st.caption(f"\U0001f504 Straightened {result['rotation_degrees_applied']:+.1f}° before analysis.")
    defects = result.get("defects", [])
    if defects:
        for d in defects:
            st.warning(d)
    elif result.get("status") == "pass":
        st.success("No defects flagged.")
    surface = result.get("surface_analysis")
    if surface and surface.get("status") != "error":
        st.caption("surface roughness (luminance-based proxy, not calibrated physical roughness)")
        r = surface.get("roughness_luminance_units", {})
        cols = st.columns(6)
        for col, key in zip(cols, ("Ra", "Rq", "Rz", "Rt", "Rsk", "Rku")):
            col.metric(key, _fmt(r.get(key)), border=True)
    _flags(result.get("flags"))
    if surface:
        _flags(surface.get("flags"))


def render_compare_to_batch_reference(result: dict) -> None:
    status_badge(result.get("status"))
    if result.get("status") == "framing_rejected":
        st.warning(result.get("user_message", "Photo framing failed the pre-analysis check."))
        issues = result.get("issues", [])
        if issues:
            st.caption("issues detected: " + ", ".join(i.replace("_", " ") for i in issues))
        return
    auto_rot = result.get("rotation_degrees_auto_detected")
    if auto_rot:
        st.caption(f"\U0001f504 Auto-straightened {auto_rot:+.2f}° to match the reference orientation before comparing.")
    m = result.get("metrics", {})
    cols = st.columns(3)
    cols[0].metric("SSIM vs. Batch Avg", _fmt(m.get("ssim_vs_batch_average")), border=True)
    pct = m.get("batch_percentile")
    cols[1].metric("Batch Percentile", f"{pct:.0f}%" if pct is not None else "—", border=True)
    cols[2].metric("Batch N", m.get("batch_n", "—"), border=True)
    _flags(result.get("flags"))


def render_compare_cv_to_batch_reference(result: dict) -> None:
    status_badge(result.get("status"))
    checked = result.get("metrics_checked")
    if isinstance(checked, dict):  # single-electrode mode
        outliers = {k: v for k, v in checked.items() if v.get("is_outlier")}
        normals = {k: v for k, v in checked.items() if not v.get("is_outlier")}
        if outliers:
            st.caption("flagged as outliers")
            cols = st.columns(min(len(outliers), 4))
            for col, (name, d) in zip(cols, outliers.items()):
                col.metric(name, _fmt(d["value"]), delta=f"z={d['modified_z']:+.1f}", delta_color="inverse", border=True)
        if normals:
            st.caption("within normal range" if outliers else "all metrics")
            cols = st.columns(min(len(normals), 5))
            for col, (name, d) in zip(cols, normals.items()):
                col.metric(name, _fmt(d["value"]), delta=f"z={d['modified_z']:+.1f}", delta_color="off", border=True)
    else:  # batch-wide sweep mode -- metrics_checked is a list of metric names here
        outlier_electrodes = result.get("outlier_electrodes", [])
        cols = st.columns(3)
        cols[0].metric("Batch N", result.get("batch_n", "—"), border=True)
        cols[1].metric("Outlier Electrodes", len(outlier_electrodes), border=True)
        cols[2].metric("Metrics Checked", len(checked) if checked else 0, border=True)
        if outlier_electrodes:
            st.warning(f"Outliers: {', '.join(outlier_electrodes)}")
    _flags(result.get("flags"))


def render_check_photo_framing(result: dict) -> None:
    ok = result.get("framing_ok")
    st.markdown(":green[✅ **FRAMING OK**]" if ok else ":orange[\U0001f4f7 **FRAMING ISSUES**]")
    st.caption(result.get("user_message", ""))
    issues = result.get("issues", [])
    if issues:
        cols = st.columns(min(len(issues), 5))
        for col, issue in zip(cols, issues):
            col.metric("issue", issue.replace("_", " "), border=True)


DASHBOARD_RENDERERS = {
    "sensor_qc": render_sensor_qc,
    "analyze_cv_stability": render_cv_stability,
    "ca_calibration": render_ca_calibration,
    "image_qc": render_image_qc,
    "compare_to_batch_reference": render_compare_to_batch_reference,
    "compare_cv_to_batch_reference": render_compare_cv_to_batch_reference,
    "check_photo_framing": render_check_photo_framing,
}


# ---------------------------------------------------------------------------
# Batch dashboard page
# ---------------------------------------------------------------------------

_DIGEST_STATUS_ICON = {"pass": "\U0001f7e2", "warn": "\U0001f7e1", "fail": "\U0001f534"}


def _list_batches() -> list:
    img_root = Path("reference_images")
    if not img_root.exists():
        return []
    return sorted(p.name for p in img_root.iterdir() if p.is_dir())


def render_batch_dashboard_page() -> None:
    st.title("\U0001f4ca Batch Dashboard")
    st.caption("At-a-glance QC status for a batch -- reads existing electrode_notes/reference_images, no model call.")

    batches = _list_batches()
    if not batches:
        st.info("No batches found under reference_images/.")
        return

    batch = st.selectbox("Batch", batches)
    if not batch:
        return

    notes_fn = agent.AVAILABLE_FUNCTIONS["list_electrode_notes"]
    meta_fn = agent.AVAILABLE_FUNCTIONS["get_batch_metadata"]
    outlier_fn = agent.AVAILABLE_FUNCTIONS["compare_cv_to_batch_reference"]

    notes = notes_fn(batch=batch)
    rows = notes.get("notes", [])

    if not rows:
        st.info(f"No electrode notes on record yet for batch '{batch}'.")
        return

    n_pass = sum(1 for r in rows if r.get("last_status") == "pass")
    n_warn = sum(1 for r in rows if r.get("last_status") == "warn")
    n_fail = sum(1 for r in rows if r.get("last_status") == "fail")
    n_other = len(rows) - n_pass - n_warn - n_fail

    cols = st.columns(4)
    cols[0].metric("Electrodes", len(rows), border=True)
    cols[1].metric("Pass", n_pass, border=True)
    cols[2].metric("Warn", n_warn, border=True)
    cols[3].metric("Fail", n_fail, border=True)
    if n_other:
        st.caption(f"{n_other} electrode(s) with a note but no QC status recorded yet.")

    fab_meta = meta_fn(batch=batch)
    if fab_meta.get("status") == "ok":
        sheets = fab_meta.get("sheets", {})
        if sheets:
            st.subheader("Fabrication")
            sheet_cols = st.columns(len(sheets))
            for col, (sheet_number, fields) in zip(sheet_cols, sorted(sheets.items())):
                with col.container(border=True):
                    st.markdown(f"**Sheet {sheet_number}**")
                    for k in ("silver_ink_formula", "carbon_ink_formula", "n_passes", "substrate"):
                        if fields.get(k) is not None:
                            st.caption(f"{k.replace('_', ' ').title()}: {fields[k]}")

    st.subheader("Electrical outliers (CV/CA vs. batch)")
    outliers = outlier_fn(batch=batch)
    if outliers.get("status") == "insufficient_data":
        st.caption(outliers.get("message"))
    elif outliers.get("status") == "error":
        st.caption(outliers.get("message"))
    else:
        flagged = set(outliers.get("outlier_electrodes", []))
        if flagged:
            st.warning(f"{len(flagged)} electrode(s) flagged: {', '.join(sorted(flagged))}")
        else:
            st.success("No electrical outliers flagged in this batch.")

    st.subheader("Electrodes")
    rows_sorted = sorted(rows, key=lambda r: ({"fail": 0, "warn": 1, "pass": 2}.get(r.get("last_status"), 3), r.get("electrode_id") or ""))
    n_cols = 6
    grid = st.columns(n_cols)
    for i, r in enumerate(rows_sorted):
        icon = _DIGEST_STATUS_ICON.get(r.get("last_status"), "⚪")
        with grid[i % n_cols].container(border=True):
            st.markdown(f"**{icon} {r.get('electrode_id')}**")
            st.caption(f"{r.get('n_qc_events', 0)} QC event(s)")
