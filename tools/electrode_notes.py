"""
Electrode notes: a persistent, per-physical-electrode record that ties
together everything the other tools learn about it -- its photo, its CV/CA
files, and every QC result across sessions. Without this, each tool only
knows about the one file it's looking at; there's no memory that "E5" has
been checked five times and failed twice.

Same format as tools/literature.py's vault notes (JSON frontmatter + free
markdown body), for the same reasons: simple to parse reliably, still
readable if opened directly in an editor. Hybrid write model:
  - qc_history entries are auto-appended by sensor_qc (CV only -- CA files
    don't carry a grid code in their filename), ca_calibration, image_qc,
    and compare_to_batch_reference whenever they can determine which
    electrode they just looked at. Best-effort and non-fatal -- a note
    failing to write never blocks the actual QC result.
  - The body is for manual notes, either hand-edited in the file directly
    or added via add_electrode_note (e.g. when the researcher tells the
    agent something worth remembering mid-conversation).

One file per (batch, electrode_code), e.g. electrode_notes/20260707/E5.md.
"batch" is the fabrication/imaging date (the leading 8-digit token), not
the folder a given file happens to sit in -- reference_images/20260707,
reference_data/20260707_cv, and a CA sample's "20260707_BLISS_2_2F8" electrode
ID all reduce to the same "20260707" batch key, so a CV run, a CA calibration,
and a photo of the same physical electrode land in one note instead of three.
"""

import json
import re
from datetime import date
from pathlib import Path

NOTES_DIR = Path("electrode_notes")
_GRID_CODE_RE = re.compile(r"([A-Za-z])(\d+)$")
_SHEET_CODE_RE = re.compile(r"-([A-Za-z]\d+)-([A-Za-z]\d+)$")  # e.g. "...-S3-A1" -> sheet="S3", code="A1"
_BATCH_DATE_RE = re.compile(r"^(\d{8})")

GET_ELECTRODE_NOTE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_electrode_note",
        "description": (
            "Read the persistent note for one electrode -- its linked photo/CV/CA files, full "
            "QC history across every past tool call, and any manual notes. Call this before "
            "answering questions about an electrode's history/track record instead of assuming "
            "you only know what's in the current conversation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "batch": {"type": "string", "description": "Fabrication/imaging batch date, e.g. '20260707'."},
                "electrode_code": {"type": "string", "description": "Grid position, e.g. 'E5'."},
            },
            "required": ["batch", "electrode_code"],
        },
    },
}

ADD_ELECTRODE_NOTE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "add_electrode_note",
        "description": (
            "Append a manual, dated note to an electrode's persistent record -- use this when "
            "the researcher tells you something about a specific electrode worth remembering "
            "for future sessions (a visual observation, a decision, context a QC tool wouldn't capture)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "batch": {"type": "string", "description": "Fabrication/imaging batch date, e.g. '20260707'."},
                "electrode_code": {"type": "string", "description": "Grid position, e.g. 'E5'."},
                "note_text": {"type": "string", "description": "The note to record."},
            },
            "required": ["batch", "electrode_code", "note_text"],
        },
    },
}

LIST_ELECTRODE_NOTES_SCHEMA = {
    "type": "function",
    "function": {
        "name": "list_electrode_notes",
        "description": "List every electrode with a note on record, optionally filtered to one batch.",
        "parameters": {
            "type": "object",
            "properties": {
                "batch": {"type": "string", "description": "Optional: restrict to one batch date, e.g. '20260707'."},
            },
        },
    },
}


def extract_batch_date(s: str) -> str:
    """Leading 8-digit date token from a folder name, filename, or electrode
    ID (e.g. "20260707", "20260707_cv", "20260707_BLISS_2_2F8" all -> "20260707").
    Falls back to the raw string if no date is found, so a note still gets
    created rather than silently skipped.
    """
    m = _BATCH_DATE_RE.match(s)
    return m.group(1) if m else s


def extract_grid_code(s: str):
    """Trailing grid code, sheet-aware where the filename encodes one.
    Handles "F8" from "20260707_BLISS_2_2F8" (CA electrode ID), "A5" from
    "707-A5-1" (CV filename), "A1" from "20260707_A1" (single-sheet image),
    and "S3-A1" from "20260804-S3-A1" (multi-sheet image -- sheet has to be
    kept here or sheet S1's A1 and sheet S3's A1 collide into one note).
    Returns None if nothing matches.
    """
    m = _SHEET_CODE_RE.search(s)
    if m:
        return f"{m.group(1).upper()}-{m.group(2).upper()}"
    m = _GRID_CODE_RE.search(s)
    return f"{m.group(1).upper()}{m.group(2)}" if m else None


def note_path(batch: str, electrode_code: str) -> Path:
    return NOTES_DIR / batch / f"{electrode_code.upper()}.md"


def _load_note(path: Path):
    text = path.read_text(encoding="utf-8")
    _, meta_block, body = text.split("---", 2)
    return json.loads(meta_block), body.strip()


def _write_note(path: Path, meta: dict, body: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\n{json.dumps(meta, indent=2)}\n---\n\n{body.strip()}\n", encoding="utf-8")


def _read_or_init(batch: str, electrode_code: str):
    path = note_path(batch, electrode_code)
    if path.exists():
        try:
            return path, *_load_note(path)
        except Exception:
            pass
    meta = {"electrode_id": electrode_code.upper(), "batch": batch, "created": date.today().isoformat()}
    return path, meta, ""


def append_qc_result(
    batch: str,
    electrode_code: str,
    tool: str,
    status: str,
    metrics: dict = None,
    file_ref: str = None,
    file_kind: str = None,
) -> None:
    """Best-effort -- callers should wrap this in try/except so a note-write
    failure never breaks the actual QC result being returned to the user.
    """
    path, meta, body = _read_or_init(batch, electrode_code)

    if file_ref and file_kind:
        refs = meta.setdefault(file_kind, [])
        if file_ref not in refs:
            refs.append(file_ref)

    history = meta.setdefault("qc_history", [])
    entry = {"date": date.today().isoformat(), "tool": tool, "status": status}
    if metrics:
        entry["metrics"] = metrics
    history.append(entry)

    _write_note(path, meta, body)
    try:
        generate_batch_digest(batch)
    except Exception:
        pass  # digest is a convenience view -- never block the actual note write over it


def add_electrode_note(batch: str, electrode_code: str, note_text: str) -> dict:
    path, meta, body = _read_or_init(batch, electrode_code)
    entry = f"### {date.today().isoformat()}\n{note_text}"
    body = f"{body}\n\n{entry}" if body else entry
    _write_note(path, meta, body)
    return {"status": "ok", "note_path": str(path)}


def get_electrode_note(batch: str, electrode_code: str) -> dict:
    path = note_path(batch, electrode_code)
    if not path.exists():
        return {"status": "error", "message": f"No note on record for {electrode_code} in batch {batch}."}
    meta, body = _load_note(path)
    return {"status": "ok", "meta": meta, "notes": body}


_STATUS_RANK = {"fail": 0, "warn": 1, "pass": 2}
DIGEST_FILENAME = "_digest.md"  # leading "_" sorts first and can't collide with a real grid code
BATCH_INFO_FILENAME = "_batch_info.md"

SET_BATCH_METADATA_SCHEMA = {
    "type": "function",
    "function": {
        "name": "set_batch_metadata",
        "description": (
            "Record or update fabrication process metadata for one sheet within a batch -- ink "
            "formulas, print passes, substrate. A batch date can have multiple sheets, each with "
            "its own ink/passes/substrate -- this is keyed by (batch, sheet_number), not batch "
            "alone. Applies to every electrode printed on that sheet, not one specific electrode "
            "(use add_electrode_note for per-electrode observations instead). Only pass the "
            "fields you're setting/changing -- existing fields are preserved, not overwritten with blanks."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "batch": {"type": "string", "description": "Fabrication/imaging batch date, e.g. '20260707'."},
                "sheet_number": {"type": "string", "description": "Print sheet/panel identifier, e.g. 'S1'."},
                "silver_ink_formula": {"type": "string", "description": "Silver (reference electrode/trace) ink formulation."},
                "carbon_ink_formula": {"type": "string", "description": "Carbon (working/counter electrode) ink formulation."},
                "n_passes": {"type": "integer", "description": "Number of screen-print passes."},
                "substrate": {"type": "string", "description": "Substrate material, e.g. 'PET', 'polyimide'."},
            },
            "required": ["batch", "sheet_number"],
        },
    },
}

GET_BATCH_METADATA_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_batch_metadata",
        "description": (
            "Read a batch's recorded fabrication process metadata, per sheet (ink formulas, "
            "passes, substrate). Pass sheet_number for just one sheet, or omit it for every "
            "sheet recorded under that batch."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "batch": {"type": "string", "description": "Fabrication/imaging batch date, e.g. '20260707'."},
                "sheet_number": {"type": "string", "description": "Optional: restrict to one sheet, e.g. 'S1'."},
            },
            "required": ["batch"],
        },
    },
}

_SHEET_METADATA_FIELDS = ("silver_ink_formula", "carbon_ink_formula", "n_passes", "substrate")


def batch_metadata_path(batch: str) -> Path:
    return NOTES_DIR / batch / BATCH_INFO_FILENAME


def set_batch_metadata(batch: str, sheet_number: str, **fields) -> dict:
    path = batch_metadata_path(batch)
    if path.exists():
        try:
            meta, body = _load_note(path)
        except Exception:
            meta, body = {"batch": batch, "sheets": {}}, ""
    else:
        meta, body = {"batch": batch, "sheets": {}}, ""
    meta.setdefault("sheets", {})

    sheet = meta["sheets"].setdefault(sheet_number, {})
    for k in _SHEET_METADATA_FIELDS:
        v = fields.get(k)
        if v is not None:
            sheet[k] = v
    sheet["updated"] = date.today().isoformat()

    _write_note(path, meta, body)
    try:
        generate_batch_digest(batch)
    except Exception:
        pass  # digest is a convenience view -- never block the actual metadata write over it
    return {"status": "ok", "batch": batch, "sheet_number": sheet_number, "metadata": sheet}


def get_batch_metadata(batch: str, sheet_number: str = None) -> dict:
    path = batch_metadata_path(batch)
    if not path.exists():
        return {"status": "error", "message": f"No fabrication metadata on record for batch '{batch}'."}
    meta, body = _load_note(path)
    sheets = meta.get("sheets", {})
    if sheet_number:
        if sheet_number not in sheets:
            return {"status": "error", "message": f"No sheet '{sheet_number}' on record for batch '{batch}'."}
        return {"status": "ok", "batch": batch, "sheet_number": sheet_number, "metadata": sheets[sheet_number], "notes": body}
    return {"status": "ok", "batch": batch, "sheets": sheets, "notes": body}


def generate_batch_digest(batch: str):
    """Compact, always-small index over a batch -- one line per electrode
    (code, last status, last tool, event count), not full history. Stays
    cheap to read regardless of how much detail accumulates in the
    per-electrode files it summarizes; see conversation for why those
    stay separate files rather than one merged per-batch note (a merged
    file would grow toward ~100k tokens at realistic per-electrode QC
    history, and every electrode's QC event would contend to rewrite the
    same shared file). Regenerated automatically by append_qc_result.
    Returns the markdown content, or None if the batch has no notes yet.
    """
    batch_dir = NOTES_DIR / batch
    if not batch_dir.exists():
        return None

    rows = []
    for path in sorted(batch_dir.glob("*.md")):
        if path.name in (DIGEST_FILENAME, BATCH_INFO_FILENAME):
            continue
        try:
            meta, _ = _load_note(path)
        except Exception:
            continue
        history = meta.get("qc_history", [])
        last = history[-1] if history else {}
        rows.append({
            "code": meta.get("electrode_id", path.stem),
            "status": last.get("status", "-"),
            "tool": last.get("tool", "-"),
            "n_events": len(history),
        })

    rows.sort(key=lambda r: (_STATUS_RANK.get(r["status"], 3), r["code"]))

    lines = [f"# Batch {batch} digest", ""]

    sheets = {}
    try:
        fab_path = batch_metadata_path(batch)
        if fab_path.exists():
            fab_meta, _ = _load_note(fab_path)
            sheets = fab_meta.get("sheets", {})
    except Exception:
        pass
    if sheets:
        lines += ["## Fabrication", ""]
        for sheet_number, fields in sorted(sheets.items()):
            lines.append(f"**Sheet {sheet_number}**")
            for k in _SHEET_METADATA_FIELDS:
                if fields.get(k) is not None:
                    lines.append(f"- **{k.replace('_', ' ').title()}**: {fields[k]}")
            lines.append("")

    lines += [
        f"_Updated {date.today().isoformat()}, {len(rows)} electrode(s) with notes. "
        f"See electrode_notes/{batch}/<code>.md for full history._",
        "",
        "| Code | Last Status | Last Tool | QC Events |",
        "| --- | --- | --- | --- |",
    ] + [f"| {r['code']} | {r['status']} | {r['tool']} | {r['n_events']} |" for r in rows]

    content = "\n".join(lines) + "\n"
    (batch_dir / DIGEST_FILENAME).write_text(content, encoding="utf-8")
    return content


GET_BATCH_DIGEST_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_batch_digest",
        "description": (
            "Get a compact, at-a-glance summary of every electrode in a batch (code, last QC "
            "status, last tool, event count) -- a small index, not full history, that stays "
            "cheap to read no matter how much detail accumulates underneath it. Call "
            "get_electrode_note for one electrode's full history instead. Regenerates "
            "automatically whenever any electrode in the batch gets a new QC result."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "batch": {"type": "string", "description": "Fabrication/imaging batch date, e.g. '20260707'."},
            },
            "required": ["batch"],
        },
    },
}


def get_batch_digest(batch: str) -> dict:
    content = generate_batch_digest(batch)
    if content is None:
        return {"status": "error", "message": f"No notes found for batch '{batch}'."}
    return {"status": "ok", "batch": batch, "digest": content}


def list_electrode_notes(batch: str = None) -> dict:
    if not NOTES_DIR.exists():
        return {"n_notes": 0, "notes": []}
    search_dir = (NOTES_DIR / batch) if batch else NOTES_DIR
    if not search_dir.exists():
        return {"n_notes": 0, "notes": []}

    notes = []
    for path in sorted(search_dir.glob("**/*.md")):
        if path.name in (DIGEST_FILENAME, BATCH_INFO_FILENAME):
            continue
        try:
            meta, _ = _load_note(path)
        except Exception:
            continue
        history = meta.get("qc_history", [])
        notes.append({
            "electrode_id": meta.get("electrode_id"),
            "batch": meta.get("batch"),
            "n_qc_events": len(history),
            "last_status": history[-1]["status"] if history else None,
        })
    return {"n_notes": len(notes), "notes": notes}
