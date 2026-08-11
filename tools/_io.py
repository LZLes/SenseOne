"""
Shared raw-instrument-CSV loading, used by sensor_qc.py and ca_calibration.py.

The exported files are messier than a plain CSV: UTF-16 or legacy
single-byte encoded, with a multi-line metadata header ("Date and time:",
"Notes:", ...) before the real data, and (for CV) several scans laid out
side by side as repeated potential/current column pairs.
"""

import numpy as np


def load_instrument_csv(path):
    """Parse a raw instrument export. Returns a 2D numpy array of the
    numeric data block, or None if no such block is found.

    The header/data boundary is detected structurally -- the first
    non-numeric row immediately followed by an all-numeric row of the same
    width -- rather than by matching exact column label text, since
    encoding quirks mangle "µA" unpredictably across files.
    """
    raw = open(path, "rb").read()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        text = raw.decode("utf-16")
    else:
        text = raw.decode("latin-1")

    lines = [ln for ln in text.splitlines() if ln.strip() != ""]

    def _row_values(line):
        try:
            return [float(p) for p in line.split(",")]
        except ValueError:
            return None

    for i, line in enumerate(lines[:-1]):
        if _row_values(line) is not None:
            continue  # numeric data row, not a header candidate
        width = len(line.split(","))
        next_vals = _row_values(lines[i + 1])
        if next_vals is not None and len(next_vals) == width:
            data = []
            for ln in lines[i + 1:]:
                vals = _row_values(ln)
                if vals is None or len(vals) != width:
                    break
                data.append(vals)
            return np.array(data)

    return None
