# reference_data/

Your raw CV/CA instrument exports go here. Not shipped with the repo
(private/large) -- this folder is empty until you add your own.

Expected layout (see the main README's "Sensor data QC" section for
how these get referenced in conversation):

```
reference_data/
  <batch>_cv/            # or cv/ -- per-scan CV CSVs, e.g. 707-A1-1.csv
  ca/                    # CA calibration CSVs
  sampleinfo_ca.txt      # concentration/timing protocol for CA runs
```

Filenames just need to encode a batch date and an electrode code the
agent can parse -- see `tools/sensor_qc.py:parse_cv_filename` if your
naming convention differs from what's already handled.
