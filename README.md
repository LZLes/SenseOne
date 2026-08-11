# SenseOne

A local, Ollama-backed research assistant for an electrochemical
biosensor (screen-printed electrode) lab. Everything runs on-device —
no data leaves your machine. It QCs raw CV/CA data, visually QCs
electrode photos, tracks every result against the same physical
electrode across sessions, searches and caches literature, and (as
enough paired data accumulates) tries to answer the actual question
the lab cares about: does an electrode's photo predict how well it'll
perform electrically, before you run it?

## Setup

```bash
# install and start Ollama first: https://ollama.com/download

./setup.sh   # pulls both models, creates a venv, installs deps
```

or by hand:

```bash
ollama pull qwen3:8b        # main chat model -- tool-calling + reasoning
ollama pull qwen2.5vl:7b    # vision model -- used internally by image tools

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Then run either interface -- both work against the same tools:

```bash
python agent.py             # terminal chat
streamlit run app.py        # local web GUI (localhost:8501) -- image upload,
                             # live thinking stream, inline plots/photos
```

If you're on limited memory (this was built/tested on an 18GB Apple
Silicon Mac), see the `OLLAMA_*` env vars mentioned in "Running
efficiently" below before you start chatting.

Everything (models, data, generated notes) stays on your machine --
nothing here talks to a remote server except the literature search,
which hits public PubMed/arXiv APIs. `reference_data/` and
`reference_images/` ship empty (see the README in each) since your lab
data is yours to keep private or share separately; everything else
(`electrode_notes/`, `literature_vault/`, `surface_plots/`,
`reference_diff/`) regenerates automatically as you use the agent.

## What to ask it, and what it can actually do

This is a tool-calling agent, not a chatbot with vision built in — it
decides which of ~17 tools to call based on what you ask, then reports
back. You don't need to name tools; plain requests work. A few notes
up front:

- **It won't guess.** If it doesn't have grounds for a claim (a tool
  result, a cited paper), it says so rather than making something up.
  For predictions specifically, see below.
- **Every QC result it runs gets remembered** against the specific
  physical electrode, automatically — ask about history days later and
  it'll still know.

### Sensor data QC (CV / CA)

```
you> QC the CV run at reference_data/20260707_cv/707-A1-1.csv
you> Check the scan-to-scan stability of electrode A1-1 in batch 20260707
you> Analyze the CA calibration for reference_data/ca/20260703_sample1_ch1.csv
```

- **`sensor_qc`** — single-scan snapshot: peak current, ΔEp, formal
  potential, ipa/ipc ratio, noise ratio.
- **`analyze_cv_stability`** — prefer this over `sensor_qc` when you
  care about more than one scan. Disregards the first scan
  (conditioning), then checks that peak current and ΔEp stay roughly
  constant across the rest — flags drift, per-scan noise, and isolated
  spikes/drops distinct from real peak shape. Adapts to however many
  scans a file actually has (varies by batch — checked directly, not
  assumed).
- **`ca_calibration`** — multi-point calibration analysis: sensitivity,
  LOD/LOQ, linearity (R²), saturation at high concentration. Looks up
  the concentration/timing protocol from `reference_data/sampleinfo_ca.txt`
  automatically.

### Electrode photos

```
you> Check electrode E5 in batch 20260707 for defects
you> Give me the surface roughness for S3-A2 in batch 20260804
you> Does A1 look unusual compared to the rest of its batch?
```

- **`image_qc`** — vision-model defect read (incomplete printing,
  contamination, cracks) plus, if asked, quantitative ISO-4287
  roughness parameters (Ra/Rq/Rz/Rt/Rsk/Rku) rendered as a 3D
  luminance-as-height surface plot. These are a photo-based proxy, not
  calibrated physical roughness (no real µm without profilometry) —
  useful for comparing your own electrodes to each other.
- **`compare_to_batch_reference`** — unsupervised outlier check: builds
  a per-pixel average across a batch and reports how much one
  electrode's print deviates from it (SSIM + percentile), no labeled
  examples needed.
- If you ask about a grid position with no photo on file, it'll
  substitute the nearest photographed neighbor and tell you plainly
  that's what happened.
- A 20260707-style photo shows **3 distinct physical electrodes** in
  one frame (top-left/top-right/bottom-left; bottom-right is always
  the unfilled reference) — ask about a specific one and it'll handle
  the crop itself. Newer batches (20260804 on) are one electrode per
  photo. Batches with more than one sheet need a `<sheet>-<code>`
  reference (e.g. "S3-A1"), since a bare code is ambiguous there.

### Uploading your own photo (GUI)

In the Streamlit app, use the sidebar uploader — the photo is saved
locally and attached to your next message. Ask something like:

```
Give me some insights on this electrode and a hedged prediction of its performance.
```

It'll run the photo through defect/roughness QC, compare it against
the relevant batch's average pattern, and give you a performance read
— **as its own uploaded photo**, never substituted with or matched
against a different cataloged file.

### Predicting electrical performance from a photo

This is the central, hardest question the agent is built around, and
it's honest about where that stands:

```
you> Will this electrode perform well electrically?
you> Just give me a preliminary prediction despite the limited data
you> What's the actual relationship between roughness and CV performance?
```

- **`predict_electrode_performance`** only asserts a real, grounded
  prediction if a *statistically significant* visual-vs-CV correlation
  currently exists in the accumulated data. As of the last check, it
  doesn't (small sample, closest relationship found was short of
  significant) — so by default it gives you the honest visual read
  plus a clearly-labeled, explicitly-hedged "preliminary" best-effort
  guess (real r/p/n shown) rather than either a fake-confident verdict
  or a flat refusal.
- **`correlate_visual_cv_performance`** — run this directly to see the
  actual current relationship (or lack of one) between every visual
  feature and every CV/CA metric on record.
- The more CV/CA data you feed in per electrode, the more this
  strengthens automatically — no extra setup needed, it just re-checks
  the correlation each time using whatever's accumulated.

### Electrode & batch memory

```
you> Has electrode E5 ever failed QC before?
you> Note that A1's trace looked fine on re-inspection today
you> Give me an overview of how batch 20260804 is doing
you> For batch 20260805, sheet S1 used Dycotec silver ink, 2 passes, PET substrate
```

Every QC tool above auto-links its result to a persistent note for
that specific physical electrode (`electrode_notes/<batch>/<code>.md`)
— tying together its photo, CV/CA files, and full history across
sessions. `get_electrode_note` / `add_electrode_note` /
`list_electrode_notes` / `get_batch_digest` / `set_batch_metadata` let
you (or the agent) read and add to that record. The digest for a batch
is a compact one-line-per-electrode summary, not a full history dump,
so it stays cheap to read as detail piles up underneath it.

### Literature

```
you> Search the literature for Nafion permselectivity in wearable biosensors
you> What's in the literature vault so far?
you> Pull the figures from that arXiv paper
you> Update the literature vault
```

- **`search_literature`** searches PubMed + arXiv; repeat queries are
  served from a local cache (`literature_vault/`) instead of
  re-fetching — pass `refresh=true` for the freshest results.
- **`analyze_literature_figures`** extracts and capitions figures from
  a paper (arXiv always; PubMed only if open access via PMC).
  **`update_literature_vault`** sweeps every paper missing figures and
  regenerates a citation-grounded synthesis (`literature_vault/INSIGHTS.md`).
- Every claim drawn from a paper gets tagged with its source inline —
  if it can't be tied to a specific paper, it's flagged as the model's
  own inference instead.

## Running efficiently (Apple Silicon / limited memory)

Ollama already uses the GPU (Metal) automatically. If you're tight on
RAM, these env vars help a lot (`launchctl setenv ...` then restart
Ollama, since it runs as a background app, not a shell process):

```bash
launchctl setenv OLLAMA_MAX_LOADED_MODELS 1   # don't keep both models resident at once
launchctl setenv OLLAMA_FLASH_ATTENTION 1
launchctl setenv OLLAMA_KV_CACHE_TYPE q8_0    # halves context-cache memory
launchctl setenv OLLAMA_KEEP_ALIVE 3m         # free idle models sooner
```

## Project layout

```
agent.py                    # CLI chat loop -- system prompt, tool registry, orchestration
app.py                      # Streamlit GUI -- same tools, adds upload + live thinking + inline images
tools/
  sensor_qc.py               # single-scan CV/CA/SWV QC
  cv_stability.py            # multi-scan CV stability (peak current/ΔEp drift, artifacts)
  ca_calibration.py          # CA calibration curve analysis (LOD, sensitivity, saturation)
  image_qc.py                 # vision defect QC + ISO-4287 surface roughness + 3D plots
  reference_diff.py           # unsupervised batch-outlier detection (SSIM vs batch average)
  performance_prediction.py   # visual-vs-CV correlation + grounded/preliminary prediction
  electrode_notes.py           # persistent per-electrode + per-batch records
  literature.py                # PubMed/arXiv search + local vault cache
  literature_figures.py        # figure extraction + captioning (arXiv, PMC)
  vault_maintenance.py          # vault-wide sweeps + insights synthesis
  _io.py                       # shared raw-instrument-CSV parsing
reference_data/               # your CV/CA CSVs
reference_images/             # your electrode photos
electrode_notes/               # auto-generated per-electrode/batch records
literature_vault/              # auto-generated literature cache
surface_plots/, reference_diff/  # auto-generated plot/analysis outputs
```

## Handing this to Claude Code

This whole folder is meant to be iterated on conversationally — e.g.
"add a tool for X", "the crop looks wrong on this new batch, check it",
"why isn't the agent finding this file". The tool boundary (schema +
registration in `agent.py`, logic in `tools/`) is deliberate so a
change to one tool doesn't require touching the orchestration loop.
