# SenseOne (Gemini fork)

A Gemini-backed research assistant for an electrochemical biosensor
(screen-printed electrode) lab. This is a fork of the original local,
Ollama-backed SenseOne -- same tools, same system prompt, same
architecture, with the LLM backend swapped from local Ollama models to
the Gemini API so it can be shared as a hosted link instead of
requiring everyone to install Ollama + pull ~11GB of models. See
[`../ai_qc_agent`](../ai_qc_agent) for the local-only original.
It QCs raw CV/CA data, visually QCs electrode photos, tracks every
result against the same physical electrode across sessions, searches
and caches literature, and (as enough paired data accumulates) tries
to answer the actual question the lab cares about: does an electrode's
photo predict how well it'll perform electrically, before you run it?

**Tradeoff of this fork:** your data and prompts go to Google's API
instead of staying fully on-device (see [Gemini API terms](https://ai.google.dev/gemini-api/terms)
for how free-tier usage is handled). Prefer the original if that
matters for your data.

## Setup

```bash
# get a free key: https://aistudio.google.com/apikey
export GEMINI_API_KEY=...

./setup.sh   # creates a venv, installs deps, checks the key is set
```

or by hand:

```bash
export GEMINI_API_KEY=...
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Then run either interface -- both work against the same tools:

```bash
python agent.py             # terminal chat
streamlit run app.py        # local web GUI (localhost:8501) -- image upload,
                             # live thinking stream, inline plots/photos
```

One Gemini model (`gemini-2.5-flash`, set in `tools/_gemini.py`) handles
tool-calling, reasoning, and vision -- unlike the Ollama version, no
separate vision model is needed. Check
[the current model list](https://ai.google.dev/gemini-api/docs/models)
if that model has since been deprecated or a better free-tier option
exists.

Everything except the LLM calls still stays local -- CV/CA/image data,
generated notes, and plots are all written to disk exactly as in the
Ollama version. `reference_data/` and `reference_images/` ship empty
(see the README in each) since your lab
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

## Deploying to Streamlit Community Cloud

Point a new Community Cloud app at this repo/branch, then set the key in
**App settings → Secrets** as a top-level entry:

```toml
GEMINI_API_KEY = "..."
```

Top-level only -- a key nested under a `[section]` won't be found (the app
checks `os.environ["GEMINI_API_KEY"]` first, then falls back to
`st.secrets["GEMINI_API_KEY"]`, and Streamlit only mirrors top-level secrets
into `os.environ`). The sidebar shows a clear error and the chat input is
disabled until a valid key is set, rather than failing on the first message.

## Rate limits and cost

Gemini's free tier has requests-per-minute/day quotas that change over
time -- check [current limits](https://ai.google.dev/gemini-api/docs/rate-limits)
before a demo with several simultaneous users. Every visitor's usage
draws on whichever `GEMINI_API_KEY` the app is running with, not
their own -- there's no per-user key handling here. Calls are retried
automatically on transient/rate-limit errors (see `tools/_gemini.py`),
and each chat turn is capped at `agent.MAX_HOPS` tool-call round-trips so
a confused model can't loop indefinitely and burn through the shared quota.

## Project layout

```
agent.py                    # CLI chat loop -- system prompt, tool registry, orchestration
app.py                      # Streamlit GUI -- same tools, adds upload + live thinking + inline images
tools/
  _gemini.py                  # shared Gemini client + model constant
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
