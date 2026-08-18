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

One Gemini model (`gemini-3.6-flash`, set in `tools/_gemini.py`) handles
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
decides which of ~18 registered tools to call based on what you ask
(plus Gemini's own built-in web search/URL-reading, see Literature
below), then reports back. You don't need to name tools; plain requests
work. A few notes up front:

- **It won't guess.** If it doesn't have grounds for a claim (a tool
  result, a cited paper), it says so rather than making something up —
  and every derived number (a ratio, a percentage, a comparison it
  computes itself) has to show its source values and arithmetic, not
  just a conclusion. For predictions specifically, see below. No LLM
  system can promise zero hallucination outright; what this one
  promises is that every claim is independently checkable — expand any
  tool-call panel in the GUI (or read the CLI's printed tool calls) and
  the exact numbers a claim is based on are right there.
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
- **`compare_cv_to_batch_reference`** — is this electrode's CV/CA data a
  statistical outlier vs. the rest of its own batch? Unsupervised (no
  labeled examples needed), using a modified z-score (median + MAD, robust
  to a couple of genuine outliers skewing the reference the way mean/stddev
  would) across peak current, ΔEp, ipa/ipc ratio, scan-to-scan stability,
  and CA sensitivity/R²/LOD. The electrical-data counterpart to
  `compare_to_batch_reference` below, which only looks at photos — needs
  ≥5 electrodes in the batch sharing a metric before it means anything.

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
- If a photo is visibly tilted (say so, or the electrode clearly isn't
  upright in frame), both tools accept `rotation_degrees` to straighten
  it before analysis — batch registration and the fixed crop boxes are
  translation-only and don't auto-correct rotation, and an uncorrected
  tilt can read as a false defect/outlier (confirmed: a 20° uncorrected
  tilt dropped a real photo's SSIM from 0.618 to 0.197 — corrected, it
  recovered to 0.616). Counterclockwise positive: `rotation_degrees=5`
  corrects a photo tilted 5° clockwise. No automatic angle detection —
  it only applies a correction you (or the researcher) specify.
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
  re-fetching — pass `refresh=true` for the freshest results, or a
  higher `max_results` (default 8) for a deliberately broad sweep.
  Every result carries `open_access` (true for all arXiv preprints, and
  for PubMed papers confirmed open access via PMC through an automatic,
  batched Europe PMC lookup) and results come back open-access-first,
  since only those can actually be pulled full-text.
- **`analyze_literature_figures`** extracts and captions figures from
  a paper (arXiv always; PubMed only if `open_access=true`).
  **`update_literature_vault`** sweeps every paper missing figures and
  regenerates a citation-grounded synthesis (`literature_vault/INSIGHTS.md`).
- Beyond the vault, the agent also has Gemini's built-in **web search**
  and **URL reading** tools (`google_search`, `url_context`) — it
  reaches for these when the vault comes up short or you want something
  PubMed/arXiv don't index (manufacturer datasheets, other journals,
  a specific link/DOI you paste in), while still preferring the vault
  first for biosensor literature specifically (cached, and gives a
  `paper_id` `analyze_literature_figures` can use).
- Every claim drawn from a paper or web result gets tagged inline with
  its source **and a real, clickable URL** — never just a name or
  `paper_id` alone. If a claim can't be tied to a specific source, it's
  flagged as the model's own inference instead.

## GUI features (Streamlit)

Beyond chat, the sidebar has:

- **Advanced settings** — temperature (default 0.1, tuned low for
  grounded answers — the system prompt assumes this baseline), a "show
  reasoning" toggle (also skips requesting thought tokens when off, not
  just hiding them), and a max-tool-call-rounds cap per turn.
- **Check connection** — sends one small real request so you can
  confirm the API key/quota are actually live right before demoing,
  rather than finding out on the first real question.
- A themed, rotating loading message covers the latency before the
  first token of each hop arrives, and clears the instant real content
  starts streaming.
- A failed turn offers a one-click **Retry** instead of asking you to
  retype the message.

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

This repo is public, so `.streamlit/config.toml` sets
`client.showErrorDetails = "type"` -- an uncaught, unanticipated exception
shows just its exception type to a visitor, not internal file paths or a
stack trace. Every failure mode this project anticipates already has its
own clean, specific message; this is only the backstop.

## Rate limits, cost, and shared-demo safeguards

Gemini's free tier has requests-per-minute/day quotas that change over
time -- check [current limits](https://ai.google.dev/gemini-api/docs/rate-limits)
before a demo with several simultaneous users. Every visitor's usage
draws on whichever `GEMINI_API_KEY` the app is running with, not their
own -- there's no per-user key handling here. `google_search` grounding
also has its own separate quota/billing beyond the base Gemini API calls
(the model decides per-turn whether to use it, per the system prompt's
guidance to prefer the local vault first).

Several layers specifically protect a live, multi-visitor demo:

- **A process-wide concurrency cap** (`tools/_gemini.py`'s `request_slot`,
  default 3) queues simultaneous Gemini requests across every visitor
  instead of letting a burst all hit the rate limit at once and cascade
  into a wave of 429s. Verified under real concurrent load: 8 simultaneous
  simulated conversations, each doing a real multi-hop tool-calling
  exchange, all succeeded with peak concurrency held exactly at the cap.
- **A per-session rate limit** (app.py, 10 messages/60s) blunts any single
  session -- an accidental loop, or a script hitting the public URL --
  from draining the shared quota alone.
- **Automatic retry** on transient/rate-limit (429/5xx) errors at the HTTP
  layer, on top of the concurrency queue above (`tools/_gemini.py`).
- Each chat turn is capped at `agent.MAX_HOPS` tool-call round-trips (user-
  adjustable in the GUI sidebar) so a confused model can't loop
  indefinitely and burn through the shared quota.
- A failed turn drops the dangling message and offers a one-click Retry,
  and 429s specifically get a "shared demo, wait a moment" message rather
  than a generic API error.

## Project layout

```
agent.py                    # CLI chat loop -- system prompt, tool registry, orchestration
app.py                      # Streamlit GUI -- same tools, adds upload + live thinking + inline images
.streamlit/config.toml      # upload-size caps, hides error details (repo/app are public)
tools/
  _gemini.py                    # shared Gemini client, request concurrency cap, retry/timeout config
  _paths.py                     # shared filesystem-path sanitization (blocks traversal from LLM-supplied ids)
  _io.py                        # shared raw-instrument-CSV parsing
  sensor_qc.py                  # single-scan CV/CA/SWV QC
  cv_stability.py                # multi-scan CV stability (peak current/ΔEp drift, artifacts)
  ca_calibration.py              # CA calibration curve analysis (LOD, sensitivity, saturation)
  image_qc.py                     # vision defect QC + ISO-4287 surface roughness + 3D plots
  reference_diff.py               # unsupervised photo batch-outlier detection (SSIM vs batch average)
  electrochem_batch_outliers.py   # unsupervised CV/CA batch-outlier detection (modified z-score)
  performance_prediction.py       # visual-vs-CV correlation + grounded/preliminary prediction
  electrode_notes.py               # persistent per-electrode + per-batch records (locked, corruption-safe writes)
  literature.py                    # PubMed/arXiv search + local vault cache + open-access detection
  literature_figures.py            # figure extraction + captioning (arXiv, PMC)
  vault_maintenance.py              # vault-wide sweeps + insights synthesis
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
