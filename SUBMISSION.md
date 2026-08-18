# SenseOne — submission notes

## Problem framing

Screen-printed electrodes (SPEs) for electrochemical biosensors are
QC'd by actually running them — cyclic voltammetry (CV) and
chronoamperometry (CA) — which costs real bench time (potentiostat
time, reagents, sample) per electrode. That's the ground-truth
signal, but it's slow, and it happens *after* fabrication, when a bad
print can no longer be caught cheaply.

The lab already has something faster available before any
electrochemistry runs at all: a photo of the printed electrode. The
open question this agent is built around is whether that photo
actually predicts electrical performance — do defects, ink coverage
evenness, or surface roughness visible in a photo correlate with CV
peak behavior — and if so, by how much, with what confidence, and
starting from how little data.

A second, more mundane problem sits underneath that: the lab's QC
data (raw CV/CA exports, electrode photos, fabrication notes per
batch/sheet, literature) accumulates across sessions in a way that's
easy to lose track of. Before any prediction question is even
interesting, the basic QC parsing and per-electrode record-keeping
needed to be automated and consistent.

## Agent architecture and workflow

SenseOne is a tool-calling agent — not a fine-tuned model, not a fixed
pipeline. This fork runs on Google's Gemini API (`gemini-3.6-flash`)
as a single model for everything: tool-calling orchestration, visible
thinking, and vision (photo reads happen via the same client, inside
specific tool implementations, not as a peer agent the orchestrator
talks to). An earlier version ran fully locally on Ollama
(`qwen3:8b` + `qwen2.5vl:7b`, one model per role); the backend was
swapped to make the agent deployable as a hosted link (Streamlit
Community Cloud, from a now-public GitHub repo) without requiring every
user to install Ollama and pull ~11GB of local models. The tools,
system prompt, and control flow are unchanged from the original —
only the LLM backend and message-shape plumbing differ.

18 custom tools are registered against a single schema/dispatch layer
(`agent.py`), grouped by what they do, plus two of Gemini's own
built-in tools (not custom function declarations — see "Web search" below):

- **Raw-data QC** — `sensor_qc`, `analyze_cv_stability`,
  `ca_calibration` parse instrument CSV exports (multiple encodings
  and column formats encountered across batches) into peak
  current/ΔEp/noise/LOD/sensitivity metrics.
- **Photo QC** — `image_qc` (vision-model defect read + ISO-4287
  surface-roughness proxy from luminance, cropped to just the working
  electrode's disc — not the counter-electrode ring, reference pad, or
  lead traces around it, since that's the surface that actually does
  the electrochemistry) and `compare_to_batch_reference` (unsupervised
  outlier detection via SSIM against a registered pixel-average
  "typical" print, no labeled examples needed).
- **Electrochemical batch QC** — `compare_cv_to_batch_reference`, the
  electrical-data counterpart to the photo-based outlier check above:
  is this electrode's CV/CA data (peak current, ΔEp, ipa/ipc ratio,
  scan-to-scan stability, CA sensitivity/R²/LOD) a statistical outlier
  vs. the rest of its own batch, via a modified z-score (median + MAD).
  See "Adding a new QC algorithm" below for why this specific method,
  and which candidate algorithms were deliberately *not* built.
- **Performance prediction** — `correlate_visual_cv_performance` and
  `predict_electrode_performance`, which pull every paired
  visual/electrical result recorded so far and only assert a grounded
  prediction when the correlation is statistically significant.
- **Persistent memory** — auto-populated per-electrode/per-batch
  Markdown records (`electrode_notes/`) that every QC tool above
  writes into without being asked, so history is available across
  sessions by default. Writes are serialized per-electrode (an
  in-process lock) so two concurrent QC calls against the same
  electrode can't silently drop each other's history, and a note that
  fails to parse gets backed up rather than silently overwritten blank.
- **Literature** — PubMed/arXiv search with a local cache
  (`literature_vault/`), automatic open-access detection, figure
  extraction, and citation-grounded synthesis.

The control flow is a standard multi-hop tool loop: the model reasons
(visible thinking), decides which tool(s) to call, the result is fed
back into the conversation, and the loop continues until the model
answers without further tool calls (or hits `agent.MAX_HOPS`, a hard
cap so a confused model can't loop forever on the shared API quota).
Two interfaces — a terminal CLI (`agent.py`) and a Streamlit GUI
(`app.py`, with live thinking stream, inline plots/photos, image
upload, and user-adjustable temperature/reasoning-visibility/hop-cap
settings) — sit on top of the identical tool layer, so no logic is
duplicated between them. The Streamlit GUI is what's deployed to
Streamlit Community Cloud; the only per-environment requirement is a
`GEMINI_API_KEY` (free tier, https://aistudio.google.com/apikey),
loaded from `.env` locally or from Streamlit's secrets manager when hosted.

## Web search: going beyond the local vault

Gemini has two built-in server-side tools — `google_search` (live web
search with citation grounding) and `url_context` (read a specific
URL/DOI) — that run alongside the 18 custom function-calling tools
above, in the same request. Combining them isn't documented anywhere
obvious: the first attempt returned a 400 error ("Please enable
tool_config.include_server_side_tool_invocations to use Built-in tools
with Function calling"), which is how the actual required flag
(`tool_config.include_server_side_tool_invocations=True`) was found.
Confirmed empirically that a server-side tool's response parts never
populate `.function_call`, so they can't collide with the existing
function-calling loop — this was verified before shipping, not assumed.

The system prompt has the agent prefer `search_literature` (the local
vault) first for biosensor literature specifically — it's cached and
gives a `paper_id` `analyze_literature_figures` can use — and reach for
`google_search`/`url_context` when the vault comes up short or the
researcher wants something PubMed/arXiv don't index at all (manufacturer
datasheets, other journals, a pasted link). Every citation, from either
source, must include the real URL inline, not just a name — addresses
both "give me the actual paper" and "search beyond just PubMed/arXiv."

## Adding a new QC algorithm: what got built, and what deliberately didn't

The pipeline had batch-level, unsupervised outlier detection for photos
(`compare_to_batch_reference`, SSIM against a pixel-average reference)
but nothing equivalent for the electrical measurements themselves —
`compare_cv_to_batch_reference` closes that gap using data that's
already being collected (no new measurement or file format), via a
modified z-score on median + median absolute deviation (MAD) rather
than mean/stddev — standard practice for outlier detection on real QC
data (Iglewicz & Hoaglin, 1993), because a couple of genuine outliers
inflate a mean/stddev calculation (the very spread being used to judge
them), which can mask them; median/MAD are robust to exactly that.
Validated against this project's real accumulated data before shipping,
not just unit-tested: it independently found a real signal a fixed-
threshold check would miss — one electrode in batch 20260805 had
scan-to-scan ΔEp variability (`delta_ep_cv_pct`) about 9x every other
electrode in its batch (~6.3% vs. ~0.7-1.4%), still well under
`analyze_cv_stability`'s own fixed 15% "unstable" threshold in absolute
terms, but a clear outlier *relative to its own batch*.

Two other candidate algorithms were deliberately not built, and it's
worth stating why rather than silently skipping them: **scan-rate-
dependent kinetics** (Randles-Ševčík, Laviron, Nicholson's k⁰ method)
would need CVs at multiple scan rates per electrode, and this lab's
protocol collects repeated scans at one fixed rate instead — building
that analysis would produce numbers with no real data behind them.
**A time-series drift/SPC control-chart check** across a batch's print
run was considered too, but `qc_history`'s `date` field is per-day, not
a genuine print-order sequence, so there's no real ordering to test a
trend against yet. Both would need a data-collection change (varying
scan rate; recording actual print sequence) before they'd be honest —
that's a fabrication-protocol conversation with the lab, not a code
change, and is noted here so it isn't silently forgotten.

## What makes this approach distinctive

- **Predictions are statistically gated, not vibes.**
  `predict_electrode_performance` only returns a real, confident
  prediction when a Pearson correlation computed from actual
  accumulated paired data clears significance (p < 0.05, n ≥ 15).
  Short of that, it returns an explicitly labeled "preliminary,
  not statistically validated" read with the real r/p/n shown —
  instead of either a fake-confident verdict or a flat refusal. Left
  alone, an LLM will answer a "will this work?" question confidently
  regardless of whether it actually knows; this agent is built to not
  do that.
- **Grounding is enforced in code, not just prompted.** Even a capable
  model doesn't reliably follow free-text "always cite your sources"
  instructions every turn. Citations go through a deterministic
  backstop (`_collect_cited_papers`) that verifies claims are actually
  tied to a retrieved paper rather than trusting the model to format
  it correctly — a carryover from the original local 8B/7B-model
  version, where the same gap was more frequent, but kept here as
  defense-in-depth rather than removed now that the backend is a
  stronger model. On top of that: temperature is tuned low (0.1) and
  the system prompt requires every derived number (a ratio, a
  percentage, anything the model computes itself rather than reads
  directly off a tool) to show its source values and arithmetic, plus
  an explicit self-check pass before the final answer is sent — every
  number/status/citation should trace to a specific tool call, or get
  removed/flagged as inference. None of this makes hallucination
  literally impossible for an LLM-based system (no honest claim can);
  what it does is make every claim independently checkable — the GUI's
  per-tool-call result panels (and the CLI's printed tool calls) show
  the exact source data, so anything that slips through is verifiable,
  not just a stated caveat.
- **Physical identity modeling matches how the lab actually generates
  data**, rather than assuming one clean convention: a single photo
  frame can contain multiple distinct sub-electrodes, a batch date can
  span multiple ink-formula sheets, and filename conventions have
  already changed multiple times across batches collected during
  development. Getting this wrong silently merges different
  electrodes' data — the record-keeping is deliberately identity-aware
  to prevent that class of bug.
- **Outlier detection needs no labeled failures.**
  `compare_to_batch_reference` builds a per-pixel average print
  pattern (via image registration) and flags deviation with SSIM —
  realistic for a research setting where confirmed electrical failures
  are scarce and hand-labeling isn't practical. By default the
  reference pool isn't limited to one batch: it auto-pools every
  batch/sheet on record (via `set_batch_metadata`) that shares the
  target's own carbon ink formula, since what actually determines
  whether two prints are comparable is the ink, not the fabrication
  date. This only activates when the ink formula is actually recorded
  — if it isn't, the tool says so and asks for either the metadata or
  an explicit batch, rather than silently falling back to a
  single-batch average or guessing.
- **Memory and literature are plain, inspectable files**, not a
  database — Markdown with JSON frontmatter, git-diffable and
  human-readable, so a researcher can open a note directly and see
  exactly what the agent has recorded about a given electrode.
- **Hosted via a frontier API, not fully local (a deliberate tradeoff
  of this fork).** The original version kept everything on-device
  (Ollama); this fork sends photos and prompts to Google's Gemini API
  so it can run as a shareable hosted link without every user
  installing local models — see the Gemini API terms for how free-tier
  usage is handled. Literature search still hits public PubMed/arXiv
  with no API key either way. Worth knowing before pointing this at
  genuinely unpublished/sensitive lab data.
- **Design choices were validated against real data, not assumed** —
  e.g., a defect-flag threshold (`luminance_cv`) was empirically found
  to flag 48/48 electrodes in a real batch with almost no spread, and
  was demoted in favor of the batch-comparison tool once that was
  measured, not guessed.

## Known limitations and edge cases identified

- **The core prediction is not yet statistically validated.** As of
  the last check: n = 21 paired electrodes, best relationship found
  (Rsk vs. ipa/ipc ratio) at r = +0.42, p = 0.056 — short of
  significance. The agent says this outright rather than hiding it;
  real predictions require more accumulated data. Note this figure
  predates the working-electrode-only crop fix below — the 21 existing
  paired readings were measured against the whole print (ring + disc +
  leads), not just the disc, so they're not strictly comparable to
  roughness values `image_qc` produces going forward until electrodes
  are re-measured.
- **Roughness metrics are a luminance proxy, not calibrated physical
  roughness.** Ra/Rq/Rz/Rt/Rsk/Rku are derived from image brightness
  as a stand-in for height — useful for comparing electrodes to each
  other under the same imaging setup, not a substitute for
  profilometry, and not expressed in real µm.
- **`luminance_cv` is a known weak signal**, kept in the tool for
  completeness but explicitly de-emphasized after confirming it
  flagged 100% of a real 48-electrode batch with near-zero
  discriminating power (values clustered 0.609–0.660).
  `compare_to_batch_reference` is the tool actually relied on for
  outlier detection.
- **Photo cropping is fixed fractions, not real detection.** Roughness
  now defaults to `WORKING_ELECTRODE_CROP_BOX`, isolating just the
  central disc rather than the whole print — but it's still a crop box
  tuned by locating the circle in a handful of real photos per format
  (single-electrode batches vs. the earliest batch's 4-sub-electrodes-
  per-photo layout, handled via `sub_pad_working_electrode_crop_box`),
  not per-photo circle detection. A genuinely new photo layout or
  camera framing needs to be told to the agent (or re-tuned in code),
  not inferred.
- **Filename conventions are not self-describing.** Both CV filenames
  and image filenames have already changed convention multiple times
  across batches collected during development; the parsers handle
  what's been seen, but a new convention requires a code update.
- **CA data has no matching photos**, so CA-derived calibration
  metrics (LOD, sensitivity) currently can't feed the visual
  correlation engine — only CV data is paired with images.
- **`compare_to_batch_reference`'s ink-formula pooling is only as good
  as the recorded metadata.** If `set_batch_metadata` was never called
  for a batch/sheet, that batch can't be found or pooled by ink formula
  — it has to be compared against explicitly by `image_dir`/`batch`
  instead. There's no way to infer ink formula from the photo itself.
- **Hosted deployment has no persistent write-back.** Streamlit
  Community Cloud's filesystem is ephemeral: `reference_images/`,
  `reference_data/`, and `electrode_notes/` are versioned into the repo
  so a fresh deploy has real data to work with, but anything the agent
  *writes* during a live session (new QC notes, new batch metadata,
  newly rendered surface/diff plots) is lost on restart. Durable
  multi-session record-keeping currently means running locally.
- **Gemini free-tier limits apply to the hosted version** — rate limits
  and quota are Google's, not something this project controls, and a
  busy shared demo can hit them.
- **Vision defect reads are single-pass**, one `gemini-3.6-flash` call
  per image with no ensemble or multi-crop voting — a borderline photo
  can plausibly get an inconsistent call between runs; this hasn't been
  quantified.
