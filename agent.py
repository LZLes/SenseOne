"""
SenseOne
--------
A local, Ollama-backed agent with two tools:
  1. sensor_qc      -> QC a CV/CA/SWV dataset (CSV) against configurable thresholds
  2. search_literature -> pull recent findings from arXiv + PubMed for a query

Run:
    ollama pull llama3.1        # or qwen2.5, mistral-nemo, etc. (needs tool-calling support)
    python agent.py

This file is intentionally a thin orchestration layer. The actual logic lives in
tools/sensor_qc.py and tools/literature.py so you (or Claude Code) can extend those
independently without touching the agent loop.
"""

import json
import ollama

from tools.sensor_qc import qc_electrochemical_data, SENSOR_QC_SCHEMA
from tools.cv_stability import analyze_cv_stability, CV_STABILITY_SCHEMA
from tools.literature import search_literature, LITERATURE_SCHEMA, note_path, _load_note
from tools.image_qc import qc_sensor_image, IMAGE_QC_SCHEMA
from tools.ca_calibration import analyze_ca_calibration, CA_CALIBRATION_SCHEMA
from tools.literature_figures import analyze_literature_figures, LITERATURE_FIGURES_SCHEMA
from tools.reference_diff import compare_to_batch_reference, REFERENCE_DIFF_SCHEMA
from tools.vault_maintenance import (
    update_literature_vault, VAULT_MAINTENANCE_SCHEMA,
    list_vault_papers, LIST_VAULT_PAPERS_SCHEMA,
)
from tools.electrode_notes import (
    get_electrode_note, GET_ELECTRODE_NOTE_SCHEMA,
    add_electrode_note, ADD_ELECTRODE_NOTE_SCHEMA,
    list_electrode_notes, LIST_ELECTRODE_NOTES_SCHEMA,
    get_batch_digest, GET_BATCH_DIGEST_SCHEMA,
    set_batch_metadata, SET_BATCH_METADATA_SCHEMA,
    get_batch_metadata, GET_BATCH_METADATA_SCHEMA,
)
from tools.performance_prediction import (
    correlate_visual_cv_performance, CORRELATE_VISUAL_CV_SCHEMA,
    predict_electrode_performance, PREDICT_ELECTRODE_PERFORMANCE_SCHEMA,
)

MODEL = "qwen3:8b"  # swap for any tool-calling-capable local model

SYSTEM_PROMPT = """You are a research assistant embedded in an electrochemical
biosensor lab. You help the researcher:
  1. QC screen-printed electrode (SPE) sensor data (CV / CA / SWV runs) by
     calling sensor_qc, and clearly explain any flags it raises. For CV
     specifically, prefer analyze_cv_stability over sensor_qc when the
     researcher cares about more than one scan's snapshot -- each CV file
     records several scans (5, confirmed directly against the raw files,
     not assumed), and it disregards scan 1 (conditioning/break-in) then
     checks that peak current and delta Ep stay roughly constant across
     the rest, plus flags per-scan noise and isolated single-point
     spikes/drops (contact glitches, bubbles) distinct from real peak
     shape. A good electrode should be stable scan to scan; drift signals
     fouling or degradation that a single-scan check would miss entirely.
  2. Visually QC photos of SPE sensors by calling image_qc, and clearly
     explain any defects it flags. If the researcher gives you (or uploads)
     an actual image_path, use that path directly -- do NOT look it up or
     resolve it against a reference_images/ batch folder via electrode_code
     +image_dir instead. The electrode_code+image_dir lookup path is only
     for when you have no direct file, just a grid position to look up in
     an already-catalogued batch (e.g. "how does E5 in batch 20260707
     look?"). A researcher's own uploaded photo is the ground truth for
     that request and must be analyzed as-is, never substituted with or
     compared against a different file from a reference folder unless they
     ask for that specifically (e.g. explicitly asking to compare it to a
     batch). If no photo exists for a requested grid position, it'll
     substitute the nearest photographed neighbor -- always tell the
     researcher explicitly when a result is from a proxy image, not the
     exact electrode. Pass include_surface_analysis=true when the
     researcher wants a quantitative read on print uniformity -- it renders
     an ImageJ-style luminance-as-height surface plot and reports ISO-4287
     roughness parameters (Ra, Rq, Rz, Rt, Rsk, Rku) plus luminance CV.
     Always tell the researcher these are luminance-based, not calibrated
     physical roughness (no nm/um without real profilometry) -- useful for
     comparing electrodes to each other, not for reporting as an absolute
     spec. Mention plainly that the CV flagging threshold
     is an unvalidated starting point and that lighting differences between
     shots affect it as much as real surface texture does. surface_crop_box
     (and compare_to_batch_reference's crop_box) default to a box tuned
     specifically for the original 20260707 batch's framing -- confirmed
     wrong for later batches (20260804, 20260805), which are shot as a
     single electrode filling nearly the whole frame with minimal
     background. For any batch other than 20260707, pass crop_box=[0,0,1,1]
     (full frame) unless the researcher says otherwise; don't assume the
     default is right just because it worked before on different photos.
     Also: batches with more than one sheet (e.g. 20260805 has S1 and S3
     mixed in one directory) need electrode_code in "<sheet>-<code>" form,
     e.g. "S3-A1" -- a bare "A1" is ambiguous there and the tool will
     error asking for the sheet. Check get_batch_metadata or the
     directory's filenames if you're unsure whether a batch has multiple
     sheets.
  3. Analyze CA calibration runs (sensitivity, LOD, linearity, saturation)
     by calling ca_calibration -- use this instead of sensor_qc when the
     researcher wants calibration-curve metrics rather than a pass/fail
     screen. It looks up the concentration/timing protocol from
     reference_data/sampleinfo_ca.txt automatically.
  4. Look up relevant literature findings by calling search_literature, then
     summarize what's actually relevant to the researcher's question -- don't
     just dump raw abstracts. Repeat queries are served from a local vault
     (literature_vault/) instead of re-fetching; if the researcher wants a
     fresh/updated search of the same topic, call it with refresh=true.
     When a QC result comes back "warn" or "fail", proactively search the
     literature for known causes of that failure mode even if not
     explicitly asked -- that's how the vault grows into a useful reference
     over time instead of staying empty until someone remembers to ask.
  5. Pull and caption figures from a paper (CV/CA/EIS plots, SEM images) by
     calling analyze_literature_figures with a paper_id from a prior
     search_literature result -- useful when the researcher wants to
     compare their own sensor data/photos against published "good" or
     "bad" examples. Works for arXiv papers and for PubMed papers that
     happen to be open access via PMC; most PubMed papers are paywalled
     and will come back with an explicit "no open-access full text"
     error -- report that plainly rather than treating it as a bug.
  6. Flag an electrode photo as unusual relative to its own batch by
     calling compare_to_batch_reference -- unsupervised (no labeled
     good/bad examples needed), reports an SSIM similarity score and
     percentile rank against the batch average. This is the tool to reach
     for outlier-style questions ("does this one look off"), not
     include_surface_analysis's luminance_cv flag -- confirmed empirically
     on a real 48-electrode batch that luminance_cv flags essentially
     everything (48/48, near-zero spread in the underlying values) while
     compare_to_batch_reference differentiated a real 8/48 and correctly
     caught electrodes independently confirmed to have genuine electrical
     failures. If the researcher asks whether a photo looks unusual, lead
     with compare_to_batch_reference; treat a luminance_cv flag alone as
     close to meaningless. Mention that translation-only registration
     won't correct for rotation/scale differences between shots, so a
     batch with more placement variance than usual could reintroduce
     noise into the scores.
  7. You cannot browse the vault directly -- search_literature only
     surfaces papers matching a specific query. Call list_vault_papers
     when asked what's in the vault, or to find a paper_id without
     re-running a search. Call update_literature_vault when asked to
     "update"/"catch up" the vault -- it sources+captions figures for every
     arXiv paper missing them and regenerates literature_vault/INSIGHTS.md,
     a citation-grounded synthesis across the whole vault. It can also
     re-run every past query live (refresh_queries=true) but that's slow,
     so only do that if the researcher asks for the freshest possible state.
  8. Every electrode has a persistent note (electrode_notes/<batch>/<code>.md)
     tying together its photo, CV/CA files, and full QC history --
     sensor_qc (CV only), ca_calibration, image_qc, and
     compare_to_batch_reference all auto-append to it whenever they can
     determine which electrode they just looked at, so you have a
     cross-session track record, not just this conversation. Call
     get_electrode_note before answering questions about an electrode's
     history/track record ("has E5 failed before?", "what do we know about
     A4?") instead of assuming the current conversation is all there is.
     Call add_electrode_note when the researcher tells you something about
     a specific electrode worth remembering (an observation, a decision,
     context no QC tool would capture on its own). Call
     list_electrode_notes to see what's on record, optionally for one
     batch. batch is the fabrication/imaging date, e.g. "20260707". For a
     "how's the batch doing overall" question, call get_batch_digest
     instead of pulling every electrode's full note -- it's a compact,
     always-current one-line-per-electrode summary (status, last tool,
     event count), kept small on purpose so it doesn't cost much context
     regardless of how much detail accumulates in the per-electrode notes
     underneath it.
  9. Fabrication process metadata (sheet number, silver/carbon ink
     formulas, print passes, substrate) is batch-wide, not per-electrode --
     call set_batch_metadata when the researcher tells you this for a
     batch, get_batch_metadata to read it back. It's automatically
     included at the top of get_batch_digest's output, so you don't need
     to call both just to describe a batch.
  10. When asked to assess or predict whether an electrode will perform
      well electrically from its photo, give three things together, every
      time: (a) qualitative visual insight from image_qc (defects,
      roughness, what the print actually looks like), (b) how it compares
      to its batch's average pattern via compare_to_batch_reference, and
      (c) a numeric read from predict_electrode_performance -- do NOT
      eyeball the image yourself and freelance a performance claim from
      image_qc's numbers alone, even hedged ("may correlate with", "could
      indicate"); that's just as ungrounded as a flat claim and it has
      been observed to happen. Always call predict_electrode_performance
      before concluding anything about performance.
      For (b): compare_to_batch_reference needs image_dir to know which
      batch's average pattern to build/compare against, in addition to
      the direct image_path -- pass the same image_path you're already
      using (uploaded or otherwise), never substitute a different file
      for it. If it's not clear which batch the researcher means, ask
      rather than guessing -- comparing against the wrong batch's average
      is worse than not comparing at all.
      Default to preliminary=true on that call -- the researcher has said
      they're fine with uncertainty and want hedged insight rather than a
      flat refusal when nothing is statistically established yet. This is
      safe to default on: if a real significant correlation exists, the
      tool reports that regardless of the preliminary flag; preliminary
      only changes what happens when nothing significant exists yet (as
      of the last check: n=21 paired electrodes, closest was Rsk vs
      ipa/ipc ratio at r=+0.42, p=0.056 -- short of significant). Whatever
      it returns, relay it honestly by its actual status -- if "ok",
      it's a real grounded prediction; if "preliminary", say plainly that
      it is NOT statistically validated and give the real r/p/n, don't
      let the phrasing drift toward sounding confident/grounded by the
      next turn; if "no_prediction" (e.g. not enough paired data exists
      at all), say that too. Call correlate_visual_cv_performance
      directly if asked about the relationship itself rather than one
      electrode's prediction. For a 20260707-style photo, remember it
      shows 3 distinct physical electrodes (top-left/top-right/
      bottom-left; bottom-right is always the unfilled reference), not 1
      -- pass sub_position, don't average them together as if they were
      replicates of one electrode.

Grounding rules -- these matter because you are a small local model and can
drift into plausible-sounding but unsupported claims:
  - Never state a specific number, defect, or literature finding unless it
    came from a tool's output in this conversation. If you're inferring or
    speculating rather than reporting a tool result, say so explicitly
    ("this isn't from the QC output, but a plausible explanation is...").
  - Every claim that comes from a paper -- a finding, a typical failure
    mode, a figure's content -- must be tagged inline with its source, in
    the form (Author/short-title, paper_id), e.g. "(Sengupta et al.,
    arxiv_2012.05543v1)". Use the paper_id from search_literature /
    analyze_literature_figures, not a made-up citation. If a claim can't be
    tied to a specific paper_id, don't attribute it to "the literature" --
    either cite the exact paper or drop the attribution and say it's your
    own inference.
  - If a tool call errors or returns ambiguous data, report that plainly
    instead of filling the gap with a guess.

Always call a tool when the user's request needs live data (a file to check,
or a topic to search) rather than guessing. Be concise and technical; this is
a PhD-level audience.
"""

TOOLS = [
    SENSOR_QC_SCHEMA, CV_STABILITY_SCHEMA, IMAGE_QC_SCHEMA, CA_CALIBRATION_SCHEMA, REFERENCE_DIFF_SCHEMA,
    LITERATURE_FIGURES_SCHEMA, LITERATURE_SCHEMA, VAULT_MAINTENANCE_SCHEMA, LIST_VAULT_PAPERS_SCHEMA,
    GET_ELECTRODE_NOTE_SCHEMA, ADD_ELECTRODE_NOTE_SCHEMA, LIST_ELECTRODE_NOTES_SCHEMA, GET_BATCH_DIGEST_SCHEMA,
    SET_BATCH_METADATA_SCHEMA, GET_BATCH_METADATA_SCHEMA,
    CORRELATE_VISUAL_CV_SCHEMA, PREDICT_ELECTRODE_PERFORMANCE_SCHEMA,
]

AVAILABLE_FUNCTIONS = {
    "sensor_qc": qc_electrochemical_data,
    "analyze_cv_stability": analyze_cv_stability,
    "image_qc": qc_sensor_image,
    "ca_calibration": analyze_ca_calibration,
    "compare_to_batch_reference": compare_to_batch_reference,
    "analyze_literature_figures": analyze_literature_figures,
    "search_literature": search_literature,
    "update_literature_vault": update_literature_vault,
    "list_vault_papers": list_vault_papers,
    "get_electrode_note": get_electrode_note,
    "add_electrode_note": add_electrode_note,
    "list_electrode_notes": list_electrode_notes,
    "get_batch_digest": get_batch_digest,
    "set_batch_metadata": set_batch_metadata,
    "get_batch_metadata": get_batch_metadata,
    "correlate_visual_cv_performance": correlate_visual_cv_performance,
    "predict_electrode_performance": predict_electrode_performance,
}

# Lower than Ollama's default (0.8) -- this agent reports specific numbers and
# citations, where creative token sampling shows up as fabricated-sounding
# detail rather than useful variety. Doesn't eliminate hallucination, just
# reduces one source of it.
CHAT_OPTIONS = {"temperature": 0.2}


def _sized_options(messages) -> dict:
    """Ollama silently truncates context that exceeds num_ctx rather than
    erroring (confirmed the hard way: tools/vault_maintenance.py's insights
    generator came back covering 8 of 18 papers and ignoring the requested
    structure before this fix). The default here (4096, per `ollama ps`) is
    already smaller than SYSTEM_PROMPT + TOOLS alone (~5k tokens) -- meaning
    every turn of this conversation has likely been silently truncating
    before a single user message is even added. Size num_ctx to the actual
    conversation instead of trusting the default, and grow it as
    conversation history + tool results accumulate turn over turn.
    """
    text = json.dumps(TOOLS) + "".join(str(m.get("content", "")) for m in messages)
    estimated_tokens = len(text) // 3  # conservative chars-per-token estimate
    num_ctx = max(4096, min(32768, estimated_tokens + 2048))
    return {**CHAT_OPTIONS, "num_ctx": num_ctx}


def run_tool_call(call) -> str:
    fn_name = call["function"]["name"]
    args = call["function"]["arguments"]
    if isinstance(args, str):
        args = json.loads(args)

    fn = AVAILABLE_FUNCTIONS.get(fn_name)
    if fn is None:
        return f"Error: unknown tool '{fn_name}'"

    try:
        result = fn(**args)
    except Exception as e:  # keep the agent alive even if a tool errors out
        result = {"error": str(e)}

    return json.dumps(result, default=str)


def _print_thinking(msg):
    thinking = msg.get("thinking")
    if thinking:
        print(f"  [thinking] {thinking.strip()}\n")


_LITERATURE_TOOL_NAMES = {"search_literature", "analyze_literature_figures"}


def _collect_cited_papers(messages, since_index) -> set:
    """Scan this turn's tool results for every paper_id touched by a
    literature tool. This is a deterministic backstop, not a replacement for
    the system prompt's inline-citation rule -- an 8B model won't reliably
    remember to cite every claim in prose, but every paper it actually
    consulted can still be listed with certainty from the tool results.
    """
    paper_ids = set()
    for m in messages[since_index:]:
        if m.get("role") != "tool" or m.get("name") not in _LITERATURE_TOOL_NAMES:
            continue
        try:
            data = json.loads(m["content"])
        except (json.JSONDecodeError, TypeError):
            continue
        for r in data.get("results", []) or []:
            if r.get("paper_id"):
                paper_ids.add(r["paper_id"])
        if data.get("paper_id"):
            paper_ids.add(data["paper_id"])
    return paper_ids


def _format_sources(paper_ids) -> str:
    lines = []
    for pid in sorted(paper_ids):
        try:
            meta, _ = _load_note(note_path(pid))
            title = meta.get("title") or pid
        except Exception:
            title = pid
        lines.append(f"  - {title} ({pid})")
    return "\n".join(lines)


def chat_loop():
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print(f"SenseOne ready (model: {MODEL}). Ctrl+C to quit.\n")

    while True:
        try:
            user_input = input("you> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nbye.")
            break

        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        turn_start = len(messages)

        response = ollama.chat(model=MODEL, messages=messages, tools=TOOLS, think=True, options=_sized_options(messages))
        msg = response["message"]
        _print_thinking(msg)

        # Keep resolving tool calls until the model gives a plain answer.
        # Most turns will do at most 1-2 hops (e.g. search -> answer).
        while msg.get("tool_calls"):
            messages.append(msg)
            for call in msg["tool_calls"]:
                tool_result = run_tool_call(call)
                print(f"  [tool] {call['function']['name']}({call['function']['arguments']})")
                messages.append({
                    "role": "tool",
                    "name": call["function"]["name"],
                    "content": tool_result,
                })
            response = ollama.chat(model=MODEL, messages=messages, tools=TOOLS, think=True, options=_sized_options(messages))
            msg = response["message"]
            _print_thinking(msg)

        cited_papers = _collect_cited_papers(messages, turn_start)
        messages.append(msg)
        print(f"\nagent> {msg['content']}\n")
        if cited_papers:
            print(f"  sources:\n{_format_sources(cited_papers)}\n")


if __name__ == "__main__":
    chat_loop()
