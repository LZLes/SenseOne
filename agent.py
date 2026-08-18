"""
SenseOne (Gemini fork)
----------------------
Same tool-calling agent as the local Ollama version, ported to Google's
Gemini API so it can run as a hosted demo without a local model. One
Gemini model (see tools/_gemini.py's MODEL) replaces both qwen3:8b (tool-calling/
thinking) and qwen2.5vl:7b (vision) -- Gemini is natively multimodal, so
image_qc.py and literature_figures.py's vision calls use the same model
and client as this file, via tools/_gemini.py.

Needs a free API key: https://aistudio.google.com/apikey
    export GEMINI_API_KEY=...
    python agent.py

This file is intentionally a thin orchestration layer. The actual logic
lives in tools/*.py so a change to one tool doesn't require touching the
agent loop -- unchanged from the original design.
"""

import json

from google.genai import errors as genai_errors
from google.genai import types

from tools._gemini import client, MODEL, api_key_configured, request_slot, RequestQueueFullError
from tools.sensor_qc import qc_electrochemical_data, SENSOR_QC_SCHEMA
from tools.cv_stability import analyze_cv_stability, CV_STABILITY_SCHEMA
from tools.literature import search_literature, LITERATURE_SCHEMA, note_path, _load_note
from tools.image_qc import qc_sensor_image, IMAGE_QC_SCHEMA
from tools.ca_calibration import analyze_ca_calibration, CA_CALIBRATION_SCHEMA
from tools.literature_figures import analyze_literature_figures, LITERATURE_FIGURES_SCHEMA
from tools.reference_diff import compare_to_batch_reference, REFERENCE_DIFF_SCHEMA
from tools.electrochem_batch_outliers import compare_cv_to_batch_reference, COMPARE_CV_TO_BATCH_SCHEMA
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
     defaults to WORKING_ELECTRODE_CROP_BOX, isolating just the working
     electrode's central disc (not the counter-electrode ring, reference
     pad, or lead traces) -- roughness should describe that surface
     specifically, since it's the one that actually does the
     electrochemistry. That default is tuned for the current
     single-electrode-per-photo framing (20260804, 20260805); don't override
     it unless the researcher wants the full print instead. For a
     20260707-style photo (2x2 cluster of 4 sub-electrodes per photo), don't
     pass surface_crop_box at all -- use predict_electrode_performance with
     sub_position instead, which selects the right sub-electrode's disc
     automatically. compare_to_batch_reference's crop_box is separate and
     still defaults to the whole-print box tuned for 20260707's framing --
     confirmed wrong for later batches, so pass crop_box=[0,0,1,1] (full
     frame) there for any batch other than 20260707 unless the researcher
     says otherwise.
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
     Each result carries open_access (true for every arXiv preprint, and for
     PubMed papers confirmed open access via PMC) -- results already come
     back open-access-first, so prefer citing/leading with those when
     several papers say similar things: the researcher can actually open
     and verify one, and analyze_literature_figures only works on them
     anyway. Don't drop a paywalled paper's finding just because it's not
     open access, but don't lead with it over an equally relevant open one.
     Pass a higher max_results (default 5) when the researcher wants a
     broad view of a topic rather than just a couple of leads -- don't
     settle for the default when "what's out there on X" is really the ask.
     You also have two built-in web tools, separate from search_literature:
     google_search can find things PubMed/arXiv don't index at all
     (other journals/publishers, datasheets, standards, non-preprint
     content), and url_context can read a specific URL/DOI the researcher
     pastes in directly. Reach for these when search_literature comes up
     short or the researcher explicitly wants something from the wider
     web/a specific link -- but default to search_literature/the vault
     first for biosensor literature specifically, since those results are
     vault-cached, carry a paper_id for analyze_literature_figures, and
     don't spend web-search quota on something the vault might already
     answer. Every citation, from either source, must include the actual
     URL inline (a paper_id alone isn't clickable) -- for search_literature
     results this is the tool result's own url field; for a web result,
     it's the real source URL the search/URL-read returned, never a
     guessed or reconstructed one.
  5. Pull and caption figures from a paper (CV/CA/EIS plots, SEM images) by
     calling analyze_literature_figures with a paper_id from a prior
     search_literature result -- useful when the researcher wants to
     compare their own sensor data/photos against published "good" or
     "bad" examples. Works for arXiv papers and for PubMed papers with
     open_access=true (open access via PMC); a paywalled PubMed paper
     (open_access=false) will come back with an explicit "no open-access
     full text" error if tried anyway -- report that plainly rather than
     treating it as a bug, and prefer reaching for an open_access=true
     paper in the first place when the researcher wants to see figures.
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
     close to meaningless. Registration here (and in image_qc/
     analyze_surface_topology) is translation-only and won't detect or
     auto-correct rotation/scale differences between shots -- if a photo
     is visibly tilted (the researcher says so, or it's obviously not
     upright), pass rotation_degrees (counterclockwise positive; e.g. 5
     corrects a photo tilted 5 degrees clockwise) to straighten it before
     analysis rather than letting the fixed crop_box/registration silently
     read the tilt as a real defect or outlier -- confirmed empirically this
     matters a lot: on a real photo with a genuine (unrotated) SSIM of 0.618
     against its own batch, an uncorrected 20-degree tilt alone dropped that
     to 0.197 (would read as a dramatic false outlier), and passing the
     matching rotation_degrees correction recovered it to 0.616 -- almost
     exactly the untilted value. There's no automatic detection, only a
     manual correction -- don't guess an angle, ask the researcher or use
     what they tell you.
  6b. compare_to_batch_reference only looks at photos -- for the same
      question about electrical performance ("is this electrode's CV
      normal for its batch", "which electrodes in this batch look
      electrically off"), call compare_cv_to_batch_reference instead. Same
      idea (unsupervised, no labeled examples, flags statistical outliers
      against the batch), but on the actual CV/CA metrics -- peak current,
      delta Ep, ipa/ipc ratio, scan-to-scan stability, CA sensitivity/R^2/LOD
      -- already logged from sensor_qc/analyze_cv_stability/ca_calibration,
      not pixels. These two tools answer different questions and neither
      substitutes for the other: an electrode can look visually normal but
      be an electrical outlier, or vice versa -- if the researcher cares
      about both, call both rather than assuming one implies the other.
      Needs at least 5 electrodes in the batch sharing a metric before the
      statistics mean anything (reports "insufficient_data" plainly below
      that, not a guess). Omit electrode_code for a batch-wide sweep.
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

Grounding rules -- treat these as hard constraints, not style preferences.
This agent's entire value proposition is that a researcher can trust what it
reports without re-verifying it by hand; a single fabricated number or
citation undermines that far more than an honest "I don't know" or "a tool
couldn't determine this" ever would. When in doubt, say less, not more.
  - Never state a specific number, defect, status, or literature finding
    unless it came from a tool's output in this conversation. If you're
    inferring or speculating rather than reporting a tool result, say so
    explicitly ("this isn't from the QC output, but a plausible explanation
    is..."). This includes numbers you derive yourself: if you compute a
    ratio, percentage, difference, or comparison from tool-reported values
    (e.g. "3x higher", "a 12% drop"), show the exact source values and the
    arithmetic, not just the derived conclusion -- a silent derivation is as
    ungrounded as an invented number if the arithmetic is ever wrong.
  - Every claim that comes from a paper or web source -- a finding, a
    typical failure mode, a figure's content -- must be tagged inline with
    its source AND a real, clickable URL, not just a name. For
    search_literature/analyze_literature_figures results: (Author/short-title,
    paper_id) plus the result's own url field, e.g. "(Sengupta et al.,
    arxiv_2012.05543v1, https://arxiv.org/abs/2012.05543)". For
    google_search/url_context results: the actual source title/domain plus
    the real URL the tool returned. Never guess, reconstruct, or omit a URL,
    and never attribute something to "the literature" or "online" in the
    abstract -- name the specific source every time or drop the claim.
  - If a tool call errors or returns ambiguous data, report that plainly
    instead of filling the gap with a guess.
  - Before sending your final answer, scan it once more: every number,
    status, defect, and citation should trace to a specific tool call
    earlier in this turn (or an earlier get_electrode_note/get_batch_digest
    you called). If you can't point to where something came from, remove it
    or explicitly mark it as your own inference rather than let it stand as
    if it were grounded.

Always call a tool when the user's request needs live data (a file to check,
or a topic to search) rather than guessing. Be concise and technical; this is
a PhD-level audience. Don't narrate your own tool-use process or planning in
the final answer ("I'll start by...", "next I need to...") -- the visible
thinking stream already shows that; the final answer should be the actual
result, not a recap of how you got there.
"""

TOOLS = [
    SENSOR_QC_SCHEMA, CV_STABILITY_SCHEMA, IMAGE_QC_SCHEMA, CA_CALIBRATION_SCHEMA, REFERENCE_DIFF_SCHEMA,
    COMPARE_CV_TO_BATCH_SCHEMA,
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
    "compare_cv_to_batch_reference": compare_cv_to_batch_reference,
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

_LITERATURE_TOOL_NAMES = {"search_literature", "analyze_literature_figures"}


def _to_gemini_tool(tools: list) -> types.Tool:
    """Our TOOLS schemas are the OpenAI/Ollama-style {"type": "function",
    "function": {name, description, parameters}} shape. Gemini's
    FunctionDeclaration takes a raw JSON-schema dict directly via
    parameters_json_schema, so this is a reshape, not a rewrite of any
    tool's actual schema.
    """
    declarations = [
        types.FunctionDeclaration(
            name=schema["function"]["name"],
            description=schema["function"].get("description", ""),
            parameters_json_schema=schema["function"].get("parameters", {"type": "object", "properties": {}}),
        )
        for schema in tools
    ]
    return types.Tool(function_declarations=declarations)


GEMINI_TOOL = _to_gemini_tool(TOOLS)

# Gemini's own built-in tools, not custom function declarations -- these run
# server-side (the model decides when to invoke them; results never come
# back as a function_call part we'd need to dispatch ourselves, confirmed
# empirically: none of a grounded response's tool_call/tool_response parts
# set .function_call, so they can't collide with run_hops'/`_stream_turn`'s
# function-call handling). google_search extends literature lookups past
# just PubMed/arXiv to the open web (any journal, publisher page, dataset,
# datasheet); url_context lets the model read a specific URL/DOI the
# researcher pastes in directly. Both require
# tool_config.include_server_side_tool_invocations=True to combine with our
# own function-calling tools in the same request -- confirmed via a 400
# error otherwise ("Please enable tool_config.include_server_side_tool_invocations
# to use Built-in tools with Function calling"), not documented anywhere obvious.
WEB_TOOLS = [types.Tool(google_search=types.GoogleSearch()), types.Tool(url_context=types.UrlContext())]

# Well below the API default -- this agent reports specific numbers and
# citations, where creative token sampling shows up as fabricated-sounding
# detail rather than useful variety. Doesn't eliminate hallucination on its
# own (that's what the grounding rules + citation backstop above are for),
# just removes one source of it. Not 0.0 -- some models get repetitive/loopy
# at the extreme, and this is already low enough that the difference isn't
# meaningfully more grounded, just more deterministic.
DEFAULT_TEMPERATURE = 0.1


def build_generate_config(temperature: float = DEFAULT_TEMPERATURE, include_thoughts: bool = True) -> types.GenerateContentConfig:
    """Builds a GenerateContentConfig -- factored out so a caller (the
    Streamlit GUI's sidebar) can offer temperature/reasoning-visibility as
    user-adjustable settings without duplicating the system prompt/tools
    wiring. The CLI just uses the module-level GENERATE_CONFIG default
    below, unchanged.
    """
    return types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[GEMINI_TOOL, *WEB_TOOLS],
        tool_config=types.ToolConfig(include_server_side_tool_invocations=True),
        thinking_config=types.ThinkingConfig(include_thoughts=include_thoughts),
        temperature=temperature,
    )


GENERATE_CONFIG = build_generate_config()


def run_tool_call(name: str, args: dict) -> dict:
    fn = AVAILABLE_FUNCTIONS.get(name)
    if fn is None:
        return {"error": f"unknown tool '{name}'"}
    try:
        result = fn(**args)
    except Exception as e:  # keep the agent alive even if a tool errors out
        result = {"error": str(e)}
    # Round-trip through json to normalize numpy/Path/etc. into plain types
    # -- the SDK needs a plain JSON-serializable dict for function_response.
    return json.loads(json.dumps(result, default=str))


class ModelCallError(RuntimeError):
    """A Gemini call failed in a way the caller should show to the user
    (network error, quota/rate-limit exhausted after retries, safety block,
    empty response) rather than let crash the process.
    """


# Hard cap on tool-call round-trips for a single user turn -- without this, a
# model that keeps reacting to a confusing tool result with more tool calls
# (rather than ever giving a final answer) would hang that session
# indefinitely and burn through the one shared GEMINI_API_KEY everyone else
# also depends on during a demo.
MAX_HOPS = 15


def _call_model(contents: list):
    """Calls Gemini and returns the first candidate, raising ModelCallError
    (instead of letting an IndexError/AttributeError/API exception escape)
    for every way that call can fail to produce a usable candidate: network/
    quota errors, and safety-blocked or otherwise empty responses.
    """
    try:
        with request_slot():
            response = client().models.generate_content(model=MODEL, contents=contents, config=GENERATE_CONFIG)
    except genai_errors.APIError as e:
        if getattr(e, "code", None) == 429:
            raise ModelCallError(
                "Gemini's rate limit was hit (this demo shares one free-tier API key across every "
                "visitor) -- wait a few seconds and try again."
            ) from e
        raise ModelCallError(f"Gemini API error ({getattr(e, 'code', '?')}): {getattr(e, 'message', e)}") from e
    except RequestQueueFullError as e:
        raise ModelCallError(str(e)) from e
    except Exception as e:
        raise ModelCallError(f"Could not reach Gemini: {e}") from e

    candidates = response.candidates or []
    if not candidates:
        feedback = getattr(response, "prompt_feedback", None)
        raise ModelCallError(f"Gemini returned no response (possibly blocked). {feedback or ''}".strip())
    candidate = candidates[0]
    if candidate.content is None:
        reason = getattr(candidate, "finish_reason", "unknown")
        raise ModelCallError(f"Gemini returned an empty response (finish_reason={reason}).")
    return candidate


def run_hops(contents: list) -> list:
    """Repeatedly calls Gemini, executing any function calls it returns and
    feeding the results back, until it answers with no further function
    calls (or MAX_HOPS is hit). Mutates `contents` in place (appends the
    model's turn and any tool-result turns), and returns the final answer
    turn's parts. Raises ModelCallError if a Gemini call itself fails --
    callers should catch that and report it rather than crash.
    """
    for _hop in range(MAX_HOPS):
        candidate = _call_model(contents)
        parts = candidate.content.parts or []

        thinking_text = "".join(p.text for p in parts if getattr(p, "thought", False) and p.text)
        if thinking_text:
            print(f"  [thinking] {thinking_text.strip()}\n")

        contents.append(candidate.content)

        function_calls = [p.function_call for p in parts if getattr(p, "function_call", None)]
        if not function_calls:
            return parts

        response_parts = []
        for fc in function_calls:
            args = dict(fc.args) if fc.args else {}
            print(f"  [tool] {fc.name}({args})")
            result = run_tool_call(fc.name, args)
            response_parts.append(types.Part.from_function_response(name=fc.name, response=result))
        # The API only accepts "user"/"model" roles -- function responses
        # ride back as a "user" turn, distinguished from real user input by
        # containing function_response parts instead of text.
        contents.append(types.Content(role="user", parts=response_parts))

    # Hit the round-trip cap without a final answer. Every function_call
    # above already got a matching function_response appended, so contents
    # stays well-formed -- we just stop asking for more hops and answer with
    # a "model" turn (keeping role alternation intact) explaining why.
    stop_text = (
        f"(Stopped after {MAX_HOPS} tool-call round-trips without a final answer -- "
        "try breaking the request into smaller steps.)"
    )
    stop_content = types.Content(role="model", parts=[types.Part.from_text(text=stop_text)])
    contents.append(stop_content)
    return stop_content.parts


def _collect_cited_papers(contents, since_index) -> set:
    """Scan this turn's tool results for every paper_id touched by a
    literature tool. Deterministic backstop, not a replacement for the
    system prompt's inline-citation rule -- a model won't reliably remember
    to cite every claim in prose, but every paper it actually consulted can
    still be listed with certainty from the tool results.
    """
    paper_ids = set()
    for c in contents[since_index:]:
        for part in c.parts or []:
            fr = getattr(part, "function_response", None)
            if fr is None or fr.name not in _LITERATURE_TOOL_NAMES:
                continue
            data = fr.response or {}
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
    contents = []
    print(f"SenseOne ready (model: {MODEL}, via Gemini API). Ctrl+C to quit.\n")

    while True:
        try:
            user_input = input("you> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nbye.")
            break

        if not user_input:
            continue

        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_input)]))
        turn_start = len(contents)

        try:
            final_parts = run_hops(contents)
        except ModelCallError as e:
            print(f"\nagent> [error] {e}\n")
            continue
        except Exception as e:  # last-resort guard -- a tool/library bug here should never kill the REPL
            print(f"\nagent> [unexpected error] {e}\n")
            continue

        cited_papers = _collect_cited_papers(contents, turn_start)
        answer = "".join(p.text for p in final_parts if p.text and not getattr(p, "thought", False))
        print(f"\nagent> {answer}\n")
        if cited_papers:
            print(f"  sources:\n{_format_sources(cited_papers)}\n")


if __name__ == "__main__":
    chat_loop()
