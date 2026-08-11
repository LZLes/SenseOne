"""
Vault maintenance tool: sweeps the whole literature_vault instead of acting
on one paper at a time.

Three things, in order:
  1. (optional) re-run every unique query already on record in the vault
     with refresh=True, to pull in newer papers -- off by default since it
     re-hits PubMed/arXiv for every past query, which is slow and eats into
     their rate limits.
  2. Source and caption figures (tools/literature_figures.py) for every
     arXiv paper in the vault that doesn't have them yet.
  3. Synthesize a grounded insights summary across everything in the vault
     -- titles, abstracts, figure captions -- written to
     literature_vault/INSIGHTS.md. Uses the main chat model directly, with
     the same citation rule as agent.py's system prompt: every claim must
     be tagged with the paper_id it came from, or it doesn't go in.

This is meant to be called either from the chat agent ("update the vault")
or run standalone/on a schedule -- it takes no conversation state.
"""

import json
import re
from datetime import date
from pathlib import Path

from google.genai import types

from tools._gemini import client, MODEL
from tools.literature import VAULT_DIR, _load_note, _normalize_query, search_literature
from tools.literature_figures import analyze_literature_figures

INSIGHTS_PATH = VAULT_DIR / "INSIGHTS.md"

VAULT_MAINTENANCE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "update_literature_vault",
        "description": (
            "Sweep the entire literature vault instead of one paper at a time: source and "
            "caption figures for every arXiv paper that doesn't have them yet, then regenerate "
            "a grounded insights summary (literature_vault/INSIGHTS.md) synthesized across all "
            "vault papers and their figures. Optionally re-run every past search query live to "
            "pull in newer papers first."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "refresh_queries": {
                    "type": "boolean",
                    "description": (
                        "Re-run every unique query already in the vault with refresh=True before "
                        "the sweep, to catch new papers. Default false -- this re-hits PubMed/arXiv "
                        "once per past query, which is slow."
                    ),
                },
                "max_figures_per_paper": {
                    "type": "integer",
                    "description": "Max figures to pull per paper. Default 5.",
                },
            },
        },
    },
}

LIST_VAULT_PAPERS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "list_vault_papers",
        "description": (
            "List every paper currently in the literature vault (paper_id, title, source, date, "
            "whether it has captioned figures). The agent has no other way to browse the vault -- "
            "search_literature only surfaces papers matching a specific query, so call this when "
            "asked what's in the vault, or to find a paper_id for analyze_literature_figures "
            "without re-running a search."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}


def list_vault_papers() -> dict:
    papers = [
        {
            "paper_id": n["paper_id"],
            "title": n["meta"].get("title"),
            "source": n["meta"].get("source"),
            "date": n["meta"].get("date"),
            "has_figures": "## Figures" in n["body"],
        }
        for n in _all_notes()
    ]
    return {"n_papers": len(papers), "papers": papers}


def _all_notes() -> list:
    notes = []
    if not VAULT_DIR.exists():
        return notes
    for path in sorted(VAULT_DIR.glob("*.md")):
        try:
            meta, body = _load_note(path)
        except Exception:
            continue
        notes.append({"paper_id": path.stem, "meta": meta, "body": body})
    return notes


def _refresh_known_queries() -> list:
    seen = set()
    for note in _all_notes():
        for q in note["meta"].get("queries", []):
            seen.add(_normalize_query(q))

    results = []
    for query in sorted(seen):
        try:
            r = search_literature(query, refresh=True)
            results.append({"query": query, "n_results": r["n_results"], "errors": r["errors"]})
        except Exception as e:
            results.append({"query": query, "error": str(e)})
    return results


def _source_missing_figures(max_figures_per_paper: int) -> list:
    results = []
    for note in _all_notes():
        # arxiv or pubmed (PMC open-access only -- see tools/literature_figures.py);
        # a paywalled PubMed paper returns an explicit error and gets re-attempted
        # on every sweep since there's no "known unavailable" cache yet -- a wasted
        # lookup each time, not a correctness issue, but worth knowing about.
        if note["meta"].get("source") not in ("arxiv", "pubmed") or "## Figures" in note["body"]:
            continue
        try:
            r = analyze_literature_figures(note["paper_id"], max_figures=max_figures_per_paper)
            results.append({"paper_id": note["paper_id"], "status": r.get("status"), "message": r.get("message")})
        except Exception as e:
            # One paper's PDF/figure issue (oversized image, malformed PDF, network
            # blip) shouldn't take down the whole vault sweep -- record and move on.
            results.append({"paper_id": note["paper_id"], "status": "error", "message": str(e)})
    return results


_INSIGHTS_PROMPT_TEMPLATE = """You are synthesizing an insights digest for a screen-printed \
electrode (SPE) biosensor lab, from a local literature vault of papers (titles, abstracts, and \
where available, captioned figures).

Rules:
- Every claim must be tagged inline with the paper_id it came from, in the form (paper_id), \
e.g. "Nafion coatings improve permselectivity in wearable sweat sensors (arxiv_2012.05543v1)."
- Only state something if it's actually supported by the content below -- do not fill gaps with \
general knowledge. If the vault doesn't have enough to say something useful on a topic, skip it \
rather than pad the summary.
- Group into: (1) electrochemical QC signatures reported (what good/bad CV or CA traces look \
like per the literature), (2) fabrication/print quality factors mentioned, (3) anything else \
notable. Skip a section entirely if nothing in the vault supports it.
- Be concise -- bullet points, not essays.

Vault contents:
{vault_content}

Write the digest now, as markdown."""


def _generate_insights() -> str:
    notes = _all_notes()
    if not notes:
        return "_Vault is empty -- nothing to synthesize yet._"

    chunks = []
    for note in notes:
        meta = note["meta"]
        chunk = f"### {meta.get('title', note['paper_id'])} ({note['paper_id']})\n{note['body'][:1500]}"
        chunks.append(chunk)
    vault_content = "\n\n".join(chunks)

    prompt = _INSIGHTS_PROMPT_TEMPLATE.format(vault_content=vault_content)

    response = client().models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2),
    )
    content = response.text.strip()

    # Models sometimes wrap markdown output in a stray ```markdown fence --
    # strip it so INSIGHTS.md renders as markdown, not a literal code block.
    if content.startswith("```"):
        content = re.sub(r"^```[a-zA-Z]*\n", "", content)
        content = re.sub(r"\n```$", "", content)

    return content


def update_literature_vault(refresh_queries: bool = False, max_figures_per_paper: int = 5) -> dict:
    query_refresh_results = _refresh_known_queries() if refresh_queries else []
    figure_results = _source_missing_figures(max_figures_per_paper)
    insights_md = _generate_insights()

    header = f"# Literature vault insights\n\n_Generated {date.today().isoformat()} from {len(_all_notes())} papers._\n\n"
    INSIGHTS_PATH.write_text(header + insights_md, encoding="utf-8")

    return {
        "status": "ok",
        "queries_refreshed": query_refresh_results,
        "figures_sourced": figure_results,
        "insights_path": str(INSIGHTS_PATH),
        "n_papers_in_vault": len(_all_notes()),
    }
