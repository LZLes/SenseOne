"""
Literature search tool.

Pulls recent findings from two free, no-API-key-needed sources:
  - PubMed (via NCBI E-utilities)   -- best for biosensor / biomedical / enzyme literature
  - arXiv                            -- best for materials science / physics preprints

Returns a compact list of {title, authors, source, date, url, summary} so the
agent's LLM layer can summarize what's relevant rather than dumping raw XML.

Results are also persisted as notes under literature_vault/ (one markdown
file per paper, with a small JSON frontmatter block), so the agent
accumulates a local, offline-capable body of literature over repeated use
instead of re-fetching the same query every session. A query that's been
searched before is served straight from the vault; pass refresh=True to
force a live re-search (e.g. to pick up newer papers).
"""

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

import requests

from tools._paths import safe_path_component as _safe_id_component

VAULT_DIR = Path("literature_vault")

LITERATURE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_literature",
        "description": (
            "Search PubMed and/or arXiv for recent papers on a topic and return "
            "titles, authors, dates, links, and abstracts/summaries. Repeat queries "
            "are served from a local vault (literature_vault/) instead of re-fetching; "
            "pass refresh=True to bypass the vault and search live."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query, e.g. 'Nafion permselectivity wearable biosensor'.",
                },
                "source": {
                    "type": "string",
                    "enum": ["pubmed", "arxiv", "both"],
                    "description": "Which database(s) to search. Default 'both'.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max results per source. Default 5.",
                },
                "refresh": {
                    "type": "boolean",
                    "description": (
                        "Bypass the local vault and search live even if this exact "
                        "query has been run before. Default false."
                    ),
                },
            },
            "required": ["query"],
        },
    },
}

PUBMED_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
PUBMED_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
ARXIV_API = "http://export.arxiv.org/api/query"


def _search_pubmed(query: str, max_results: int) -> list:
    search_resp = requests.get(
        PUBMED_ESEARCH,
        params={"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"},
        timeout=15,
    )
    search_resp.raise_for_status()
    ids = search_resp.json().get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []

    fetch_resp = requests.get(
        PUBMED_EFETCH,
        params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml"},
        timeout=15,
    )
    fetch_resp.raise_for_status()
    root = ET.fromstring(fetch_resp.content)

    results = []
    for article in root.findall(".//PubmedArticle"):
        title_el = article.find(".//ArticleTitle")
        title = "".join(title_el.itertext()).strip() if title_el is not None else "(no title)"

        abstract_parts = article.findall(".//AbstractText")
        abstract = " ".join("".join(a.itertext()).strip() for a in abstract_parts) or "(no abstract)"

        authors = []
        for author in article.findall(".//Author"):
            last = author.findtext("LastName")
            fore = author.findtext("ForeName")
            if last:
                authors.append(f"{fore} {last}".strip() if fore else last)

        pmid_el = article.find(".//PMID")
        pmid = pmid_el.text if pmid_el is not None else None

        year = article.findtext(".//PubDate/Year") or article.findtext(".//PubDate/MedlineDate", "")

        results.append({
            "source": "pubmed",
            "title": title,
            "authors": authors[:5],
            "date": year,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
            "summary": abstract[:1000],
        })
    return results


def _search_arxiv(query: str, max_results: int) -> list:
    resp = requests.get(
        ARXIV_API,
        params={"search_query": f"all:{query}", "start": 0, "max_results": max_results},
        timeout=15,
    )
    resp.raise_for_status()
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(resp.content)

    results = []
    for entry in root.findall("atom:entry", ns):
        title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
        summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
        published = (entry.findtext("atom:published", default="", namespaces=ns) or "")[:10]
        url = entry.findtext("atom:id", default="", namespaces=ns)
        authors = [
            a.findtext("atom:name", default="", namespaces=ns)
            for a in entry.findall("atom:author", ns)
        ]

        results.append({
            "source": "arxiv",
            "title": title,
            "authors": authors[:5],
            "date": published,
            "url": url,
            "summary": summary[:1000],
        })
    return results


def _normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().lower())


def _paper_id(paper: dict) -> str:
    url = paper.get("url") or ""
    if paper.get("source") == "pubmed":
        m = re.search(r"/(\d+)/?$", url)
        if m:
            return f"pubmed_{m.group(1)}"
    if paper.get("source") == "arxiv":
        # Old-style arXiv IDs include a "/" (e.g. "physics/0310029") --
        # capture past it too, or different papers collide into one file.
        m = re.search(r"abs/([\w./\-]+)", url)
        if m:
            return f"arxiv_{m.group(1).replace('/', '_')}"
    digest = hashlib.sha1((paper.get("title") or "").encode()).hexdigest()[:10]
    return f"{paper.get('source', 'paper')}_{digest}"


def _load_note(path: Path):
    text = path.read_text(encoding="utf-8")
    _, meta_block, body = text.split("---", 2)
    return json.loads(meta_block), body.strip()


def _save_note(paper: dict, query: str):
    VAULT_DIR.mkdir(parents=True, exist_ok=True)
    path = note_path(_paper_id(paper))

    if path.exists():
        try:
            meta, body = _load_note(path)
        except Exception:
            meta, body = {}, paper.get("summary", "")
    else:
        meta, body = {}, paper.get("summary", "")

    meta.setdefault("title", paper.get("title"))
    meta.setdefault("source", paper.get("source"))
    meta.setdefault("authors", paper.get("authors", []))
    meta.setdefault("date", paper.get("date"))
    meta.setdefault("url", paper.get("url"))
    meta.setdefault("saved", date.today().isoformat())
    queries = meta.setdefault("queries", [])
    if _normalize_query(query) not in [_normalize_query(q) for q in queries]:
        queries.append(query)

    path.write_text(f"---\n{json.dumps(meta, indent=2)}\n---\n\n{body}\n", encoding="utf-8")


def _search_vault(query: str) -> list:
    if not VAULT_DIR.exists():
        return []
    norm_q = _normalize_query(query)
    matches = []
    for path in VAULT_DIR.glob("*.md"):
        try:
            meta, body = _load_note(path)
        except Exception:
            continue
        if norm_q in [_normalize_query(q) for q in meta.get("queries", [])]:
            matches.append({
                "paper_id": path.stem,
                "source": meta.get("source"),
                "title": meta.get("title"),
                "authors": meta.get("authors", []),
                "date": meta.get("date"),
                "url": meta.get("url"),
                "summary": body,
            })
    return matches


def note_path(paper_id: str) -> Path:
    return VAULT_DIR / f"{_safe_id_component(paper_id)}.md"


def append_figures_section(paper_id: str, figures_md: str):
    """Replace the '## Figures' section of a vault note (if any) with fresh
    content -- used by tools/literature_figures.py so re-analyzing a paper
    doesn't pile up duplicate sections.
    """
    path = note_path(paper_id)
    meta, body = _load_note(path)
    body = re.split(r"\n## Figures\n", body, maxsplit=1)[0].rstrip()
    body = f"{body}\n\n## Figures\n\n{figures_md}"
    path.write_text(f"---\n{json.dumps(meta, indent=2)}\n---\n\n{body}\n", encoding="utf-8")


def search_literature(query: str, source: str = "both", max_results: int = 5, refresh: bool = False) -> dict:
    if not refresh:
        cached = _search_vault(query)
        if cached:
            return {"query": query, "n_results": len(cached), "results": cached, "errors": [], "from_vault": True}

    results = []
    errors = []

    if source in ("pubmed", "both"):
        try:
            results.extend(_search_pubmed(query, max_results))
        except Exception as e:
            errors.append(f"pubmed: {e}")

    if source in ("arxiv", "both"):
        try:
            results.extend(_search_arxiv(query, max_results))
        except Exception as e:
            errors.append(f"arxiv: {e}")

    for paper in results:
        paper["paper_id"] = _paper_id(paper)
        try:
            _save_note(paper, query)
        except Exception:
            pass  # vault write failures shouldn't break the search response

    return {"query": query, "n_results": len(results), "results": results, "errors": errors, "from_vault": False}
