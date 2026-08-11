"""
Literature figure analysis tool.

Many biosensor papers show their own CV/CA/EIS traces or SEM/microscopy
images of the electrode surface -- useful reference points for judging
whether your own sensor data/photos look "good" or "bad" by comparison.
This tool pulls those figures out of a paper already in the literature
vault (see tools/literature.py) and captions each one with the vision
model, then appends the captions back onto that paper's vault note.

Two sources:
  - arXiv: papers there are open PDFs, pulled directly.
  - PubMed: most are paywalled, but some are open access via PMC. Looked up
    through Europe PMC's search API (EXT_ID:<pmid> AND SRC:MED) -- if
    isOpenAccess is "Y" and a pmcid is returned, the PDF is fetched via
    Europe PMC's render endpoint (europepmc.org/articles/<pmcid>?pdf=render,
    found by inspecting the search result's fullTextUrlList rather than
    guessing a URL pattern -- an earlier guess, NCBI's ptpmcrender.fcgi,
    doesn't respond at all). Paywalled PubMed papers return an explicit
    "no open-access full text" error rather than silently doing nothing.
"""

import io
import json
import re

import fitz  # PyMuPDF
import requests
from google.genai import types
from PIL import Image

from tools._gemini import client, MODEL
from tools.literature import VAULT_DIR, _load_note, append_figures_section, note_path

FIGURES_DIR = VAULT_DIR / "figures"
MIN_FIGURE_DIM = 200  # px; filters out logos/icons/decorative marks

LITERATURE_FIGURES_SCHEMA = {
    "type": "function",
    "function": {
        "name": "analyze_literature_figures",
        "description": (
            "Extract and caption the figures (CV/CA/EIS plots, SEM/microscopy images, "
            "schematics, etc.) from a paper already found via search_literature. Downloads "
            "the PDF, pulls out embedded images above a minimum size, and uses the vision "
            "model to describe what each shows -- useful as reference 'good vs bad' examples "
            "when characterizing your own sensor data or photos. Captions are saved back onto "
            "the paper's entry in literature_vault/ for future reuse. Works for arXiv papers "
            "and PubMed papers that are open access via PMC; paywalled PubMed papers return "
            "an explicit error."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "paper_id": {
                    "type": "string",
                    "description": (
                        "The paper_id returned by search_literature (e.g. 'arxiv_2012.05543v1')."
                    ),
                },
                "max_figures": {
                    "type": "integer",
                    "description": "Max figures to extract and caption. Default 5.",
                },
                "context": {
                    "type": "string",
                    "description": "Optional context to focus the captions, e.g. 'looking for CV peak separation examples'.",
                },
            },
            "required": ["paper_id"],
        },
    },
}

_CAPTION_PROMPT_TEMPLATE = """This image is a figure extracted from a biosensor research paper. \
Describe what it shows and classify it as one of: "electrochemical plot" (CV/CA/EIS/DPV/SWV \
trace), "microscopy/SEM image" (electrode surface characterization), "schematic/diagram", or \
"other" (e.g. a data table rendered as an image, a device photo).

If it's an electrochemical plot, note things useful for judging sensor quality: peak shape and \
separation, noise level, baseline drift, linearity of a calibration series -- and whether it \
reads as a "good" (clean, well-defined, low-noise) or "poor" (broken/absent electron transfer, \
high baseline noise, drifting baseline) example.

If it's a microscopy image, note surface uniformity, visible defects, cracking, or contamination.
{context_line}
Respond ONLY with JSON: {{"figure_type": "...", "description": "...", "quality_note": "..."}}
quality_note should be empty string if not applicable (e.g. for schematics)."""


# Old-style arXiv IDs are "<category>/<7digits><version>" (e.g. "cond-mat/0602636v1").
# tools/literature.py's _paper_id replaces that "/" with "_" for the vault filename
# (arxiv_cond-mat_0602636v1), so it has to be converted back for the real download URL --
# missing this the first time caused live 404s during a vault sweep, not just in theory.
_OLD_STYLE_ARXIV_RE = re.compile(r"^([a-zA-Z\-]+)_(\d.*)$")


def _reconstruct_arxiv_id(paper_id: str) -> str:
    raw = paper_id[len("arxiv_"):]
    m = _OLD_STYLE_ARXIV_RE.match(raw)
    return f"{m.group(1)}/{m.group(2)}" if m else raw


def _download_arxiv_pdf(arxiv_id: str) -> bytes:
    # arxiv_id here is the real arXiv ID, e.g. "2012.05543v1" or "cond-mat/0602636v1"
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.content


EUROPEPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


def _lookup_pmc(pmid: str):
    """Returns a pmcid string if this PubMed article is open access via PMC, else None."""
    resp = requests.get(EUROPEPMC_SEARCH, params={"query": f"EXT_ID:{pmid} AND SRC:MED", "format": "json"}, timeout=15)
    resp.raise_for_status()
    results = resp.json().get("resultList", {}).get("result", [])
    if not results:
        return None
    r = results[0]
    if r.get("isOpenAccess") != "Y" or not r.get("pmcid"):
        return None
    return r["pmcid"]


def _download_pmc_pdf(pmcid: str) -> bytes:
    # Found via a search result's fullTextUrlList, not guessed -- the more
    # "obvious" NCBI endpoint (ptpmcrender.fcgi) doesn't respond at all.
    resp = requests.get(f"https://europepmc.org/articles/{pmcid}?pdf=render", timeout=30)
    resp.raise_for_status()
    if not resp.content.startswith(b"%PDF"):
        raise ValueError(f"Europe PMC did not return a PDF for {pmcid}")
    return resp.content


def _extract_figures(pdf_bytes: bytes, max_figures: int):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    figures = []
    for page_index in range(len(doc)):
        for img in doc[page_index].get_images(full=True):
            xref = img[0]
            base = doc.extract_image(xref)
            if base["width"] < MIN_FIGURE_DIM or base["height"] < MIN_FIGURE_DIM:
                continue
            figures.append({
                "page": page_index + 1,
                "ext": base["ext"],
                "bytes": base["image"],
            })
            if len(figures) >= max_figures:
                return figures
    return figures


_MAX_FIGURE_DIM = 1024  # downsize before captioning -- PDF-extracted figures can be large enough
                        # to blow past Ollama's default 4096-token context on their own (hit in practice)


def _downsize_for_captioning(image_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(image_bytes))
    if max(img.size) > _MAX_FIGURE_DIM:
        scale = _MAX_FIGURE_DIM / max(img.size)
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _caption_figure(image_bytes: bytes, context: str = "") -> dict:
    context_line = f"\nAdditional context: {context}\n" if context else ""
    prompt = _CAPTION_PROMPT_TEMPLATE.format(context_line=context_line)
    response = client().models.generate_content(
        model=MODEL,
        contents=[prompt, types.Part.from_bytes(data=_downsize_for_captioning(image_bytes), mime_type="image/jpeg")],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    try:
        parsed = json.loads(response.text)
    except json.JSONDecodeError:
        parsed = {"figure_type": "other", "description": "(caption failed to parse)", "quality_note": ""}
    parsed.setdefault("figure_type", "other")
    parsed.setdefault("description", "")
    parsed.setdefault("quality_note", "")
    return parsed


def analyze_literature_figures(paper_id: str, max_figures: int = 5, context: str = "") -> dict:
    note_file = note_path(paper_id)
    if not note_file.exists():
        return {
            "status": "error",
            "message": f"No vault entry for paper_id '{paper_id}'. Run search_literature first.",
        }

    meta, _ = _load_note(note_file)
    source = meta.get("source")

    if source == "arxiv":
        arxiv_id = _reconstruct_arxiv_id(paper_id)
        try:
            pdf_bytes = _download_arxiv_pdf(arxiv_id)
        except Exception as e:
            return {"status": "error", "message": f"Could not download PDF: {e}"}
    elif source == "pubmed":
        pmid = paper_id[len("pubmed_"):]
        try:
            pmcid = _lookup_pmc(pmid)
        except Exception as e:
            return {"status": "error", "message": f"PMC lookup failed: {e}"}
        if pmcid is None:
            return {
                "status": "error",
                "message": f"'{paper_id}' has no open-access full text in PMC -- figure extraction isn't possible.",
            }
        try:
            pdf_bytes = _download_pmc_pdf(pmcid)
        except Exception as e:
            return {"status": "error", "message": f"Could not download PMC PDF ({pmcid}): {e}"}
    else:
        return {"status": "error", "message": f"Unsupported source '{source}' for figure extraction."}

    try:
        raw_figures = _extract_figures(pdf_bytes, max_figures)
    except Exception as e:
        return {"status": "error", "message": f"Could not parse PDF: {e}"}

    if not raw_figures:
        return {"status": "warn", "message": "No figures above the size threshold were found.", "figures": []}

    out_dir = FIGURES_DIR / paper_id
    out_dir.mkdir(parents=True, exist_ok=True)

    figures = []
    for i, fig in enumerate(raw_figures):
        img_path = out_dir / f"fig_{i}_p{fig['page']}.{fig['ext']}"
        img_path.write_bytes(fig["bytes"])
        caption = _caption_figure(fig["bytes"], context)
        figures.append({
            "image_path": str(img_path),
            "page": fig["page"],
            **caption,
        })

    figures_md = "\n\n".join(
        f"**Figure {i + 1} (p.{f['page']}, {f['figure_type']})**: {f['description']}"
        + (f" _Quality note: {f['quality_note']}_" if f["quality_note"] else "")
        for i, f in enumerate(figures)
    )
    try:
        append_figures_section(paper_id, figures_md)
    except Exception:
        pass  # captions are still returned even if the vault write fails

    return {"status": "ok", "paper_id": paper_id, "n_figures": len(figures), "figures": figures}
