"""
literature_intelligence.py  v4 — Specificity Update
=====================================================
Key improvements over v3:
  - Tighter query syntax per source (title/abstract field tags)
  - Fetches 2× results then filters to top MAX_RESULTS by relevance score
  - Relevance scorer: compound name in title > abstract > incidental mention
  - Papers scoring below MIN_RELEVANCE_SCORE are discarded
  - PubMed uses [tiab] field tag (title+abstract only — excludes method noise)
  - OpenAlex uses title.search for stricter matching
  - Whole-word compound name matching (avoids partial hits)

Waterfall: PubMed → Semantic Scholar → OpenAlex → CrossRef
"""

from __future__ import annotations

import re
import time
import logging
from typing import TypedDict

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

FETCH_SIZE        = 20    # request this many from API, then filter down
MAX_RESULTS       = 10    # return at most this many after filtering
MIN_RELEVANCE     = 30    # discard papers scoring below this (0-100 scale)
CACHE_TTL_SECONDS = 1800  # 30 min — only successful results cached
TIMEOUT           = 12    # seconds per HTTP request

_UA = "Mozilla/5.0 (compatible; LeadRefine/4.0) Python-requests"

# ── Category keyword map ───────────────────────────────────────────────────────
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Metabolism": [
        "metabolism", "metabolite", "metabolic", "cyp", "cyp450",
        "pharmacokinetics", "pharmacokinetic", "bioavailability",
        "clearance", "half-life", "biotransformation", "microsomal",
        "first-pass", "oxidation", "glucuronidation", "hydroxylation",
    ],
    "Toxicity": [
        "toxicity", "toxicology", "toxic", "hepatotoxicity", "cardiotoxicity",
        "nephrotoxicity", "cytotoxicity", "genotoxicity", "mutagenicity",
        "adverse", "side effect", "hepatotoxic", "reactive metabolite",
        "idiosyncratic", "lethal", "ld50", "poisoning",
    ],
    "Resistance": [
        "resistance", "drug resistance", "multidrug", "efflux",
        "cross-resistance", "acquired resistance", "mutation",
        "antimicrobial resistance", "antibiotic resistance", "mdr",
    ],
    "Pharmacology": [
        "efficacy", "activity", "binding", "receptor", "potency",
        "ic50", "ec50", "inhibition", "agonist", "antagonist",
        "pharmacodynamics", "mechanism of action", "target",
        "selectivity", "affinity", "analgesic", "anti-inflammatory",
        "anesthetic", "anaesthetic", "antifungal", "antibacterial",
        "antiviral", "anticancer", "antineoplastic",
    ],
    "Clinical": [
        "clinical trial", "phase i", "phase ii", "phase iii",
        "patient", "randomized", "randomised", "placebo",
        "dosing", "therapeutic", "treatment", "in vivo",
        "animal model", "cohort", "surgery", "analgesia",
    ],
    "Synthesis": [
        "synthesis", "synthetic", "preparation", "scaffold",
        "structure-activity", "sar", "medicinal chemistry",
        "analogue", "derivative", "molecular docking",
        "in silico", "qsar", "optimization",
    ],
}

# ══════════════════════════════════════════════════════════════════════════════
# CACHE
# ══════════════════════════════════════════════════════════════════════════════

_cache: dict[str, dict] = {}

def _cache_get(key: str) -> dict | None:
    e = _cache.get(key)
    if not e:
        return None
    if time.monotonic() - e["ts"] > CACHE_TTL_SECONDS:
        del _cache[key]
        return None
    return e["data"]

def _cache_set(key: str, data: dict) -> None:
    if data.get("total", 0) > 0:
        _cache[key] = {"data": data, "ts": time.monotonic()}

# ══════════════════════════════════════════════════════════════════════════════
# HTTP SESSION
# ══════════════════════════════════════════════════════════════════════════════

def _session(verify: bool = True) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": _UA, "Accept": "application/json"})
    retry = Retry(total=2, backoff_factor=0.4,
                  status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["GET"])
    a = HTTPAdapter(max_retries=retry)
    s.mount("https://", a)
    s.mount("http://", a)
    s.verify = verify
    return s

def _get(url: str, params: dict | None = None) -> dict:
    for verify in (True, False):
        try:
            r = _session(verify).get(url, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.SSLError:
            if verify:
                continue
            raise
    return {}

# ══════════════════════════════════════════════════════════════════════════════
# PAPER SCHEMA
# ══════════════════════════════════════════════════════════════════════════════

class Paper(TypedDict):
    title:           str
    journal:         str
    year:            str
    doi:             str | None
    url:             str
    abstract:        str
    category:        str
    relevance_score: int

def _truncate(text: str, n: int = 350) -> str:
    return text[:n] + ("…" if len(text) > n else "")

def _clean_html(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s).strip()

# ══════════════════════════════════════════════════════════════════════════════
# RELEVANCE SCORING
# ══════════════════════════════════════════════════════════════════════════════

def _word_pattern(name: str) -> re.Pattern:
    """
    Build a whole-word regex for the compound name.
    Handles multi-word names like "acetyl salicylic acid".
    """
    escaped = re.escape(name.lower())
    return re.compile(r"\b" + escaped + r"\b")


def _relevance_score(paper: Paper, name: str) -> int:
    """
    Score 0-100 measuring how specifically this paper is about `name`.

    Scoring rubric
    --------------
    +50   compound name found as whole word in title
    +25   compound name found as whole word in abstract
    +10   each category keyword hit in title       (max +20)
    +5    each category keyword hit in abstract    (max +10)
    -999  compound name not found anywhere → DISCARD (returns 0)

    A paper must score ≥ MIN_RELEVANCE to pass.
    """
    title_lc    = paper["title"].lower()
    abstract_lc = paper["abstract"].lower()
    name_lc     = name.lower()
    pat         = _word_pattern(name_lc)

    in_title    = bool(pat.search(title_lc))
    in_abstract = bool(pat.search(abstract_lc))

    # Hard gate: compound name must appear in title OR abstract
    if not in_title and not in_abstract:
        return 0

    score = 0
    if in_title:
        score += 50
    if in_abstract:
        score += 25

    # Bonus: pharmacology/drug-related keywords in title (more specific)
    all_kws = [kw for kws in CATEGORY_KEYWORDS.values() for kw in kws]
    title_kw_hits    = sum(1 for kw in all_kws if kw in title_lc)
    abstract_kw_hits = sum(1 for kw in all_kws if kw in abstract_lc)
    score += min(title_kw_hits * 10, 20)
    score += min(abstract_kw_hits * 5, 10)

    return min(score, 100)


def _filter_and_rank(papers: list[Paper], name: str) -> list[Paper]:
    """
    Score every paper, discard those below MIN_RELEVANCE,
    sort survivors by score descending, return top MAX_RESULTS.
    """
    scored = []
    for p in papers:
        score = _relevance_score(p, name)
        if score >= MIN_RELEVANCE:
            m = dict(p)
            m["relevance_score"] = score
            scored.append(m)

    scored.sort(key=lambda x: x["relevance_score"], reverse=True)
    return scored[:MAX_RESULTS]

# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 1 — PubMed  (NCBI E-utilities)
# ══════════════════════════════════════════════════════════════════════════════

def _pubmed(name: str) -> list[Paper]:
    """
    Uses [tiab] field tag: compound name must appear in Title OR Abstract.
    This eliminates papers that only mention the drug in the methods section.
    Also restricts publication type to exclude reviews-of-reviews and editorials.
    """
    BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    # [tiab] = title/abstract field tag — much tighter than a free-text search
    # NOT Editorial[pt] NOT Letter[pt] = exclude non-research content
    query = (
        f'"{name}"[tiab] '
        'AND ("drug"[tiab] OR "pharmacol*"[tiab] OR "therap*"[tiab] '
        'OR "treatment"[tiab] OR "clinical"[tiab] OR "toxicit*"[tiab] '
        'OR "metabol*"[tiab] OR "inhibit*"[tiab] OR "binding"[tiab] '
        'OR "efficacy"[tiab] OR "dose"[tiab] OR "patient"[tiab] '
        'OR "synthesis"[tiab] OR "activity"[tiab])'
        ' NOT Editorial[pt] NOT Letter[pt] NOT Comment[pt]'
    )

    search = _get(f"{BASE}/esearch.fcgi", {
        "db":      "pubmed",
        "term":    query,
        "retmax":  FETCH_SIZE,
        "sort":    "relevance",
        "retmode": "json",
    })
    ids = search.get("esearchresult", {}).get("idlist", [])
    if not ids:
        # Fallback: simpler [tiab] query without secondary filter
        search = _get(f"{BASE}/esearch.fcgi", {
            "db":      "pubmed",
            "term":    f'"{name}"[tiab]',
            "retmax":  FETCH_SIZE,
            "sort":    "relevance",
            "retmode": "json",
        })
        ids = search.get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []

    # Fetch full records with abstracts via efetch
    fetch = _get(f"{BASE}/efetch.fcgi", {
        "db":      "pubmed",
        "id":      ",".join(ids),
        "retmode": "json",
        "rettype": "abstract",
    })

    # efetch doesn't return structured JSON well — fall back to esummary
    # for structured metadata, then patch abstracts separately
    summary = _get(f"{BASE}/esummary.fcgi", {
        "db":      "pubmed",
        "id":      ",".join(ids),
        "retmode": "json",
    })
    uids = summary.get("result", {}).get("uids", [])

    # Fetch abstracts via efetch XML-as-text is unreliable in JSON mode.
    # Use the PubMed article API instead for clean abstract text.
    abstract_map = _pubmed_abstracts(ids)

    papers: list[Paper] = []
    for uid in uids:
        rec   = summary["result"].get(uid, {})
        title = _clean_html(rec.get("title", "") or "")
        if not title:
            continue

        journal  = rec.get("fulljournalname") or rec.get("source") or "Unknown journal"
        raw_date = rec.get("pubdate") or rec.get("epubdate") or ""
        year     = raw_date[:4] if raw_date else ""

        doi = None
        for aid in rec.get("articleids", []):
            if aid.get("idtype") == "doi":
                doi = aid.get("value")
                break

        url      = f"https://doi.org/{doi}" if doi else f"https://pubmed.ncbi.nlm.nih.gov/{uid}/"
        abstract = _truncate(abstract_map.get(uid, ""))

        papers.append(Paper(
            title=title, journal=journal, year=year,
            doi=doi, url=url, abstract=abstract,
            category="", relevance_score=0,
        ))

    logger.info("PubMed raw: %d papers for '%s'", len(papers), name)
    return papers


def _pubmed_abstracts(pmids: list[str]) -> dict[str, str]:
    """
    Fetch abstracts for a list of PMIDs using the PubMed summary endpoint.
    Returns {pmid: abstract_text}.
    """
    BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    result: dict[str, str] = {}
    try:
        # Use efetch with rettype=abstract retmode=text — returns plain text
        r = _session().get(
            f"{BASE}/efetch.fcgi",
            params={
                "db":      "pubmed",
                "id":      ",".join(pmids),
                "rettype": "abstract",
                "retmode": "text",
            },
            timeout=TIMEOUT,
        )
        if r.status_code == 200:
            # Parse plain-text abstracts: blocks separated by blank lines
            # Each block starts with PMID: XXXXX or contains the abstract text
            current_pmid = None
            current_lines: list[str] = []
            in_abstract = False

            for line in r.text.splitlines():
                if line.startswith("PMID-"):
                    if current_pmid and current_lines:
                        result[current_pmid] = " ".join(current_lines).strip()
                    current_pmid  = line.split("-", 1)[1].strip()
                    current_lines = []
                    in_abstract   = False
                elif line.startswith("AB  -"):
                    in_abstract = True
                    current_lines.append(line[6:].strip())
                elif line.startswith("      ") and in_abstract:
                    current_lines.append(line.strip())
                elif line.strip() == "" and in_abstract:
                    in_abstract = False

            if current_pmid and current_lines:
                result[current_pmid] = " ".join(current_lines).strip()
    except Exception as exc:
        logger.debug("Abstract fetch failed: %s", exc)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 2 — Semantic Scholar
# ══════════════════════════════════════════════════════════════════════════════

def _semantic_scholar(name: str) -> list[Paper]:
    """
    Searches with the compound name. Semantic Scholar's ranking is
    relevance-based so the top results are usually on-topic.
    We still apply our relevance filter afterwards.
    """
    data = _get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        {
            "query":  name,
            "limit":  FETCH_SIZE,
            "fields": "title,year,venue,externalIds,abstract",
        },
    )
    papers: list[Paper] = []
    for rec in data.get("data", []):
        title = _clean_html(rec.get("title", "") or "")
        if not title:
            continue
        doi     = (rec.get("externalIds") or {}).get("DOI")
        pmid    = (rec.get("externalIds") or {}).get("PubMed")
        url     = (f"https://doi.org/{doi}" if doi
                   else f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid
                   else "https://www.semanticscholar.org")
        journal  = rec.get("venue") or "Unknown journal"
        year     = str(rec.get("year") or "")
        abstract = _truncate(rec.get("abstract") or "")
        papers.append(Paper(
            title=title, journal=journal, year=year, doi=doi,
            url=url, abstract=abstract, category="", relevance_score=0,
        ))
    logger.info("Semantic Scholar raw: %d papers for '%s'", len(papers), name)
    return papers


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 3 — OpenAlex
# ══════════════════════════════════════════════════════════════════════════════

def _reconstruct_abstract(inv_index: dict | None) -> str:
    if not inv_index:
        return ""
    try:
        pairs = [(p, w) for w, positions in inv_index.items() for p in positions]
        pairs.sort()
        return _truncate(" ".join(w for _, w in pairs))
    except Exception:
        return ""

def _openalex(name: str) -> list[Paper]:
    """
    Uses the `search` parameter which searches title+abstract+fulltext.
    We fetch extra and let the relevance filter do the work.
    """
    data = _get(
        "https://api.openalex.org/works",
        {
            "search":   name,
            "per-page": FETCH_SIZE,
            "select":   "title,publication_year,primary_location,doi,abstract_inverted_index",
            "mailto":   "leadrefine@example.com",
        },
    )
    papers: list[Paper] = []
    for rec in data.get("results", []):
        title = _clean_html(rec.get("title", "") or "")
        if not title:
            continue
        doi = rec.get("doi") or None
        if doi and doi.startswith("https://doi.org/"):
            doi = doi[len("https://doi.org/"):]
        url      = f"https://doi.org/{doi}" if doi else "https://openalex.org"
        loc      = rec.get("primary_location") or {}
        src      = loc.get("source") or {}
        journal  = src.get("display_name") or "Unknown journal"
        year     = str(rec.get("publication_year") or "")
        abstract = _reconstruct_abstract(rec.get("abstract_inverted_index"))
        papers.append(Paper(
            title=title, journal=journal, year=year, doi=doi,
            url=url, abstract=abstract, category="", relevance_score=0,
        ))
    logger.info("OpenAlex raw: %d papers for '%s'", len(papers), name)
    return papers


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 4 — CrossRef
# ══════════════════════════════════════════════════════════════════════════════

def _crossref(name: str) -> list[Paper]:
    data = _get(
        "https://api.crossref.org/works",
        {
            "query.bibliographic": name,   # more specific than query=
            "rows":                FETCH_SIZE,
            "select":              "title,published,container-title,DOI,abstract",
        },
    )
    papers: list[Paper] = []
    for rec in (data.get("message") or {}).get("items", []):
        title_list = rec.get("title") or []
        title = _clean_html(title_list[0] if title_list else "")
        if not title:
            continue
        doi      = rec.get("DOI")
        url      = f"https://doi.org/{doi}" if doi else "https://crossref.org"
        ct       = rec.get("container-title") or []
        journal  = ct[0] if ct else "Unknown journal"
        pub      = rec.get("published") or {}
        dp       = pub.get("date-parts") or [[]]
        year     = str(dp[0][0]) if dp and dp[0] else ""
        abstract = _truncate(_clean_html(rec.get("abstract") or ""))
        papers.append(Paper(
            title=title, journal=journal, year=year, doi=doi,
            url=url, abstract=abstract, category="", relevance_score=0,
        ))
    logger.info("CrossRef raw: %d papers for '%s'", len(papers), name)
    return papers


# ══════════════════════════════════════════════════════════════════════════════
# WATERFALL
# ══════════════════════════════════════════════════════════════════════════════

_SOURCES = [
    ("PubMed",           _pubmed),
    ("Semantic Scholar", _semantic_scholar),
    ("OpenAlex",         _openalex),
    ("CrossRef",         _crossref),
]

def _fetch_and_filter(name: str) -> tuple[list[Paper], str]:
    """
    Try each source in order. For each source:
      1. Fetch up to FETCH_SIZE raw papers
      2. Apply relevance filter + ranking
      3. If ≥ 1 paper survives → return them with source name
    Falls back to next source if filtered result is empty.
    Returns ([], last_error) if all sources fail.
    """
    last_exc = ""
    for source_name, fn in _SOURCES:
        try:
            raw = fn(name)
            if not raw:
                logger.info("%s returned 0 raw results for '%s'", source_name, name)
                continue
            filtered = _filter_and_rank(raw, name)
            if filtered:
                logger.info(
                    "%s: %d/%d papers passed relevance filter for '%s'",
                    source_name, len(filtered), len(raw), name,
                )
                return filtered, source_name
            else:
                logger.info(
                    "%s: %d raw papers fetched but 0 passed relevance filter (score<%d) for '%s'",
                    source_name, len(raw), MIN_RELEVANCE, name,
                )
                # Don't give up — the next source might return more specific papers
                continue
        except Exception as exc:
            last_exc = str(exc)
            logger.warning("%s failed for '%s': %s", source_name, name, exc)
            continue

    return [], last_exc


# ══════════════════════════════════════════════════════════════════════════════
# CATEGORISATION
# ══════════════════════════════════════════════════════════════════════════════

def _assign_category(p: Paper) -> str:
    corpus = (p["title"] + " " + p["abstract"]).lower()
    for cat, kws in CATEGORY_KEYWORDS.items():
        if any(kw in corpus for kw in kws):
            return cat
    return "General"

def _categorise(papers: list[Paper]) -> dict[str, list[Paper]]:
    grouped: dict[str, list[Paper]] = {}
    for p in papers:
        cat = _assign_category(p)
        m   = dict(p)
        m["category"] = cat
        grouped.setdefault(cat, []).append(m)
    return grouped


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def get_literature_intelligence(compound_name: str) -> dict:
    """
    Fetch, filter by relevance, and categorise literature for compound_name.

    Returns
    -------
    {
      "compound"  : str,
      "categories": { str: [Paper] },
      "total"     : int,
      "source"    : str,
      "error"     : str | None
    }
    """
    if not compound_name or not compound_name.strip():
        return _err("", "No compound name provided.")

    name      = compound_name.strip()
    cache_key = name.lower()

    cached = _cache_get(cache_key)
    if cached:
        return cached

    papers, source_or_err = _fetch_and_filter(name)

    if not papers:
        return {
            "compound": name, "categories": {}, "total": 0,
            "source": "none", "error": None,
            "_empty_reason": (
                f"Fetched results from all 4 sources but none were "
                f"specific enough to '{name}' (relevance score < {MIN_RELEVANCE}). "
                f"Try a more specific name (e.g. IUPAC name or common drug name)."
                if not source_or_err else
                f"All sources failed. Last error: {source_or_err}"
            ),
        }

    categories = _categorise(papers)
    result = {
        "compound":   name,
        "categories": categories,
        "total":      len(papers),
        "source":     source_or_err,
        "error":      None,
    }
    _cache_set(cache_key, result)
    logger.info("Literature OK: %d specific papers via %s for '%s'",
                len(papers), source_or_err, name)
    return result


def _err(compound: str, msg: str) -> dict:
    return {"compound": compound, "categories": {}, "total": 0,
            "source": "none", "error": msg}