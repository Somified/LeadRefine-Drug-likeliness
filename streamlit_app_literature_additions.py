# ══════════════════════════════════════════════════════════════════════════════
# LITERATURE INTELLIGENCE — Integration Notes
# ══════════════════════════════════════════════════════════════════════════════
#
# NEW FILES ADDED (do not modify existing files):
# ────────────────────────────────────────────────
#   literature_intelligence.py    ← new standalone backend module
#   streamlit_app.py              ← patched (3 additions only, nothing removed)
#   streamlit_app_literature_additions.py  ← diff/instructions reference file
#
# REQUIREMENTS ADDITIONS:
# ───────────────────────
# The `requests` library is already in your requirements.txt.
# No new pip packages are needed.
# The only stdlib modules used are: time, logging, re, typing.
#
# PLACEMENT IN YOUR PROJECT:
# ──────────────────────────
#   LEADREFINE/
#   ├── adme_analysis.py               (unchanged)
#   ├── analysis.py                    (unchanged)
#   ├── streamlit_app.py               (patched — 3 additions marked [NEW])
#   ├── literature_intelligence.py     ← NEW — place here
#   ├── streamlit_app_literature_additions.py  ← NEW — reference / docs
#   ├── app.py                         (unchanged)
#   ├── run.py                         (unchanged)
#   └── requirements.txt               (unchanged — requests already present)
#
# HOW THE FEATURE WORKS:
# ──────────────────────
# 1. After ADME analysis completes and results render, the new
#    "📚 Literature Intelligence" section appears below render_detail().
#
# 2. The compound name is auto-filled from PubChem metadata when available.
#    Users can also type any name manually and click "🔍 Search".
#
# 3. literature_intelligence.py calls the Europe PMC REST API:
#      GET https://www.ebi.ac.uk/europepmc/webservices/rest/search
#          ?query=<compound_name>&format=json&pageSize=10
#          &resultType=core&sort=date desc
#
# 4. Up to 10 papers are returned, normalised, and assigned to categories:
#      Metabolism  · Toxicity  · Resistance  · Pharmacology
#      Clinical    · Synthesis · General
#    Category assignment uses keyword matching on title + abstract text.
#
# 5. Results are cached in-memory for 30 minutes (per compound name).
#    Re-searching the same compound within 30 min hits the cache instantly.
#
# 6. Error handling covers: timeout, connection error, HTTP error, empty result.
#    All surface a clear user-facing message. No crash or traceback exposed.
#
# 7. In batch mode, each molecule gets its own Literature section inside
#    its expander, with unique Streamlit widget keys to avoid conflicts.
#
# WHAT WAS NOT TOUCHED:
# ─────────────────────
#   • All ADME calculation logic (analysis.py, adme_analysis.py) — untouched
#   • All visualisation code (build_adme_panels, build_adme_figure) — untouched
#   • All PubChem integration (get_pubchem_metadata) — untouched
#   • All existing UI sections (KPI metrics, charts, detail expanders) — untouched
#   • The sidebar, glossary, batch progress bar — all untouched
#   • No existing function was modified or deleted