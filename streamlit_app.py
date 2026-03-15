"""
streamlit_app.py — ADME Analysis Dashboard
Run with:  streamlit run streamlit_app.py
Requires adme_analysis.py and literature_intelligence.py in the same folder.
"""

import io, time
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from adme_analysis import (
    analyze_smiles,
    build_adme_panels,
    build_adme_figure,
    generate_optimization_advice,
)
from literature_intelligence import get_literature_intelligence

st.set_page_config(page_title="ADME Analyzer", page_icon="🧬", layout="wide")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main { background-color: #0D1117; }
.stTextInput > div > div > input,
.stTextArea  > div > div > textarea {
    background-color: #161B22; color: #E6EDF3;
    border: 1px solid #30363D; border-radius: 6px; font-family: monospace;
}
.sec-head {
    font-size: 1.22rem; font-weight: 700; color: #58A6FF;
    border-bottom: 1px solid #30363D; padding-bottom: 4px;
    margin: 1.2rem 0 0.5rem 0;
}
.sub-head { font-size: 1.0rem; font-weight: 600; color: #E6EDF3; margin: 0.7rem 0 0.25rem 0; }
.gloss-term { font-weight: 700; color: #58A6FF; }
.gloss-def  { color: #C9D1D9; font-size: 0.92rem; }
.pubchem-name { font-size: 1.5rem; font-weight: 800; color: #E6EDF3; margin-bottom: 4px; }
.pubchem-meta { font-size: 0.97rem; color: #8B949E; }
.info-pill {
    display:inline-block; background:#21262D; border:1px solid #30363D;
    border-radius:20px; padding:2px 11px; margin:3px 3px;
    font-size:0.82rem; color:#C9D1D9;
}

/* ── Literature Intelligence ──────────────────────────────────────────────── */
.lit-header { font-size:1.25rem; font-weight:800; color:#E6EDF3;
    border-bottom:2px solid #30363D; padding-bottom:6px; margin:1.6rem 0 0.3rem 0; }
.lit-sub { font-size:0.86rem; color:#8B949E; margin-bottom:1rem; }
.lit-source { font-size:0.75rem; color:#3FB950; font-weight:600; }
.lit-cat { font-size:1rem; font-weight:700; padding-left:4px;
    border-left:3px solid var(--ac); color:var(--ac);
    margin:1rem 0 0.3rem 0; }
.lit-card { background:#161B22; border:1px solid #30363D; border-radius:8px;
    padding:10px 14px; margin-bottom:7px; }
.lit-card:hover { border-color:#58A6FF; }
.lit-title { font-size:0.92rem; font-weight:600; color:#E6EDF3;
    margin-bottom:2px; line-height:1.35; }
.lit-meta { font-size:0.78rem; color:#8B949E; }
.lit-abs { font-size:0.77rem; color:#8B949E; font-style:italic;
    margin-top:4px; line-height:1.4; }
.lit-pill { display:inline-block; background:#21262D; border:1px solid #30363D;
    border-radius:14px; padding:2px 11px; margin:2px; font-size:0.78rem; color:#C9D1D9; }
.lit-empty { background:#161B22; border:1px dashed #30363D; border-radius:8px;
    padding:18px; text-align:center; color:#8B949E; font-size:0.9rem; }

</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def fig_to_bytes(fig, dpi=300):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()

def risk_icon(val):
    return {"HIGH":"🟠","CRITICAL":"🔴","MODERATE":"🟡","LOW":"🟢",
            "CLEAN":"🟢","POSITIVE":"🔴","NEGATIVE":"🟢","FLAGGED":"🟠"
            }.get(str(val).upper(), "⚪")

def sh(t):  return f'<p class="sec-head">{t}</p>'
def sub(t): return f'<p class="sub-head">{t}</p>'

@st.cache_data(show_spinner=False)
def cached_analyze(smiles, fetch_pubchem):
    return analyze_smiles(smiles, fetch_pubchem=fetch_pubchem)


# ── GLOSSARY ──────────────────────────────────────────────────────────────────
GLOSSARY = {
    "Drug-likeness Score (0–100)": (
        "A composite heuristic score estimating how suitable a molecule is as an oral drug. "
        "≥ 70 = ACCEPT, 50–69 = borderline, < 50 = REJECT."
    ),
    "Lipinski Rule-of-Five": (
        "Oral bioavailability filter: MW ≤ 500 Da, LogP ≤ 5, H-bond donors ≤ 5, "
        "H-bond acceptors ≤ 10. Failing ≥ 2 rules predicts poor absorption."
    ),
    "LogP": (
        "Octanol–water partition coefficient — measure of lipophilicity. "
        "High LogP (> 5) → poor solubility, high metabolism. Low LogP (< 0) → poor membrane permeation."
    ),
    "TPSA (Topological Polar Surface Area)": (
        "Sum of polar atom surface areas in Å². Predicts gut absorption and BBB penetration. "
        "≤ 90 Å² → good oral absorption; ≤ 60 Å² → likely CNS penetration; > 140 Å² → poor oral bioavailability."
    ),
    "CYP450 (Cytochrome P450)": (
        "Family of liver enzymes responsible for metabolising ~75% of drugs. "
        "Key isoforms: CYP3A4, CYP2D6, CYP2C9, CYP2C19, CYP1A2. "
        "Structural alerts predict inhibition risk and potential drug–drug interactions (DDIs)."
    ),
    "hERG Liability": (
        "hERG is a cardiac ion channel. Inhibition prolongs the QT interval, "
        "risking fatal arrhythmia (torsades de pointes). High LogP + basic nitrogen → HIGH risk."
    ),
    "Ames Mutagenicity": (
        "Bacterial reverse-mutation assay predicting genotoxicity. "
        "POSITIVE indicates structural alerts (nitroaromatics, alkylating agents) associated with DNA damage."
    ),
    "Microsomal Stability / t½": (
        "Predicted in vitro half-life in liver microsomes. "
        "HIGH (t½ > 60 min) = metabolically stable. LOW (t½ < 30 min) = rapid clearance → short in vivo half-life."
    ),
    "Reactive Metabolites": (
        "Some functional groups (furans, thiophenes, catechols) are bioactivated to electrophilic species "
        "that covalently modify proteins → idiosyncratic toxicity. CRITICAL = strong concern."
    ),
    "GSH Trapping": (
        "Glutathione (GSH) trapping assay detects reactive intermediates in vitro. "
        "POSITIVE = molecule forms adducts with GSH, indicating bioactivation liability."
    ),
    "MBI (Mechanism-Based Inactivation)": (
        "Irreversible CYP inhibition caused by a reactive metabolite binding covalently to the enzyme. "
        "Results in non-linear DDIs and requires time-dependent inhibition (TDI) follow-up assays."
    ),
    "PAINS (Pan-Assay Interference Compounds)": (
        "Structural motifs (e.g. rhodanines, catechols, quinones) that produce false positives in HTS assays "
        "through non-specific mechanisms (aggregation, redox cycling, fluorescence). FLAGGED = assay artefact risk."
    ),
    "Brenk Alerts": (
        "Substructure filter from Brenk et al. (2008) identifying fragments with unfavourable "
        "physicochemical properties or known toxicity, used for lead-hopping and library design."
    ),
    "SureChEMBL / ICH S2(R1)": (
        "Genotoxic impurity alerts based on structural classes from regulatory guidelines (ICH S2(R1), REACH). "
        "HIGH = regulatory concern requiring dedicated genotoxicity battery before clinical trials."
    ),
    "PPB (Plasma Protein Binding)": (
        "Fraction of drug bound to plasma proteins (albumin, AGP). "
        "Only unbound (free) fraction is pharmacologically active. "
        "> 95% binding → reduced free drug; may affect efficacy and duration."
    ),
    "Caco-2 Permeability (Papp)": (
        "Human colon carcinoma cell monolayer assay modelling intestinal absorption. "
        "Papp > 20 nm/s = HIGH (good oral absorption); < 5 nm/s = LOW (poor absorption)."
    ),
    "P-gp (P-glycoprotein) Efflux": (
        "ABC transporter (MDR1/ABCB1) that pumps substrates out of cells — intestinal epithelium, BBB, tumour cells. "
        "HIGH P-gp risk → reduced oral absorption and CNS penetration; potential multidrug resistance."
    ),
    "Metabolic Soft Spots": (
        "Structural positions predicted to undergo preferential oxidative metabolism (CYP-mediated). "
        "Blocking these sites (fluorination, deuteration, steric shielding) can improve metabolic stability."
    ),
}


def render_glossary():
    st.markdown(sh("📖 Term Glossary"), unsafe_allow_html=True)
    st.caption("Every heading used in this report is explained here. Expand to read.")
    with st.expander("📖 Click to read the full glossary", expanded=False):
        terms = list(GLOSSARY.items())
        mid   = (len(terms) + 1) // 2
        c1, c2 = st.columns(2)
        for col, chunk in [(c1, terms[:mid]), (c2, terms[mid:])]:
            with col:
                for term, defn in chunk:
                    st.markdown(
                        f'<span class="gloss-term">{term}</span><br>'
                        f'<span class="gloss-def">{defn}</span>',
                        unsafe_allow_html=True,
                    )
                    st.markdown("---")


# ── PubChem block ─────────────────────────────────────────────────────────────
def render_pubchem(pc):
    if not pc.get("found"):
        st.warning(f"⚠️ {pc.get('error', 'Compound not found in PubChem database.')}")
        return

    cid  = pc["cid"]
    name = pc.get("common_name") or pc.get("iupac_name") or f"CID {cid}"

    img_col, id_col = st.columns([1, 3])
    with img_col:
        st.image(pc["links"]["structure_image"], use_container_width=True,
                 caption=f"CID {cid}")
    with id_col:
        st.markdown(f'<div class="pubchem-name">{name}</div>', unsafe_allow_html=True)
        meta = (
            f'Formula: <b>{pc.get("formula","")}</b> &nbsp;|&nbsp; '
            f'MW: <b>{pc.get("molecular_weight","")} Da</b> &nbsp;|&nbsp; '
            f'Charge: <b>{pc.get("charge","")}</b>'
        )
        if pc.get("xlogp") is not None:
            meta += f' &nbsp;|&nbsp; XLogP: <b>{pc["xlogp"]}</b>'
        if pc.get("exact_mass"):
            meta += f' &nbsp;|&nbsp; Exact mass: <b>{pc["exact_mass"]} Da</b>'
        st.markdown(f'<div class="pubchem-meta">{meta}</div>', unsafe_allow_html=True)
        st.markdown(
            f'🔗 **[Open on PubChem]({pc["links"]["compound_page"]})**'
            f' &nbsp;|&nbsp; [SDF]({pc["links"]["sdf_download"]})'
            f' &nbsp;|&nbsp; [JSON]({pc["links"]["json_api"]})'
        )
        if pc.get("synonyms"):
            pills = "".join(f'<span class="info-pill">{s}</span>'
                            for s in pc["synonyms"][:8])
            st.markdown(f"**Known names:** {pills}", unsafe_allow_html=True)

    st.markdown("---")

    with st.expander("🔑 Chemical Identifiers", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**CID:** `{cid}`")
            st.markdown("**IUPAC name:**"); st.code(pc.get("iupac_name",""), language=None)
            st.markdown("**Canonical SMILES:**"); st.code(pc.get("canonical_smiles",""), language=None)
        with c2:
            st.markdown(f"**InChIKey:** `{pc.get('inchikey','')}`")
            if pc.get("inchi"):
                st.markdown("**InChI:**"); st.code(pc["inchi"], language=None)


# ── Detail sections ───────────────────────────────────────────────────────────
def render_detail(r):
    props = r["properties"]

    with st.expander("⚗️ Physicochemical Properties", expanded=False):
        ca, cb = st.columns(2)
        with ca:
            st.markdown(sub("Lipinski Rule-of-Five"), unsafe_allow_html=True)
            for pname, pval, plim in [
                ("Molecular Weight", props["molecular_weight"], 500),
                ("LogP",             props["logp"],             5),
                ("H-bond Donors",    props["hbd"],              5),
                ("H-bond Acceptors", props["hba"],              10),
            ]:
                st.markdown(f"{'✅' if pval<=plim else '❌'} **{pname}:** `{pval:.2f}` *(≤ {plim})*")
        with cb:
            st.markdown(sub("Extended"), unsafe_allow_html=True)
            st.markdown(f"{'✅' if props['tpsa']<=140 else '❌'} **TPSA:** `{props['tpsa']:.1f} Å²` *(≤ 140)*")
            st.markdown(f"{'✅' if props['rotatable_bonds']<=10 else '❌'} **Rot. Bonds:** `{props['rotatable_bonds']}` *(≤ 10)*")
        if r["violations"]:
            st.warning(f"⚠️ **{len(r['violations'])} violation(s)**")
            for v in r["violations"]:
                st.markdown(f"- **{v['property']}** = {v['value']:.2f} (limit {v['limit']}) — {v['issue']}")

    with st.expander("🔬 Metabolic Stability", expanded=False):
        ms, rm, gs, mb, ss = (
            r["microsomal_stability"], r["reactive_metabolites"],
            r["gsh_trapping"],        r["mbi_risk"],
            r["metabolic_soft_spots"],
        )
        mc1, mc2 = st.columns(2)
        with mc1:
            cls  = ms.get("stability_class","N/A")
            icon = {"HIGH":"🟢","MODERATE":"🟡","LOW":"🔴"}.get(cls,"⚪")
            st.markdown(sub("Microsomal Stability"), unsafe_allow_html=True)
            st.markdown(f"{icon} **{cls}** — Score `{ms.get('microsomal_stability_score')}/100`, "
                        f"t½ `{ms.get('predicted_t_half')}`")
            st.markdown(sub("Metabolic Soft Spots"), unsafe_allow_html=True)
            if ss["soft_spots"]:
                for s in ss["soft_spots"]: st.markdown(f"- {s}")
                st.caption(f"Clearance: **{ss.get('clearance_prediction')}**")
            else:
                st.success("None detected.")
        with mc2:
            rm_risk = rm.get("reactive_metabolite_risk","N/A")
            st.markdown(sub("Reactive Metabolites"), unsafe_allow_html=True)
            st.markdown(f"{risk_icon(rm_risk)} **{rm_risk}** — {rm.get('alert_count',0)} alert(s)")
            for a in rm.get("reactive_metabolite_alerts",[]): st.markdown(f"- {a}")
            for a in rm.get("recommended_assays",[]): st.markdown(f"  ↳ {a}")
            st.markdown(sub("GSH Trapping"), unsafe_allow_html=True)
            gsh_r = gs.get("gsh_trapping_risk","N/A")
            st.markdown(f"{risk_icon(gsh_r)} **{gsh_r}**")
            for a in gs.get("gsh_trap_alerts",[]): st.markdown(f"- {a}")
            st.markdown(sub("MBI Risk"), unsafe_allow_html=True)
            mbi_r = mb.get("mbi_risk","N/A")
            st.markdown(f"{risk_icon(mbi_r)} **{mbi_r}**")
            for a in mb.get("mbi_alerts",[]): st.markdown(f"- {a}")

    with st.expander("☠️ Toxicophore Alerts", expanded=False):
        summary = r["toxicophore_summary"]
        overall = summary.get("overall_toxicophore_risk","N/A")
        oi = {"CRITICAL":"🔴","HIGH":"🟠","MODERATE":"🟡","CLEAN":"🟢"}.get(overall,"⚪")
        st.markdown(f"### {oi} Overall: **{overall}**")
        st.info(summary.get("filter_recommendation",""))
        bd = summary.get("breakdown",{})
        tc1, tc2, tc3, tc4, tc5 = st.columns(5)
        for col, (lbl, key) in zip(
            [tc1,tc2,tc3,tc4,tc5],
            [("PAINS","pains"),("Brenk","brenk"),("SureChEMBL","surechembl"),
             ("Ames","ames"),("Reactive","reactive")]
        ):
            cnt = bd.get(key,0)
            col.metric(lbl, f"{'🔴' if cnt>=3 else '🟠' if cnt>=1 else '🟢'} {cnt}")
        tc_a, tc_b, tc_c = st.columns(3)
        for col, key, lbl in [(tc_a,"pains","PAINS"),(tc_b,"brenk","Brenk"),(tc_c,"surechembl","SureChEMBL")]:
            with col:
                alerts = r[key].get(f"{key}_alerts",[])
                flag   = r[key].get(f"{key}_flag", r[key].get("regulatory_risk",""))
                st.markdown(f"**{lbl} — {flag}**")
                if alerts:
                    for a in alerts: st.markdown(f"- {a}")
                else:
                    st.success(f"No {lbl} alerts.")

    with st.expander("🧬 ADME Properties", expanded=False):
        ac1, ac2 = st.columns(2)
        with ac1:
            cyp  = r["cyp450"]
            herg = r["herg"]
            ames = r["mutagenicity"]
            st.markdown(sub("CYP450"), unsafe_allow_html=True)
            st.markdown(f"{risk_icon(cyp['cyp_risk'])} **{cyp['cyp_risk']}** ({cyp['isoforms_flagged_count']} isoform(s))")
            for iso, alerts in cyp.get("flagged_isoforms",{}).items():
                st.markdown(f"  *{iso}:* {len(alerts)} alert(s)")
            st.markdown(sub("hERG Cardiac Liability"), unsafe_allow_html=True)
            st.markdown(f"{risk_icon(herg['herg_risk'])} **{herg['herg_risk']}** (score {herg['herg_risk_score']})")
            st.markdown(sub("Ames Mutagenicity"), unsafe_allow_html=True)
            st.markdown(f"{risk_icon(ames['mutagenicity_flag'])} **{ames['mutagenicity_flag']}** — {ames['alert_count']} alert(s)")
            for a in ames.get("alerts",[]): st.markdown(f"- {a}")
        with ac2:
            ppb   = r["plasma_protein_binding"]
            caco2 = r["caco2_permeability"]
            pgp   = r["pgp_efflux"]
            st.markdown(sub("Plasma Protein Binding (PPB)"), unsafe_allow_html=True)
            st.markdown(f"{risk_icon(ppb['ppb_class'])} **{ppb['ppb_class']}** — ~{ppb['ppb_estimate_pct']}% bound")
            st.caption(ppb.get("ppb_note",""))
            st.markdown(sub("Caco-2 Permeability"), unsafe_allow_html=True)
            ci = {"HIGH":"🟢","MEDIUM":"🟡","LOW":"🔴"}.get(caco2["caco2_class"],"⚪")
            st.markdown(f"{ci} **{caco2['caco2_class']}** — {caco2['caco2_papp_nm_s']} nm/s")
            st.caption(caco2.get("caco2_note",""))
            st.markdown(sub("P-glycoprotein Efflux"), unsafe_allow_html=True)
            st.markdown(f"{risk_icon(pgp['pgp_risk'])} **{pgp['pgp_risk']}** (score {pgp['pgp_risk_score']})")
            for reason in pgp.get("pgp_risk_reasons",[]): st.markdown(f"- {reason}")

    with st.expander("💡 Optimisation Advice", expanded=False):
        adv = generate_optimization_advice(r)
        if not adv:
            st.success("✅ No optimisation needed — all filters passed.")
        else:
            for a in sorted(adv):
                st.markdown(f"{'⚠️' if a.startswith('PRIORITY') else '▸'} {a}")


# ── Panel display helper ──────────────────────────────────────────────────────
PANEL_META = {
    "overview":    ("📊 Overview",                  "Drug-likeness score and physicochemical radar chart"),
    "risk_matrix": ("🚦 ADME Risk Matrix",           "Full 14-row risk rating across all ADME endpoints"),
    "alerts":      ("🔎 Alerts & Stability",         "Structural alert counts + microsomal stability breakdown"),
    "cyp_tox":     ("⚠️ CYP450 & Toxicophores",      "CYP isoform liability and toxicophore alert distribution"),
    "properties":  ("📋 Property Table",             "Numerical properties vs. drug-likeness thresholds"),
    "advice":      ("💡 Optimisation Recommendations","Priority chemistry modifications suggested"),
}

def render_panels(result, key_prefix=""):
    panels = build_adme_panels(result)
    if not panels:
        st.error("Could not build charts.")
        return

    panel_order = ["overview", "risk_matrix", "alerts", "cyp_tox", "properties", "advice"]

    fig_combined = build_adme_figure(result)
    if fig_combined:
        combined_bytes = fig_to_bytes(fig_combined, dpi=300)
        plt.close(fig_combined)
        st.download_button(
            label="⬇️ Download combined report (300 DPI)",
            data=combined_bytes,
            file_name=f"adme_combined_{key_prefix or 'report'}.png",
            mime="image/png",
            key=f"dl_combined_{key_prefix}",
        )

    st.markdown("")
    for key in panel_order:
        fig = panels.get(key)
        if fig is None:
            continue
        title, caption = PANEL_META[key]
        st.markdown(sh(title), unsafe_allow_html=True)
        st.caption(caption)
        st.pyplot(fig, use_container_width=True)
        png_bytes = fig_to_bytes(fig, dpi=300)
        plt.close(fig)
        st.download_button(
            label=f"⬇️ Download — {title} (300 DPI)",
            data=png_bytes,
            file_name=f"adme_{key}_{key_prefix or 'report'}.png",
            mime="image/png",
            key=f"dl_{key}_{key_prefix}",
        )
        st.markdown("")



# ══════════════════════════════════════════════════════════════════════════════
# LITERATURE INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════

_LIT_CATS = {
    "Metabolism":   ("⚗️",  "#58A6FF"),
    "Toxicity":     ("☠️",  "#F85149"),
    "Resistance":   ("🛡️", "#D29922"),
    "Pharmacology": ("💊",  "#3FB950"),
    "Clinical":     ("🏥",  "#BC8CFF"),
    "Synthesis":    ("🔬",  "#F97316"),
    "General":      ("📄",  "#8B949E"),
}
_LIT_ORDER = ["Metabolism","Toxicity","Resistance","Pharmacology","Clinical","Synthesis","General"]


def _lit_paper_card(paper: dict, accent: str) -> None:
    parts = []
    if paper.get("year"):    parts.append(f"📅 {paper['year']}")
    if paper.get("journal"): parts.append(f"📰 {paper['journal']}")
    if paper.get("doi"):     parts.append(f"DOI: {paper['doi']}")
    meta = "  ·  ".join(parts)
    url  = paper.get("url", "")
    link = (f'<a href="{url}" target="_blank" style="color:{accent};'
            f'font-size:0.79rem;text-decoration:none;">🔗 View</a>') if url else ""
    abs_html = (f'<div class="lit-abs">"{paper["abstract"]}"</div>'
                if paper.get("abstract") else "")
    st.markdown(
        f'<div class="lit-card">'
        f'<div class="lit-title">{paper.get("title","Untitled")}</div>'
        f'<div class="lit-meta">{meta} &nbsp; {link}</div>'
        + abs_html + "</div>",
        unsafe_allow_html=True,
    )


def render_literature_intelligence(pubchem_metadata: dict, key_prefix: str = "") -> None:
    st.markdown(
        '<div class="lit-header">📚 Literature Intelligence</div>'
        '<div class="lit-sub">Recent papers retrieved automatically '
        '(PubMed → Semantic Scholar → OpenAlex → CrossRef) '
        '— grouped by research area.</div>',
        unsafe_allow_html=True,
    )

    pc_name = ""
    if pubchem_metadata and pubchem_metadata.get("found"):
        pc_name = (pubchem_metadata.get("common_name") or
                   pubchem_metadata.get("iupac_name") or "").strip()

    c1, c2 = st.columns([4, 1])
    with c1:
        query = st.text_input(
            "lit_input", value=pc_name,
            placeholder="e.g. lidocaine, aspirin, ibuprofen",
            key=f"lit_q_{key_prefix}", label_visibility="collapsed",
        )
    with c2:
        clicked = st.button("🔍 Search", key=f"lit_btn_{key_prefix}",
                            type="primary", use_container_width=True)

    if not pc_name and not clicked:
        st.caption("💡 Enable Fetch PubChem in sidebar for auto-fill, or type a name and press Search.")

    auto_k = f"lit_done_{key_prefix}"
    if clicked:
        st.session_state[auto_k] = False
        should = True
    elif pc_name and not st.session_state.get(auto_k, False):
        should = True
    else:
        should = False

    if not should or not query.strip():
        return

    if not clicked:
        st.session_state[auto_k] = True

    with st.spinner(f"Searching for **{query.strip()}** across PubMed, Semantic Scholar, OpenAlex…"):
        res = get_literature_intelligence(query.strip())

    if res.get("error"):
        st.error(f"⚠️ {res['error']}")
        return

    if res["total"] == 0:
        reason = res.get("_empty_reason", "")
        st.markdown(
            '<div class="lit-empty">📭 No papers found for this compound.'
            + (f"<br><small style='color:#8B949E'>{reason}</small>" if reason else "")
            + "</div>",
            unsafe_allow_html=True,
        )
        return

    cats  = res["categories"]
    total = res["total"]
    src   = res.get("source", "")

    pills = f'<span class="lit-pill">🗂️ {total} paper{"s" if total!=1 else ""}</span>'
    for cat, papers in cats.items():
        em, _ = _LIT_CATS.get(cat, ("📄", "#8B949E"))
        pills += f'<span class="lit-pill">{em} {cat}: {len(papers)}</span>'
    if src and src != "none":
        pills += f'<span class="lit-source" style="margin-left:8px;">via {src}</span>'
    st.markdown(pills, unsafe_allow_html=True)
    st.markdown("")

    ordered = [c for c in _LIT_ORDER if c in cats] + [c for c in cats if c not in _LIT_ORDER]
    for cat in ordered:
        papers = cats.get(cat, [])
        if not papers:
            continue
        em, ac = _LIT_CATS.get(cat, ("📄", "#8B949E"))
        n = len(papers)
        st.markdown(
            f'<div class="lit-cat" style="--ac:{ac};">{em} {cat} '
            f'<span style="font-size:0.8rem;font-weight:400;color:#8B949E;">'
            f'({n} paper{"s" if n!=1 else ""})</span></div>',
            unsafe_allow_html=True,
        )
        for p in papers:
            _lit_paper_card(p, ac)




# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧬 ADME Analyzer")
    st.markdown("---")
    mode = st.radio("Mode", ["Single molecule", "Batch (multiple SMILES)"])

    fetch_pubchem = st.toggle(
        "🔗 Fetch PubChem data",
        value=True,
        help="Queries PubChem for compound name, structure, identifiers and synonyms (~3–5 s). "
             "Turn OFF for instant results.",
    )
    if fetch_pubchem:
        st.caption("~3–5 s per molecule (name + structure only).")
    else:
        st.info("⚡ Fast mode — PubChem section skipped.")

    st.markdown("---")
    st.markdown("**Demo molecules**")
    demos = {
        "Aspirin":      "CC(=O)Oc1ccccc1C(=O)O",
        "CCl4":         "ClC(Cl)(Cl)Cl",
        "Ibuprofen":    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "Atorvastatin": "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CCC(O)CC(O)CC(=O)O",
        "Caffeine":     "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
        "Morphine":     "OC1=CC=C2CC3N(CCC34CCc5c4cc(O)c(c5)OC)C2=C1",
    }
    selected_demo = st.selectbox("Load a demo", ["— select —"] + list(demos.keys()))
    st.markdown("---")
    st.caption("Rule-based screening.\nNot a substitute for experimental data.")


# ── Title + Glossary (always shown) ──────────────────────────────────────────
st.title("🧬 ADME Analysis Dashboard")
st.markdown(
    "Screen molecules for drug-likeness, metabolic stability, toxicophore alerts, "
    "and ADME properties — enriched with live PubChem data."
)
render_glossary()

# ══════════════════════════════════════════════════════════════════
# SINGLE MOLECULE
# ══════════════════════════════════════════════════════════════════
if mode == "Single molecule":

    default_smiles = demos[selected_demo] if selected_demo != "— select —" else ""
    smiles_input = st.text_input(
        "SMILES string",
        value=default_smiles,
        placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O",
    )
    if fetch_pubchem:
        st.caption("⏱️ With PubChem: ~5–10 s  |  Toggle OFF in sidebar for < 1 s")

    if st.button("Analyse", type="primary", use_container_width=True) and smiles_input.strip():
        _t0 = time.time()
        _st = st.empty()
        _st.info("🔬 Running ADME analysis…")
        _res = cached_analyze(smiles_input.strip(), fetch_pubchem)
        _st.success(f"✅ Done in {time.time()-_t0:.1f} s")
        if not _res["valid"]:
            st.error("❌ Invalid SMILES string.")
            st.stop()
        st.session_state["single_result"] = _res
        st.session_state["lit_done_single"] = False

    # ── Render results ───────────────────────────────────────────────────────
    result = st.session_state.get("single_result")
    if result and result.get("valid"):

        # KPI
        st.markdown("---")
        k1, k2, k3, k4, k5 = st.columns(5)
        d_icon = "🟢" if result["decision"]=="ACCEPT" else "🔴"
        k1.metric("Score",       f"{result['score']} / 100")
        k2.metric("Status",      result["physchem_status"])
        k3.metric("Decision",    f"{d_icon} {result['decision']}")
        k4.metric("Lipinski",    "✅ PASS" if result["lipinski"]["passes"] else "❌ FAIL")
        k5.metric("Tox Alerts",  result["toxicophore_summary"]["total_toxicophore_alerts"])

        # Charts
        st.markdown(sh("📊 ADME Charts"), unsafe_allow_html=True)
        render_panels(result, key_prefix="single")

        # PubChem
        st.markdown(sh("🔗 PubChem Information"), unsafe_allow_html=True)
        render_pubchem(result.get("pubchem_metadata", {}))

        # Detail
        st.markdown(sh("🔬 Detailed Analysis"), unsafe_allow_html=True)
        st.caption("Click any section to expand.")
        render_detail(result)

        # Literature Intelligence
        st.markdown("---")
        render_literature_intelligence(
            pubchem_metadata=result.get("pubchem_metadata", {}),
            key_prefix="single",
        )


# ══════════════════════════════════════════════════════════════════
# BATCH MODE
# ══════════════════════════════════════════════════════════════════
else:
    st.markdown("Enter one SMILES per line:")
    batch_input = st.text_area(
        "SMILES list", height=180,
        placeholder="CC(=O)Oc1ccccc1C(=O)O\nClC(Cl)(Cl)Cl\nCC(C)Cc1ccc(cc1)C(C)C(=O)O",
    )
    if fetch_pubchem:
        st.caption("⏱️ PubChem ON: ~5–10 s per molecule. Toggle OFF in sidebar for instant results.")
    else:
        st.caption("⚡ PubChem OFF: < 1 s per molecule.")

    if st.button("Analyse all", type="primary", use_container_width=True) and batch_input.strip():
        smiles_list = [s.strip() for s in batch_input.strip().splitlines() if s.strip()]
        n = len(smiles_list)

        est = n * (7 if fetch_pubchem else 1)
        est_str = f"{est//60} min {est%60} s" if est >= 60 else f"~{est} s"
        st.info(f"⏱️ Analysing **{n} molecule(s)** — estimated time: **{est_str}**")

        progress_bar = st.progress(0)
        status_text  = st.empty()
        results      = []
        t_start      = time.time()

        for idx, smi in enumerate(smiles_list):
            status_text.markdown(f"🔬 Molecule **{idx+1}/{n}**: `{smi[:50]}`")
            r = analyze_smiles(smi.strip(), fetch_pubchem=fetch_pubchem)
            results.append(r)
            elapsed = time.time() - t_start
            if idx > 0:
                rem = (elapsed/(idx+1)) * (n-idx-1)
                rem_str = f"{int(rem//60)} min {int(rem%60)} s" if rem>=60 else f"{int(rem)} s"
                status_text.markdown(f"✅ **{idx+1}/{n}** done  |  ⏳ ~{rem_str} remaining")
            progress_bar.progress((idx+1)/n)

        total = time.time() - t_start
        progress_bar.progress(1.0)
        status_text.success(f"✅ All {n} done in {total:.1f} s ({total/n:.1f} s/molecule avg)")

        valid   = [r for r in results if r["valid"]]
        invalid = [r for r in results if not r["valid"]]
        st.markdown(f"**{len(valid)} valid** | **{len(invalid)} invalid**")
        if invalid:
            with st.expander("❌ Invalid SMILES"):
                for r in invalid: st.markdown(f"- `{r['smiles']}`")
        if not valid: st.stop()

        # Summary table
        st.markdown("---")
        st.markdown(sh("📋 Results Summary"), unsafe_allow_html=True)
        import pandas as pd
        rows = []
        for r in valid:
            pc      = r.get("pubchem_metadata",{})
            pc_name = (pc.get("common_name") or pc.get("iupac_name","")) if pc.get("found") else ""
            rows.append({
                "Name":            pc_name[:30] if pc_name else r["smiles"][:30],
                "Score":           r["score"],
                "Decision":        r["decision"],
                "Lipinski":        "PASS" if r["lipinski"]["passes"] else "FAIL",
                "Reactive Met.":   r["reactive_metabolites"]["reactive_metabolite_risk"],
                "Tox Alerts":      r["toxicophore_summary"]["total_toxicophore_alerts"],
                "hERG":            r["herg"]["herg_risk"],
                "CYP Risk":        r["cyp450"]["cyp_risk"],
                "Microsomal":      r["microsomal_stability"]["stability_class"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # Per-molecule reports
        st.markdown("---")
        st.markdown(sh("📊 Individual Reports"), unsafe_allow_html=True)
        for i, r in enumerate(valid):
            pc   = r.get("pubchem_metadata",{})
            name = (pc.get("common_name") or pc.get("iupac_name","")) if pc.get("found") else ""
            label = f"Molecule {i+1}"
            if name: label += f" — {name}"
            label += f":  `{r['smiles'][:50]}`"

            with st.expander(label, expanded=False):
                d_icon = "🟢" if r["decision"]=="ACCEPT" else "🔴"
                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric("Score",     f"{r['score']} / 100")
                sc2.metric("Decision",  f"{d_icon} {r['decision']}")
                sc3.metric("Lipinski",  "✅ PASS" if r["lipinski"]["passes"] else "❌ FAIL")
                sc4.metric("Tox Alerts",r["toxicophore_summary"]["total_toxicophore_alerts"])
                st.markdown("---")

                st.markdown(sub("📊 ADME Charts"), unsafe_allow_html=True)
                render_panels(r, key_prefix=f"mol{i}")

                st.markdown("---")
                st.markdown(sub("🔗 PubChem Information"), unsafe_allow_html=True)
                render_pubchem(pc)

                st.markdown("---")
                st.markdown(sub("🔬 Detailed Analysis"), unsafe_allow_html=True)
                render_detail(r)

                # Literature Intelligence
                st.markdown("---")
                render_literature_intelligence(
                    pubchem_metadata=r.get("pubchem_metadata", {}),
                    key_prefix=f"mol{i}",
                )