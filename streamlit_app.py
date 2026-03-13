"""
streamlit_app.py — ADME Analysis Dashboard
Run with:  streamlit run streamlit_app.py
Requires adme_analysis.py in the same folder.
"""

import io
import time
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from adme_analysis import (
    analyze_smiles,
    build_adme_figure,
    generate_optimization_advice,
)

# ── Page config ───────────────────────────────────────────────────────────────
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
    .section-heading {
        font-size: 1.25rem; font-weight: 700; color: #58A6FF;
        margin: 1.1rem 0 0.4rem 0;
        border-bottom: 1px solid #30363D; padding-bottom: 4px;
    }
    .sub-heading {
        font-size: 1.05rem; font-weight: 600; color: #E6EDF3;
        margin: 0.8rem 0 0.3rem 0;
    }
    .pubchem-name    { font-size: 1.6rem; font-weight: 800; color: #E6EDF3; margin-bottom: 4px; }
    .pubchem-formula { font-size: 1.05rem; color: #8B949E; margin-bottom: 10px; }
    .info-pill {
        display: inline-block; background: #21262D;
        border: 1px solid #30363D; border-radius: 20px;
        padding: 2px 12px; margin: 3px 4px; font-size: 0.85rem; color: #C9D1D9;
    }
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
    return {"HIGH":"🟠","CRITICAL":"🔴","MODERATE":"🟡",
            "LOW":"🟢","CLEAN":"🟢","POSITIVE":"🔴",
            "NEGATIVE":"🟢","FLAGGED":"🟠"}.get(str(val).upper(),"⚪")

def sh(text):
    return f'<p class="section-heading">{text}</p>'

def subh(text):
    return f'<p class="sub-heading">{text}</p>'


# ── Cached single-molecule analysis ───────────────────────────────────────────
@st.cache_data(show_spinner=False)
def cached_analyze(smiles, fetch_pubchem):
    return analyze_smiles(smiles, fetch_pubchem=fetch_pubchem)


# ── PubChem block — always fully visible, never hidden in a tab ───────────────
def render_pubchem(pc):
    if not pc.get("found"):
        st.warning(f"⚠️ {pc.get('error', 'Compound not found in PubChem database.')}")
        return

    cid  = pc["cid"]
    name = pc.get("common_name") or pc.get("iupac_name") or f"CID {cid}"

    img_col, id_col = st.columns([1, 3])
    with img_col:
        st.image(pc["links"]["structure_image"], use_container_width=True,
                 caption=f"PubChem CID {cid}")
    with id_col:
        st.markdown(f'<div class="pubchem-name">{name}</div>', unsafe_allow_html=True)
        formula_parts = (
            f'Formula: <b>{pc["formula"]}</b> &nbsp;|&nbsp; '
            f'MW: <b>{pc["molecular_weight"]} Da</b> &nbsp;|&nbsp; '
            f'Charge: <b>{pc["charge"]}</b>'
        )
        if pc.get("xlogp") is not None:
            formula_parts += f' &nbsp;|&nbsp; XLogP: <b>{pc["xlogp"]}</b>'
        if pc.get("exact_mass"):
            formula_parts += f' &nbsp;|&nbsp; Exact mass: <b>{pc["exact_mass"]} Da</b>'
        st.markdown(f'<div class="pubchem-formula">{formula_parts}</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f'🔗 **[Open on PubChem]({pc["links"]["compound_page"]})**'
            f' &nbsp;|&nbsp; [Download SDF]({pc["links"]["sdf_download"]})'
            f' &nbsp;|&nbsp; [JSON API]({pc["links"]["json_api"]})'
        )
        if pc.get("synonyms"):
            pills = "".join(f'<span class="info-pill">{s}</span>'
                            for s in pc["synonyms"][:8])
            st.markdown(f"**Synonyms:** {pills}", unsafe_allow_html=True)

    st.markdown("---")

    with st.expander("🔑 Chemical Identifiers", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**CID:** `{cid}`")
            st.markdown("**IUPAC name:**")
            st.code(pc.get("iupac_name",""), language=None)
            st.markdown("**Canonical SMILES:**")
            st.code(pc.get("canonical_smiles",""), language=None)
        with c2:
            st.markdown(f"**InChIKey:** `{pc.get('inchikey','')}`")
            if pc.get("inchi"):
                st.markdown("**InChI:**")
                st.code(pc["inchi"], language=None)

    if pc.get("pharmacology"):
        with st.expander("💊 Pharmacology & Biochemistry", expanded=True):
            st.markdown(pc["pharmacology"])

    if pc.get("drug_info"):
        with st.expander("🏥 Drug & Medication Information", expanded=True):
            for heading, items in pc["drug_info"].items():
                st.markdown(f"**{heading}**")
                for item in items:
                    st.markdown(f"- {item}")

    ba = pc.get("bioactivity", {})
    with st.expander(f"🧪 Bioactivity Summary ({ba.get('assay_count',0)} assays)", expanded=True):
        b1, b2, b3 = st.columns(3)
        b1.metric("Assays tested",     ba.get("assay_count", 0))
        b2.metric("Active outcomes",   ba.get("active_count", 0))
        b3.metric("Inactive outcomes", ba.get("inactive_count", 0))
        if ba.get("note"):
            st.caption(ba["note"])

    targets = pc.get("targets", [])
    with st.expander(f"🎯 Protein / Enzyme Targets ({len(targets)} found)",
                     expanded=bool(targets)):
        if targets:
            for t in targets: st.markdown(f"- {t}")
        else:
            st.caption("No target data retrieved from PubChem BioAssays.")

    if pc.get("pathways"):
        with st.expander(f"🔄 Biological Pathways ({len(pc['pathways'])} found)"):
            for pw in pc["pathways"]: st.markdown(f"- {pw}")

    if pc.get("toxicity"):
        with st.expander(f"⚠️ Safety & Toxicity ({len(pc['toxicity'])} entries)"):
            for tox in pc["toxicity"]: st.markdown(f"- {tox}")

    if pc.get("literature"):
        with st.expander(f"📚 PubMed References ({len(pc['literature'])} found)"):
            for ref in pc["literature"]: st.markdown(f"- {ref}")


# ── Detail sections — all collapsible expanders ───────────────────────────────
def render_detail(r):
    props = r["properties"]

    with st.expander("⚗️ Physicochemical Properties", expanded=False):
        ca, cb = st.columns(2)
        with ca:
            st.markdown(subh("Lipinski Rule-of-Five"), unsafe_allow_html=True)
            for pname, pval, plim in [
                ("Molecular Weight", props["molecular_weight"], 500),
                ("LogP",             props["logp"],             5),
                ("H-bond Donors",    props["hbd"],              5),
                ("H-bond Acceptors", props["hba"],              10),
            ]:
                icon = "✅" if pval <= plim else "❌"
                st.markdown(f"{icon} **{pname}:** `{pval:.2f}` *(limit ≤ {plim})*")
        with cb:
            st.markdown(subh("Extended Rules"), unsafe_allow_html=True)
            st.markdown(f"{'✅' if props['tpsa']<=140 else '❌'} **TPSA:** "
                        f"`{props['tpsa']:.1f} Å²` *(≤ 140)*")
            st.markdown(f"{'✅' if props['rotatable_bonds']<=10 else '❌'} **Rotatable Bonds:** "
                        f"`{props['rotatable_bonds']}` *(≤ 10)*")
        if r["violations"]:
            st.warning(f"⚠️ **{len(r['violations'])} violation(s)**")
            for v in r["violations"]:
                st.markdown(f"- **{v['property']}** = {v['value']:.2f} "
                            f"(limit {v['limit']}) — {v['issue']}")

    with st.expander("🔬 Metabolic Stability", expanded=False):
        ms = r["microsomal_stability"]
        rm = r["reactive_metabolites"]
        gs = r["gsh_trapping"]
        mb = r["mbi_risk"]
        ss = r["metabolic_soft_spots"]
        mc1, mc2 = st.columns(2)
        with mc1:
            cls  = ms.get("stability_class","N/A")
            icon = {"HIGH":"🟢","MODERATE":"🟡","LOW":"🔴"}.get(cls,"⚪")
            st.markdown(subh("Microsomal Stability"), unsafe_allow_html=True)
            st.markdown(f"{icon} **{cls}** — Score `{ms.get('microsomal_stability_score')}/100`  "
                        f"t½ `{ms.get('predicted_t_half')}`")
            st.markdown(subh("Soft Spots"), unsafe_allow_html=True)
            if ss["soft_spots"]:
                for s in ss["soft_spots"]: st.markdown(f"- {s}")
                st.caption(f"Clearance: **{ss.get('clearance_prediction')}**")
            else:
                st.success("No metabolic soft spots detected.")
        with mc2:
            rm_risk = rm.get("reactive_metabolite_risk","N/A")
            st.markdown(subh("Reactive Metabolites"), unsafe_allow_html=True)
            st.markdown(f"{risk_icon(rm_risk)} **{rm_risk}** — {rm.get('alert_count',0)} alert(s)")
            if rm["reactive_metabolite_alerts"]:
                for a in rm["reactive_metabolite_alerts"]: st.markdown(f"- {a}")
            if rm.get("recommended_assays"):
                for a in rm["recommended_assays"]: st.markdown(f"  - {a}")
            gsh_r = gs.get("gsh_trapping_risk","N/A")
            st.markdown(subh("GSH Trapping"), unsafe_allow_html=True)
            st.markdown(f"{risk_icon(gsh_r)} **{gsh_r}**")
            if gs.get("gsh_trap_alerts"):
                for a in gs["gsh_trap_alerts"]: st.markdown(f"- {a}")
            mbi_r = mb.get("mbi_risk","N/A")
            st.markdown(subh("MBI Risk"), unsafe_allow_html=True)
            st.markdown(f"{risk_icon(mbi_r)} **{mbi_r}**")
            if mb.get("mbi_alerts"):
                for a in mb["mbi_alerts"]: st.markdown(f"- {a}")

    with st.expander("☠️ Toxicophore Alerts", expanded=False):
        summary = r["toxicophore_summary"]
        overall = summary.get("overall_toxicophore_risk","N/A")
        oi = {"CRITICAL":"🔴","HIGH":"🟠","MODERATE":"🟡","CLEAN":"🟢"}.get(overall,"⚪")
        st.markdown(f"### {oi} Overall Risk: **{overall}**")
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
        for col, key, lbl in [
            (tc_a,"pains","PAINS"),(tc_b,"brenk","Brenk"),(tc_c,"surechembl","SureChEMBL")
        ]:
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
            cyp = r["cyp450"]
            st.markdown(subh("CYP450"), unsafe_allow_html=True)
            st.markdown(f"{risk_icon(cyp['cyp_risk'])} **{cyp['cyp_risk']}** "
                        f"({cyp['isoforms_flagged_count']} isoform(s))")
            if cyp["flagged_isoforms"]:
                for iso, alerts in cyp["flagged_isoforms"].items():
                    st.markdown(f"  *{iso}:* {len(alerts)} alert(s)")
            herg = r["herg"]
            st.markdown(subh("hERG"), unsafe_allow_html=True)
            st.markdown(f"{risk_icon(herg['herg_risk'])} **{herg['herg_risk']}** "
                        f"(score {herg['herg_risk_score']})")
            ames = r["mutagenicity"]
            st.markdown(subh("Mutagenicity (Ames)"), unsafe_allow_html=True)
            st.markdown(f"{risk_icon(ames['mutagenicity_flag'])} **{ames['mutagenicity_flag']}** "
                        f"— {ames['alert_count']} alert(s)")
            if ames["alerts"]:
                for a in ames["alerts"]: st.markdown(f"- {a}")
        with ac2:
            ppb = r["plasma_protein_binding"]
            st.markdown(subh("Plasma Protein Binding"), unsafe_allow_html=True)
            st.markdown(f"{risk_icon(ppb['ppb_class'])} **{ppb['ppb_class']}** "
                        f"~{ppb['ppb_estimate_pct']}% bound")
            st.caption(ppb["ppb_note"])
            caco2 = r["caco2_permeability"]
            st.markdown(subh("Caco-2 Permeability"), unsafe_allow_html=True)
            ci = {"HIGH":"🟢","MEDIUM":"🟡","LOW":"🔴"}.get(caco2["caco2_class"],"⚪")
            st.markdown(f"{ci} **{caco2['caco2_class']}** — {caco2['caco2_papp_nm_s']} nm/s")
            st.caption(caco2["caco2_note"])
            pgp = r["pgp_efflux"]
            st.markdown(subh("P-glycoprotein Efflux"), unsafe_allow_html=True)
            st.markdown(f"{risk_icon(pgp['pgp_risk'])} **{pgp['pgp_risk']}** "
                        f"(score {pgp['pgp_risk_score']})")
            if pgp["pgp_risk_reasons"]:
                for reason in pgp["pgp_risk_reasons"]: st.markdown(f"- {reason}")

    with st.expander("💡 Optimisation Advice", expanded=False):
        adv = generate_optimization_advice(r)
        if not adv:
            st.success("✅ No optimisation needed — all filters passed.")
        else:
            for a in sorted(adv):
                st.markdown(f"{'⚠️' if a.startswith('PRIORITY') else '▸'} {a}")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧬 ADME Analyzer")
    st.markdown("---")
    mode = st.radio("Mode", ["Single molecule", "Batch (multiple SMILES)"])

    fetch_pubchem = st.toggle(
        "🔗 Fetch PubChem data",
        value=True,
        help=(
            "Queries the PubChem API for compound names, bioactivity, targets, etc. "
            "Each molecule takes ~5–15 s. Turn OFF for instant results."
        ),
    )
    if fetch_pubchem:
        st.caption("~5–15 s per molecule for PubChem lookup.")
    else:
        st.info("⚡ Fast mode — PubChem section will show 'skipped'.")

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
    st.caption("Rule-based ADME screening.\nNot a substitute for experimental data.")


# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🧬 ADME Analysis Dashboard")
st.markdown(
    "Screen molecules for drug-likeness, metabolic stability, toxicophore alerts, "
    "and ADME properties — enriched with live PubChem data."
)

# ═══════════════════════════════════════════════════════════════════
# SINGLE MOLECULE
# ═══════════════════════════════════════════════════════════════════
if mode == "Single molecule":

    default_smiles = demos[selected_demo] if selected_demo != "— select —" else ""
    smiles_input = st.text_input(
        "SMILES string",
        value=default_smiles,
        placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O",
    )
    if fetch_pubchem:
        st.caption("⏱️ With PubChem: ~10–20 s  |  Toggle OFF for < 1 s")

    analyse_btn = st.button("Analyse", type="primary", use_container_width=True)

    if analyse_btn and smiles_input.strip():
        t0       = time.time()
        status   = st.empty()
        status.info("🔬 Running ADME analysis…")
        result   = cached_analyze(smiles_input.strip(), fetch_pubchem)
        elapsed  = time.time() - t0
        status.success(f"✅ Done in {elapsed:.1f} s")

        if not result["valid"]:
            st.error("❌ Invalid SMILES string — could not parse molecule.")
            st.stop()

        # KPI row
        st.markdown("---")
        k1, k2, k3, k4, k5 = st.columns(5)
        d_icon = "🟢" if result["decision"] == "ACCEPT" else "🔴"
        k1.metric("Score",    f"{result['score']} / 100")
        k2.metric("Status",   result["physchem_status"])
        k3.metric("Decision", f"{d_icon} {result['decision']}")
        k4.metric("Lipinski", "✅ PASS" if result["lipinski"]["passes"] else "❌ FAIL")
        k5.metric("Tox Alerts", result["toxicophore_summary"]["total_toxicophore_alerts"])

        # Chart
        st.markdown(sh("📊 Full ADME Report"), unsafe_allow_html=True)
        fig = build_adme_figure(result)
        if fig:
            st.pyplot(fig, use_container_width=True)
            png_bytes = fig_to_bytes(fig, dpi=300)
            plt.close(fig)
            st.download_button(
                label="⬇️  Download high-res PNG (300 DPI)",
                data=png_bytes,
                file_name=f"adme_{smiles_input[:20].replace('/','_')}.png",
                mime="image/png",
            )

        # PubChem — always shown inline
        st.markdown(sh("🔗 PubChem Information"), unsafe_allow_html=True)
        render_pubchem(result.get("pubchem_metadata", {}))

        # Collapsible detail sections
        st.markdown(sh("🔬 Detailed Analysis"), unsafe_allow_html=True)
        st.caption("Click any section below to expand.")
        render_detail(result)


# ═══════════════════════════════════════════════════════════════════
# BATCH MODE
# ═══════════════════════════════════════════════════════════════════
else:
    st.markdown("Enter one SMILES per line:")
    batch_input = st.text_area(
        "SMILES list",
        height=180,
        placeholder="CC(=O)Oc1ccccc1C(=O)O\nClC(Cl)(Cl)Cl\nCC(C)Cc1ccc(cc1)C(C)C(=O)O",
    )

    if fetch_pubchem:
        st.caption("⏱️ PubChem ON: ~10–20 s per molecule. Toggle OFF for instant results.")
    else:
        st.caption("⚡ PubChem OFF: < 1 s per molecule.")

    run_batch = st.button("Analyse all", type="primary", use_container_width=True)

    if run_batch and batch_input.strip():
        smiles_list = [s.strip() for s in batch_input.strip().splitlines() if s.strip()]
        n = len(smiles_list)

        est_secs = n * (12 if fetch_pubchem else 1)
        est_str  = (f"{est_secs // 60} min {est_secs % 60} s"
                    if est_secs >= 60 else f"~{est_secs} s")
        st.info(f"⏱️ Analysing **{n} molecule(s)** — estimated time: **{est_str}**")

        progress_bar = st.progress(0)
        status_text  = st.empty()
        results = []
        t_start = time.time()

        for idx, smi in enumerate(smiles_list):
            status_text.markdown(f"🔬 Molecule **{idx+1} / {n}**: `{smi[:50]}`")
            r = analyze_smiles(smi.strip(), fetch_pubchem=fetch_pubchem)
            results.append(r)
            elapsed = time.time() - t_start
            if idx > 0:
                remaining = (elapsed / (idx+1)) * (n - idx - 1)
                rem_str   = (f"{int(remaining//60)} min {int(remaining%60)} s"
                             if remaining >= 60 else f"{int(remaining)} s")
                status_text.markdown(
                    f"✅ **{idx+1} / {n}** done  &nbsp;|&nbsp;  ⏳ ~{rem_str} remaining"
                )
            progress_bar.progress((idx+1) / n)

        total_elapsed = time.time() - t_start
        progress_bar.progress(1.0)
        status_text.success(
            f"✅ All {n} molecule(s) done in {total_elapsed:.1f} s "
            f"({total_elapsed/n:.1f} s/molecule avg)"
        )

        valid   = [r for r in results if r["valid"]]
        invalid = [r for r in results if not r["valid"]]

        st.markdown(f"**{len(valid)} valid** | **{len(invalid)} invalid**")
        if invalid:
            with st.expander("❌ Invalid SMILES"):
                for r in invalid:
                    st.markdown(f"- `{r['smiles']}`")

        if not valid:
            st.stop()

        # Summary table
        st.markdown("---")
        st.markdown(sh("📋 Results Summary"), unsafe_allow_html=True)
        import pandas as pd
        rows = []
        for r in valid:
            pc      = r.get("pubchem_metadata", {})
            pc_name = (pc.get("common_name") or pc.get("iupac_name","")
                       if pc.get("found") else "")
            rows.append({
                "Name":             pc_name[:30] if pc_name else r["smiles"][:30],
                "Score":            r["score"],
                "Status":           r["physchem_status"],
                "Decision":         r["decision"],
                "Lipinski":         "PASS" if r["lipinski"]["passes"] else "FAIL",
                "Reactive Met.":    r["reactive_metabolites"]["reactive_metabolite_risk"],
                "Tox Alerts":       r["toxicophore_summary"]["total_toxicophore_alerts"],
                "hERG":             r["herg"]["herg_risk"],
                "CYP Risk":         r["cyp450"]["cyp_risk"],
                "Microsomal Stab.": r["microsomal_stability"]["stability_class"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # Per-molecule full reports
        st.markdown("---")
        st.markdown(sh("📊 Individual Reports"), unsafe_allow_html=True)

        for i, r in enumerate(valid):
            pc   = r.get("pubchem_metadata", {})
            name = (pc.get("common_name") or pc.get("iupac_name","")
                    if pc.get("found") else "")
            label = f"Molecule {i+1}"
            if name:
                label += f" — {name}"
            label += f":  `{r['smiles'][:55]}`"

            with st.expander(label, expanded=False):
                # KPI row inside expander
                d_icon = "🟢" if r["decision"] == "ACCEPT" else "🔴"
                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric("Score",      f"{r['score']} / 100")
                sc2.metric("Decision",   f"{d_icon} {r['decision']}")
                sc3.metric("Lipinski",   "✅ PASS" if r["lipinski"]["passes"] else "❌ FAIL")
                sc4.metric("Tox Alerts", r["toxicophore_summary"]["total_toxicophore_alerts"])

                st.markdown("---")

                # Chart + download
                st.markdown(subh("📊 ADME Report Chart"), unsafe_allow_html=True)
                fig = build_adme_figure(r)
                if fig:
                    st.pyplot(fig, use_container_width=True)
                    png_bytes = fig_to_bytes(fig, dpi=300)
                    plt.close(fig)
                    st.download_button(
                        label=f"⬇️ Download high-res PNG — Molecule {i+1}",
                        data=png_bytes,
                        file_name=f"adme_mol_{i+1}.png",
                        mime="image/png",
                        key=f"dl_{i}",
                    )

                # PubChem — always visible, inline
                st.markdown("---")
                st.markdown(subh("🔗 PubChem Information"), unsafe_allow_html=True)
                render_pubchem(pc)

                # Collapsible detail sections
                st.markdown("---")
                st.markdown(subh("🔬 Detailed Analysis"), unsafe_allow_html=True)
                render_detail(r)