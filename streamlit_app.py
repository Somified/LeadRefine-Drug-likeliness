"""
streamlit_app.py
================
Streamlit front-end for the ADME analysis module.

Run with:
    streamlit run streamlit_app.py

Requires adme_analysis.py to be in the same folder.
"""

import io
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from adme_analysis import analyze_smiles, analyze_multiple_smiles, build_adme_figure

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ADME Analyzer",
    page_icon="🧬",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0D1117; }
    .stTextInput > div > div > input {
        background-color: #161B22;
        color: #E6EDF3;
        border: 1px solid #30363D;
        border-radius: 6px;
        font-family: monospace;
    }
    .stTextArea > div > div > textarea {
        background-color: #161B22;
        color: #E6EDF3;
        border: 1px solid #30363D;
        font-family: monospace;
    }
    .metric-card {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 4px 0;
    }
    .badge-green  { color: #3FB950; font-weight: bold; }
    .badge-yellow { color: #D29922; font-weight: bold; }
    .badge-orange { color: #F97316; font-weight: bold; }
    .badge-red    { color: #F85149; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
RISK_CLASS = {
    "LOW":        "badge-green",
    "CLEAN":      "badge-green",
    "NEGATIVE":   "badge-green",
    "OPTIMAL":    "badge-green",
    "ACCEPT":     "badge-green",
    "PASS":       "badge-green",
    "HIGH_STAB":  "badge-green",
    "MODERATE":   "badge-yellow",
    "NEAR_OPTIMAL": "badge-yellow",
    "WEAK":       "badge-orange",
    "HIGH":       "badge-orange",
    "FLAGGED":    "badge-orange",
    "POSITIVE":   "badge-red",
    "CRITICAL":   "badge-red",
    "POOR":       "badge-red",
    "REJECT":     "badge-red",
}

def badge(value):
    cls = RISK_CLASS.get(str(value).upper(), "")
    return f'<span class="{cls}">{value}</span>'


def fig_to_bytes(fig):
    """Convert a matplotlib figure to PNG bytes for st.download_button."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧬 ADME Analyzer")
    st.markdown("---")

    mode = st.radio("Mode", ["Single molecule", "Batch (multiple SMILES)"])

    st.markdown("---")
    st.markdown("**Demo molecules**")
    demos = {
        "Aspirin":      "CC(=O)Oc1ccccc1C(=O)O",
        "CCl4":         "ClC(Cl)(Cl)Cl",
        "Ibuprofen":    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "Atorvastatin": "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CCC(O)CC(O)CC(=O)O",
        "Caffeine":     "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
    }
    selected_demo = st.selectbox("Load a demo", ["— select —"] + list(demos.keys()))

    st.markdown("---")
    st.caption("Rule-based ADME screening.\nNot a substitute for experimental data.")


# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🧬 ADME Analysis Dashboard")
st.markdown(
    "Enter a SMILES string to screen for drug-likeness, metabolic stability, "
    "toxicophore alerts, and full ADME properties."
)

# ── Single molecule mode ──────────────────────────────────────────────────────
if mode == "Single molecule":

    default_smiles = demos[selected_demo] if selected_demo != "— select —" else ""
    smiles_input = st.text_input(
        "SMILES string",
        value=default_smiles,
        placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O",
    )

    analyse_btn = st.button("Analyse", type="primary", use_container_width=True)

    if analyse_btn and smiles_input.strip():
        with st.spinner("Running ADME analysis…"):
            result = analyze_smiles(smiles_input.strip())

        if not result["valid"]:
            st.error("❌ Invalid SMILES string — could not parse molecule.")
            st.stop()

        # ── Top KPI row ───────────────────────────────────────────────────────
        st.markdown("---")
        k1, k2, k3, k4, k5 = st.columns(5)

        decision_color = "🟢" if result["decision"] == "ACCEPT" else "🔴"
        k1.metric("Drug-likeness Score", f"{result['score']} / 100")
        k2.metric("Status",   result["physchem_status"])
        k3.metric("Decision", f"{decision_color} {result['decision']}")
        k4.metric("Lipinski", "✅ PASS" if result["lipinski"]["passes"] else "❌ FAIL")
        k5.metric("Toxicophore Alerts",
                  result["toxicophore_summary"]["total_toxicophore_alerts"])

        # ── Full visual report ────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📊 Full ADME Report")

        with st.spinner("Building charts…"):
            fig = build_adme_figure(result)

        if fig:
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            # Download button
            png_bytes = fig_to_bytes(build_adme_figure(result))
            plt.close("all")
            st.download_button(
                label="⬇️  Download report as PNG",
                data=png_bytes,
                file_name=f"adme_report_{smiles_input[:20].replace('/','_')}.png",
                mime="image/png",
            )

        # ── Detailed breakdown tables ─────────────────────────────────────────
        st.markdown("---")
        st.subheader("🔬 Detailed Breakdown")

        tab_phys, tab_metab, tab_tox, tab_adme, tab_advice = st.tabs([
            "Physicochemical",
            "Metabolic Stability",
            "Toxicophores",
            "ADME Properties",
            "Optimisation Advice",
        ])

        # ── Tab 1: Physicochemical ────────────────────────────────────────────
        with tab_phys:
            props = result["properties"]
            lip   = result["lipinski"]

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Lipinski Rule-of-Five**")
                rows = [
                    ("Molecular Weight", props["molecular_weight"], "≤ 500 Da"),
                    ("LogP",             props["logp"],             "≤ 5"),
                    ("H-bond Donors",    props["hbd"],              "≤ 5"),
                    ("H-bond Acceptors", props["hba"],              "≤ 10"),
                ]
                for name, val, limit in rows:
                    icon = "✅" if str(val) <= limit.split()[0].replace("≤","").strip() else "❌"
                    st.markdown(f"**{name}:** {val:.2f}  `{limit}`  {icon}")

            with col_b:
                st.markdown("**Extended Properties**")
                st.markdown(f"**TPSA:** {props['tpsa']:.1f} Å²  `≤ 140`  "
                            f"{'✅' if props['tpsa'] <= 140 else '❌'}")
                st.markdown(f"**Rotatable Bonds:** {props['rotatable_bonds']}  `≤ 10`  "
                            f"{'✅' if props['rotatable_bonds'] <= 10 else '❌'}")

            if result["violations"]:
                st.warning(f"⚠️ {len(result['violations'])} physicochemical violation(s) detected.")
                for v in result["violations"]:
                    st.markdown(f"- **{v['property']}** = {v['value']:.2f} (limit {v['limit']}) — {v['issue']}")

        # ── Tab 2: Metabolic Stability ────────────────────────────────────────
        with tab_metab:
            ms = result["microsomal_stability"]
            rm = result["reactive_metabolites"]
            gs = result["gsh_trapping"]
            mb = result["mbi_risk"]
            ss = result["metabolic_soft_spots"]

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Microsomal Stability**")
                stab_cls = ms.get("stability_class", "N/A")
                color = {"HIGH": "🟢", "MODERATE": "🟡", "LOW": "🔴"}.get(stab_cls, "⚪")
                st.markdown(f"{color} **{stab_cls}** — Score: {ms.get('microsomal_stability_score')}/100")
                st.markdown(f"Predicted t½: `{ms.get('predicted_t_half')}`")
                st.markdown(f"Aromatic rings: `{ms.get('n_aromatic_rings')}` | "
                            f"Soft spots: `{ms.get('soft_spot_count')}`")

                st.markdown("**Metabolic Soft Spots**")
                if ss["soft_spots"]:
                    for s in ss["soft_spots"]:
                        st.markdown(f"- {s}")
                else:
                    st.success("No metabolic soft spots detected.")

            with col2:
                st.markdown("**Reactive Metabolite Risk**")
                rm_risk = rm.get("reactive_metabolite_risk", "N/A")
                risk_icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MODERATE": "🟡", "LOW": "🟢"}.get(rm_risk, "⚪")
                st.markdown(f"{risk_icon} **{rm_risk}** — {rm.get('alert_count', 0)} alert(s)")
                if rm["reactive_metabolite_alerts"]:
                    with st.expander("View alerts"):
                        for a in rm["reactive_metabolite_alerts"]:
                            st.markdown(f"- {a}")
                if rm.get("recommended_assays"):
                    st.markdown("**Recommended assays:**")
                    for assay in rm["recommended_assays"]:
                        st.markdown(f"- {assay}")

                st.markdown("**GSH Trapping Risk**")
                gsh_risk = gs.get("gsh_trapping_risk", "N/A")
                gsh_icon = "🔴" if gsh_risk == "POSITIVE" else "🟢"
                st.markdown(f"{gsh_icon} **{gsh_risk}**")
                if gs["gsh_trap_alerts"]:
                    for a in gs["gsh_trap_alerts"]:
                        st.markdown(f"- {a}")

                st.markdown("**MBI Risk (CYP inactivation)**")
                mbi_risk = mb.get("mbi_risk", "N/A")
                mbi_icon = {"HIGH": "🔴", "MODERATE": "🟡", "LOW": "🟢"}.get(mbi_risk, "⚪")
                st.markdown(f"{mbi_icon} **{mbi_risk}**")
                if mb["mbi_alerts"]:
                    for a in mb["mbi_alerts"]:
                        st.markdown(f"- {a}")

        # ── Tab 3: Toxicophores ───────────────────────────────────────────────
        with tab_tox:
            summary = result["toxicophore_summary"]
            overall = summary.get("overall_toxicophore_risk", "N/A")
            overall_icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MODERATE": "🟡", "CLEAN": "🟢"}.get(overall, "⚪")

            st.markdown(f"### {overall_icon} Overall Toxicophore Risk: **{overall}**")
            st.info(summary.get("filter_recommendation", ""))

            bd = summary.get("breakdown", {})
            col1, col2, col3, col4, col5 = st.columns(5)
            for col, (label, key) in zip(
                [col1, col2, col3, col4, col5],
                [("PAINS","pains"), ("Brenk","brenk"), ("SureChEMBL","surechembl"),
                 ("Ames","ames"), ("Reactive","reactive")]
            ):
                count = bd.get(key, 0)
                icon  = "🔴" if count >= 3 else "🟠" if count >= 1 else "🟢"
                col.metric(label, f"{icon} {count}")

            st.markdown("---")
            col_a, col_b, col_c = st.columns(3)

            with col_a:
                pains = result["pains"]
                st.markdown(f"**PAINS — {pains['pains_flag']}**")
                if pains["pains_alerts"]:
                    for a in pains["pains_alerts"]:
                        st.markdown(f"- {a}")
                else:
                    st.success("No PAINS alerts.")

            with col_b:
                brenk = result["brenk"]
                st.markdown(f"**Brenk — {brenk['brenk_flag']}**")
                if brenk["brenk_alerts"]:
                    for a in brenk["brenk_alerts"]:
                        st.markdown(f"- {a}")
                else:
                    st.success("No Brenk alerts.")

            with col_c:
                sure = result["surechembl"]
                st.markdown(f"**SureChEMBL — {sure['regulatory_risk']}**")
                if sure["surechembl_alerts"]:
                    for a in sure["surechembl_alerts"]:
                        st.markdown(f"- {a}")
                else:
                    st.success("No SureChEMBL alerts.")

        # ── Tab 4: ADME Properties ────────────────────────────────────────────
        with tab_adme:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**CYP450 Liability**")
                cyp = result["cyp450"]
                cyp_icon = {"HIGH": "🔴", "MODERATE": "🟡", "LOW": "🟢"}.get(cyp["cyp_risk"], "⚪")
                st.markdown(f"{cyp_icon} Overall: **{cyp['cyp_risk']}** "
                            f"({cyp['isoforms_flagged_count']} isoform(s) flagged)")
                if cyp["flagged_isoforms"]:
                    for iso, alerts in cyp["flagged_isoforms"].items():
                        with st.expander(f"{iso} — {len(alerts)} alert(s)"):
                            for a in alerts:
                                st.markdown(f"- {a}")

                st.markdown("**hERG Cardiac Liability**")
                herg = result["herg"]
                herg_icon = {"HIGH": "🔴", "MODERATE": "🟡", "LOW": "🟢"}.get(herg["herg_risk"], "⚪")
                st.markdown(f"{herg_icon} **{herg['herg_risk']}** (score: {herg['herg_risk_score']})")

                st.markdown("**Mutagenicity (Ames)**")
                ames = result["mutagenicity"]
                ames_icon = "🔴" if ames["mutagenicity_flag"] == "POSITIVE" else "🟢"
                st.markdown(f"{ames_icon} **{ames['mutagenicity_flag']}** — {ames['alert_count']} alert(s)")
                if ames["alerts"]:
                    for a in ames["alerts"]:
                        st.markdown(f"- {a}")

            with col2:
                st.markdown("**Plasma Protein Binding (PPB)**")
                ppb = result["plasma_protein_binding"]
                ppb_icon = {"HIGH": "🔴", "MODERATE": "🟡", "LOW": "🟢"}.get(ppb["ppb_class"], "⚪")
                st.markdown(f"{ppb_icon} **{ppb['ppb_class']}** — ~{ppb['ppb_estimate_pct']}% bound")
                st.caption(ppb["ppb_note"])

                st.markdown("**Caco-2 Permeability**")
                caco2 = result["caco2_permeability"]
                caco2_icon = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(caco2["caco2_class"], "⚪")
                st.markdown(f"{caco2_icon} **{caco2['caco2_class']}** — {caco2['caco2_papp_nm_s']} nm/s")
                st.caption(caco2["caco2_note"])

                st.markdown("**P-glycoprotein Efflux**")
                pgp = result["pgp_efflux"]
                pgp_icon = {"HIGH": "🔴", "MODERATE": "🟡", "LOW": "🟢"}.get(pgp["pgp_risk"], "⚪")
                st.markdown(f"{pgp_icon} **{pgp['pgp_risk']}** (score: {pgp['pgp_risk_score']})")
                if pgp["pgp_risk_reasons"]:
                    for r in pgp["pgp_risk_reasons"]:
                        st.markdown(f"- {r}")

        # ── Tab 5: Optimisation Advice ────────────────────────────────────────
        with tab_advice:
            from adme_analysis import generate_optimization_advice
            advice = generate_optimization_advice(result)

            if not advice:
                st.success("✅ No optimisation needed — molecule passes all filters.")
            else:
                for a in sorted(advice):
                    icon = "⚠️" if a.startswith("PRIORITY") else "▸"
                    st.markdown(f"{icon} {a}")


# ── Batch mode ────────────────────────────────────────────────────────────────
else:
    st.markdown("Enter one SMILES per line:")
    batch_input = st.text_area(
        "SMILES list",
        height=180,
        placeholder="CC(=O)Oc1ccccc1C(=O)O\nClC(Cl)(Cl)Cl\nCC(C)Cc1ccc(cc1)C(C)C(=O)O",
    )

    run_batch = st.button("Analyse all", type="primary", use_container_width=True)

    if run_batch and batch_input.strip():
        smiles_list = [s.strip() for s in batch_input.strip().splitlines() if s.strip()]

        with st.spinner(f"Analysing {len(smiles_list)} molecule(s)…"):
            results = analyze_multiple_smiles(smiles_list)

        valid   = [r for r in results if r["valid"]]
        invalid = [r for r in results if not r["valid"]]

        st.markdown(f"**{len(valid)} valid** | **{len(invalid)} invalid**")

        if invalid:
            with st.expander("Invalid SMILES"):
                for r in invalid:
                    st.markdown(f"- `{r['smiles']}`")

        if not valid:
            st.stop()

        # Summary table
        st.markdown("---")
        st.subheader("📋 Results Summary")

        import pandas as pd
        table_rows = []
        for r in valid:
            table_rows.append({
                "SMILES":           r["smiles"][:40],
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

        df = pd.DataFrame(table_rows)
        st.dataframe(df, use_container_width=True)

        # Per-molecule charts
        st.markdown("---")
        st.subheader("📊 Individual Reports")

        for i, r in enumerate(valid):
            with st.expander(f"Molecule {i+1}: `{r['smiles'][:60]}`"):
                with st.spinner("Building chart…"):
                    fig = build_adme_figure(r)
                if fig:
                    st.pyplot(fig, use_container_width=True)
                    png_bytes = fig_to_bytes(fig)
                    plt.close(fig)
                    st.download_button(
                        label=f"⬇️ Download report {i+1}",
                        data=png_bytes,
                        file_name=f"adme_report_{i+1}.png",
                        mime="image/png",
                        key=f"dl_{i}",
                    )