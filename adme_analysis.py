"""
adme_analysis.py
================
Complete self-contained ADME analysis module.

All physicochemical screening, metabolic stability rules, toxicophore
filters, and visual report generation live in this one file.

Basic usage
-----------
    from adme_analysis import analyze_smiles

    # Analysis only:
    result = analyze_smiles("CC(=O)Oc1ccccc1C(=O)O")

    # Analysis + PNG report in one call:
    result = analyze_smiles("CC(=O)Oc1ccccc1C(=O)O", output_path="aspirin.png")

    # Or generate the report separately:
    from adme_analysis import generate_adme_report
    generate_adme_report(result, output_path="aspirin.png")

Run directly for a demo:
    python adme_analysis.py
"""

# ── Standard library ──────────────────────────────────────────────────────────
import math
import os
import time
import requests
import functools
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── RDKit ─────────────────────────────────────────────────────────────────────
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

# ── Plotting (Agg backend — works without a display) ──────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap


# ============================================================
# 1. RULE DEFINITIONS  (Baselines & Domain Knowledge)
# ============================================================

LIPINSKI_RULES = {
    "molecular_weight": 500,
    "logp":             5,
    "hbd":              5,
    "hba":              10,
}

EXTENDED_ADMET_RULES = {
    "tpsa":            140,
    "rotatable_bonds": 10,
}

ALL_RULES = {**LIPINSKI_RULES, **EXTENDED_ADMET_RULES}

VIOLATION_MEANINGS = {
    "molecular_weight": {"category": "size",         "issue": "Too large for oral absorption"},
    "logp":             {"category": "lipophilicity", "issue": "Poor solubility risk"},
    "hbd":              {"category": "polarity",      "issue": "Too many H-bond donors"},
    "hba":              {"category": "polarity",      "issue": "Too many H-bond acceptors"},
    "tpsa":             {"category": "polarity",      "issue": "Poor membrane permeability"},
    "rotatable_bonds":  {"category": "flexibility",   "issue": "Excessive flexibility"},
}

SUGGESTION_MAP = {
    "size":          ["Remove bulky groups", "Simplify scaffold"],
    "lipophilicity": ["Add polar groups", "Reduce alkyl chains"],
    "polarity":      ["Reduce donors/acceptors", "Use bioisosteres"],
    "flexibility":   ["Reduce rotatable bonds", "Add rings"],
}


# ============================================================
# 1b. CYP450 / hERG / AMES / REACTIVE SMARTS
# ============================================================

CYP_SMARTS = {
    "CYP3A4": [
        ("[#7;!$(N-C=O);!$(N-S=O);!$(N=*)]", "Basic nitrogen (non-amide/sulfonamide)"),
        ("c1ccncc1",                           "Pyridine ring (heme coordination)"),
        ("c1cnc[nH]1",                         "Imidazole ring (heme coordination)"),
        ("c1cn[nH]c1",                         "Pyrazole ring"),
        ("c1ncnn1",                            "Triazole ring (heme coordination)"),
    ],
    "CYP2D6": [
        ("[NH2,NH;!$(NC=O);!$(NS=O)]c1ccccc1",                "Aromatic amine near aryl ring"),
        ("[NH2,NH,N;!$(NC=O);!$(NS=O)]-[CX4]-[CX4]-c1ccccc1","Aliphatic amine 2C from aryl"),
        ("[NH2,NH,N;!$(NC=O)]-[CX4]-c1ccccc1",               "Aliphatic amine adjacent to aryl"),
    ],
    "CYP2C9": [
        ("C(=O)[OH]",     "Carboxylic acid"),
        ("S(=O)(=O)[NH]", "Sulfonamide (acidic NH)"),
        ("c1ccccc1O",     "Phenol"),
        ("c1cc(F)ccc1",   "Fluorinated aryl (metabolic soft spot)"),
    ],
    "CYP1A2": [
        ("c1ccc2ccccc2c1",  "Fused polycyclic aromatic (naphthalene-like)"),
        ("[NH2]c1ccccc1",   "Aniline"),
        ("c1ccc2[nH]ccc2c1","Indole scaffold"),
        ("C#N",             "Nitrile group"),
    ],
    "CYP2C19": [
        ("c1cncc(c1)",             "Substituted pyridine"),
        ("[CH2;!R](c1ccccc1)C=O", "Benzyl carbonyl"),
        ("c1ncc[nH]1",            "Imidazole"),
    ],
}

HERG_SMARTS = [
    ("[NH,NH2,NH3+;!$(NC=O);!$(NS=O)]", "Basic nitrogen"),
    ("c1ccccc1",                          "Phenyl ring (hydrophobic bulk)"),
    ("[nH]1cccc1",                        "Basic heteroaromatic N"),
]

AMES_SMARTS = [
    ("[c][N+](=O)[O-]",                         "Nitroaromatic"),
    ("[NH2]c1ccccc1",                            "Primary aromatic amine (aniline)"),
    ("N=N",                                      "Azo group"),
    ("[N;!$(NC=O)]N",                            "Hydrazine"),
    ("C1CO1",                                    "Epoxide"),
    ("C(=O)Cl",                                  "Acid chloride"),
    ("[CH]=O",                                   "Aldehyde"),
    ("S(=O)(=O)Cl",                              "Sulfonyl chloride"),
    ("[c][NH][NH2]",                             "Arylhydrazine"),
    ("C(=S)N",                                   "Thioamide / dithiocarbamate"),
    ("[N;!$(NC=O)]-[N;!$(NC=O)]-[N;!$(NC=O)]", "Triazene"),
]

REACTIVE_SMARTS = [
    ("C1CO1",                  "Epoxide (alkylating agent)"),
    ("C(=O)Cl",                "Acid chloride (acylating agent)"),
    ("[CH2]=[CH]C(=O)",        "Alpha-beta-unsaturated carbonyl (Michael acceptor)"),
    ("C(=C)C(=O)[#7,#8]",     "Acrylamide / acrylate"),
    ("N=C=O",                  "Isocyanate"),
    ("N=C=S",                  "Isothiocyanate"),
    ("C(=O)OC(=O)",           "Anhydride"),
    ("O=S(=O)(O)Cl",          "Sulfonyl chloride"),
    ("[SH]",                   "Free thiol"),
    ("[OH]C=C",                "Enol"),
    ("[N;!$(NC=O)]-O",        "Hydroxylamine"),
    ("C(=O)-O-N",              "NHS ester"),
    ("[C;!R]=C-[C;!R]=O",     "Conjugated Michael acceptor"),
]

CACO2_THRESHOLDS = {"high": 20, "medium": 2}
PPB_THRESHOLDS   = {"high": 90, "moderate": 70}


# ============================================================
# 1c. METABOLIC STABILITY SMARTS  (most critical category)
# ============================================================

REACTIVE_METABOLITE_SMARTS = [
    ("Oc1ccc(O)cc1",          "Catechol → ortho-quinone (GSH trapping, protein adducts)"),
    ("Oc1ccc(N)cc1",          "4-aminophenol → quinone imine (paracetamol-type toxicity)"),
    ("Oc1cccc(N)c1",          "3-aminophenol → quinone imine"),
    ("Nc1ccc(O)cc1",          "Aminophenol → iminoquinone (idiosyncratic hepatotoxin)"),
    ("Oc1ccc(O)c(O)c1",      "Pyrogallol-type → ortho-quinone"),
    ("O=C1C=CC(=O)C=C1",     "Para-quinone (direct electrophile)"),
    ("c1ccoc1",               "Furan → cis-2-enedione (hepatotoxin — CCl4-like mechanism)"),
    ("C1=COC=C1",             "Dihydrofuran → epoxide intermediate"),
    ("[CH2]c1ccoc1",          "Alkylfuran (bioactivated by CYP-mediated epoxidation)"),
    ("c1ccsc1",               "Thiophene → S-oxide / thiolactone (CYP-mediated, GSH trapping)"),
    ("c1csc(N)c1",            "2-aminothiophene → reactive sulfoxide"),
    ("[CX4]([Cl,Br,F])([Cl,Br,F])[Cl,Br,F]",
                              "Polyhalogenated carbon → carbon radical (liver necrosis, CCl4 mechanism)"),
    ("[CX4]([Cl])([Cl])Cl",  "Trichloromethyl group → CCl3• radical (direct hepatotoxin)"),
    ("C(Cl)(Cl)Cl",          "CHCl3-type → trichloromethanol radical"),
    ("[C;H0]#[C;H1]",         "Terminal alkyne → ketene (irreversible CYP inactivation)"),
    ("[C;H0]#[C;H0]",         "Internal alkyne (MBI risk)"),
    ("c1cc2c(cc1)OCO2",      "Methylenedioxyphenyl → carbene (CYP mechanism-based inactivation)"),
    ("[NH2]c1ccccc1",        "Aniline → nitroso / hydroxylamine (GSH adducts, Met-Hb)"),
    ("[NH2]c1ccc([N+](=O)[O-])cc1",
                              "Nitroaniline → reactive nitroso species"),
    ("NN",                    "Hydrazine → acyl nitroso / diazonium (DNA alkylation)"),
    ("c1ccc2ccccc2c1",       "Polycyclic aromatic → arene oxide / diol epoxide (DNA adduct)"),
    ("[CX3](=O)[CX4][Cl,Br,I]", "Alpha-haloketone → alkylating species"),
    ("C(=O)[OH]",            "Carboxylic acid → acyl glucuronide (protein acylation)"),
]

GSH_TRAP_SMARTS = [
    ("[CH2]=[CH]C(=O)",      "Michael acceptor (direct GSH adduct)"),
    ("C(=O)Cl",              "Acyl chloride (acylates GSH)"),
    ("C1CO1",                "Epoxide (GSH conjugation)"),
    ("O=C1C=CC(=O)C=C1",    "Quinone (1,4-addition with GSH)"),
    ("Oc1ccc(O)cc1",         "Catechol → quinone → GSH adduct"),
    ("c1ccoc1",              "Furan → enedione → GSH adduct"),
    ("[N+](=O)[O-]",         "Nitro → nitroso → sulfinyl adducts"),
    ("c1ccsc1",              "Thiophene → S-oxide → GSH adduct"),
    ("[CX4]([Cl,Br])([Cl,Br])[Cl,Br]",
                             "Polyhalogenated → radical → GSH adduct (via oxidative stress)"),
]

SOFT_SPOT_SMARTS = [
    ("[CH3]c1ccccc1",        "Benzylic methyl (O-dealkylation / benzylic hydroxylation)"),
    ("[CH2]c1ccccc1",        "Benzylic CH2 (rapid CYP oxidation)"),
    ("[CH3][O,N;!$(NC=O)]", "N-methyl / O-methyl (N/O-dealkylation)"),
    ("C(C)(C)c1ccccc1",     "Tert-butyl aryl (hydroxylation at benzylic position)"),
    ("[CX4H2][NH,NH2]",     "Alpha-carbon to amine (amine oxidation / carbinolamine)"),
    ("c1ccccc1[OH]",         "Phenol (glucuronidation / sulfation soft spot)"),
    ("[CX4][F]",             "Aliphatic C-F (defluorination in some CYPs)"),
    ("c1ccc(OC)cc1",         "Methoxyphenyl (O-demethylation → phenol → catechol cascade)"),
    ("[CH2][CH2][CH2][CH3]","Omega-terminal carbon (omega-oxidation)"),
]

MBI_SMARTS = [
    ("c1cc2c(cc1)OCO2",  "Methylenedioxy → carbene (prototypical MBI — safrole, paroxetine)"),
    ("[C;H0]#[C;H1]",    "Terminal alkyne → ketene (MBI of CYP3A4/2B6)"),
    ("c1ccsc1",          "Thiophene → S-oxide (clopidogrel-type MBI)"),
    ("c1ccoc1",          "Furan → cis-enedione (MBI via Michael addition)"),
    ("[NX3]c1ccccc1",    "Aryl amine → nitroso (MBI by nitroso binding to heme Fe)"),
    ("[CX4]([Cl])([Cl])Cl", "CCl3 type → CCl3• radical (CYP2E1 inactivation)"),
]


# ============================================================
# 1d. TOXICOPHORE SMARTS  (PAINS / Brenk / SureChEMBL)
# ============================================================

PAINS_SMARTS = [
    ("O=C1NC(=S)SC1",           "Rhodanine (PAINS — redox cycling / aggregation)"),
    ("c1cc(O)c(O)cc1",          "Catechol (PAINS — metal chelation / quinone formation)"),
    ("O=C1C=CC(=O)C=C1",       "para-Quinone (PAINS — redox / thiol reactive)"),
    ("O=C1C=CC(=O)c2ccccc21",  "Naphthoquinone (PAINS)"),
    ("[CH]=N",                   "Aliphatic imine/Schiff base (PAINS — hydrolysis lability)"),
    ("C=CC(=O)c1ccccc1",       "Chalcone enone (PAINS — Michael acceptor)"),
    ("[CH]=[CH]C(=O)[#6]",     "Alpha-beta-unsaturated carbonyl (PAINS)"),
    ("c1ccc(/N=N/c2ccccc2)cc1","Azo dye (PAINS — redox interference)"),
    ("C(=O)NO",                  "Hydroxamic acid (PAINS — metal chelation)"),
    ("[SH]",                     "Free thiol (PAINS — thiol reactive, assay interference)"),
    ("CCCCCCCC",                 "Long aliphatic chain (aggregation-prone)"),
    ("c1ccc2ccccc2c1",          "Naphthalene (may interfere with fluorescence assays)"),
    ("C(=C)C#N",                "Alpha-cyanoacrylate (Michael acceptor PAINS)"),
]

BRENK_SMARTS = [
    ("[CX4][Cl,Br,I]",         "Brenk: Alkyl halide (electrophilic reactivity)"),
    ("[F,Cl,Br,I][CX4][F,Cl,Br,I]",
                                 "Brenk: Gem-dihalide (reactive)"),
    ("[N;!$(NC=O)]=O",         "Brenk: Nitroso group (reactive, mutagen)"),
    ("[N+]([O-])(=O)c1ccccc1", "Brenk: Nitroaromatic (hepatotoxin / Ames+)"),
    ("OO",                       "Brenk: Peroxide (oxidative stress)"),
    ("C(=O)OO",                  "Brenk: Peracid (strongly oxidising)"),
    ("P(=S)",                    "Brenk: Thiophosphate (organophosphate — neurotoxin)"),
    ("P(=O)(Cl)",               "Brenk: Phosphoryl chloride (hydrolysis product toxic)"),
    ("[#6]S[#6]S[#6]",         "Brenk: Polythioether (metal chelation / interference)"),
    ("C(=O)NC(=O)",             "Brenk: Imide (reactivity, protein binding)"),
    ("[N+](C)(C)(C)C",          "Brenk: Quaternary ammonium (absorption limited)"),
    ("C(=S)",                    "Brenk: Thiocarbonyl (hepatotoxin / metal reactive)"),
    ("ClCCN",                    "Brenk: Nitrogen mustard-like (alkylating agent)"),
    ("C=CC(=O)O",               "Brenk: Acrylate ester (Michael acceptor / contact sensitiser)"),
    ("[CH]=O",                   "Brenk: Aldehyde (protein reactive, sensitisation)"),
    ("C=[N+]=[N-]",             "Brenk: Diazo compound (reactive nitrogen species)"),
    ("[#6][SH]",                "Brenk: Free thiol on carbon (assay interference / reactive)"),
    ("OC(=O)N",                  "Brenk: Carbamate (metabolic lability)"),
    ("C1OC1",                    "Brenk: Epoxide (alkylating agent)"),
    ("C(=O)NN",                  "Brenk: Hydrazide (reactive metabolite)"),
    ("C=NO",                     "Brenk: Oxime (hydrolysis lability / reactive)"),
    ("C1CC(=O)N1",              "Brenk: Beta-lactam (unless intentional antibiotic scaffold)"),
    ("[Hg,Pb,As,Cd,Tl]",       "Brenk: Heavy metal atom (systemic toxicity)"),
    ("c1ccc2c(c1)ccc3ccccc23", "Brenk: Anthracene/PAH (DNA intercalation / mutagenesis)"),
]

SURECHEMBL_SMARTS = [
    ("[N+](=O)[O-]",            "SureChEMBL: Nitro group (genotoxic via nitroreduction)"),
    ("[NH2]c1ccccc1",           "SureChEMBL: Primary aromatic amine (genotoxic)"),
    ("c1ccc(N)cc1-c2ccccc2N",  "SureChEMBL: Diaminobiphenyl (carcinogen)"),
    ("N=Nc1ccccc1",             "SureChEMBL: Azo to aryl (azo dye — carcinogen)"),
    ("C(=O)Cl",                 "SureChEMBL: Acyl chloride"),
    ("S(=O)(=O)Cl",             "SureChEMBL: Sulfonyl chloride"),
    ("C1CO1",                   "SureChEMBL: Epoxide (alkylating agent)"),
    ("C=CC#N",                  "SureChEMBL: Acrylonitrile (Michael acceptor, hepatotoxin)"),
    ("c1cc(O)ccc1",             "SureChEMBL: Phenol (estrogen receptor binding risk)"),
    ("c1ccc(Cl)cc1",            "SureChEMBL: Chlorobenzene (persistent, bioaccumulates)"),
    ("[CX4]([Cl])([Cl])[Cl]",  "SureChEMBL: Trichloromethyl (hepatotoxin — CCl4 mechanism)"),
    ("[CX4]([Cl])([Cl])([Cl])[Cl]",
                                 "SureChEMBL: Carbon tetrachloride scaffold (liver necrosis)"),
    ("[CX4]([Br])([Br])[Br]",  "SureChEMBL: Tribromomethane type"),
    ("[Hg,Pb,As,Cd,Se,Te,Tl]", "SureChEMBL: Heavy metal (organ toxicity)"),
    ("C(=O)CCCC(=O)",           "SureChEMBL: 1,5-diketone (reproductive toxin — hexanedione type)"),
    ("P(=S)(O)(O)O",            "SureChEMBL: Organothiophosphate (neurotoxin — AChE inhibitor)"),
    ("[Cl]C(Cl)=CCl",           "SureChEMBL: Trichloroethylene-type (carcinogen)"),
    ("C=CC(=O)",                "SureChEMBL: Alpha,beta-unsaturated ketone (skin sensitiser)"),
    ("c1ccc2[nH]ccc2c1",       "SureChEMBL: Indole (potential 5-HT liability at high dose)"),
]


# ============================================================
# 2. ADMET FEATURE COMPUTATION
# ============================================================

def compute_admet_features(smiles):
    """Compute physicochemical and ADMET-related descriptors."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

    features = {
        "molecular_weight": Descriptors.MolWt(mol),
        "logp":             Descriptors.MolLogP(mol),
        "hbd":              Lipinski.NumHDonors(mol),
        "hba":              Lipinski.NumHAcceptors(mol),
        "tpsa":             Descriptors.TPSA(mol),
        "rotatable_bonds":  Lipinski.NumRotatableBonds(mol),
    }
    return mol, features


# ============================================================
# 3. LIPINSKI BASELINE
# ============================================================

def lipinski_filter(features):
    """Traditional Lipinski Rule-of-Five. Returns pass/fail + violated rules."""
    violations = [p for p, limit in LIPINSKI_RULES.items() if features[p] > limit]
    return {"passes": len(violations) <= 1, "violations": violations}


# ============================================================
# 4. EXTENDED ADMET RULE CHECKING
# ============================================================

def check_admet_violations(features):
    """Checks all ADMET rule violations (Lipinski + extended)."""
    violations = []
    for prop, limit in ALL_RULES.items():
        if features[prop] > limit:
            v = VIOLATION_MEANINGS[prop]
            violations.append({
                "property":    prop,
                "value":       features[prop],
                "limit":       limit,
                "issue":       v["issue"],
                "suggestions": SUGGESTION_MAP[v["category"]],
            })
    return violations


# ============================================================
# 5. RULE-BASED COMPOSITE SCORING
# ============================================================

def heuristic_drug_likeness_score(features, violations):
    """Hand-crafted composite drug-likeness score (0–100)."""
    score = 100
    score -= 15 * len(violations)
    for prop, limit in ALL_RULES.items():
        score -= max(0, features[prop] - limit) * 0.1

    reject_reason = None
    if features["molecular_weight"] < 150:
        reject_reason = "Too small to be a viable drug scaffold"
    if features["hbd"] == 0 and features["hba"] == 0:
        reject_reason = "No functional groups for target binding"
    if reject_reason:
        score -= 40

    return max(score, 0), reject_reason


# ============================================================
# 6a. CYP450 METABOLIC LIABILITY
# ============================================================

def compute_cyp450_liability(mol):
    """Structural alerts for CYP3A4, 2D6, 2C9, 1A2, 2C19 isoforms."""
    cyp_flags = {}
    for isoform, patterns in CYP_SMARTS.items():
        matched = [desc for smarts, desc in patterns
                   if (q := Chem.MolFromSmarts(smarts)) and mol.HasSubstructMatch(q)]
        if matched:
            cyp_flags[isoform] = matched

    n = len(cyp_flags)
    return {
        "flagged_isoforms":       cyp_flags,
        "isoforms_flagged_count": n,
        "cyp_risk":               "HIGH" if n >= 3 else "MODERATE" if n >= 1 else "LOW",
    }


# ============================================================
# 6b. hERG CARDIAC LIABILITY
# ============================================================

def compute_herg_risk(features, mol):
    """Estimates hERG potassium channel liability (cardiac toxicity risk)."""
    smarts_hits = [desc for smarts, desc in HERG_SMARTS
                   if (q := Chem.MolFromSmarts(smarts)) and mol.HasSubstructMatch(q)]

    has_basic_n   = any("nitrogen" in h.lower() for h in smarts_hits)
    is_lipophilic = features["logp"] > 1.0
    is_large      = features["molecular_weight"] > 300

    risk_score = 0
    if has_basic_n:             risk_score += 2
    if is_lipophilic:           risk_score += 1
    if is_large:                risk_score += 1
    if features["logp"] > 3.5: risk_score += 1

    return {
        "herg_risk":       "HIGH" if risk_score >= 4 else "MODERATE" if risk_score >= 2 else "LOW",
        "herg_risk_score": risk_score,
        "structural_flags": smarts_hits,
        "contributing_factors": {
            "basic_nitrogen_present": has_basic_n,
            "lipophilic_logp_gt_1":   is_lipophilic,
            "large_mw_gt_300":        is_large,
            "logp_value":             features["logp"],
        },
    }


# ============================================================
# 6c. MUTAGENICITY — AMES TEST
# ============================================================

def compute_mutagenicity_alerts(mol):
    """Screens for Ames-test-positive structural alerts (Kazius/Benigni)."""
    alerts = [desc for smarts, desc in AMES_SMARTS
              if (q := Chem.MolFromSmarts(smarts)) and mol.HasSubstructMatch(q)]
    return {
        "alerts":            alerts,
        "alert_count":       len(alerts),
        "mutagenicity_flag": "POSITIVE" if alerts else "NEGATIVE",
        "confidence":        "STRUCTURAL_ALERT_BASED",
    }


# ============================================================
# 6d. REACTIVE GROUPS / PAINS
# ============================================================

def compute_reactive_alerts(mol):
    """Flags reactive functional groups (PAINS, covalent warheads)."""
    alerts = [desc for smarts, desc in REACTIVE_SMARTS
              if (q := Chem.MolFromSmarts(smarts)) and mol.HasSubstructMatch(q)]
    return {
        "reactive_groups": alerts,
        "reactive_count":  len(alerts),
        "reactivity_flag": "REACTIVE" if alerts else "CLEAN",
        "notes": (
            "Reactive groups detected — may cause covalent off-target modification "
            "or interfere with assay signals." if alerts else "No reactive groups detected."
        ),
    }


# ============================================================
# 6e. PLASMA PROTEIN BINDING
# ============================================================

def estimate_plasma_protein_binding(features):
    """Heuristic PPB estimate. Formula: 40 + 8·logP + 0.05·MW (Yamazaki)."""
    ppb = round(min(99.9, max(0.0, 40 + features["logp"] * 8 + features["molecular_weight"] * 0.05)), 1)
    if ppb >= PPB_THRESHOLDS["high"]:
        cls, note = "HIGH",     "Limited free drug fraction — may affect efficacy and dosing"
    elif ppb >= PPB_THRESHOLDS["moderate"]:
        cls, note = "MODERATE", "Moderate protein binding — generally acceptable"
    else:
        cls, note = "LOW",      "Low protein binding — higher free drug fraction"
    return {"ppb_estimate_pct": ppb, "ppb_class": cls, "ppb_note": note}


# ============================================================
# 6f. CACO-2 PERMEABILITY
# ============================================================

def estimate_caco2_permeability(features):
    """Clark (1999) log-linear Papp model from TPSA and MW."""
    log_papp  = -0.0131 * features["tpsa"] - 0.003 * (features["molecular_weight"] - 200) + 1.5
    papp_nm_s = round(10 ** log_papp, 2)
    if papp_nm_s >= CACO2_THRESHOLDS["high"]:
        cls, note = "HIGH",   "Good intestinal permeability predicted"
    elif papp_nm_s >= CACO2_THRESHOLDS["medium"]:
        cls, note = "MEDIUM", "Moderate permeability — absorption may be variable"
    else:
        cls, note = "LOW",    "Poor permeability predicted — oral absorption likely limited"
    return {"caco2_papp_nm_s": papp_nm_s, "caco2_class": cls, "caco2_note": note}


# ============================================================
# 6g. P-GLYCOPROTEIN EFFLUX
# ============================================================

def estimate_pgp_efflux(features, mol):
    """Didziapetris (2003) MW + HBD + HBA scoring for P-gp substrate risk."""
    mw, hbd, hba, rb = (features["molecular_weight"], features["hbd"],
                        features["hba"], features["rotatable_bonds"])
    pgp_score, reasons = 0, []
    if mw  > 400: pgp_score += 2; reasons.append(f"High MW ({mw:.1f} > 400 Da)")
    if hbd > 2:   pgp_score += 1; reasons.append(f"Multiple H-bond donors (HBD = {hbd})")
    if hba > 3:   pgp_score += 1; reasons.append(f"Multiple H-bond acceptors (HBA = {hba})")
    if rb  > 6:   pgp_score += 1; reasons.append(f"High flexibility ({rb} rotatable bonds)")

    if pgp_score >= 4:
        cls, note = "HIGH",     "Probable P-gp substrate — efflux-limited absorption or CNS exclusion"
    elif pgp_score >= 2:
        cls, note = "MODERATE", "Possible P-gp substrate — monitor in permeability assays"
    else:
        cls, note = "LOW",      "Unlikely P-gp substrate"
    return {"pgp_risk": cls, "pgp_risk_score": pgp_score,
            "pgp_risk_reasons": reasons, "pgp_note": note}


# ============================================================
# 7a. METABOLIC STABILITY — reactive metabolite precursors
# ============================================================

def compute_reactive_metabolite_risk(mol):
    """
    Screens for structural precursors bioactivated into reactive / toxic
    metabolites by CYP enzymes.  #1 cause of post-market drug withdrawals.

    Risk: CRITICAL ≥3 alerts | HIGH ≥2 | MODERATE ≥1 | LOW 0
    """
    alerts = [desc for smarts, desc in REACTIVE_METABOLITE_SMARTS
              if (q := Chem.MolFromSmarts(smarts)) and mol.HasSubstructMatch(q)]
    n = len(alerts)
    risk = "CRITICAL" if n >= 3 else "HIGH" if n >= 2 else "MODERATE" if n >= 1 else "LOW"
    return {
        "reactive_metabolite_alerts": alerts,
        "alert_count":               n,
        "reactive_metabolite_risk":  risk,
        "recommended_assays": (
            ["GSH trapping assay (LC-MS/MS)", "KCN trapping assay",
             "CYP TDI (time-dependent inhibition) shift assay"] if n >= 1 else []
        ),
        "interpretation": (
            f"{n} reactive metabolite precursor alert(s) detected. Structural features "
            "known to be bioactivated into electrophilic species." if n >= 1
            else "No reactive metabolite precursors detected."
        ),
    }


# ============================================================
# 7b. METABOLIC STABILITY — GSH trapping
# ============================================================

def compute_gsh_trapping_risk(mol):
    """
    Flags structural features that form glutathione (GSH) adducts in vitro.
    Positive GSH trapping correlates with idiosyncratic hepatotoxicity risk.
    """
    alerts = [desc for smarts, desc in GSH_TRAP_SMARTS
              if (q := Chem.MolFromSmarts(smarts)) and mol.HasSubstructMatch(q)]
    return {
        "gsh_trap_alerts":   alerts,
        "gsh_alert_count":   len(alerts),
        "gsh_trapping_risk": "POSITIVE" if alerts else "NEGATIVE",
        "clinical_implication": (
            "Idiosyncratic hepatotoxicity risk — GSH depletion in hepatocytes leads to "
            "oxidative stress and cell death (mechanism: CCl4, APAP, troglitazone)."
            if alerts else "No GSH trapping structural alerts detected."
        ),
    }


# ============================================================
# 7c. METABOLIC STABILITY — soft spots
# ============================================================

def identify_metabolic_soft_spots(mol):
    """
    Identifies structural features preferentially oxidised by CYP enzymes.
    High soft-spot count → rapid hepatic clearance → short t½.
    """
    spots = [desc for smarts, desc in SOFT_SPOT_SMARTS
             if (q := Chem.MolFromSmarts(smarts)) and mol.HasSubstructMatch(q)]
    return {
        "soft_spots":           spots,
        "soft_spot_count":      len(spots),
        "clearance_prediction": "HIGH" if len(spots) >= 3 else "MODERATE" if len(spots) >= 1 else "LOW",
        "optimization_strategies": (
            ["Fluorine blocking of benzylic/allylic soft spots",
             "Deuterium substitution at N-methyl or O-methyl positions",
             "Replace methoxyphenyl with difluorophenyl",
             "Cyclise or rigidify flexible chains"] if spots else []
        ),
    }


# ============================================================
# 7d. METABOLIC STABILITY — mechanism-based inactivation
# ============================================================

def compute_mechanism_based_inactivation_risk(mol):
    """
    Screens for CYP mechanism-based inactivation (MBI) alerts.
    MBI causes irreversible CYP inhibition and major DDI risk.
    """
    alerts = [desc for smarts, desc in MBI_SMARTS
              if (q := Chem.MolFromSmarts(smarts)) and mol.HasSubstructMatch(q)]
    return {
        "mbi_alerts": alerts,
        "mbi_count":  len(alerts),
        "mbi_risk":   "HIGH" if len(alerts) >= 2 else "MODERATE" if len(alerts) >= 1 else "LOW",
        "ddi_concern": len(alerts) >= 1,
        "recommended_assays": (
            ["CYP TDI IC50 shift assay", "Kinact/KI determination",
             "DDI risk ratio per FDA/EMA guidance"] if alerts else []
        ),
    }


# ============================================================
# 7e. METABOLIC STABILITY — microsomal stability score
# ============================================================

def estimate_microsomal_stability(features, mol):
    """
    Heuristic hepatic microsomal stability score (0–100).
    Based on Di et al. (2012) J. Med. Chem. 55:4669 descriptor relationships.

    HIGH  ≥70  → t½ > 60 min
    MOD   40-70 → t½ 30–60 min
    LOW   <40  → t½ < 30 min
    """
    logp = features["logp"]
    mw   = features["molecular_weight"]
    try:
        n_ar = rdMolDescriptors.CalcNumAromaticRings(mol)
    except Exception:
        n_ar = 0

    n_soft = identify_metabolic_soft_spots(mol)["soft_spot_count"]

    score  = 100
    score -= logp * 6
    score -= n_ar * 5
    score -= n_soft * 8
    score += min(mw, 500) * 0.04
    score  = round(min(100, max(0, score)), 1)

    cls   = "HIGH" if score >= 70 else "MODERATE" if score >= 40 else "LOW"
    t_half = "> 60 min (predicted)" if cls == "HIGH" else \
             "30–60 min (predicted)" if cls == "MODERATE" else "< 30 min (predicted)"

    return {
        "microsomal_stability_score": score,
        "stability_class":            cls,
        "predicted_t_half":           t_half,
        "n_aromatic_rings":           n_ar,
        "soft_spot_count":            n_soft,
        "logp_penalty":               round(logp * 6, 1),
        "aromatic_ring_penalty":      n_ar * 5,
        "soft_spot_penalty":          n_soft * 8,
        "stability_interpretation": (
            f"Predicted microsomal stability: {cls} (score {score}/100, t½ {t_half}). "
            f"{n_ar} aromatic ring(s) and {n_soft} soft spot(s) detected."
        ),
    }


# ============================================================
# 8a. TOXICOPHORES — PAINS
# ============================================================

def compute_pains_alerts(mol):
    """
    Pan-Assay INterference compound (PAINS) filter.
    Baell & Holloway (2010) J. Med. Chem. 53:2719
    """
    alerts = [desc for smarts, desc in PAINS_SMARTS
              if (q := Chem.MolFromSmarts(smarts)) and mol.HasSubstructMatch(q)]
    return {
        "pains_alerts": alerts,
        "pains_count":  len(alerts),
        "pains_flag":   "FLAGGED" if alerts else "CLEAN",
        "assay_validity": (
            "Counter-screening strongly recommended — compound may produce "
            "false positive HTS results independent of target engagement."
            if alerts else "No PAINS alerts detected. Assay validity unaffected."
        ),
    }


# ============================================================
# 8b. TOXICOPHORES — Brenk
# ============================================================

def compute_brenk_alerts(mol):
    """
    Brenk structural alerts for lead-likeness.
    Brenk et al. (2008) ChemMedChem 3:435
    """
    alerts = [desc for smarts, desc in BRENK_SMARTS
              if (q := Chem.MolFromSmarts(smarts)) and mol.HasSubstructMatch(q)]
    n = len(alerts)
    return {
        "brenk_alerts": alerts,
        "brenk_count":  n,
        "brenk_flag":   "HIGH" if n >= 3 else "MODERATE" if n >= 1 else "CLEAN",
        "lead_likeness": (
            "POOR — multiple Brenk alerts; scaffold likely unsuitable for lead development"
            if n >= 3 else
            "MARGINAL — Brenk alert(s) present; optimise or confirm necessity"
            if n >= 1 else "GOOD — No Brenk alerts detected."
        ),
    }


# ============================================================
# 8c. TOXICOPHORES — SureChEMBL
# ============================================================

def compute_surechembl_alerts(mol):
    """
    SureChEMBL / ICH S2(R1) toxicity structural alerts.
    Covers genotoxins, carcinogens, endocrine disruptors, neurotoxins.
    """
    alerts = [desc for smarts, desc in SURECHEMBL_SMARTS
              if (q := Chem.MolFromSmarts(smarts)) and mol.HasSubstructMatch(q)]
    n = len(alerts)
    return {
        "surechembl_alerts": alerts,
        "surechembl_count":  n,
        "regulatory_risk":   "HIGH" if n >= 2 else "MODERATE" if n >= 1 else "LOW",
        "regulatory_note": (
            "Regulatory safety concern — compound contains patterns flagged under "
            "ICH S2(R1), REACH SVHC, or EMA genotoxicity guidance."
            if n >= 1 else "No SureChEMBL regulatory alerts detected."
        ),
    }


# ============================================================
# 8d. TOXICOPHORES — Overall summary
# ============================================================

def compute_overall_toxicophore_summary(pains, brenk, surechembl, mutagenicity, reactive):
    """Aggregates all toxicophore results into a single risk profile."""
    total = (pains["pains_count"] + brenk["brenk_count"] +
             surechembl["surechembl_count"] + mutagenicity["alert_count"] +
             reactive["reactive_count"])

    overall = ("CRITICAL" if total >= 5 else "HIGH" if total >= 3 else
               "MODERATE" if total >= 1 else "CLEAN")

    return {
        "total_toxicophore_alerts": total,
        "overall_toxicophore_risk": overall,
        "breakdown": {
            "pains":      pains["pains_count"],
            "brenk":      brenk["brenk_count"],
            "surechembl": surechembl["surechembl_count"],
            "ames":       mutagenicity["alert_count"],
            "reactive":   reactive["reactive_count"],
        },
        "filter_recommendation": (
            "REJECT — compound unlikely to progress through safety pharmacology."
            if overall == "CRITICAL" else
            "FLAG — requires significant medicinal chemistry optimisation."
            if overall == "HIGH" else
            "CAUTION — verify alerts are acceptable in therapeutic context."
            if overall == "MODERATE" else
            "PASS — no toxicophore alerts detected across all filter sets."
        ),
    }


# ============================================================
# 8e. PUBCHEM METADATA RETRIEVAL  (NEW — additive only)
# ============================================================

@functools.lru_cache(maxsize=256)
def get_pubchem_metadata(smiles):
    """
    Fast PubChem lookup — 3 API calls only (CID, properties, synonyms).
    Returns compound identity, naming and chemical identifiers.
    Bioassays, targets, pathways and literature are excluded for speed.
    """
    BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

    def _json(url, params=None):
        try:
            r = requests.get(url, params=params, timeout=8)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        return {}

    # Step 1: SMILES → CID
    cids = (_json(f"{BASE}/compound/smiles/cids/JSON", params={"smiles": smiles})
            .get("IdentifierList", {}).get("CID", []))
    if not cids:
        return {"found": False, "error": "Compound not found in PubChem database."}
    cid = int(cids[0])

    # Step 2: Core properties
    props_wanted = ("IUPACName,MolecularFormula,MolecularWeight,CanonicalSMILES,"
                    "InChI,InChIKey,Charge,XLogP,ExactMass")
    prop_data = (_json(f"{BASE}/compound/cid/{cid}/property/{props_wanted}/JSON")
                 .get("PropertyTable", {}).get("Properties", [{}])[0])

    # Step 3: Synonyms
    all_syns = (_json(f"{BASE}/compound/cid/{cid}/synonyms/JSON")
                .get("InformationList", {})
                .get("Information", [{}])[0]
                .get("Synonym", []))
    synonyms    = all_syns[:10]
    common_name = all_syns[0] if all_syns else prop_data.get("IUPACName", "")

    return {
        "found":            True,
        "cid":              cid,
        "iupac_name":       prop_data.get("IUPACName", ""),
        "common_name":      common_name,
        "canonical_smiles": prop_data.get("CanonicalSMILES", ""),
        "formula":          prop_data.get("MolecularFormula", ""),
        "molecular_weight": prop_data.get("MolecularWeight"),
        "inchi":            prop_data.get("InChI", ""),
        "inchikey":         prop_data.get("InChIKey", ""),
        "charge":           prop_data.get("Charge"),
        "xlogp":            prop_data.get("XLogP"),
        "exact_mass":       prop_data.get("ExactMass"),
        "synonyms":         synonyms,
        "links": {
            "compound_page":   f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
            "structure_image": f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/PNG",
            "sdf_download":    f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/SDF",
            "json_api":        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/JSON",
        },
    }


# ============================================================
# 9. MASTER ANALYSIS PIPELINE
# ============================================================

def analyze_smiles(smiles, output_path=None, show=False, fetch_pubchem=True):
    """
    Full ADME analysis pipeline.

    Parameters
    ----------
    smiles      : SMILES string of the molecule to analyse
    output_path : optional path to save a PNG visual report,
                  e.g. "report.png". If None, no chart is generated.
    show          : if True, call plt.show() (requires a GUI display)
    fetch_pubchem : if True (default), query the PubChem API for compound
                    metadata. Set to False for fast batch runs where the
                    extra network latency is unwanted.

    Returns
    -------
    dict with all ADME analysis results

    Examples
    --------
    # Analysis only:
    result = analyze_smiles("CC(=O)Oc1ccccc1C(=O)O")

    # Analysis + auto-generate PNG report:
    result = analyze_smiles("CC(=O)Oc1ccccc1C(=O)O", output_path="aspirin.png")
    """
    mol, features = compute_admet_features(smiles)

    if features is None:
        return {"smiles": smiles, "valid": False}

    lipinski_result = lipinski_filter(features)
    violations      = check_admet_violations(features)
    score, reject_reason = heuristic_drug_likeness_score(features, violations)

    # Original ADME modules
    cyp450       = compute_cyp450_liability(mol)
    herg         = compute_herg_risk(features, mol)
    mutagenicity = compute_mutagenicity_alerts(mol)
    reactive     = compute_reactive_alerts(mol)
    ppb          = estimate_plasma_protein_binding(features)
    caco2        = estimate_caco2_permeability(features)
    pgp          = estimate_pgp_efflux(features, mol)

    # Metabolic stability modules
    reactive_metabolites = compute_reactive_metabolite_risk(mol)
    gsh_trapping         = compute_gsh_trapping_risk(mol)
    soft_spots           = identify_metabolic_soft_spots(mol)
    mbi_risk             = compute_mechanism_based_inactivation_risk(mol)
    microsomal_stability = estimate_microsomal_stability(features, mol)

    # Toxicophore modules
    pains_alerts      = compute_pains_alerts(mol)
    brenk_alerts      = compute_brenk_alerts(mol)
    surechembl_alerts = compute_surechembl_alerts(mol)
    toxicophore_summary = compute_overall_toxicophore_summary(
        pains_alerts, brenk_alerts, surechembl_alerts, mutagenicity, reactive
    )

    # Status labels
    if score >= 85 and not violations:
        status = "OPTIMAL"
    elif score >= 70:
        status = "NEAR_OPTIMAL"
    elif score >= 50:
        status = "WEAK"
    else:
        status = "POOR"

    result = {
        "smiles":          smiles,
        "valid":           True,
        "mol":             mol,
        "properties":      features,
        "lipinski":        lipinski_result,
        "violations":      violations,
        "score":           round(score, 2),
        "physchem_status": status,
        "decision":        "REJECT" if reject_reason else "ACCEPT",
        "reject_reason":   reject_reason,
        # Original ADME
        "cyp450":                 cyp450,
        "herg":                   herg,
        "mutagenicity":           mutagenicity,
        "reactive_alerts":        reactive,
        "plasma_protein_binding": ppb,
        "caco2_permeability":     caco2,
        "pgp_efflux":             pgp,
        # Metabolic stability
        "reactive_metabolites":  reactive_metabolites,
        "gsh_trapping":          gsh_trapping,
        "metabolic_soft_spots":  soft_spots,
        "mbi_risk":              mbi_risk,
        "microsomal_stability":  microsomal_stability,
        # Toxicophores
        "pains":               pains_alerts,
        "brenk":               brenk_alerts,
        "surechembl":          surechembl_alerts,
        "toxicophore_summary": toxicophore_summary,
        # PubChem enrichment (fetched live; opt-out with fetch_pubchem=False)
        "pubchem_metadata":    get_pubchem_metadata(smiles) if fetch_pubchem
                               else {"found": False, "error": "PubChem lookup skipped (fetch_pubchem=False)."},
    }

    # Auto-generate the visual report if caller supplied a path
    if output_path:
        generate_adme_report(result, output_path=output_path, show=show)

    return result


# ============================================================
# 10. DATASET-LEVEL UTILITIES
# ============================================================

def get_pubchem_metadata_batch(smiles_list):
    """
    Fetch PubChem metadata for a list of SMILES using batch API calls.

    Instead of N × 10 sequential calls (the single-molecule path), this does:
      Phase 1 — resolve all SMILES → CIDs in parallel  (N calls, concurrent)
      Phase 2 — fetch ALL properties in ONE API call   (1 call)
      Phase 3 — fetch ALL synonyms in ONE API call     (1 call)
      Phase 4 — lightweight bioactivity count per CID  (N calls, concurrent)

    Returns a list of metadata dicts in the same order as smiles_list.
    """
    BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

    def _get(url, params=None, timeout=10):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r
        except Exception:
            pass
        return None

    def _json(url, params=None):
        r = _get(url, params=params)
        if r:
            try:
                return r.json()
            except Exception:
                pass
        return {}

    # ── Phase 1: Resolve SMILES → CID in parallel ────────────────────────────
    cid_map = {}   # smiles -> cid (or None)

    def _resolve_cid(smiles):
        data = _json(f"{BASE}/compound/smiles/cids/JSON", params={"smiles": smiles})
        cids = data.get("IdentifierList", {}).get("CID", [])
        return smiles, int(cids[0]) if cids else None

    with ThreadPoolExecutor(max_workers=8) as ex:
        for smiles, cid in ex.map(_resolve_cid, smiles_list):
            cid_map[smiles] = cid

    valid_cids   = [cid for cid in cid_map.values() if cid]
    cid_to_smiles = {cid: smi for smi, cid in cid_map.items() if cid}

    if not valid_cids:
        return [{"found": False, "error": "Compound not found in PubChem database."}
                for _ in smiles_list]

    cid_str = ",".join(str(c) for c in valid_cids)

    # ── Phase 2: Batch property fetch — ONE call for all CIDs ────────────────
    props_wanted = ("IUPACName,MolecularFormula,MolecularWeight,CanonicalSMILES,"
                    "InChI,InChIKey,Charge,XLogP,ExactMass,MonoisotopicMass")
    prop_url  = f"{BASE}/compound/cid/{cid_str}/property/{props_wanted}/JSON"
    prop_data = _json(prop_url).get("PropertyTable", {}).get("Properties", [])
    props_by_cid = {p["CID"]: p for p in prop_data}

    # ── Phase 3: Batch synonym fetch — ONE call for all CIDs ─────────────────
    syn_url  = f"{BASE}/compound/cid/{cid_str}/synonyms/JSON"
    syn_info = _json(syn_url).get("InformationList", {}).get("Information", [])
    syns_by_cid = {s["CID"]: s.get("Synonym", []) for s in syn_info}

    # ── Phase 4: Bioassay count per CID — parallel, lightweight ──────────────
    def _assay_count(cid):
        r = _get(f"{BASE}/compound/cid/{cid}/assaysummary/JSON")
        if not r:
            return cid, {"assay_count": 0, "active_count": 0,
                         "inactive_count": 0, "note": "No bioassay data."}
        try:
            table    = r.json().get("Table", {})
            cols     = [c.get("Name","") for c in table.get("Column", [])]
            rows     = table.get("Row", [])
            n        = len(rows)
            active   = inactive = 0
            if "Activity Outcome" in cols:
                idx      = cols.index("Activity Outcome")
                outcomes = [row.get("Cell",[])[idx] if len(row.get("Cell",[])) > idx
                            else "" for row in rows]
                active   = outcomes.count("Active")
                inactive = outcomes.count("Inactive")
            return cid, {"assay_count": n, "active_count": active,
                         "inactive_count": inactive,
                         "note": f"{n} bioassay record(s) found."}
        except Exception:
            return cid, {"assay_count": 0, "active_count": 0,
                         "inactive_count": 0, "note": "Parse error."}

    bioact_by_cid = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for cid, ba in ex.map(_assay_count, valid_cids):
            bioact_by_cid[cid] = ba

    # ── Assemble per-molecule results ─────────────────────────────────────────
    results = []
    for smiles in smiles_list:
        cid = cid_map.get(smiles)
        if not cid:
            results.append({"found": False,
                            "error": "Compound not found in PubChem database."})
            continue

        p    = props_by_cid.get(cid, {})
        syns = syns_by_cid.get(cid, [])
        results.append({
            "found":            True,
            "cid":              cid,
            "iupac_name":       p.get("IUPACName", ""),
            "common_name":      syns[0] if syns else p.get("IUPACName", ""),
            "canonical_smiles": p.get("CanonicalSMILES", ""),
            "formula":          p.get("MolecularFormula", ""),
            "molecular_weight": p.get("MolecularWeight"),
            "inchi":            p.get("InChI", ""),
            "inchikey":         p.get("InChIKey", ""),
            "charge":           p.get("Charge"),
            "xlogp":            p.get("XLogP"),
            "exact_mass":       p.get("ExactMass"),
            "monoisotopic_mass":p.get("MonoisotopicMass"),
            "synonyms":         syns[:10],
            "bioactivity":      bioact_by_cid.get(cid, {}),
            # Fields skipped in batch mode for speed:
            "targets":      [],
            "pharmacology": "",
            "drug_info":    {},
            "toxicity":     [],
            "pathways":     [],
            "literature":   [],
            "links": {
                "compound_page":  f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
                "structure_image":f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/PNG",
                "sdf_download":   f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/SDF",
                "json_api":       f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/JSON",
            },
            "_batch_mode": True,   # flag — detailed fields omitted for speed
        })
    return results


def analyze_multiple_smiles(smiles_list, output_dir=None, fetch_pubchem=False,
                             max_workers=6):
    """
    Analyse a list of SMILES strings **in parallel**.

    Parameters
    ----------
    smiles_list   : list of SMILES strings
    output_dir    : optional directory — if provided, a PNG report is saved
                    for each valid molecule as <output_dir>/adme_report_<i>.png
    fetch_pubchem : if True, enrich results with PubChem batch API (fast —
                    uses 2 batch calls + parallel CID resolution instead of
                    N×10 sequential calls). Defaults to False.
    max_workers   : thread pool size for parallel ADME analysis (default 6).

    Speed characteristics
    ---------------------
    - ADME analysis runs in parallel across all molecules simultaneously.
    - PubChem (when enabled) uses batch endpoints: ~3 s for any batch size
      instead of N × 10–30 s.
    """
    results = [None] * len(smiles_list)

    def _analyse_one(args):
        i, smiles = args
        out = os.path.join(output_dir, f"adme_report_{i}.png") if output_dir else None
        # Never fetch PubChem here — we do it in one batch call below
        return i, analyze_smiles(smiles, output_path=out, fetch_pubchem=False)

    # ── Parallel ADME analysis ────────────────────────────────────────────────
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_analyse_one, (i, s)): i
                   for i, s in enumerate(smiles_list)}
        for future in as_completed(futures):
            i, result = future.result()
            results[i] = result

    # ── Batch PubChem enrichment (optional) ───────────────────────────────────
    if fetch_pubchem:
        batch_meta = get_pubchem_metadata_batch(smiles_list)
        for i, meta in enumerate(batch_meta):
            if results[i] is not None:
                results[i]["pubchem_metadata"] = meta
    else:
        for r in results:
            if r is not None:
                r["pubchem_metadata"] = {"found": False,
                                          "error": "PubChem lookup skipped."}

    return results


def rank_molecules(results):
    valid = [r for r in results if r["valid"]]
    return sorted(valid, key=lambda x: x["score"], reverse=True)


def dataset_decision(results):
    decisions = [r["decision"] for r in results if r["valid"]]
    if all(d == "REJECT" for d in decisions): return "ALL_REJECTED"
    if all(d == "ACCEPT" for d in decisions): return "ALL_ACCEPTED"
    return "MIXED"


# ============================================================
# 11. INTERPRETATION & USER FEEDBACK
# ============================================================

def explain_imperfection(mol, is_top=False, tied=False):
    reasons = []

    if mol["violations"]:
        reasons.append("Violates one or more physicochemical drug-likeness rules")
    if mol["properties"]["logp"] > 4.5:
        reasons.append("Lipophilicity is close to the upper acceptable limit")
    if mol["properties"]["rotatable_bonds"] > 8:
        reasons.append("Molecular flexibility may reduce binding specificity")
    if mol["properties"]["tpsa"] > 120:
        reasons.append("High polarity may reduce membrane permeability")
    if mol.get("cyp450", {}).get("cyp_risk") == "HIGH":
        reasons.append("High CYP450 liability across multiple isoforms (metabolic risk)")
    if mol.get("herg", {}).get("herg_risk") == "HIGH":
        reasons.append("High hERG liability (cardiac toxicity risk)")
    if mol.get("mutagenicity", {}).get("mutagenicity_flag") == "POSITIVE":
        reasons.append(f"Ames mutagenicity alerts: {mol['mutagenicity']['alerts']}")
    if mol.get("reactive_alerts", {}).get("reactivity_flag") == "REACTIVE":
        reasons.append(f"Reactive functional groups: {mol['reactive_alerts']['reactive_groups']}")
    if mol.get("caco2_permeability", {}).get("caco2_class") == "LOW":
        reasons.append("Poor predicted Caco-2 permeability (limited oral absorption)")
    if mol.get("pgp_efflux", {}).get("pgp_risk") == "HIGH":
        reasons.append("High P-gp efflux risk (may limit bioavailability/CNS access)")
    if mol.get("plasma_protein_binding", {}).get("ppb_class") == "HIGH":
        ppb = mol["plasma_protein_binding"]["ppb_estimate_pct"]
        reasons.append(f"High plasma protein binding (~{ppb}%) limits free drug fraction")
    rm_risk = mol.get("reactive_metabolites", {}).get("reactive_metabolite_risk")
    if rm_risk in ("CRITICAL", "HIGH"):
        n = mol["reactive_metabolites"]["alert_count"]
        reasons.append(f"CRITICAL: {n} reactive metabolite precursor alert(s) — hepatotoxicity risk")
    if mol.get("gsh_trapping", {}).get("gsh_trapping_risk") == "POSITIVE":
        reasons.append("GSH trapping alerts detected — idiosyncratic hepatotoxicity risk")
    if mol.get("mbi_risk", {}).get("mbi_risk") in ("HIGH", "MODERATE"):
        reasons.append("Mechanism-based CYP inactivation risk — potential DDI liability")
    if mol.get("microsomal_stability", {}).get("stability_class") == "LOW":
        reasons.append("Low predicted microsomal stability (t½ < 30 min) — rapid hepatic clearance")
    if mol.get("pains", {}).get("pains_flag") == "FLAGGED":
        reasons.append(f"PAINS alerts: {mol['pains']['pains_alerts']}")
    if mol.get("brenk", {}).get("brenk_flag") in ("HIGH", "MODERATE"):
        reasons.append(f"Brenk alerts ({mol['brenk']['brenk_count']}): {mol['brenk']['brenk_alerts']}")
    if mol.get("surechembl", {}).get("regulatory_risk") in ("HIGH", "MODERATE"):
        reasons.append(f"SureChEMBL regulatory alerts: {mol['surechembl']['surechembl_alerts']}")

    if not reasons:
        if tied:
            reasons.append("Equally optimal by all evaluated drug-likeness criteria")
        elif is_top:
            reasons.append("Satisfies all evaluated drug-likeness rules — selected as top candidate")
        else:
            reasons.append("Optimal but ranks lower due to relative differences")

    return reasons


def generate_optimization_advice(mol):
    advice = []

    for v in mol["violations"]:
        advice.extend(v["suggestions"])
    if mol.get("cyp450", {}).get("cyp_risk") in ("HIGH", "MODERATE"):
        advice.append("Block metabolic soft spots via fluorination or deuterium substitution")
        advice.append("Run CYP inhibition panel (fluorescence or LC-MS) to confirm liability")
    if mol.get("herg", {}).get("herg_risk") == "HIGH":
        advice.append("Reduce basicity of nitrogen centers to lower hERG affinity")
        advice.append("Lower logP to < 3 to reduce hERG binding")
    if mol.get("mutagenicity", {}).get("mutagenicity_flag") == "POSITIVE":
        advice.append("Replace mutagenic alerts (e.g. nitroaromatics → sulfonamides)")
    if mol.get("reactive_alerts", {}).get("reactivity_flag") == "REACTIVE":
        advice.append("Replace reactive groups with non-reactive bioisosteres")
    if mol.get("caco2_permeability", {}).get("caco2_class") == "LOW":
        advice.append("Reduce TPSA < 90 Å² and MW < 400 Da to improve permeability")
    if mol.get("pgp_efflux", {}).get("pgp_risk") == "HIGH":
        advice.append("Reduce HBD ≤ 2 and MW to minimise P-gp recognition")
    if mol.get("plasma_protein_binding", {}).get("ppb_class") == "HIGH":
        advice.append("Lower logP to reduce albumin/AGP binding")
    rm_risk = mol.get("reactive_metabolites", {}).get("reactive_metabolite_risk")
    if rm_risk in ("CRITICAL", "HIGH"):
        advice.append("PRIORITY: Remove or block reactive metabolite precursor groups")
        advice.append("Replace furan → pyridine, thiophene → thiazole, catechol → fluorophenol")
        advice.append("Run GSH and KCN trapping assays before any in vivo work")
    if mol.get("gsh_trapping", {}).get("gsh_trapping_risk") == "POSITIVE":
        advice.append("Introduce blocking substituents ortho/para to GSH-reactive site")
    if mol.get("mbi_risk", {}).get("mbi_risk") in ("HIGH", "MODERATE"):
        advice.append("Replace MBI-prone groups (furans → thiophenes → pyridines)")
        advice.append("Run TDI IC50 shift assay and determine kinact/KI for DDI assessment")
    soft = mol.get("metabolic_soft_spots", {})
    if soft.get("clearance_prediction") in ("HIGH", "MODERATE"):
        advice.extend(soft.get("optimization_strategies", []))
    if mol.get("pains", {}).get("pains_flag") == "FLAGGED":
        advice.append("Counter-screen PAINS hits with orthogonal assays (SPR, NMR, thermal shift)")
        advice.append("Remove PAINS-flagged substructure if not essential to pharmacophore")
    if mol.get("brenk", {}).get("brenk_flag") in ("HIGH", "MODERATE"):
        advice.append("Address Brenk alerts — prioritise removal of reactive/genotoxic groups")
    if mol.get("surechembl", {}).get("regulatory_risk") in ("HIGH", "MODERATE"):
        advice.append("Consult ICH S2(R1) genotoxicity battery before clinical submission")
        advice.append("Substitute genotoxic fragment with non-alerting bioisostere")

    return list(set(advice))


# ============================================================
# 12. VISUALISATION — colour palette & helpers
# ============================================================

BG        = "#0D1117"
PANEL     = "#161B22"
BORDER    = "#30363D"
TEXT_MAIN = "#E6EDF3"
TEXT_DIM  = "#8B949E"
GREEN     = "#3FB950"
YELLOW    = "#D29922"
ORANGE    = "#F97316"
RED       = "#F85149"
BLUE      = "#58A6FF"
PURPLE    = "#BC8CFF"

RISK_COLORS = {
    "LOW": GREEN, "MODERATE": YELLOW, "HIGH": ORANGE, "CRITICAL": RED,
    "CLEAN": GREEN, "FLAGGED": ORANGE,
    "POSITIVE": RED, "NEGATIVE": GREEN,
    "OPTIMAL": GREEN, "NEAR_OPTIMAL": YELLOW, "WEAK": ORANGE, "POOR": RED,
    "ACCEPT": GREEN, "REJECT": RED, "PASS": GREEN,
}


def _rc(value):
    return RISK_COLORS.get(str(value).upper(), TEXT_DIM)


def _panel(ax):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
        spine.set_linewidth(0.8)


def _title(ax, text, fontsize=9):
    ax.set_title(text, color=TEXT_DIM, fontsize=fontsize,
                 loc="left", fontweight="bold", pad=6)


# ============================================================
# 13. VISUALISATION — individual panel drawers
# ============================================================

def _draw_score_gauge(ax, score, status, decision):
    ax.set_facecolor(PANEL)
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-0.3, 1.4)
    ax.axis("off")

    theta   = np.linspace(math.pi, 0, 200)
    r_o, r_i = 1.2, 0.8
    ax.fill_between(r_o * np.cos(theta), r_o * np.sin(theta),
                    r_i * np.cos(theta), r_i * np.sin(theta),
                    color=BORDER, alpha=0.4)

    frac    = score / 100
    cmap    = LinearSegmentedColormap.from_list("g", [RED, YELLOW, GREEN])
    color   = cmap(frac)
    theta_s = np.linspace(math.pi, math.pi - frac * math.pi, 200)
    ax.fill_between(r_o * np.cos(theta_s), r_o * np.sin(theta_s),
                    r_i * np.cos(theta_s), r_i * np.sin(theta_s),
                    color=color, alpha=0.9)

    ax.text(0,  0.35, f"{score:.0f}",       ha="center", va="center", color=color,      fontsize=26, fontweight="bold")
    ax.text(0,  0.10, "/ 100",              ha="center", va="center", color=TEXT_DIM,   fontsize=9)
    ax.text(0, -0.05, status,               ha="center", va="center", color=_rc(status),fontsize=9,  fontweight="bold")
    ax.text(0, -0.20, f"Decision: {decision}", ha="center", va="center", color=_rc(decision), fontsize=8)
    _title(ax, "Drug-likeness Score", fontsize=8)


def _draw_radar(ax, features):
    labels = ["MW/500", "LogP/5", "HBD/5", "HBA/10", "TPSA/140", "RotB/10"]
    limits = [500, 5, 5, 10, 140, 10]
    vals   = [features[k] for k in ("molecular_weight","logp","hbd","hba","tpsa","rotatable_bonds")]
    norm   = [min(v / l, 1.5) for v, l in zip(vals, limits)]
    N      = len(labels)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    norm  += norm[:1]; angles += angles[:1]

    ax.set_facecolor(PANEL)
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color=TEXT_DIM, fontsize=7)
    ax.set_ylim(0, 1.5)
    ax.set_yticks([0.5, 1.0, 1.5])
    ax.set_yticklabels(["0.5×", "1×\n(limit)", "1.5×"], color=TEXT_DIM, fontsize=6)
    ax.tick_params(colors=BORDER)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.fill(angles, norm, alpha=0.3, color=BLUE)
    ax.plot(angles, norm, color=BLUE, linewidth=1.5)
    ax.plot(angles, [1.0] * (N + 1), color=YELLOW, linewidth=0.8, linestyle="--", alpha=0.6)
    _title(ax, "Physicochemical Profile", fontsize=8)


def _draw_risk_matrix(ax, result):
    ax.set_facecolor(PANEL)
    ax.axis("off")
    rows = [
        ("CYP450 Risk",            result.get("cyp450",{}).get("cyp_risk","N/A")),
        ("hERG Risk",              result.get("herg",{}).get("herg_risk","N/A")),
        ("Mutagenicity (Ames)",    result.get("mutagenicity",{}).get("mutagenicity_flag","N/A")),
        ("Reactive Groups",        result.get("reactive_alerts",{}).get("reactivity_flag","N/A")),
        ("Plasma Protein Binding", result.get("plasma_protein_binding",{}).get("ppb_class","N/A")),
        ("Caco-2 Permeability",    result.get("caco2_permeability",{}).get("caco2_class","N/A")),
        ("P-gp Efflux",            result.get("pgp_efflux",{}).get("pgp_risk","N/A")),
        ("Reactive Metabolites",   result.get("reactive_metabolites",{}).get("reactive_metabolite_risk","N/A")),
        ("GSH Trapping",           result.get("gsh_trapping",{}).get("gsh_trapping_risk","N/A")),
        ("MBI Risk",               result.get("mbi_risk",{}).get("mbi_risk","N/A")),
        ("Microsomal Stability",   result.get("microsomal_stability",{}).get("stability_class","N/A")),
        ("PAINS",                  result.get("pains",{}).get("pains_flag","N/A")),
        ("Brenk Alerts",           result.get("brenk",{}).get("brenk_flag","N/A")),
        ("SureChEMBL",             result.get("surechembl",{}).get("regulatory_risk","N/A")),
    ]
    per_c = math.ceil(len(rows) / 2)
    for i, (label, val) in enumerate(rows):
        col = i // per_c
        row = i % per_c
        x   = col * 0.52
        y   = 1 - (row + 1) * (1 / per_c)
        color = _rc(val)
        ax.add_patch(FancyBboxPatch(
            (x, y + 0.01), 0.48, 0.055,
            boxstyle="round,pad=0.005",
            facecolor=color + "22", edgecolor=color,
            linewidth=0.8, transform=ax.transAxes,
        ))
        ax.text(x + 0.01, y + 0.038, label,
                transform=ax.transAxes, color=TEXT_MAIN, fontsize=6.5, va="center")
        ax.text(x + 0.47, y + 0.038, str(val),
                transform=ax.transAxes, color=color, fontsize=6.5,
                va="center", ha="right", fontweight="bold")
    _title(ax, "Risk Matrix — All ADME Modules", fontsize=8)


def _draw_alert_bars(ax, result):
    _panel(ax)
    cats   = ["PAINS","Brenk","SureChEMBL","Ames","Reactive\nGroups",
              "React.\nMetabolites","GSH Trap","MBI","Soft Spots"]
    counts = [
        result.get("pains",{}).get("pains_count",0),
        result.get("brenk",{}).get("brenk_count",0),
        result.get("surechembl",{}).get("surechembl_count",0),
        result.get("mutagenicity",{}).get("alert_count",0),
        result.get("reactive_alerts",{}).get("reactive_count",0),
        result.get("reactive_metabolites",{}).get("alert_count",0),
        result.get("gsh_trapping",{}).get("gsh_alert_count",0),
        result.get("mbi_risk",{}).get("mbi_count",0),
        result.get("metabolic_soft_spots",{}).get("soft_spot_count",0),
    ]
    colors = [RED if c >= 3 else ORANGE if c >= 1 else GREEN for c in counts]
    bars   = ax.barh(np.arange(len(cats)), counts, color=colors, alpha=0.85, height=0.6)
    ax.set_yticks(np.arange(len(cats)))
    ax.set_yticklabels(cats, color=TEXT_MAIN, fontsize=7)
    ax.set_xlabel("Alert count", color=TEXT_DIM, fontsize=7)
    ax.tick_params(axis="x", colors=TEXT_DIM, labelsize=7)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                    str(count), va="center", color=TEXT_MAIN, fontsize=7)
    _title(ax, "Structural Alert Counts by Filter Category", fontsize=8)


def _draw_stability_bars(ax, result):
    _panel(ax)
    stab = result.get("microsomal_stability", {})
    if not stab:
        ax.text(0.5, 0.5, "No stability data", ha="center", va="center",
                color=TEXT_DIM, transform=ax.transAxes)
        return

    score    = stab.get("microsomal_stability_score", 0)
    lp_pen   = stab.get("logp_penalty", 0)
    ring_pen = stab.get("aromatic_ring_penalty", 0)
    spot_pen = stab.get("soft_spot_penalty", 0)

    cats   = ["Base\nScore","LogP\nPenalty","Aromatic\nRings","Soft\nSpots","Final\nScore"]
    values = [100, -lp_pen, -ring_pen, -spot_pen, score]
    colors = [BLUE, RED, ORANGE, YELLOW,
              GREEN if score >= 70 else YELLOW if score >= 40 else RED]

    bars = ax.bar(cats, [abs(v) for v in values], color=colors, alpha=0.85, width=0.55)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{'−' if val < 0 else ''}{abs(val):.1f}",
                ha="center", color=TEXT_MAIN, fontsize=7.5, fontweight="bold")

    ax.set_ylabel("Score", color=TEXT_DIM, fontsize=7)
    ax.set_ylim(0, 115)
    ax.tick_params(axis="x", colors=TEXT_MAIN, labelsize=7)
    ax.tick_params(axis="y", colors=TEXT_DIM, labelsize=7)
    cls = stab.get("stability_class","")
    t12 = stab.get("predicted_t_half","")
    ax.text(0.98, 0.95, f"{cls}  |  t½ {t12}",
            transform=ax.transAxes, ha="right", va="top",
            color=_rc(cls), fontsize=7.5, fontweight="bold")
    _title(ax, "Microsomal Stability Score Breakdown", fontsize=8)


def _draw_cyp_bars(ax, result):
    _panel(ax)
    cyp     = result.get("cyp450", {})
    flagged = cyp.get("flagged_isoforms", {})
    isos    = ["CYP3A4","CYP2D6","CYP2C9","CYP1A2","CYP2C19"]
    counts  = [len(flagged.get(iso, [])) for iso in isos]
    colors  = [RED if c >= 2 else ORANGE if c == 1 else GREEN for c in counts]

    bars = ax.bar(isos, counts, color=colors, alpha=0.85, width=0.55)
    ax.set_ylabel("Alert count", color=TEXT_DIM, fontsize=7)
    ax.set_ylim(0, max(counts + [3]) + 1)
    ax.tick_params(axis="x", colors=TEXT_MAIN, labelsize=7)
    ax.tick_params(axis="y", colors=TEXT_DIM, labelsize=7)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                str(count), ha="center", color=TEXT_MAIN, fontsize=7.5, fontweight="bold")
    risk = cyp.get("cyp_risk","N/A")
    ax.text(0.98, 0.95, f"Overall CYP Risk: {risk}",
            transform=ax.transAxes, ha="right", va="top",
            color=_rc(risk), fontsize=7.5, fontweight="bold")
    _title(ax, "CYP450 Isoform Structural Alerts", fontsize=8)


def _draw_toxicophore_donut(ax, result):
    summary   = result.get("toxicophore_summary", {})
    breakdown = summary.get("breakdown", {})
    labels    = list(breakdown.keys())
    values    = [breakdown[k] for k in labels]
    total     = sum(values)

    if total == 0:
        pie_vals   = [1] * len(labels)
        pie_colors = [GREEN] * len(labels)
        center_txt  = "CLEAN"
        center_col  = GREEN
    else:
        pie_vals   = values
        pie_colors = [RED if v >= 3 else ORANGE if v >= 1 else BORDER for v in values]
        center_txt  = summary.get("overall_toxicophore_risk","N/A")
        center_col  = _rc(center_txt)

    ax.set_facecolor(PANEL)
    ax.pie(pie_vals, colors=pie_colors, startangle=90,
           wedgeprops={"width": 0.55, "edgecolor": PANEL, "linewidth": 1.5})
    ax.text(0,  0.1,  center_txt,    ha="center", va="center", color=center_col, fontsize=10, fontweight="bold")
    ax.text(0, -0.18, f"{total} alerts", ha="center", va="center", color=TEXT_DIM, fontsize=7)
    ax.legend(
        handles=[mpatches.Patch(color=c, label=f"{l} ({v})")
                 for l, v, c in zip(labels, breakdown.values(), pie_colors)],
        loc="lower center", bbox_to_anchor=(0.5, -0.45), ncol=3,
        fontsize=5.5, frameon=False, labelcolor=TEXT_MAIN,
    )
    _title(ax, "Toxicophore Alert Distribution", fontsize=8)


def _draw_property_table(ax, features, result):
    ax.set_facecolor(PANEL)
    ax.axis("off")

    headers   = ["Property", "Value", "Limit", "Status"]
    rows_data = [
        ("Mol. Weight (Da)", f"{features['molecular_weight']:.1f}", "≤ 500",
         "OK" if features["molecular_weight"] <= 500 else "FAIL"),
        ("LogP",             f"{features['logp']:.2f}",             "≤ 5",
         "OK" if features["logp"] <= 5 else "FAIL"),
        ("HBD",              f"{features['hbd']}",                  "≤ 5",
         "OK" if features["hbd"] <= 5 else "FAIL"),
        ("HBA",              f"{features['hba']}",                  "≤ 10",
         "OK" if features["hba"] <= 10 else "FAIL"),
        ("TPSA (Å²)",        f"{features['tpsa']:.1f}",             "≤ 140",
         "OK" if features["tpsa"] <= 140 else "FAIL"),
        ("Rotatable Bonds",  f"{features['rotatable_bonds']}",      "≤ 10",
         "OK" if features["rotatable_bonds"] <= 10 else "FAIL"),
        ("PPB (%)",
         f"{result.get('plasma_protein_binding',{}).get('ppb_estimate_pct','N/A')}",
         "< 90", result.get("plasma_protein_binding",{}).get("ppb_class","N/A")),
        ("Caco-2 (nm/s)",
         f"{result.get('caco2_permeability',{}).get('caco2_papp_nm_s','N/A')}",
         "> 20 high", result.get("caco2_permeability",{}).get("caco2_class","N/A")),
        ("Microsomal Stab.",
         f"{result.get('microsomal_stability',{}).get('microsomal_stability_score','N/A')}",
         "≥ 70 high", result.get("microsomal_stability",{}).get("stability_class","N/A")),
    ]

    col_x    = [0.00, 0.38, 0.60, 0.80]
    row_h    = 0.095
    header_y = 0.96

    for cx, h in zip(col_x, headers):
        ax.text(cx, header_y, h, transform=ax.transAxes,
                color=TEXT_DIM, fontsize=7, fontweight="bold", va="top")
    ax.plot([0, 1], [header_y - 0.02, header_y - 0.02],
            color=BORDER, linewidth=0.8, transform=ax.transAxes, clip_on=False)

    for i, (prop, val, lim, status) in enumerate(rows_data):
        y = header_y - 0.04 - (i + 1) * row_h
        if i % 2 == 0:
            ax.add_patch(FancyBboxPatch(
                (0, y - 0.02), 1, row_h * 0.95,
                boxstyle="square,pad=0", facecolor="#1C2128", edgecolor="none",
                transform=ax.transAxes, clip_on=False,
            ))
        ax.text(col_x[0], y + 0.02, prop,   transform=ax.transAxes, color=TEXT_MAIN, fontsize=6.5, va="center")
        ax.text(col_x[1], y + 0.02, val,    transform=ax.transAxes, color=BLUE,      fontsize=6.5, va="center")
        ax.text(col_x[2], y + 0.02, lim,    transform=ax.transAxes, color=TEXT_DIM,  fontsize=6.5, va="center")
        sc = _rc(status) if status not in ("OK","FAIL") else (GREEN if status == "OK" else RED)
        ax.text(col_x[3], y + 0.02, status, transform=ax.transAxes, color=sc,        fontsize=6.5, va="center",
                fontweight="bold")
    _title(ax, "Property Table vs Thresholds", fontsize=8)


def _draw_advice(ax, advice_list):
    ax.set_facecolor(PANEL)
    ax.axis("off")
    ax.text(0, 1.0, "Optimisation Recommendations",
            transform=ax.transAxes, color=TEXT_DIM, fontsize=8, fontweight="bold", va="top")
    if not advice_list:
        ax.text(0, 0.88, "✓  No optimisation required — all filters passed.",
                transform=ax.transAxes, color=GREEN, fontsize=7.5, va="top")
        return
    y = 0.90
    for advice in advice_list[:12]:
        marker = "⚠ " if advice.startswith("PRIORITY") else "▸ "
        color  = ORANGE if advice.startswith("PRIORITY") else TEXT_MAIN
        ax.text(0.01, y, f"{marker}{advice}",
                transform=ax.transAxes, color=color, fontsize=6.3, va="top")
        y -= 0.075
        if y < 0:
            break


# ============================================================
# 14. MASTER REPORT GENERATOR
# ============================================================


def _make_fig(w, h, title_left="", title_right=""):
    """Create a styled blank figure."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.facecolor": BG, "axes.facecolor": PANEL,
        "text.color": TEXT_MAIN, "axes.labelcolor": TEXT_DIM,
        "xtick.color": TEXT_DIM, "ytick.color": TEXT_DIM,
        "grid.color": BORDER, "grid.alpha": 0.3,
    })
    fig = plt.figure(figsize=(w, h), facecolor=BG)
    fig.patch.set_facecolor(BG)
    if title_left:
        fig.text(0.02, 0.97, title_left, color=TEXT_MAIN, fontsize=12,
                 fontweight="bold", va="top")
    if title_right:
        fig.text(0.98, 0.012, title_right, color=TEXT_DIM, fontsize=6,
                 ha="right", va="bottom", style="italic")
    return fig


def build_adme_panels(result):
    """
    Returns 6 separate high-resolution matplotlib figures instead of one
    cramped combined figure.  Each can be displayed individually in Streamlit
    for much better readability.

    Keys returned
    -------------
    "overview"    : Score gauge + physicochemical radar
    "risk_matrix" : 14-row ADME risk badge grid
    "alerts"      : Structural alert counts + microsomal stability breakdown
    "cyp_tox"     : CYP450 isoform bars + toxicophore donut
    "properties"  : Property table vs thresholds
    "advice"      : Optimisation recommendations

    Returns {} if result is invalid.
    """
    if not result.get("valid"):
        return {}

    footer = "Rule-based ADME screening | Not a substitute for experimental measurement"
    features = result["properties"]
    panels = {}

    # ── 1. Overview: gauge + radar ────────────────────────────────────────────
    fig = _make_fig(14, 6, "Overview", footer)
    ax_g = fig.add_axes([0.03, 0.08, 0.30, 0.82])
    ax_r = fig.add_axes([0.38, 0.08, 0.58, 0.82], polar=True)
    _draw_score_gauge(ax_g, result["score"], result["physchem_status"], result["decision"])
    _draw_radar(ax_r, features)
    panels["overview"] = fig

    # ── 2. Risk matrix ────────────────────────────────────────────────────────
    fig = _make_fig(14, 7, "ADME Risk Matrix", footer)
    ax = fig.add_axes([0.01, 0.04, 0.98, 0.88])
    _draw_risk_matrix(ax, result)
    panels["risk_matrix"] = fig

    # ── 3. Alerts + stability ─────────────────────────────────────────────────
    fig = _make_fig(16, 6, "Structural Alerts & Metabolic Stability", footer)
    ax_a = fig.add_axes([0.04, 0.10, 0.44, 0.82])
    ax_s = fig.add_axes([0.55, 0.10, 0.42, 0.82])
    _draw_alert_bars(ax_a, result)
    _draw_stability_bars(ax_s, result)
    panels["alerts"] = fig

    # ── 4. CYP + toxicophore donut ────────────────────────────────────────────
    fig = _make_fig(16, 6, "CYP450 Liability & Toxicophore Distribution", footer)
    ax_c = fig.add_axes([0.04, 0.10, 0.52, 0.82])
    ax_d = fig.add_axes([0.62, 0.08, 0.35, 0.84])
    _draw_cyp_bars(ax_c, result)
    _draw_toxicophore_donut(ax_d, result)
    panels["cyp_tox"] = fig

    # ── 5. Property table ─────────────────────────────────────────────────────
    fig = _make_fig(14, 6, "Property Table vs Thresholds", footer)
    ax = fig.add_axes([0.02, 0.04, 0.96, 0.88])
    _draw_property_table(ax, features, result)
    panels["properties"] = fig

    # ── 6. Advice ─────────────────────────────────────────────────────────────
    fig = _make_fig(14, 5, "Optimisation Recommendations", footer)
    ax = fig.add_axes([0.02, 0.04, 0.96, 0.88])
    _draw_advice(ax, generate_optimization_advice(result))
    panels["advice"] = fig

    return panels


def build_adme_figure(result):
    """
    Builds and returns the matplotlib Figure WITHOUT saving or closing it.
    Use this inside Streamlit with st.pyplot(fig).

    Parameters
    ----------
    result : dict returned by analyze_smiles()

    Returns
    -------
    matplotlib.figure.Figure, or None if result is invalid.

    Streamlit example
    -----------------
        import streamlit as st
        from adme_analysis import analyze_smiles, build_adme_figure

        result = analyze_smiles(smiles_input)
        fig = build_adme_figure(result)
        if fig:
            st.pyplot(fig)
    """
    if not result.get("valid"):
        return None

    plt.rcParams.update({
        "font.family":      "DejaVu Sans",
        "figure.facecolor": BG,
        "axes.facecolor":   PANEL,
        "text.color":       TEXT_MAIN,
        "axes.labelcolor":  TEXT_DIM,
        "xtick.color":      TEXT_DIM,
        "ytick.color":      TEXT_DIM,
        "grid.color":       BORDER,
        "grid.alpha":       0.3,
    })

    fig = plt.figure(figsize=(20, 16), facecolor=BG)
    fig.patch.set_facecolor(BG)

    fig.text(0.04, 0.975, "ADME Analysis Report",
             color=TEXT_MAIN, fontsize=18, fontweight="bold", va="top")
    fig.text(0.04, 0.957,
             f"SMILES: {result['smiles']}    |    "
             f"Lipinski: {'PASS' if result['lipinski']['passes'] else 'FAIL'}    |    "
             f"Decision: {result['decision']}",
             color=TEXT_DIM, fontsize=9, va="top")

    gs = gridspec.GridSpec(4, 4, figure=fig,
                           top=0.94, bottom=0.04, left=0.04, right=0.97,
                           hspace=0.48, wspace=0.38)

    ax_gauge  = fig.add_subplot(gs[0, 0])
    ax_radar  = fig.add_subplot(gs[0, 1], polar=True)
    ax_risk   = fig.add_subplot(gs[0, 2:4])
    ax_alert  = fig.add_subplot(gs[1, 0:2])
    ax_stab   = fig.add_subplot(gs[1, 2:4])
    ax_cyp    = fig.add_subplot(gs[2, 0:2])
    ax_donut  = fig.add_subplot(gs[2, 2])
    ax_table  = fig.add_subplot(gs[2, 3])
    ax_advice = fig.add_subplot(gs[3, :])

    features = result["properties"]
    _draw_score_gauge(ax_gauge, result["score"], result["physchem_status"], result["decision"])
    _draw_radar(ax_radar, features)
    _draw_risk_matrix(ax_risk, result)
    _draw_alert_bars(ax_alert, result)
    _draw_stability_bars(ax_stab, result)
    _draw_cyp_bars(ax_cyp, result)
    _draw_toxicophore_donut(ax_donut, result)
    _draw_property_table(ax_table, features, result)
    _draw_advice(ax_advice, generate_optimization_advice(result))

    fig.text(0.97, 0.012,
             "Rule-based ADME screening | Not a substitute for experimental measurement",
             color=TEXT_DIM, fontsize=6.5, ha="right", va="bottom", style="italic")

    return fig


def generate_adme_report(result, output_path="adme_report.png", show=False):
    """
    Generates and saves a 9-panel ADME visual report as a PNG.

    Parameters
    ----------
    result      : dict returned by analyze_smiles()
    output_path : file path for the saved PNG (default: "adme_report.png")
    show        : if True, call plt.show() — requires a display / GUI

    Returns
    -------
    output_path on success, None if the result is invalid.
    """
    fig = build_adme_figure(result)
    if fig is None:
        print(f"[ERROR] Invalid SMILES — cannot generate report for: {result.get('smiles')}")
        return None

    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    print(f"[OK] Report saved → {output_path}")

    if show:
        plt.show()
    plt.close(fig)
    return output_path


# ============================================================
# ENTRY POINT  — run directly to see demo reports
# ============================================================

if __name__ == "__main__":
    demo_molecules = {
        "Aspirin":      "CC(=O)Oc1ccccc1C(=O)O",
        "CCl4":         "ClC(Cl)(Cl)Cl",
        "Ibuprofen":    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "Atorvastatin": "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CCC(O)CC(O)CC(=O)O",
    }

    for name, smiles in demo_molecules.items():
        print(f"\nAnalysing: {name}  ({smiles})")
        result = analyze_smiles(smiles, output_path=f"adme_report_{name.lower()}.png")
        if not result["valid"]:
            print("  [SKIP] Invalid SMILES")