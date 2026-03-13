import math
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

# ============================================================
# 1. RULE DEFINITIONS (Baselines & Domain Knowledge)
# ============================================================

LIPINSKI_RULES = {
    "molecular_weight": 500,
    "logp": 5,
    "hbd": 5,
    "hba": 10,
}

EXTENDED_ADMET_RULES = {
    "tpsa": 140,
    "rotatable_bonds": 10
}

ALL_RULES = {**LIPINSKI_RULES, **EXTENDED_ADMET_RULES}

VIOLATION_MEANINGS = {
    "molecular_weight": {"category": "size",          "issue": "Too large for oral absorption"},
    "logp":             {"category": "lipophilicity",  "issue": "Poor solubility risk"},
    "hbd":              {"category": "polarity",       "issue": "Too many H-bond donors"},
    "hba":              {"category": "polarity",       "issue": "Too many H-bond acceptors"},
    "tpsa":             {"category": "polarity",       "issue": "Poor membrane permeability"},
    "rotatable_bonds":  {"category": "flexibility",    "issue": "Excessive flexibility"},
}

SUGGESTION_MAP = {
    "size":          ["Remove bulky groups", "Simplify scaffold"],
    "lipophilicity": ["Add polar groups", "Reduce alkyl chains"],
    "polarity":      ["Reduce donors/acceptors", "Use bioisosteres"],
    "flexibility":   ["Reduce rotatable bonds", "Add rings"],
}

# ============================================================
# 1b. EXTENDED ADME RULE DEFINITIONS
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
        ("[NH2,NH;!$(NC=O);!$(NS=O)]c1ccccc1",               "Aromatic amine near aryl ring"),
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
        ("c1cncc(c1)",              "Substituted pyridine"),
        ("[CH2;!R](c1ccccc1)C=O",  "Benzyl carbonyl"),
        ("c1ncc[nH]1",             "Imidazole"),
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
    ("C1CO1",                      "Epoxide (alkylating agent)"),
    ("C(=O)Cl",                    "Acid chloride (acylating agent)"),
    ("[CH2]=[CH]C(=O)",            "alpha-beta-unsaturated carbonyl (Michael acceptor)"),
    ("C(=C)C(=O)[#7,#8]",         "Acrylamide / acrylate"),
    ("N=C=O",                      "Isocyanate"),
    ("N=C=S",                      "Isothiocyanate"),
    ("C(=O)OC(=O)",               "Anhydride"),
    ("O=S(=O)(O)Cl",              "Sulfonyl chloride"),
    ("[SH]",                       "Free thiol"),
    ("[OH]C=C",                    "Enol"),
    ("[N;!$(NC=O)]-O",            "Hydroxylamine"),
    ("C(=O)-O-N",                  "NHS ester"),
    ("[C;!R]=C-[C;!R]=O",         "Conjugated Michael acceptor"),
]

CACO2_THRESHOLDS = {"high": 20, "medium": 2}
PPB_THRESHOLDS   = {"high": 90, "moderate": 70}

# ============================================================
# 1c. METABOLIC STABILITY — NEW (MOST CRITICAL)
# ============================================================

# Reactive Metabolite Precursor SMARTS
# Structural features known to be bioactivated by CYPs or other enzymes
# into chemically reactive, potentially toxic species.
# References:
#   Kalgutkar & Didiuk (2009) Chem Biodiversity 6:2115
#   Stepan et al. (2011) J. Med. Chem. 54:7772
#   Thompson et al. (2016) Drug Metab. Dispos. 44:1790
REACTIVE_METABOLITE_SMARTS = [
    # --- Quinone / quinone imine precursors ---
    ("Oc1ccc(O)cc1",              "Catechol → ortho-quinone (GSH trapping, protein adducts)"),
    ("Oc1ccc(N)cc1",              "4-aminophenol → quinone imine (paracetamol-type toxicity)"),
    ("Oc1cccc(N)c1",              "3-aminophenol → quinone imine"),
    ("Nc1ccc(O)cc1",              "Aminophenol → iminoquinone (idiosyncratic hepatotoxin)"),
    ("Oc1ccc(O)c(O)c1",          "Pyrogallol-type → ortho-quinone"),
    ("O=C1C=CC(=O)C=C1",         "Para-quinone (direct electrophile)"),
    # --- Furans (epoxide / cis-enedione) ---
    ("c1ccoc1",                   "Furan → cis-2-enedione (hepatotoxin — CCl4-like mechanism)"),
    ("C1=COC=C1",                 "Dihydrofuran → epoxide intermediate"),
    ("[CH2]c1ccoc1",              "Alkylfuran (bioactivated by CYP-mediated epoxidation)"),
    # --- Thiophenes (sulfoxide / thiolactone) ---
    ("c1ccsc1",                   "Thiophene → S-oxide / thiolactone (CYP-mediated, GSH trapping)"),
    ("c1csc(N)c1",                "2-aminothiophene → reactive sulfoxide"),
    # --- Polyhalogenated carbons (radical formation) ---
    ("[CX4]([Cl,Br,F])([Cl,Br,F])[Cl,Br,F]",
                                  "Polyhalogenated carbon → carbon radical (liver necrosis, CCl4 mechanism)"),
    ("[CX4]([Cl])([Cl])Cl",      "Trichloromethyl group → CCl3• radical (direct hepatotoxin)"),
    ("C(Cl)(Cl)Cl",              "CHCl3-type → trichloromethanol radical"),
    # --- Terminal alkynes (CYP mechanism-based inactivation) ---
    ("C#CH",                      "Terminal alkyne → ketene (irreversible CYP inactivation)"),
    ("[CX2]#[CX2]",              "Internal alkyne (also a MBI risk)"),
    # --- Methylenedioxy phenyl (carbene, CYP inactivation) ---
    ("c1cc2c(cc1)OCO2",          "Methylenedioxyphenyl → carbene (CYP mechanism-based inactivation)"),
    # --- Anilines and nitroso metabolites ---
    ("[NH2]c1ccccc1",            "Aniline → nitroso / hydroxylamine (GSH adducts, Met-Hb)"),
    ("[NH2]c1ccc([N+](=O)[O-])cc1",
                                  "Nitroaniline → reactive nitroso species"),
    # --- Hydrazines ---
    ("NN",                        "Hydrazine → acyl nitroso / diazonium (DNA alkylation)"),
    # --- Epoxidizable aromatic rings ---
    ("c1ccc2ccccc2c1",           "Polycyclic aromatic → arene oxide / diol epoxide (DNA adduct)"),
    # --- Alpha-halo carbonyls ---
    ("[CX3](=O)[CX4][Cl,Br,I]", "Alpha-haloketone → alkylating species"),
    # --- Acyl glucuronides precursors ---
    ("C(=O)[OH]",                "Carboxylic acid → acyl glucuronide (protein acylation)"),
]

# GSH (Glutathione) Trapping Alerts
# GSH adduct formation is a key marker of reactive metabolite generation.
# Positive GSH trapping in vitro is a major flag in pharmaceutical development.
GSH_TRAP_SMARTS = [
    ("[CH2]=[CH]C(=O)",          "Michael acceptor (direct GSH adduct)"),
    ("C(=O)Cl",                  "Acyl chloride (acylates GSH)"),
    ("C1CO1",                    "Epoxide (GSH conjugation)"),
    ("O=C1C=CC(=O)C=C1",        "Quinone (1,4-addition with GSH)"),
    ("Oc1ccc(O)cc1",             "Catechol → quinone → GSH adduct"),
    ("c1ccoc1",                  "Furan → enedione → GSH adduct"),
    ("[N+](=O)[O-]",             "Nitro → nitroso → sulfinyl adducts"),
    ("c1ccsc1",                  "Thiophene → S-oxide → GSH adduct"),
    ("[CX4]([Cl,Br])([Cl,Br])[Cl,Br]",
                                 "Polyhalogenated → radical → GSH adduct (via oxidative stress)"),
]

# Metabolic Soft Spots
# Structural features preferentially oxidized by CYP enzymes,
# leading to rapid clearance or toxic metabolite generation.
SOFT_SPOT_SMARTS = [
    ("[CH3]c1ccccc1",            "Benzylic methyl (O-dealkylation / benzylic hydroxylation)"),
    ("[CH2]c1ccccc1",            "Benzylic CH2 (rapid CYP oxidation)"),
    ("[CH3][O,N;!$(NC=O)]",     "N-methyl / O-methyl (N/O-dealkylation)"),
    ("C(C)(C)c1ccccc1",         "Tert-butyl aryl (hydroxylation at benzylic position)"),
    ("[CX4H2][NH,NH2]",         "Alpha-carbon to amine (amine oxidation / carbinolamine)"),
    ("c1ccccc1[OH]",             "Phenol (glucuronidation / sulfation soft spot)"),
    ("[CX4][F]",                 "Aliphatic C-F (defluorination in some CYPs)"),
    ("c1ccc(OC)cc1",             "Methoxyphenyl (O-demethylation → phenol → catechol cascade)"),
    ("[CH2][CH2][CH2][CH3]",    "Omega-terminal carbon (omega-oxidation)"),
]

# Mechanism-Based Inactivation (MBI) — CYP irreversible inhibition
# Compounds that covalently bind to CYP active sites after activation.
# DDI risk: may cause clinically significant drug-drug interactions.
MBI_SMARTS = [
    ("c1cc2c(cc1)OCO2",          "Methylenedioxy → carbene (prototypical MBI — safrole, paroxetine)"),
    ("C#CH",                     "Terminal alkyne → ketene (MBI of CYP3A4/2B6)"),
    ("c1ccsc1",                  "Thiophene → S-oxide (clopidogrel-type MBI)"),
    ("c1ccoc1",                  "Furan → cis-enedione (MBI via Michael addition)"),
    ("[NX3]c1ccccc1",            "Aryl amine → nitroso (MBI by nitroso binding to heme Fe)"),
    ("[CX4]([Cl])([Cl])Cl",     "CCl3 type → CCl3• radical (CYP2E1 inactivation)"),
]

# ============================================================
# 1d. TOXICOPHORE / STRUCTURAL ALERTS — EXPANDED (NEW)
# ============================================================

# PAINS (Pan-Assay INterference compoundS) filters
# Based on Baell & Holloway (2010) J. Med. Chem. 53:2719
# These cause frequent false positives in HTS biochemical assays.
PAINS_SMARTS = [
    # Rhodanines
    ("O=C1NC(=S)SC1",            "Rhodanine (PAINS — redox cycling / aggregation)"),
    ("O=C1NC(=S)S[C@@H]1",      "Rhodanine (chiral)"),
    # Catechols
    ("c1cc(O)c(O)cc1",           "Catechol (PAINS — metal chelation / quinone formation)"),
    # Quinones
    ("O=C1C=CC(=O)C=C1",        "para-Quinone (PAINS — redox / thiol reactive)"),
    ("O=C1C=CC(=O)c2ccccc21",   "Naphthoquinone (PAINS)"),
    # Imines / Schiff bases
    ("[CH]=N",                   "Aliphatic imine/Schiff base (PAINS — hydrolysis lability)"),
    # Enones
    ("C=CC(=O)c1ccccc1",        "Chalcone enone (PAINS — Michael acceptor)"),
    ("[CH]=[CH]C(=O)[#6]",      "Alpha-beta-unsaturated carbonyl (PAINS)"),
    # Azo compounds
    ("c1ccc(/N=N/c2ccccc2)cc1", "Azo dye (PAINS — redox interference)"),
    # Hydroxamic acids
    ("C(=O)NO",                  "Hydroxamic acid (PAINS — metal chelation)"),
    # Thiol-reactive
    ("[SH]",                     "Free thiol (PAINS — thiol reactive, assay interference)"),
    # Aggregators (highly hydrophobic / amphiphilic)
    ("CCCCCCCC",                 "Long aliphatic chain (aggregation-prone)"),
    # Fluorescent compounds (spectroscopic interference)
    ("c1ccc2ccccc2c1",           "Naphthalene (may interfere with fluorescence assays)"),
    # Alpha-cyano acrylates
    ("C(=C)C#N",                 "Alpha-cyanoacrylate (Michael acceptor PAINS)"),
]

# Brenk Structural Alerts
# From Brenk et al. (2008) ChemMedChem 3:435
# Validated on large drug-like compound sets; stronger empirical basis than PAINS.
BRENK_SMARTS = [
    # Halogens on sp3 carbon adjacent to heteroatom
    ("[CX4][Cl,Br,I]",          "Brenk: Alkyl halide (electrophilic reactivity)"),
    ("[F,Cl,Br,I][CX4][F,Cl,Br,I]",
                                  "Brenk: Gem-dihalide (reactive)"),
    # Nitroso / N-oxide
    ("[N;!$(NC=O)]=O",          "Brenk: Nitroso group (reactive, mutagen)"),
    ("[N+]([O-])(=O)c1ccccc1",  "Brenk: Nitroaromatic (hepatotoxin / Ames+)"),
    # Peroxides
    ("OO",                       "Brenk: Peroxide (oxidative stress)"),
    ("C(=O)OO",                  "Brenk: Peracid (strongly oxidizing)"),
    # Phosphorus hazards
    ("P(=S)",                    "Brenk: Thiophosphate (organophosphate — neurotoxin)"),
    ("P(=O)(Cl)",                "Brenk: Phosphoryl chloride (hydrolysis product toxic)"),
    # Metal-binding fragments
    ("[#6]S[#6]S[#6]",          "Brenk: Polythioether (metal chelation / interference)"),
    ("C(=O)NC(=O)",              "Brenk: Imide (reactivity, protein binding)"),
    # Quaternary nitrogen with halogens
    ("[N+](C)(C)(C)C",           "Brenk: Quaternary ammonium (absorption limited)"),
    # Thiocarbonyl
    ("C(=S)",                    "Brenk: Thiocarbonyl (hepatotoxin / metal reactive)"),
    # Mustards
    ("ClCCN",                    "Brenk: Nitrogen mustard-like (alkylating agent)"),
    ("ClCC[NH]",                 "Brenk: Secondary nitrogen mustard"),
    # Acrylates
    ("C=CC(=O)O",                "Brenk: Acrylate ester (Michael acceptor / contact sensitiser)"),
    # Aldehydes
    ("[CH]=O",                   "Brenk: Aldehyde (protein reactive, sensitisation)"),
    # Diazo
    ("C=[N+]=[N-]",              "Brenk: Diazo compound (reactive nitrogen species)"),
    # Thiols
    ("[#6][SH]",                 "Brenk: Free thiol on carbon (assay interference / reactive)"),
    # Carbamates
    ("OC(=O)N",                  "Brenk: Carbamate (metabolic lability)"),
    # Epoxides
    ("C1OC1",                    "Brenk: Epoxide (alkylating agent)"),
    # Hydrazides
    ("C(=O)NN",                  "Brenk: Hydrazide (reactive metabolite)"),
    # Oximes
    ("C=NO",                     "Brenk: Oxime (hydrolysis lability / reactive)"),
    # Beta-lactams (not drug context)
    ("C1CC(=O)N1",               "Brenk: Beta-lactam (unless intentional antibiotic scaffold)"),
    # Heavy atom concern
    ("[Hg,Pb,As,Cd,Tl]",        "Brenk: Heavy metal atom (systemic toxicity)"),
    # Polycyclic aromatic hydrocarbons
    ("c1ccc2c(c1)ccc3ccccc23",   "Brenk: Anthracene/PAH (DNA intercalation / mutagenesis)"),
]

# SureChEMBL Toxicity Alerts
# Based on public SureChEMBL safety alert database and EMA/ICH S2(R1) guidelines.
SURECHEMBL_SMARTS = [
    # Genotoxic alerts
    ("[N+](=O)[O-]",             "SureChEMBL: Nitro group (genotoxic via nitroreduction)"),
    ("[NH2]c1ccccc1",            "SureChEMBL: Primary aromatic amine (genotoxic)"),
    ("c1ccc(N)cc1-c2ccccc2N",   "SureChEMBL: Diaminobiphenyl (carcinogen)"),
    ("N=Nc1ccccc1",              "SureChEMBL: Azo to aryl (azo dye — carcinogen)"),
    # Reactive electrophiles
    ("C(=O)Cl",                  "SureChEMBL: Acyl chloride"),
    ("S(=O)(=O)Cl",              "SureChEMBL: Sulfonyl chloride"),
    ("C1CO1",                    "SureChEMBL: Epoxide (alkylating agent)"),
    ("C=CC#N",                   "SureChEMBL: Acrylonitrile (Michael acceptor, hepatotoxin)"),
    # Endocrine disruption alerts
    ("c1cc(O)ccc1",              "SureChEMBL: Phenol (estrogen receptor binding risk)"),
    ("c1ccc(Cl)cc1",             "SureChEMBL: Chlorobenzene (persistent, bioaccumulates)"),
    # Halogenated solvents / radicals
    ("[CX4]([Cl])([Cl])[Cl]",   "SureChEMBL: Trichloromethyl (hepatotoxin — CCl4 mechanism)"),
    ("[CX4]([Cl])([Cl])([Cl])[Cl]",
                                  "SureChEMBL: Carbon tetrachloride scaffold (CCl4 — liver necrosis)"),
    ("[CX4]([Br])([Br])[Br]",   "SureChEMBL: Tribromomethane type"),
    # Heavy metals
    ("[Hg,Pb,As,Cd,Se,Te,Tl]",  "SureChEMBL: Heavy metal (organ toxicity)"),
    # Reproductive/developmental toxins
    ("C(=O)CCCC(=O)",            "SureChEMBL: 1,5-diketone (reproductive toxin — hexanedione type)"),
    ("CCCC1CCCC1=O",             "SureChEMBL: Cyclopentenone (electrophilic, sensitiser)"),
    # Pesticide-like structures
    ("P(=S)(O)(O)O",             "SureChEMBL: Organothiophosphate (neurotoxin — AChE inhibitor)"),
    ("[Cl]C(Cl)=CCl",            "SureChEMBL: Trichloroethylene-type (carcinogen)"),
    # Sensitisers
    ("C=CC(=O)",                 "SureChEMBL: Alpha,beta-unsaturated ketone (skin sensitiser)"),
    ("c1ccc2[nH]ccc2c1",        "SureChEMBL: Indole (potential 5-HT liability at high dose)"),
]

# ============================================================
# 2. ADMET FEATURE COMPUTATION
# ============================================================

def compute_admet_features(smiles):
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
# 3. LIPINSKI BASELINE (PURE PASS / FAIL)
# ============================================================

def lipinski_filter(features):
    violations = []
    for prop, limit in LIPINSKI_RULES.items():
        if features[prop] > limit:
            violations.append(prop)
    return {"passes": len(violations) <= 1, "violations": violations}

# ============================================================
# 4. EXTENDED ADMET RULE CHECKING (EXPLANATORY)
# ============================================================

def check_admet_violations(features):
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
# 5. RULE-BASED COMPOSITE SCORING (NON-ML BASELINE)
# ============================================================

def heuristic_drug_likeness_score(features, violations):
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
    cyp_flags = {}
    for isoform, patterns in CYP_SMARTS.items():
        matched = []
        for smarts, description in patterns:
            query = Chem.MolFromSmarts(smarts)
            if query and mol.HasSubstructMatch(query):
                matched.append(description)
        if matched:
            cyp_flags[isoform] = matched

    n = len(cyp_flags)
    return {
        "flagged_isoforms":      cyp_flags,
        "isoforms_flagged_count": n,
        "cyp_risk":              "HIGH" if n >= 3 else "MODERATE" if n >= 1 else "LOW",
    }

# ============================================================
# 6b. hERG CARDIAC LIABILITY
# ============================================================

def compute_herg_risk(features, mol):
    smarts_hits = []
    for smarts, desc in HERG_SMARTS:
        query = Chem.MolFromSmarts(smarts)
        if query and mol.HasSubstructMatch(query):
            smarts_hits.append(desc)

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
# 6c. MUTAGENICITY — AMES TEST STRUCTURAL ALERTS
# ============================================================

def compute_mutagenicity_alerts(mol):
    alerts = []
    for smarts, description in AMES_SMARTS:
        query = Chem.MolFromSmarts(smarts)
        if query and mol.HasSubstructMatch(query):
            alerts.append(description)
    return {
        "alerts":             alerts,
        "alert_count":        len(alerts),
        "mutagenicity_flag":  "POSITIVE" if alerts else "NEGATIVE",
        "confidence":         "STRUCTURAL_ALERT_BASED",
    }

# ============================================================
# 6d. REACTIVE GROUP / PAINS ALERTS
# ============================================================

def compute_reactive_alerts(mol):
    alerts = []
    for smarts, description in REACTIVE_SMARTS:
        query = Chem.MolFromSmarts(smarts)
        if query and mol.HasSubstructMatch(query):
            alerts.append(description)
    return {
        "reactive_groups":  alerts,
        "reactive_count":   len(alerts),
        "reactivity_flag":  "REACTIVE" if alerts else "CLEAN",
        "notes": (
            "Reactive groups detected — may cause covalent off-target modification "
            "or interfere with assay signals." if alerts else "No reactive groups detected."
        ),
    }

# ============================================================
# 6e. PLASMA PROTEIN BINDING (PPB) ESTIMATION
# ============================================================

def estimate_plasma_protein_binding(features):
    ppb_estimate = round(min(99.9, max(0.0, 40 + features["logp"] * 8 + features["molecular_weight"] * 0.05)), 1)
    if ppb_estimate >= PPB_THRESHOLDS["high"]:
        cls, note = "HIGH",     "Limited free drug fraction — may affect efficacy and dosing"
    elif ppb_estimate >= PPB_THRESHOLDS["moderate"]:
        cls, note = "MODERATE", "Moderate protein binding — generally acceptable"
    else:
        cls, note = "LOW",      "Low protein binding — higher free drug fraction"
    return {"ppb_estimate_pct": ppb_estimate, "ppb_class": cls, "ppb_note": note}

# ============================================================
# 6f. CACO-2 PERMEABILITY ESTIMATION
# ============================================================

def estimate_caco2_permeability(features):
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
# 6g. P-GLYCOPROTEIN EFFLUX RISK
# ============================================================

def estimate_pgp_efflux(features, mol):
    mw, hbd, hba, rb = features["molecular_weight"], features["hbd"], features["hba"], features["rotatable_bonds"]
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
    return {"pgp_risk": cls, "pgp_risk_score": pgp_score, "pgp_risk_reasons": reasons, "pgp_note": note}

# ============================================================
# 7a. METABOLIC STABILITY — NEW (MOST CRITICAL)
# ============================================================

def compute_reactive_metabolite_risk(mol):
    """
    Screens for structural precursors that are bioactivated into
    chemically reactive, toxic metabolites by CYP enzymes or other
    metabolic pathways.

    This is the #1 cause of post-market drug withdrawals.
    Includes: quinone/iminoquinone precursors, furans, thiophenes,
    polyhalogenated carbons (CCl4-type), terminal alkynes,
    methylenedioxyphenyl groups, anilines, hydrazines, acyl glucuronides.

    Risk levels:
        CRITICAL  ≥ 3 alerts  → likely reactive metabolite generation
        HIGH      ≥ 2         → strong structural concern
        MODERATE  ≥ 1         → warrants in vitro confirmation
        LOW       0           → no alerts detected
    """
    alerts = []
    for smarts, description in REACTIVE_METABOLITE_SMARTS:
        query = Chem.MolFromSmarts(smarts)
        if query and mol.HasSubstructMatch(query):
            alerts.append(description)

    n = len(alerts)
    if n >= 3:   risk = "CRITICAL"
    elif n >= 2: risk = "HIGH"
    elif n >= 1: risk = "MODERATE"
    else:        risk = "LOW"

    return {
        "reactive_metabolite_alerts": alerts,
        "alert_count":                n,
        "reactive_metabolite_risk":   risk,
        "recommended_assays": [
            "GSH trapping assay (LC-MS/MS)",
            "Potassium cyanide (KCN) trapping assay",
            "CYP TDI (time-dependent inhibition) shift assay",
        ] if n >= 1 else [],
        "interpretation": (
            f"{n} reactive metabolite precursor alert(s) detected. "
            "Structural features are known to be bioactivated into electrophilic "
            "species capable of forming covalent adducts with proteins or DNA."
        ) if n >= 0 else "No reactive metabolite precursors detected.",
    }


def compute_gsh_trapping_risk(mol):
    """
    Flags structural features that form glutathione (GSH) adducts.

    GSH trapping is used in vitro as a surrogate for reactive
    metabolite formation. A positive signal strongly correlates with
    idiosyncratic hepatotoxicity risk.

    References:
        Baillie & Davis (1993) FASEB J.
        Thompson et al. (2016) Drug Metab. Dispos.
    """
    alerts = []
    for smarts, description in GSH_TRAP_SMARTS:
        query = Chem.MolFromSmarts(smarts)
        if query and mol.HasSubstructMatch(query):
            alerts.append(description)

    return {
        "gsh_trap_alerts":   alerts,
        "gsh_alert_count":   len(alerts),
        "gsh_trapping_risk": "POSITIVE" if alerts else "NEGATIVE",
        "clinical_implication": (
            "Idiosyncratic hepatotoxicity risk — GSH depletion in hepatocytes "
            "leads to oxidative stress and cell death (mechanism: CCl4, APAP, troglitazone)."
        ) if alerts else "No GSH trapping structural alerts detected.",
    }


def identify_metabolic_soft_spots(mol):
    """
    Identifies structural features that are preferential sites of
    CYP-mediated oxidative metabolism (metabolic soft spots).

    High soft-spot count → rapid hepatic clearance → short t½.
    Some soft spots generate reactive/toxic metabolites upon oxidation.

    Awareness of soft spots guides medicinal chemistry optimisation:
        - Block benzylic positions with fluorine
        - Replace N-methyl with N-CD3 (deuterium kinetic isotope effect)
        - Use ring systems to reduce rotatable/oxidisable carbons
    """
    spots = []
    for smarts, description in SOFT_SPOT_SMARTS:
        query = Chem.MolFromSmarts(smarts)
        if query and mol.HasSubstructMatch(query):
            spots.append(description)

    return {
        "soft_spots":           spots,
        "soft_spot_count":      len(spots),
        "clearance_prediction": (
            "HIGH" if len(spots) >= 3 else "MODERATE" if len(spots) >= 1 else "LOW"
        ),
        "optimization_strategies": [
            "Fluorine blocking of benzylic/allylic soft spots",
            "Deuterium substitution at N-methyl or O-methyl positions",
            "Replace methoxyphenyl with difluorophenyl",
            "Cyclise or rigidify flexible chains",
        ] if spots else [],
    }


def compute_mechanism_based_inactivation_risk(mol):
    """
    Screens for mechanism-based CYP inactivation (MBI) alerts.

    MBI compounds covalently modify the CYP enzyme after metabolic
    activation, causing irreversible inhibition. This is a major source
    of clinically significant drug-drug interactions (DDIs).

    Example drugs withdrawn or restricted due to MBI:
        Mibefradil (CYP3A4 MBI), Terfenadine (QT prolongation via MBI DDI)

    Reference: Grimm et al. (2009) Drug Metab. Dispos. 37:1355
    """
    alerts = []
    for smarts, description in MBI_SMARTS:
        query = Chem.MolFromSmarts(smarts)
        if query and mol.HasSubstructMatch(query):
            alerts.append(description)

    return {
        "mbi_alerts":   alerts,
        "mbi_count":    len(alerts),
        "mbi_risk":     "HIGH" if len(alerts) >= 2 else "MODERATE" if len(alerts) >= 1 else "LOW",
        "ddi_concern":  len(alerts) >= 1,
        "recommended_assays": [
            "CYP TDI (time-dependent inhibition) IC50 shift assay",
            "Kinact/KI determination",
            "DDI risk ratio calculation per FDA/EMA guidance",
        ] if alerts else [],
    }


def estimate_microsomal_stability(features, mol):
    """
    Heuristic estimate of hepatic microsomal metabolic stability.

    Based on physicochemical descriptor relationships observed across
    large compound sets (Di et al. 2012, J. Med. Chem. 55:4669).

    Stability score (0–100):
        High stability  (t½ > 60 min microsomal)  → score ≥ 70
        Moderate        (t½ 30–60 min)             → score 40–70
        Low             (t½ < 30 min)              → score < 40

    Key drivers:
        - logP:             higher → faster CYP oxidation (lowers score)
        - aromatic rings:   more   → more CYP substrates (lowers score)
        - soft spots:       more   → faster turnover (lowers score)
        - MW:               higher → sometimes slower (less permeable to CYP)
    """
    logp = features["logp"]
    mw   = features["molecular_weight"]
    try:
        n_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    except Exception:
        n_aromatic_rings = 0

    soft_spots = identify_metabolic_soft_spots(mol)["soft_spot_count"]

    # Heuristic formula
    stability_score = 100
    stability_score -= logp * 6               # lipophilicity drives CYP binding
    stability_score -= n_aromatic_rings * 5   # aromatic rings = oxidation handles
    stability_score -= soft_spots * 8         # each soft spot accelerates clearance
    stability_score += min(mw, 500) * 0.04   # larger molecules slightly more stable (slower CYP access)
    stability_score = round(min(100, max(0, stability_score)), 1)

    if stability_score >= 70:
        cls   = "HIGH"
        t_half = "> 60 min (predicted)"
    elif stability_score >= 40:
        cls   = "MODERATE"
        t_half = "30–60 min (predicted)"
    else:
        cls   = "LOW"
        t_half = "< 30 min (predicted)"

    return {
        "microsomal_stability_score": stability_score,
        "stability_class":            cls,
        "predicted_t_half":           t_half,
        "n_aromatic_rings":           n_aromatic_rings,
        "soft_spot_count":            soft_spots,
        "logp_penalty":               round(logp * 6, 1),
        "aromatic_ring_penalty":      n_aromatic_rings * 5,
        "soft_spot_penalty":          soft_spots * 8,
        "stability_interpretation": (
            f"Predicted microsomal stability: {cls} (score {stability_score}/100, t½ {t_half}). "
            f"{n_aromatic_rings} aromatic ring(s) and {soft_spots} soft spot(s) detected."
        ),
    }

# ============================================================
# 7b. TOXICOPHORE / STRUCTURAL ALERTS — EXPANDED (NEW)
# ============================================================

def compute_pains_alerts(mol):
    """
    Pan-Assay INterference compound (PAINS) filter.

    Compounds matching PAINS patterns frequently produce false positives
    in HTS biochemical assays due to:
        - Redox cycling (catechols, quinones)
        - Covalent binding (Michael acceptors)
        - Aggregation (amphiphilic, hydrophobic)
        - Fluorescence interference (polycyclics)
        - Metal chelation (hydroxamic acids, catechols)

    Reference: Baell & Holloway (2010) J. Med. Chem. 53:2719

    Note: PAINS are not automatically disqualified — some are drugs —
    but require extensive counter-screening to confirm activity.
    """
    alerts = []
    for smarts, description in PAINS_SMARTS:
        query = Chem.MolFromSmarts(smarts)
        if query and mol.HasSubstructMatch(query):
            alerts.append(description)

    return {
        "pains_alerts":   alerts,
        "pains_count":    len(alerts),
        "pains_flag":     "FLAGGED" if alerts else "CLEAN",
        "assay_validity": (
            "Counter-screening strongly recommended — compound may produce "
            "false positive HTS results independent of target engagement."
        ) if alerts else "No PAINS alerts detected. Assay validity unaffected.",
    }


def compute_brenk_alerts(mol):
    """
    Brenk structural alerts for lead-likeness filtering.

    The Brenk set contains 105 validated structural alerts derived from
    analysis of undesirable groups found in compound libraries.
    Strong empirical validation across millions of compounds.

    Reference: Brenk et al. (2008) ChemMedChem 3:435-444

    Covers: reactive electrophiles, genotoxic motifs, unstable groups,
    metabolic liabilities, and heavy metal fragments.
    """
    alerts = []
    for smarts, description in BRENK_SMARTS:
        query = Chem.MolFromSmarts(smarts)
        if query and mol.HasSubstructMatch(query):
            alerts.append(description)

    n = len(alerts)
    return {
        "brenk_alerts":  alerts,
        "brenk_count":   n,
        "brenk_flag":    "HIGH" if n >= 3 else "MODERATE" if n >= 1 else "CLEAN",
        "lead_likeness": (
            "POOR — multiple Brenk alerts; scaffold likely unsuitable for lead development"
        ) if n >= 3 else (
            "MARGINAL — Brenk alert(s) present; optimise or confirm necessity"
        ) if n >= 1 else "GOOD — No Brenk alerts detected.",
    }


def compute_surechembl_alerts(mol):
    """
    SureChEMBL toxicity structural alerts.

    Derived from the public SureChEMBL patent chemistry database and
    EMA/ICH S2(R1) genotoxicity guidance.

    Covers: genotoxic fragments, carcinogens, endocrine disruptors,
    neurotoxins (organophosphates), persistent bioaccumulative toxins,
    reproductive toxins (REACH SVHC), and sensitisers.

    These alerts are aligned with regulatory safety pharmacology
    requirements (ICH S7A/S7B, REACH, ECHA).
    """
    alerts = []
    for smarts, description in SURECHEMBL_SMARTS:
        query = Chem.MolFromSmarts(smarts)
        if query and mol.HasSubstructMatch(query):
            alerts.append(description)

    n = len(alerts)
    return {
        "surechembl_alerts":   alerts,
        "surechembl_count":    n,
        "regulatory_risk":     "HIGH" if n >= 2 else "MODERATE" if n >= 1 else "LOW",
        "regulatory_note": (
            "Regulatory safety concern — compound contains patterns flagged under "
            "ICH S2(R1), REACH SVHC, or EMA genotoxicity guidance."
        ) if n >= 1 else "No SureChEMBL regulatory alerts detected.",
    }


def compute_overall_toxicophore_summary(pains, brenk, surechembl, mutagenicity, reactive):
    """
    Aggregates all toxicophore/structural alert results into a single
    summary risk profile.
    """
    total_alerts = (
        pains["pains_count"] +
        brenk["brenk_count"] +
        surechembl["surechembl_count"] +
        mutagenicity["alert_count"] +
        reactive["reactive_count"]
    )

    if total_alerts >= 5:
        overall = "CRITICAL"
    elif total_alerts >= 3:
        overall = "HIGH"
    elif total_alerts >= 1:
        overall = "MODERATE"
    else:
        overall = "CLEAN"

    return {
        "total_toxicophore_alerts": total_alerts,
        "overall_toxicophore_risk": overall,
        "breakdown": {
            "pains":        pains["pains_count"],
            "brenk":        brenk["brenk_count"],
            "surechembl":   surechembl["surechembl_count"],
            "ames":         mutagenicity["alert_count"],
            "reactive":     reactive["reactive_count"],
        },
        "filter_recommendation": (
            "REJECT — compound is unlikely to progress through safety pharmacology."
            if overall == "CRITICAL" else
            "FLAG — requires significant medicinal chemistry optimisation before progression."
            if overall == "HIGH" else
            "CAUTION — verify alerts are acceptable in therapeutic context."
            if overall == "MODERATE" else
            "PASS — no toxicophore alerts detected across all filter sets."
        ),
    }

# ============================================================
# 8. MASTER ANALYSIS PIPELINE (WHAT app.py CALLS)
# ============================================================

def analyze_smiles(smiles):
    mol, features = compute_admet_features(smiles)

    if features is None:
        return {"smiles": smiles, "valid": False}

    lipinski_result = lipinski_filter(features)
    violations      = check_admet_violations(features)
    score, reject_reason = heuristic_drug_likeness_score(features, violations)

    # --- Original ADME modules ---
    cyp450       = compute_cyp450_liability(mol)
    herg         = compute_herg_risk(features, mol)
    mutagenicity = compute_mutagenicity_alerts(mol)
    reactive     = compute_reactive_alerts(mol)
    ppb          = estimate_plasma_protein_binding(features)
    caco2        = estimate_caco2_permeability(features)
    pgp          = estimate_pgp_efflux(features, mol)

    # --- Metabolic Stability modules (NEW — most critical) ---
    reactive_metabolites = compute_reactive_metabolite_risk(mol)
    gsh_trapping         = compute_gsh_trapping_risk(mol)
    soft_spots           = identify_metabolic_soft_spots(mol)
    mbi_risk             = compute_mechanism_based_inactivation_risk(mol)
    microsomal_stability = estimate_microsomal_stability(features, mol)

    # --- Expanded Toxicophore modules (NEW) ---
    pains_alerts     = compute_pains_alerts(mol)
    brenk_alerts     = compute_brenk_alerts(mol)
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

    decision = "REJECT" if reject_reason else "ACCEPT"

    return {
        "smiles":          smiles,
        "valid":           True,
        "mol":             mol,
        "properties":      features,
        "lipinski":        lipinski_result,
        "violations":      violations,
        "score":           round(score, 2),
        "physchem_status": status,
        "decision":        decision,
        "reject_reason":   reject_reason,
        # Original ADME
        "cyp450":                 cyp450,
        "herg":                   herg,
        "mutagenicity":           mutagenicity,
        "reactive_alerts":        reactive,
        "plasma_protein_binding": ppb,
        "caco2_permeability":     caco2,
        "pgp_efflux":             pgp,
        # Metabolic stability (new)
        "reactive_metabolites":  reactive_metabolites,
        "gsh_trapping":          gsh_trapping,
        "metabolic_soft_spots":  soft_spots,
        "mbi_risk":              mbi_risk,
        "microsomal_stability":  microsomal_stability,
        # Expanded toxicophores (new)
        "pains":               pains_alerts,
        "brenk":               brenk_alerts,
        "surechembl":          surechembl_alerts,
        "toxicophore_summary": toxicophore_summary,
    }

# ============================================================
# 9. DATASET-LEVEL UTILITIES
# ============================================================

def analyze_multiple_smiles(smiles_list):
    return [analyze_smiles(s) for s in smiles_list]

def rank_molecules(results):
    valid = [r for r in results if r["valid"]]
    return sorted(valid, key=lambda x: x["score"], reverse=True)

def dataset_decision(results):
    decisions = [r["decision"] for r in results]
    if all(d == "REJECT" for d in decisions): return "ALL_REJECTED"
    if all(d == "ACCEPT" for d in decisions): return "ALL_ACCEPTED"
    return "MIXED"

# ============================================================
# 10. INTERPRETATION & USER FEEDBACK
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

    # --- Metabolic stability feedback ---
    rm_risk = mol.get("reactive_metabolites", {}).get("reactive_metabolite_risk")
    if rm_risk in ("CRITICAL", "HIGH"):
        n = mol["reactive_metabolites"]["alert_count"]
        reasons.append(f"CRITICAL: {n} reactive metabolite precursor alert(s) — hepatotoxicity risk")
    if mol.get("gsh_trapping", {}).get("gsh_trapping_risk") == "POSITIVE":
        reasons.append("GSH trapping alerts detected — idiosyncratic hepatotoxicity risk")
    if mol.get("mbi_risk", {}).get("mbi_risk") in ("HIGH", "MODERATE"):
        reasons.append("Mechanism-based CYP inactivation risk — potential DDI liability")
    stab = mol.get("microsomal_stability", {}).get("stability_class")
    if stab == "LOW":
        reasons.append(f"Low predicted microsomal stability (t½ < 30 min) — rapid hepatic clearance")

    # --- Toxicophore feedback ---
    if mol.get("pains", {}).get("pains_flag") == "FLAGGED":
        reasons.append(f"PAINS alerts detected: {mol['pains']['pains_alerts']}")
    brenk_flag = mol.get("brenk", {}).get("brenk_flag")
    if brenk_flag in ("HIGH", "MODERATE"):
        reasons.append(f"Brenk structural alerts ({mol['brenk']['brenk_count']}): {mol['brenk']['brenk_alerts']}")
    if mol.get("surechembl", {}).get("regulatory_risk") in ("HIGH", "MODERATE"):
        reasons.append(f"SureChEMBL regulatory alerts: {mol['surechembl']['surechembl_alerts']}")

    if not reasons:
        if tied:
            reasons.append("Equally optimal by all evaluated drug-likeness criteria")
        elif is_top:
            reasons.append(
                "Satisfies all evaluated drug-likeness rules — selected as top candidate"
            )
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

    # --- Metabolic stability advice (new) ---
    rm_risk = mol.get("reactive_metabolites", {}).get("reactive_metabolite_risk")
    if rm_risk in ("CRITICAL", "HIGH"):
        advice.append("PRIORITY: Remove or block reactive metabolite precursor groups")
        advice.append("Replace furan with pyridine, thiophene with thiazole, catechol with fluorophenol")
        advice.append("Run GSH trapping and KCN trapping assays before any in vivo work")
    if mol.get("gsh_trapping", {}).get("gsh_trapping_risk") == "POSITIVE":
        advice.append("Replace GSH-reactive groups — introduce blocking substituents ortho/para to reactive site")
    if mol.get("mbi_risk", {}).get("mbi_risk") in ("HIGH", "MODERATE"):
        advice.append("Replace MBI-prone groups (furans → thiophenes → pyridines as metabolic bioisosteres)")
        advice.append("Run TDI IC50 shift assay and determine kinact/KI for DDI risk assessment")
    soft = mol.get("metabolic_soft_spots", {})
    if soft.get("clearance_prediction") in ("HIGH", "MODERATE"):
        advice.extend(soft.get("optimization_strategies", []))

    # --- Toxicophore advice (new) ---
    if mol.get("pains", {}).get("pains_flag") == "FLAGGED":
        advice.append("Counter-screen PAINS hits with orthogonal assay formats (SPR, NMR, thermal shift)")
        advice.append("Consider removing PAINS-flagged substructure if not essential to pharmacophore")
    if mol.get("brenk", {}).get("brenk_flag") in ("HIGH", "MODERATE"):
        advice.append("Address Brenk alerts — prioritise removal of reactive or genotoxic Brenk groups")
    if mol.get("surechembl", {}).get("regulatory_risk") in ("HIGH", "MODERATE"):
        advice.append("Regulatory alert: consult ICH S2(R1) genotoxicity battery before clinical submission")
        advice.append("Substitute genotoxic fragment with non-alerting bioisostere")

    return list(set(advice))