# 🧪 LeadRefine: Drug-Likeness & ADME Analyzer

LeadRefine is a comprehensive, rule-based screening tool designed to evaluate the drug-likeliness and ADME (Absorption, Distribution, Metabolism, and Excretion) profiles of small molecules using SMILES strings. It provides medicinal chemists with actionable insights and optimization suggestions to improve lead candidates.

## 🚀 Features

### 1. Physicochemical Analysis
- **Lipinski Rule of Five:** MW ≤ 500, LogP ≤ 5, HBD ≤ 5, HBA ≤ 10.
- **Extended Rules:** TPSA ≤ 140 Å², Rotatable Bonds ≤ 10.
- **Composite Scoring:** A heuristic drug-likeness score (0–100) based on property deviations.

### 2. ADME & Toxicity Predictions (Heuristic)
- **Metabolic Liability:** Structural alerts for major CYP450 isoforms (3A4, 2D6, 2C9, 1A2, 2C19).
- **Cardiac Safety:** hERG potassium channel inhibition risk estimation.
- **Genotoxicity:** Ames mutagenicity structural alerts (Kazius/Benigni).
- **Absorption:** Caco-2 permeability and P-glycoprotein (P-gp) efflux risk.
- **Distribution:** Plasma Protein Binding (PPB) percentage estimation.

### 3. Advanced Metabolic Stability
- **Reactive Metabolites:** Flags precursors for quinones, furans, thiophenes, etc.
- **GSH Trapping:** Detects motifs prone to glutathione conjugation (hepatotoxicity risk).
- **Soft Spots:** Identifies sites of preferential metabolic oxidation.
- **MBI Risk:** Screens for Mechanism-Based Inactivation of CYP enzymes.

### 4. Toxicophore & Interference Filters
- **PAINS:** Pan-Assay Interference Compounds (Baell & Holloway).
- **Brenk Alerts:** Undesirable groups for lead development.
- **SureChEMBL:** Regulatory safety alerts (carcinogens, endocrine disruptors).

### 5. PubChem Enrichment
- Automatically fetches IUPAC names, synonyms, bioactivity summaries, and literature links for known compounds.

---

## 🛠 Installation & Setup

### Prerequisites
- Python 3.8+
- [Conda](https://docs.conda.io/en/latest/) (Recommended for RDKit) or `pip`

### 1. Clone the Repository
```bash
git clone https://github.com/Woeter69/LeadRefine-Drug-likeliness.git
cd LeadRefine-Drug-likeliness
```

### 2. Set up Virtual Environment
**Using `venv`:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Using `conda`:**
```bash
conda create -n leadrefine python=3.9
conda activate leadrefine
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
*Note: RDKit is a core dependency. If `pip install` fails, try `conda install -c conda-forge rdkit`.*

---

## 🖥 Usage

### Interactive Web App
The easiest way to use LeadRefine is via the Streamlit dashboard:
```bash
streamlit run app.py
```
Paste your SMILES strings (one per line) and get instant visual reports and optimization advice.

### Python API / Batch Processing
You can use `adme_analysis.py` for programmatic analysis and generating 9-panel PNG reports:

```python
from adme_analysis import analyze_smiles

# Analyze a single molecule
result = analyze_smiles("CC(=O)Oc1ccccc1C(=O)O", output_path="aspirin_report.png")

print(f"Score: {result['score']}")
print(f"Decision: {result['decision']}")
```

### Batch Analysis
To process a list of molecules and save reports to a folder:
```bash
python run.py
```

---

## 📊 Visual Reports
LeadRefine generates comprehensive reports (`reports/`) containing:
- Radar charts for physicochemical properties.
- Risk matrices for ADME/Tox modules.
- Structural alert counts.
- Optimization suggestions (e.g., "Fluorine blocking", "Reduce TPSA").

---

Rule-based ADME screening | Not a substitute for experimental measurement
