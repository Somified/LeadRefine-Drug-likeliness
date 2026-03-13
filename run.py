from adme_analysis import analyze_smiles, analyze_multiple_smiles

# Single molecule — analysis only
result = analyze_smiles("CC(=O)Oc1ccccc1C(=O)O")
print(result["score"])
print(result["decision"])

# Single molecule — analysis + chart saved as aspirin.png
result = analyze_smiles("CC(=O)Oc1ccccc1C(=O)O", output_path="aspirin.png")

# Multiple molecules — saves one chart per molecule into a reports folder
import os
os.makedirs("reports", exist_ok=True)
results = analyze_multiple_smiles(["CCO", "c1ccccc1"], output_dir="reports")