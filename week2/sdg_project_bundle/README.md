# SDG Project Bundle

Contents:
- synthetic_co2_dataset.csv — small demonstration dataset
- sdg_project_notebook.py — annotated notebook/script to run the pipeline
- report.md — 1-page summary
- presentation.md — 5-minute demo slides & speaker notes
- requirements.txt — suggested Python packages

How to run:
1. (Optional) create a virtual environment and install requirements:
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

2. Run the notebook script (or paste into a Jupyter notebook):
   python sdg_project_notebook.py

3. Outputs produced by the script:
   - true_vs_predicted.png (diagnostic plot)
   - printed evaluation metrics and a small 2025 forecast table in the console

Notes:
- The dataset is synthetic for demonstration. Replace with real datasets (World Bank / Kaggle) to create a robust submission.
- The code is intentionally compact; feel free to expand (cross-validation, feature scaling, richer models).
