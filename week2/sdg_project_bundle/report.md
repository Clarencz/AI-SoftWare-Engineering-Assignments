# SDG 13 Project — 1-page Summary

**Project title:** Forecasting CO₂ emissions per capita (SDG 13: Climate Action)

**Problem addressed**
Predict near-term CO₂ emissions per capita for countries using readily available socio-economic indicators (year, GDP per capita, population, energy consumption per capita). Accurate short-term forecasts can help policymakers plan mitigation and adaptation actions.

**ML approach**
Supervised learning — Linear Regression (baseline).  
Pipeline: data loading → feature engineering → train/test split → model training → evaluation → simple forecasting.

**Dataset**
A small synthetic dataset included for demonstration (synthetic_co2_dataset.csv). In a production project, use World Bank CO₂ and energy datasets or Kaggle historical emissions datasets.

**Results**
Baseline Linear Regression on the synthetic dataset:
- Evaluation metrics produced by the notebook (MAE, R²) — see notebook for exact numbers.
- Visual diagnostic saved: `true_vs_predicted.png`.
- Example 2025 forecast generated for sample countries.

**Ethical considerations**
- **Bias & representativeness:** Synthetic / limited historical data may not reflect real-world variability (e.g., sudden policy shifts, pandemics, economic crises). Using incomplete data leads to biased forecasts.
- **Fairness:** Models must consider vulnerable populations and avoid decisions that disproportionately harm low-income regions.
- **Transparency:** Provide clear uncertainty estimates and avoid overreliance on point predictions for policy decisions.
- **Sustainability:** Encourage use of model outputs to plan equitable decarbonization strategies (invest in renewables, energy efficiency).



