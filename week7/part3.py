## Part 3: Practical Audit (25%)

### Task: Audit COMPAS Recidivism Dataset

## 1. Python Code (Analysis)


import pandas as pd
import numpy as np
from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric
import matplotlib.pyplot as plt

# 1. Load and Preprocess Data (Simulated structure)
def load_compas_data():
    url = "[https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv](https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv)"
    df = pd.read_csv(url)
    df = df[df['race'].isin(['African-American', 'Caucasian'])]
    # Create Binary target: High Risk (1) vs Low Risk (0)
    df['high_risk'] = np.where(df['decile_score'] > 4, 1, 0)
    # Define protected attribute: race (0: African-American, 1: Caucasian)
    df['race_code'] = np.where(df['race'] == 'Caucasian', 1, 0)
    return df

df = load_compas_data()

# Convert to AIF360 format
dataset = StandardDataset(df, 
                          label_name='high_risk', 
                          favorable_classes=[0], 
                          protected_attribute_names=['race_code'], 
                          privileged_classes=[[1]])

# 2. Metric Analysis
metric_orig = BinaryLabelDatasetMetric(dataset, 
                                       privileged_groups=[{'race_code': 1}],
                                       unprivileged_groups=[{'race_code': 0}])

print(f"Disparate Impact: {metric_orig.disparate_impact():.4f}")
print(f"Mean Difference: {metric_orig.mean_difference():.4f}")

# 3. Visualization
groups = ['African-American', 'Caucasian']
fpr = [44.9, 23.5] # Actual ProPublica stats for False Positive Rate
fnr = [28.0, 47.7] # Actual ProPublica stats for False Negative Rate

x = np.arange(len(groups))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, fpr, width, label='False Positive')
rects2 = ax.bar(x + width/2, fnr, width, label='False Negative')

ax.set_ylabel('Percentage')
ax.set_title('Bias in COMPAS Risk Scores')
ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.legend()
plt.show()