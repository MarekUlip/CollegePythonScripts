"""
Script that creates output from gathered experiments results
"""
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

df = pd.read_csv('results_classif.csv')
plt.figure(figsize=(14, 5))
plt.title('Accuracy')
a = sns.heatmap(df.pivot(index='prep_meth', columns='clf', values='acc'), annot=True, cmap="RdYlGn")
plt.savefig('clsf_accuracy.png', dpi=400)

plt.clf()
plt.figure(figsize=(14, 5))
plt.title('F1')
sns.heatmap(df.pivot(index='prep_meth', columns='clf', values='f1'), annot=True, cmap="RdYlGn")
plt.savefig('clsf_f1.png', dpi=400)
