import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

accuracy_data = {
    "Dataset": ["MALICIOUS", "NSL-KDD", "UNSW15", "CIC-IDS-2017", "BCCC-CIC-IDS2017", "BOTNETIOT"],
    "GaussianNB":     [0.7956, 0.8962, 0.8061, 0.9895, 0.7834, 0.7834],
    "KNN":            [0.9963, 0.9962, 0.8756, 0.9980, 0.9979, 0.9979],
    "RandomForest":   [0.9967, 0.9930, 0.8889, 0.9989, 0.9979, 0.9980],
    "AdaBoost":       [0.9919, 0.9765, 0.8576, 0.9895, 0.9865, 0.9865],
    "LogisticRegression":    [0.9524, 0.9485, 0.8178, 0.9185, 0.9270, 0.9270],
    "DecisionTree":    [0.9966, 0.9972, 0.9981, 0.7782, 0.9981, 0.9913]
}

df = pd.DataFrame(accuracy_data)

# Plot settings
models = df.columns[1:]
datasets = df["Dataset"]
num_datasets = len(datasets)
num_models = len(models)
bar_width = 0.12
x = np.arange(num_datasets)

colors = ['#91c7f3', '#f4a582', '#92c28c', '#ffffff', '#cccccc', '#fada5e']
edgecolors = ['#91c7f3', '#f4a582', '#92c28c', '#666666', '#666666', '#fada5e']
hatches = [None, None, None, '//', None, None]  # Hatch for NeuralHD (like your image)

# Plotting
fig, ax = plt.subplots(figsize=(10, 5))

for i, model in enumerate(models):
    values = df[model] * 100  # Convert to %
    ax.bar(x + i * bar_width, values,
           width=bar_width,
           label=model,
           color=colors[i],
           edgecolor=edgecolors[i],
           hatch=hatches[i])

# Style
ax.set_xticks(x + bar_width * (num_models-1) / 2)
ax.set_xticklabels(datasets, fontsize=10)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_ylim(70, 100)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fontsize=10)
ax.grid(axis="y", linestyle="--", linewidth=0.5)
plt.xticks(rotation=15)
plt.tight_layout()

# Save or show
plt.savefig("accuracy_comparison.png", dpi=300)
plt.show()