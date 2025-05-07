import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

accuracy_data = {
    "Dataset": ["MALICIOUS", "NSL-KDD", "UNSW15", "CIC-IDS-2017", "BCCC-CIC-IDS2017", "BOTNETIOT"],
    "BinHD":     [0.4567, 0.4084, 0.5135, 0.4983, 0.4805, 0.5073],
    "AdaptHD":   [0.9264, 0.9494, 0.8519, 0.9218, 0.9366, 0.9403],
    "OnlineHD":  [0.9325, 0.9628, 0.8403, 0.9127, 0.9270, ],
    "NeuralHD":  [0.9286, 0.9737, 0.8379, 0.9076, 0.9175, 0.9450],
    "DistHD":    [0.9299, 0.9223, 0.7574, 0.4991, 0.8955, 0.7348],
    "CompHD":    [0.8737, 0.8863, 0.8158, 0.7586, 0.7361, 0.9213]
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