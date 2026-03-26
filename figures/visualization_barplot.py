import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# df = pd.read_excel("PathologyBenchmark.xlsx", sheet_name="hyperparameter")
df = pd.read_csv("hyperparameter.csv")

# Filter out LR = 2e-6
df = df[df["LR"] != 2e-6]

df["LR_str"] = df["LR"].astype(str)

encoders = sorted(df["Encoder"].unique())
optimizers = sorted(df["Optimizer"].unique())
lrs = sorted(df["LR_str"].unique())

# Create list of all encoder-optimizer pairs
group_keys = sorted(df[["Encoder", "Optimizer"]].drop_duplicates().apply(tuple, axis=1).tolist())

# Build color mapping based on encoder name group
mstar_pairs = [pair for pair in group_keys if "mSTAR" in pair[0]]
conch_pairs = [pair for pair in group_keys if "conchv15" in pair[0]]
fold_pairs = [pair for pair in group_keys if "10folds" in pair[0]]
other_pairs = [pair for pair in group_keys if pair not in mstar_pairs + conch_pairs + fold_pairs]

# Generate colors
color_map = {}

reds = cm.get_cmap("Reds", len(mstar_pairs) + 2)
for i, pair in enumerate(mstar_pairs):
    color_map[pair] = reds(i + 1)

blues = cm.get_cmap("Blues", len(conch_pairs) + 2)
for i, pair in enumerate(conch_pairs):
    color_map[pair] = blues(i + 1)

greens = cm.get_cmap("Greens", len(fold_pairs) + 2)
for i, pair in enumerate(fold_pairs):
    color_map[pair] = greens(i + 1)

grays = cm.get_cmap("Greys", len(other_pairs) + 2)
for i, pair in enumerate(other_pairs):
    color_map[pair] = grays(i + 1)

# Bar plot
x = np.arange(len(lrs))
bar_width = 0.8 / len(group_keys)

plt.figure(figsize=(16, 6))

for i, (encoder, optimizer) in enumerate(group_keys):
    means, stds = [], []
    for lr in lrs:
        row = df[(df["Encoder"] == encoder) & (df["Optimizer"] == optimizer) & (df["LR_str"] == lr)]
        if not row.empty:
            means.append(row["Macro_AUC_mean"].values[0])
            stds.append(row["Macro_AUC_std"].values[0])
        else:
            means.append(np.nan)
            stds.append(0)

    offset = (i - len(group_keys) / 2) * bar_width + bar_width / 2
    color = color_map[(encoder, optimizer)]
    plt.bar(x + offset, means, yerr=stds, width=bar_width, capsize=2,
            label=f"{encoder}-{optimizer}", color=color, edgecolor="black")

plt.xticks(x, lrs, rotation=45)
plt.xlabel("Learning Rate")
plt.ylabel("Macro AUC")
plt.title("Macro AUC vs Learning Rate (per Encoder-Optimizer)")
plt.grid(axis="y")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Encoder-Optimizer")
plt.tight_layout()
plt.show()
