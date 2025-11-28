# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 14:25:34 2025

@author: joms0005
"""

# -*- coding: utf-8 -*-
"""
K-Fold Result Analysis Script
Matches the style of your PSPNet/DeepLab/AttentionUNet analysis,
but built for your UNet K-fold training output.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

sns.set_theme(style="whitegrid")

# -----------------------------------------------------------
# CONFIG: Set this to the output folder of your K-fold script
# -----------------------------------------------------------
root_dir = Path(r"//storage-ume.slu.se/home$/joms0005/Desktop/SLU/Vindeln_2025/DRONE_FLIGHTS/work/train_kfold_v2")

# -----------------------------------------------------------
# Load history.json from each fold
# -----------------------------------------------------------
records = []

for fold_dir in sorted(root_dir.glob("fold_*")):
    hist_file = fold_dir / "history.json"
    if not hist_file.exists():
        continue

    with open(hist_file, "r") as f:
        hist = json.load(f)

    # Extract last epoch OR best epoch?
    # → best based on minimal val_loss
    df_hist = pd.DataFrame(hist)
    best_row = df_hist.loc[df_hist["val_loss"].idxmin()]

    records.append({
        "Fold": fold_dir.name,
        "Best_Epoch": int(best_row["epoch"]),
        "Train_Loss": float(best_row["train_loss"]),
        "Val_Loss": float(best_row["val_loss"]),
        "IoU": float(best_row["val_iou"]),
        "Dice": float(best_row["val_dice"]),
    })

if not records:
    raise RuntimeError("No fold results found!")

df = pd.DataFrame(records)
print("\n=== Loaded K-Fold Results ===")
print(df)

# -----------------------------------------------------------
# ANOVA Example (here for IoU)
# -----------------------------------------------------------
groups = [df[df["Fold"] == f]["IoU"] for f in df["Fold"].unique()]
F, p = f_oneway(*groups)

print(f"\nANOVA for IoU across folds: F={F:.4f}, p={p:.6f}")

if p < 0.05:
    tukey = pairwise_tukeyhsd(df["IoU"], df["Fold"])
    print("\nTukey HSD (IoU):")
    print(tukey)

# -----------------------------------------------------------
# PLOTS
# -----------------------------------------------------------

# ---- Boxplot: IoU per fold ----
plt.figure(figsize=(8, 5))
sns.boxplot(
    x="Fold",
    y="IoU",
    data=df,
    palette="Set2"
)
plt.title("IoU Distribution per Fold")
plt.tight_layout()
plt.show()

# ---- Barplot: Mean IoU ----
plt.figure(figsize=(8, 5))
sns.barplot(
    x="Fold",
    y="IoU",
    data=df,
    palette="Set2",
    errorbar=None
)
plt.title("Mean IoU per Fold")
plt.tight_layout()
plt.show()

# ---- Lineplot: IoU across folds ----
plt.figure(figsize=(8, 5))
sns.lineplot(
    x="Fold",
    y="IoU",
    data=df,
    marker="o"
)
plt.title("IoU Across Folds")
plt.tight_layout()
plt.show()

# ---- Boxplot for all metrics ----
metrics = ["Train_Loss", "Val_Loss", "IoU", "Dice"]

for metric in metrics:
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        x="Fold",
        y=metric,
        data=df,
        palette="Set3"
    )
    plt.title(f"{metric} Distribution per Fold")
    plt.tight_layout()
    plt.show()

print("\n✔ Analysis Complete.")
