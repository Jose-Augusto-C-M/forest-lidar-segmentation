# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 13:22:30 2025

@author: joms0005
"""

import laspy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# ============================================================
# 1. SETTINGS – CHANGE THESE PATHS
# ============================================================

LABELED_LAS_PATH = "D:/SLU_biodiversity/STORAENSO_X/MATCHES_overlap_las_2/Ground pointscropR100_reheadered.las"   # has classification 1/2
RAW_LAS_PATH     = "D:/SLU_biodiversity/STORAENSO_X/MATCHES_overlap_las_2/_ - Channel 2 - 230917_163625_Channel_2 - originalpoints_cropR100.las"                  # unlabeled point cloud
OUTPUT_LAS_PATH  = "D:/SLU_biodiversity/STORAENSO_X/MATCHES_overlap_las_2/_ - Channel 2 - 230917_163625_Channel_2 - originalpoints_raw_cloud_segmented.las"

# For plotting (subsample to avoid plotting millions of points)
MAX_POINTS_PLOT = 100_000
# Raster resolution (for 2D ground map where voids = ground)
RASTER_RESOLUTION = 1.0  # in same units as X/Y (e.g. meters)

# ============================================================
# 1. LOCAL HEIGHT + FEATURE EXTRACTION
# ============================================================

def compute_local_relative_height(x, y, z, cell_size=1.0):
    """
    Compute local relative height = z - min(z_in_cell)
    per XY grid cell.

    For each point we always have at least its own cell, so there are
    no 'void' cells at point level. (Void cells matter only for rasters.)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    xmin, ymin = x.min(), y.min()

    # Compute integer grid coordinates
    ix = ((x - xmin) / cell_size).astype(np.int32)
    iy = ((y - ymin) / cell_size).astype(np.int32)

    # Define grid size
    nx = ix.max() + 1

    # Flatten cell ID
    cell_id = iy * nx + ix

    # Sort points by cell ID
    order = np.argsort(cell_id)
    cid_sorted = cell_id[order]
    z_sorted = z[order]

    # Unique cells and start positions
    unique_cid, idx_start = np.unique(cid_sorted, return_index=True)

    # Min-Z per cell
    min_z_sorted = np.minimum.reduceat(z_sorted, idx_start)

    # Map each point's cell to its min-z
    cell_to_min = dict(zip(unique_cid, min_z_sorted))
    min_z = np.array([cell_to_min[c] for c in cell_id], dtype=z.dtype)

    # Local height
    z_local = z - min_z
    z_local[z_local < 0] = 0
    return z_local


def las_to_features(las, cell_size=1.0):
    """
    Build a feature matrix from a LAS file.

    Uses:
      - local relative height (z - min(z) in XY cell)
      - intensity (if present)
      - return_number (if present)
      - number_of_returns (if present)

    Does NOT use absolute x,y,z directly to reduce height/location artefacts.
    """
    dim_names = set(las.point_format.dimension_names)
    feats = []
    feat_names = []

    x = las.x
    y = las.y
    z = las.z

    # Local relative height
    z_local = compute_local_relative_height(x, y, z, cell_size=cell_size)
    feats.append(z_local)
    feat_names.append("z_local")

    # Standard dims
    if "intensity" in dim_names:
        feats.append(las.intensity)
        feat_names.append("intensity")

    if "return_number" in dim_names:
        feats.append(las.return_number)
        feat_names.append("return_number")

    if "number_of_returns" in dim_names:
        feats.append(las.number_of_returns)
        feat_names.append("number_of_returns")

    X = np.vstack(feats).T.astype(np.float64)
    return X, feat_names

# ============================================================
# 2. PLOTTING HELPERS
# ============================================================

def plot_3d_point_cloud(points, labels=None, title="Point Cloud",
                        s=1.0, alpha=0.8):
    """
    points : (N, 3) array -> [x, y, z]
    labels : (N,) array of 0/1 or None
             0 = off-ground, 1 = ground
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    if labels is None:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   s=s, alpha=alpha)
    else:
        colors = np.where(labels == 1, "green", "brown")
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   c=colors, s=s, alpha=alpha)

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', linestyle='',
                   label='Ground (1)', markerfacecolor='green'),
            Line2D([0], [0], marker='o', linestyle='',
                   label='Off-ground (0)', markerfacecolor='brown')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_class_hist(labels, title="Class distribution"):
    """
    Simple bar plot of class counts.
    labels: 0 = off-ground, 1 = ground
    """
    unique, counts = np.unique(labels, return_counts=True)
    label_names = {0: "off-ground", 1: "ground"}
    names = [label_names.get(u, str(u)) for u in unique]

    plt.figure(figsize=(5, 4))
    plt.bar(names, counts)
    plt.ylabel("Number of points")
    plt.title(title)
    for i, c in enumerate(counts):
        plt.text(i, c, str(c), ha="center", va="bottom")
    plt.tight_layout()
    plt.show()


def plot_2d_profile(points, labels=None, axis=("x", "z"), title="2D profile"):
    """
    2D profile view, e.g. x vs z or y vs z.
    axis: tuple of axis names, e.g. ("x", "z") or ("y", "z").
    labels: 0 = off-ground, 1 = ground
    """
    idx_map = {"x": 0, "y": 1, "z": 2}
    i = idx_map[axis[0]]
    j = idx_map[axis[1]]

    plt.figure(figsize=(6, 5))
    if labels is None:
        plt.scatter(points[:, i], points[:, j], s=1, alpha=0.5)
    else:
        colors = np.where(labels == 1, "green", "brown")
        plt.scatter(points[:, i], points[:, j], c=colors, s=1, alpha=0.5)

    plt.xlabel(axis[0].upper())
    plt.ylabel(axis[1].upper())
    plt.title(title)
    plt.tight_layout()
    plt.show()


def make_ground_map(points_xy, labels, resolution=1.0, title="Ground map"):
    """
    Create a simple 2D raster (ground/off-ground) from point labels.
    IMPORTANT: void cells (no points) are treated as GROUND.

    points_xy : (N, 2) array -> [x, y]
    labels    : (N,) array of 0/1 (0 = off-ground, 1 = ground)
    resolution: grid size in same units as x/y
    """
    x = points_xy[:, 0]
    y = points_xy[:, 1]

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    nx = int(np.ceil((xmax - xmin) / resolution)) + 1
    ny = int(np.ceil((ymax - ymin) / resolution)) + 1

    # Initialize all cells as ground (1) to respect "void = ground" rule
    grid = np.ones((ny, nx), dtype=np.uint8)

    # Convert point coords to grid indices
    ix = ((x - xmin) / resolution).astype(int)
    iy = ((y - ymin) / resolution).astype(int)

    # For any cell where there is at least one off-ground point, set to 0
    off_ground_mask = (labels == 0)
    ix_off = ix[off_ground_mask]
    iy_off = iy[off_ground_mask]
    grid[iy_off, ix_off] = 0

    # Plot
    plt.figure(figsize=(8, 6))
    img = plt.imshow(grid, origin="lower", extent=[xmin, xmax, ymin, ymax],
                     interpolation="nearest", cmap="viridis")
    plt.colorbar(img, label="1 = ground, 0 = off-ground")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# ============================================================
# 3. MAIN WORKFLOW
# ============================================================

def main():
    print(f"laspy version: {laspy.__version__}")

    # ---------- 3.1 Load labeled LAS ----------
    print("Reading labeled LAS...")
    las_labeled = laspy.read(LABELED_LAS_PATH)

    X_all, feat_names = las_to_features(las_labeled, cell_size=RASTER_RESOLUTION)
    print(f"Features used: {feat_names}")
    print(f"Labeled cloud: {X_all.shape[0]} points, {X_all.shape[1]} features")

    # Classification as NumPy array (SubFieldView-safe)
    cls = np.asarray(las_labeled.classification, dtype=int)

    # --- Inspect available classes in the labeled file ---
    unique_cls, counts_cls = np.unique(cls, return_counts=True)
    print("Raw classification codes in labeled LAS:")
    for c, n in zip(unique_cls, counts_cls):
        print(f"  class {c}: {n} points")

    # ---------- Build training labels y (0 = off-ground, 1 = ground) ----------
    if len(unique_cls) < 2:
        raise ValueError(
            f"Only one classification code found: {unique_cls}. "
            "Need at least two classes to train (ground vs off-ground)."
        )

    if 1 in unique_cls and 2 in unique_cls:
        print("Using explicit mapping: 1 = off-ground (0), 2 = ground (1)")
        mask = np.isin(cls, [1, 2])
        X = X_all[mask]
        cls_used = cls[mask]
        y = np.where(cls_used == 2, 1, 0)
    else:
        print("Classes 1 and 2 not both present → auto-selecting top two classes")
        order = np.argsort(counts_cls)[::-1]  # descending by frequency
        top_two = unique_cls[order][:2]
        code_off = int(min(top_two))
        code_ground = int(max(top_two))

        print(f"Using mapping: {code_off} → off-ground (0), "
              f"{code_ground} → ground (1)")

        mask = np.isin(cls, top_two)
        X = X_all[mask]
        cls_used = cls[mask]
        y = np.where(cls_used == code_ground, 1, 0)

    if X.shape[0] == 0:
        raise ValueError(
            f"No points selected for training. "
            f"Classification codes found: {unique_cls}"
        )

    unique_y, counts_y = np.unique(y, return_counts=True)
    print("Training label counts (after mapping):")
    for u, c in zip(unique_y, counts_y):
        name = "ground" if u == 1 else "off-ground"
        print(f"  {u} ({name}): {c}")

    if len(unique_y) < 2:
        raise ValueError(
            "Only one class present after mapping. Cannot train classifier.\n"
            f"Available cls codes: {unique_cls}"
        )

    # ---------- 3.2 Train ML model ----------
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
            class_weight="balanced",
            random_state=42
        ))
    ])

    print("Training classifier...")
    clf.fit(X_train, y_train)

    y_val_pred = clf.predict(X_val)
    print("\nValidation report (from labeled LAS):")
    print(classification_report(y_val, y_val_pred,
                                target_names=["off-ground", "ground"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_val, y_val_pred))

    # ============ 3.3. Apply model to raw LAS ============
    print("\nReading raw LAS and predicting classes...")
    las_raw = laspy.read(RAW_LAS_PATH)

    X_raw, feat_names_raw = las_to_features(las_raw, cell_size=RASTER_RESOLUTION)
    print(f"Raw cloud: {X_raw.shape[0]} points, features: {feat_names_raw}")

    y_raw_pred = clf.predict(X_raw)        # 0/1
    y_raw_proba = clf.predict_proba(X_raw) # probabilities
    confidence = np.max(y_raw_proba, axis=1)

    print("Prediction done.")
    unique_pred, counts_pred = np.unique(y_raw_pred, return_counts=True)
    print("Predicted label counts in raw cloud:")
    for u, c in zip(unique_pred, counts_pred):
        name = "ground" if u == 1 else "off-ground"
        print(f"  {u} ({name}): {c}")

    # ============ 3.4. Write segmented LAS ============
    print(f"\nWriting segmented LAS to {OUTPUT_LAS_PATH} ...")

    # Convert our labels back to LAS classification codes:
    #   off-ground (0) -> 1
    #   ground (1)     -> 2
    classification_values = np.where(y_raw_pred == 1, 2, 1).astype(np.uint8)
    las_raw.classification = classification_values

    # Store confidence (0–255) in user_data if exists
    if "user_data" in las_raw.point_format.dimension_names:
        las_raw.user_data = (confidence * 255).astype(np.uint8)

    las_raw.write(OUTPUT_LAS_PATH)
    print("Segmented LAS written.")

    # ============ 3.5. Plots ============
    coords_raw = np.vstack([las_raw.x, las_raw.y, las_raw.z]).T

    # Subsample for plotting if necessary
    if coords_raw.shape[0] > MAX_POINTS_PLOT:
        idx = np.random.choice(coords_raw.shape[0], MAX_POINTS_PLOT,
                               replace=False)
        coords_plot = coords_raw[idx]
        y_plot = y_raw_pred[idx]
    else:
        coords_plot = coords_raw
        y_plot = y_raw_pred

    # 3D segmented cloud
    plot_3d_point_cloud(
        coords_plot,
        labels=y_plot,
        title="Segmented raw point cloud (ground vs off-ground)",
        s=1.0,
        alpha=0.6
    )

    # Histogram
    plot_class_hist(y_raw_pred,
                    title="Predicted class distribution in raw cloud")

    # 2D profiles
    plot_2d_profile(coords_plot, labels=y_plot,
                    axis=("x", "z"),
                    title="X-Z profile (segmented)")
    plot_2d_profile(coords_plot, labels=y_plot,
                    axis=("y", "z"),
                    title="Y-Z profile (segmented)")

    # 2D ground map where void cells = ground
    points_xy = coords_raw[:, :2]
    make_ground_map(points_xy, y_raw_pred,
                    resolution=RASTER_RESOLUTION,
                    title="Ground map (void cells treated as ground)")


if __name__ == "__main__":

    main()
