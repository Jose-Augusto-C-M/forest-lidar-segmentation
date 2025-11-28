# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 14:24:09 2025

@author: joms0005
"""
from __future__ import annotations
#!/usr/bin/env python3
"""
Batch overlap between a survey table and many LAS/LAZ files.

For each LAS that has at least one overlapping table row, this script creates:

    <stem>_overlap.csv
    <stem>_table_points.las
    <stem>_cropR<radius>.las
    <stem>_overlap_map.png
    <stem>_features.tif  (GeoTIFF with geometric features)

Features in *_features.tif (bands):
  1: mean elevation
  2: elevation std-dev (vertical roughness)
  3: point density (points / m²)
  4: slope (from mean elevation)
"""



import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Iterable, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import laspy
from laspy import ExtraBytesParams
from pyproj import CRS, Transformer
from tqdm import tqdm
import h5py


import rasterio
from rasterio.transform import from_origin

# =========================================================
#                    USER SETTINGS
# =========================================================

# Folders where LAS/LAZ are stored (searched recursively)
ROOT_DIRS: List[str] = [
    "D:/SLU_biodiversity/STORAENSO_X/01_SLU/01_LAS",
    "D:/SLU_biodiversity/STORAENSO_X/02_SLU/01_LAS",
    "D:/SLU_biodiversity/STORAENSO_X/03_SLU/01_LAS",
    "D:/SLU_biodiversity/STORAENSO_X/04_SLU/01_LAS",
    "D:/SLU_biodiversity/STORAENSO_X/05_SLU/01_LAS",
    "D:/SLU_biodiversity/STORAENSO_X/06_SLU/01_LAS",
    "D:/SLU_biodiversity/STORAENSO_X/07_SLU/01_LAS",
    "D:/SLU_biodiversity/STORAENSO_X/08_SLU/01_LAS",
    "D:/SLU_biodiversity/STORAENSO_X/09_SLU/01_LAS",
    "D:/SLU_biodiversity/STORAENSO_X/10_SLU/01_LAS",
    "D:/SLU_biodiversity/STORAENSO_X/11_SLU/01_LAS",
    "D:/SLU_biodiversity/STORAENSO_X/12_SLU/01_LAS",
    "D:/SLU_biodiversity/STORAENSO_X/13_SLU/01_LAS",
    "D:/SLU_biodiversity/STORAENSO_X/14_SLU/01_LAS",
    "D:/SLU_biodiversity/STORAENSO_X/15_SLU/01_LAS",
    "D:/SLU_biodiversity/STORAENSO_X/16_SLU/01_LAS",
    "D:/SLU_biodiversity/STORAENSO_X/17_SLU/01_LAS",
    "D:/SLU_biodiversity/STORAENSO_X/18_SLU/01_LAS",
    # add more paths as needed, e.g., "./block_A", "./block_B"
]

# Where to put all outputs. The script mirrors the input subfolders under here.
OUTPUT_DIR = "D:/SLU_biodiversity/STORAENSO_X/MATCHES_overlap_las_6"

# Your table file (CSV or Excel) with the header you provided.
TABLE_PATH = r"//storage-ume.slu.se/home$/joms0005/Desktop/SLU/Vindeln_2025/tables/Halsingland_Oct2025.csv"


# Column names in your table
COL_E   = "Easting"
COL_N   = "Northing"
COL_Z   = "Elevation"         # table Z column to use
COL_LON = "Longitude"
COL_LAT = "Latitude"
COL_NAME= "Name"
COL_CODE= "Code"

# CRS:
TABLE_EPSG = 3006             # your case: SWEREF 99 TM
LAS_EPSG_FALLBACK = 3006      # if LAS has no CRS in header

# Overlap settings
BUFFER_METERS = 0.0           # extra around LAS bbox when selecting table points

# Vertical offset (table -> LAS)
# If the table elevations are systematically higher or lower than LAS by ~6 m:
#   Z_las ≈ Z_table + TABLE_Z_OFFSET
APPLY_TABLE_Z_OFFSET = False
TABLE_Z_OFFSET = -6.0         # if table Z ~6 m higher than LAS, use -6.0

# Cropped LAS settings (rectangular box around the group + radius)
WRITE_CROPPED_LAS = True
CROP_RADIUS = 100.0          # meters beyond min/max XY of matched points
CROP_SUFFIX = f"_cropR{int(CROP_RADIUS)}"

# Table points LAS
WRITE_TABLE_POINTS_LAS = True
TABLE_CLASSIFICATION = 12     # classification value for table points

# PNG figure settings
MAKE_FIGURES = True
MAX_POINTS_TO_PLOT = 30_000   # decimated points for background cloud
FIG_DPI = 120                 # slightly lower DPI for speed

# Raster settings (GeoTIFF)
MAKE_RASTERS = False
RASTER_RES = 1.0              # cell size in meters (1.0 or 2.0 typically)

# Write an HDF5 copy of the LAS (cropped)
WRITE_H5 = False

# Execution
EXTS = (".las", ".laz")
MAX_WORKERS = 1               # keep it simple & predictable (no multiprocessing)
DEBUG_EXCEPTIONS = True
AUTO_RETRY_WGS84 = True       # if 3006 projection finds no overlaps, try 4326
VERBOSE = True

# Chunk size for reading LAS
CHUNK_SIZE = 1_000_000


# =========================================================
#                  UTILITY FUNCTIONS
# =========================================================

def _log(msg: str):
    if VERBOSE:
        print(msg)


def _trace():
    if DEBUG_EXCEPTIONS:
        import traceback
        traceback.print_exc()


def find_las_files(root_dirs: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for root in root_dirs:
        p = Path(root)
        if not p.exists():
            continue
        for ext in EXTS:
            files.extend(p.rglob(f"*{ext}"))
    return sorted(set(files))


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def las_header_bounds(path: Path) -> Tuple[float, float, float, float, int]:
    with laspy.open(path) as r:
        hdr = r.header
        xmin, ymin = hdr.mins[0], hdr.mins[1]
        xmax, ymax = hdr.maxs[0], hdr.maxs[1]
        npts = int(hdr.point_count)
    return float(xmin), float(xmax), float(ymin), float(ymax), npts


def las_epsg_for(path: Path) -> Optional[int]:
    try:
        with laspy.open(path) as r:
            crs = r.header.parse_crs()
        if crs is None:
            return None
        return CRS.from_user_input(crs).to_epsg()
    except Exception:
        return None


def build_transformer(src_epsg: Optional[int], dst_epsg: Optional[int]) -> Optional[Transformer]:
    if src_epsg is None or dst_epsg is None:
        return None
    return Transformer.from_crs(
        CRS.from_epsg(src_epsg),
        CRS.from_epsg(dst_epsg),
        always_xy=True,
    )


def choose_input_xy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, str]:
    have_EN = (
        COL_E in df and COL_N in df and
        df[COL_E].notna().any() and df[COL_N].notna().any()
    )
    have_LL = (
        COL_LON in df and COL_LAT in df and
        df[COL_LON].notna().any() and df[COL_LAT].notna().any()
    )
    if have_EN:
        return (
            df[COL_E].astype(float).to_numpy(),
            df[COL_N].astype(float).to_numpy(),
            "EN",
        )
    if have_LL:
        return (
            df[COL_LON].astype(float).to_numpy(),
            df[COL_LAT].astype(float).to_numpy(),
            "LL",
        )
    raise ValueError("Table missing coordinate columns (Easting/Northing or Longitude/Latitude).")


def project_table_xy(df: pd.DataFrame, table_epsg: Optional[int], las_epsg: Optional[int]) -> pd.DataFrame:
    x_in, y_in, _ = choose_input_xy(df)
    tr = build_transformer(table_epsg, las_epsg)
    if tr:
        x, y = tr.transform(x_in, y_in)
    else:
        x, y = x_in, y_in
    out = df.copy()
    out["_x_las"] = x
    out["_y_las"] = y
    return out


def bbox_overlap(df_proj: pd.DataFrame, las_bbox: Tuple[float, float, float, float], buffer_m: float) -> pd.DataFrame:
    xmin, xmax, ymin, ymax = las_bbox
    xmin -= buffer_m
    ymin -= buffer_m
    xmax += buffer_m
    ymax += buffer_m

    mask = (
        (df_proj["_x_las"] >= xmin) & (df_proj["_x_las"] <= xmax) &
        (df_proj["_y_las"] >= ymin) & (df_proj["_y_las"] <= ymax)
    )
    return df_proj.loc[mask].copy()


def decimate_points(x: np.ndarray, y: np.ndarray, z: np.ndarray, max_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = x.size
    if n <= max_points:
        return x, y, z
    rng = np.random.default_rng(0)  # reproducible
    idx = rng.choice(n, size=max_points, replace=False)
    return x[idx], y[idx], z[idx]


def make_figure(las: laspy.LasData, df_overlap_xy: pd.DataFrame, bbox, out_path: Path):
    xmin, xmax, ymin, ymax = bbox

    x = np.asarray(las.x)
    y = np.asarray(las.y)
    z = np.asarray(getattr(las, "z", np.zeros_like(x, dtype=float)))

    x, y, z = decimate_points(x, y, z, MAX_POINTS_TO_PLOT)

    plt.figure(figsize=(8, 7), dpi=FIG_DPI)
    sc = plt.scatter(x, y, c=z, s=1, alpha=0.8, rasterized=True)
    cbar = plt.colorbar(sc)
    cbar.set_label("Elevation (LAS z)")

    if not df_overlap_xy.empty:
        plt.scatter(
            df_overlap_xy["_x_las"],
            df_overlap_xy["_y_las"],
            s=25,
            marker="o",
            edgecolors="k",
            linewidths=0.5,
            c="none",
        )

        name_series = df_overlap_xy[COL_NAME] if COL_NAME in df_overlap_xy.columns else None
        code_series = df_overlap_xy[COL_CODE] if COL_CODE in df_overlap_xy.columns else None

        for _, row in df_overlap_xy.iterrows():
            bits = []
            if name_series is not None and pd.notna(row.get(COL_NAME, None)) and str(row.get(COL_NAME, "")).strip():
                bits.append(str(row[COL_NAME]))
            if code_series is not None and pd.notna(row.get(COL_CODE, None)) and str(row.get(COL_CODE, "")).strip():
                bits.append(f"[{row[COL_CODE]}]")
            label = " ".join(bits)
            if label:
                plt.annotate(
                    label,
                    (row["_x_las"], row["_y_las"]),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.6, ec="none"),
                )

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("X (LAS CRS)")
    plt.ylabel("Y (LAS CRS)")
    plt.title("Table–LAS Overlap (top-down)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


# =========================================================
#     TABLE-POINTS LAS WITH VERTICAL OFFSET APPLIED
# =========================================================

def write_table_points_las(
    template_las_path: Path,
    df_overlap: pd.DataFrame,
    out_points_las: Path,
    classification_value: int = TABLE_CLASSIFICATION,
):
    """Write a LAS containing ONLY the overlapping table rows as points."""
    if df_overlap.empty:
        return

    with laspy.open(template_las_path) as reader:
        hdr = reader.header
        pf = hdr.point_format
        scales = hdr.scales
        offsets = hdr.offsets
        crs = hdr.parse_crs()
        try:
            min_z = float(hdr.mins[2])
        except Exception:
            min_z = 0.0

    new_hdr = laspy.LasHeader(point_format=pf, version=hdr.version)
    new_hdr.scales = scales
    new_hdr.offsets = offsets
    if crs is not None:
        new_hdr.parse_crs(crs)

    las = laspy.LasData(new_hdr)

    x = df_overlap["_x_las"].to_numpy(dtype=np.float64)
    y = df_overlap["_y_las"].to_numpy(dtype=np.float64)

    # Z with optional offset
    if COL_Z in df_overlap.columns and df_overlap[COL_Z].notna().any():
        z_vals = df_overlap[COL_Z].astype(float).to_numpy()
        if APPLY_TABLE_Z_OFFSET:
            z_vals = z_vals + TABLE_Z_OFFSET
        z = z_vals
    else:
        z = np.full(len(x), min_z, dtype=np.float64)

    las.x = x
    las.y = y
    las.z = z

    # Basic dims
    if "classification" in las.point_format.dimension_names:
        las.classification = np.full(len(x), classification_value, dtype=np.uint8)
    if "intensity" in las.point_format.dimension_names:
        las.intensity = np.zeros(len(x), dtype=np.uint16)
    if "user_data" in las.point_format.dimension_names:
        las.user_data = np.zeros(len(x), dtype=np.uint8)

    # One numeric ExtraByte: table_id
    las.add_extra_dims([
        ExtraBytesParams(name="table_id", type=np.uint32),
    ])
    las["table_id"] = np.arange(len(x), dtype=np.uint32)

    out_points_las.parent.mkdir(parents=True, exist_ok=True)
    las.write(out_points_las)


# =========================================================
#  CROPPED LAS: RECTANGULAR BUFFER AROUND GROUP (FAST)
# =========================================================

def write_cropped_las_box_buffer(
    src_las_path: Path,
    crop_bounds: Tuple[float, float, float, float],  # (xmin, xmax, ymin, ymax)
    out_cropped_path: Path,
    chunk_size: int = CHUNK_SIZE,
):
    """
    Fast crop: keep ALL original LAS points inside a rectangular crop window:

        [xmin, xmax] × [ymin, ymax]

    Uses writer.write_points; compatible with laspy 2.6.1.
    """
    xmin, xmax, ymin, ymax = crop_bounds

    with laspy.open(src_las_path) as src:
        header = src.header

        new_hdr = laspy.LasHeader(
            point_format=header.point_format,
            version=header.version,
        )
        new_hdr.scales = header.scales
        new_hdr.offsets = header.offsets

        crs = header.parse_crs()
        if crs is not None:
            new_hdr.parse_crs(crs)

        out_cropped_path.parent.mkdir(parents=True, exist_ok=True)
        total_selected = 0

        with laspy.open(out_cropped_path, mode="w", header=new_hdr) as writer:
            for pts in src.chunk_iterator(chunk_size):
                x = np.asarray(pts.x, dtype=np.float64)
                y = np.asarray(pts.y, dtype=np.float64)

                mask = (
                    (x >= xmin) & (x <= xmax) &
                    (y >= ymin) & (y <= ymax)
                )

                if mask.any():
                    writer.write_points(pts[mask])
                    total_selected += int(mask.sum())

        if VERBOSE:
            print(
                f"[WRITE] LAS(crop box) -> "
                f"{out_cropped_path.name} ({total_selected} pts; bounds = "
                f"{xmin:.2f},{ymin:.2f}–{xmax:.2f},{ymax:.2f})"
            )



# =========================================================
#  RASTER (GeoTIFF) FROM CROPPED LAS: DSM (Z ONLY)
# =========================================================

def make_feature_raster_from_las(
    las_path: Path,
    out_tif: Path,
    res: float,
    epsg: Optional[int],
):
    """
    Create a single-band DSM raster (mean elevation Z) from a LAS file.

    IMPORTANT:
        - Call this with the *cropped* LAS as input (so the raster
          extent matches the cropped area).
        - Raster extent = [xmin, xmax] × [ymin, ymax] of this LAS.
        - Pixel values   = mean Z in each cell.

    This is a standard orthogonal X–Y grid with Z as the cell value.
    """
    las = laspy.read(las_path)
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)

    if x.size == 0:
        return

    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())

    width  = int(np.ceil((xmax - xmin) / res))
    height = int(np.ceil((ymax - ymin) / res))
    if width <= 0 or height <= 0:
        return

    # indices into grid
    ix = np.floor((x - xmin) / res).astype(int)
    # raster row 0 at top (ymax)
    iy = np.floor((ymax - y) / res).astype(int)

    valid = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
    ix = ix[valid]
    iy = iy[valid]
    z  = z[valid]

    idx = iy * width + ix
    n_cells = width * height

    count = np.bincount(idx, minlength=n_cells).reshape(height, width)
    sum_z = np.bincount(idx, weights=z, minlength=n_cells).reshape(height, width)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_z = sum_z / np.where(count > 0, count, np.nan)

    # GeoTIFF profile: top-left origin at (xmin, ymax)
    transform = from_origin(xmin, ymax, res, res)
    out_tif.parent.mkdir(parents=True, exist_ok=True)

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,               # ONE band = Z
        "dtype": "float32",
        "transform": transform,
        "nodata": np.nan,
        "compress": "lzw",
    }
    if epsg is not None:
        profile["crs"] = CRS.from_epsg(epsg)

    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(mean_z.astype(np.float32), 1)

    if VERBOSE:
        print(f"[WRITE] DSM GeoTIFF -> {out_tif.name} (extent = cropped LAS)")



# =========================================================
#        SAVE LAS (CROPPED) AS HDF5 (H5) FILE
# =========================================================

def write_las_to_h5(las_path: Path, out_h5: Path):
    """
    Save a LAS/LAZ file into HDF5 format.

    Structure:
        /attrs: basic metadata (epsg, scales, offsets, point_format)
        /points/<dimension>: datasets for each LAS dimension
            e.g. X, Y, Z, intensity, classification, etc.

    Call this with the *cropped* LAS so the H5 represents the cropped area.
    """
    las = laspy.read(las_path)

    dims = list(las.point_format.dimension_names)
    npts = las.header.point_count

    out_h5.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_h5, "w") as f:
        # file-level metadata
        f.attrs["source_las"] = str(las_path)
        f.attrs["point_count"] = int(npts)

        try:
            crs = las.header.parse_crs()
            if crs is not None:
                epsg = CRS.from_user_input(crs).to_epsg()
                if epsg is not None:
                    f.attrs["epsg"] = int(epsg)
        except Exception:
            pass

        f.attrs["point_format_id"] = int(las.header.point_format.id)
        f.attrs["version_major"] = int(las.header.version.major)
        f.attrs["version_minor"] = int(las.header.version.minor)
        f.attrs["scales"] = np.array(las.header.scales, dtype="float64")
        f.attrs["offsets"] = np.array(las.header.offsets, dtype="float64")

        # point data
        grp = f.create_group("points")
        for dim in dims:
            arr = np.asarray(getattr(las, dim))
            grp.create_dataset(
                dim,
                data=arr,
                compression="gzip",
                shuffle=True,
                fletcher32=True,
            )

    if VERBOSE:
        print(f"[WRITE] H5 -> {out_h5.name} (from {las_path.name})")




# =========================================================
#                  PROCESS ONE LAS
# =========================================================

@dataclass
class JobResult:
    las_file: str
    overlap_rows: int
    out_csv: Optional[str]
    out_png: Optional[str]
    out_table_las: Optional[str]
    out_crop_las: Optional[str]
    out_raster: Optional[str]
    out_h5: Optional[str]


_table_proj_cache: Dict[Optional[int], pd.DataFrame] = {}


def get_projected_table(df_table: pd.DataFrame, las_epsg: Optional[int]) -> pd.DataFrame:
    if las_epsg not in _table_proj_cache:
        _table_proj_cache[las_epsg] = project_table_xy(df_table, TABLE_EPSG, las_epsg)
    return _table_proj_cache[las_epsg]


def process_one(
    las_path: Path,
    df_table: pd.DataFrame,
    search_roots: List[Path],
) -> Optional[JobResult]:
    try:
        xmin, xmax, ymin, ymax, npts = las_header_bounds(las_path)
        las_bbox = (xmin, xmax, ymin, ymax)

        las_epsg = las_epsg_for(las_path) or LAS_EPSG_FALLBACK
        if las_epsg is None:
            _log(
                f"[NO-CRS] {las_path.name}: LAS has no CRS and "
                f"LAS_EPSG_FALLBACK=None -> assuming table already in LAS units."
            )

        df_proj = get_projected_table(df_table, las_epsg)
        df_overlap = bbox_overlap(df_proj, las_bbox, BUFFER_METERS)

        retried = False
        if df_overlap.empty and AUTO_RETRY_WGS84 and TABLE_EPSG != 4326:
            retried = True
            _log(
                f"[RETRY] {las_path.name}: 0 matches with TABLE_EPSG={TABLE_EPSG}; trying 4326..."
            )
            df_proj_retry = project_table_xy(df_table, 4326, las_epsg)
            df_overlap = bbox_overlap(df_proj_retry, las_bbox, BUFFER_METERS)
            if not df_overlap.empty:
                _log(
                    f"[RETRY-SUCCESS] {las_path.name}: "
                    f"{len(df_overlap)} matches with TABLE_EPSG=4326."
                )
                df_proj = df_proj_retry
            else:
                _log(
                    f"[RETRY-FAIL] {las_path.name}: 0 matches with TABLE_EPSG=4326."
                )

        if df_overlap.empty:
            _log(
                f"[SKIP] {las_path.name}: 0 matches "
                f"(bbox {xmin:.2f},{ymin:.2f}–{xmax:.2f},{ymax:.2f}; "
                f"EPSG={las_epsg}, retried={retried})."
            )
            return None

        _log(f"[MATCH] {las_path.name}: {len(df_overlap)} rows (EPSG={las_epsg}).")

        parent_root = next(
            (r for r in search_roots if str(las_path).startswith(str(r))),
            None,
        )
        rel = las_path.parent.relative_to(parent_root) if parent_root else Path(".")
        out_dir = Path(OUTPUT_DIR) / rel
        out_dir.mkdir(parents=True, exist_ok=True)

        stem = las_path.stem
        out_csv       = out_dir / f"{stem}_overlap.csv"
        out_png       = out_dir / f"{stem}_overlap_map.png" if MAKE_FIGURES else None
        out_table_las = out_dir / f"{stem}_table_points.las" if WRITE_TABLE_POINTS_LAS else None
        out_crop_las  = out_dir / f"{stem}{CROP_SUFFIX}.las" if WRITE_CROPPED_LAS else None
        out_raster    = out_dir / f"{stem}_features.tif" if MAKE_RASTERS else None

        # CSV without helper XY columns
        df_overlap_nohelper = df_overlap.drop(columns=["_x_las", "_y_las"], errors="ignore")
        df_overlap_nohelper.to_csv(out_csv, index=False)
        _log(f"[WRITE] CSV -> {out_csv.name}")

        # Table-points LAS
        if WRITE_TABLE_POINTS_LAS and out_table_las is not None:
            write_table_points_las(
                template_las_path=las_path,
                df_overlap=df_overlap,
                out_points_las=out_table_las,
                classification_value=TABLE_CLASSIFICATION,
            )
            _log(f"[WRITE] LAS(table) -> {out_table_las.name}")

        # Cropped LAS (original points in rectangular buffer)
        # Cropped LAS (original points in rectangular buffer)
        crop_source_path = las_path
        crop_bounds = las_bbox  # fallback to full tile if needed

        if WRITE_CROPPED_LAS and out_crop_las is not None:
            overlap_xy = df_overlap[["_x_las", "_y_las"]].to_numpy()

            if overlap_xy.size > 0:
                xmin_pts = float(overlap_xy[:, 0].min())
                xmax_pts = float(overlap_xy[:, 0].max())
                ymin_pts = float(overlap_xy[:, 1].min())
                ymax_pts = float(overlap_xy[:, 1].max())
                crop_bounds = (
                    xmin_pts - CROP_RADIUS,
                    xmax_pts + CROP_RADIUS,
                    ymin_pts - CROP_RADIUS,
                    ymax_pts + CROP_RADIUS,
                )
            else:
                crop_bounds = las_bbox

            write_cropped_las_box_buffer(
                src_las_path=las_path,
                crop_bounds=crop_bounds,
                out_cropped_path=out_crop_las,
                chunk_size=CHUNK_SIZE,
            )
            crop_source_path = out_crop_las  # for rasters & figures

        # Raster from cropped LAS, using the same crop_bounds
        if MAKE_RASTERS and out_raster is not None:
            make_feature_raster_from_las(
                las_path=crop_source_path,   # should be out_crop_las
                out_tif=out_raster,
                res=RASTER_RES,
                epsg=las_epsg,
            )


        # PNG figure (use cropped LAS + crop_bounds)
        if MAKE_FIGURES and out_png is not None:
            las_data = laspy.read(crop_source_path)
            make_figure(
                las=las_data,
                df_overlap_xy=df_overlap,
                bbox=crop_bounds,
                out_path=out_png,
            )
            _log(f"[WRITE] PNG -> {out_png.name}")
            
        # HDF5 (H5) from cropped LAS
        out_h5 = None
        if WRITE_H5:
            h5_stem = Path(crop_source_path).stem
            out_h5_path = out_dir / f"{h5_stem}.h5"
            write_las_to_h5(crop_source_path, out_h5_path)
            out_h5 = str(out_h5_path)




        return JobResult(
            las_file=str(las_path),
            overlap_rows=len(df_overlap),
            out_csv=str(out_csv),
            out_png=str(out_png) if out_png else None,
            out_table_las=str(out_table_las) if out_table_las else None,
            out_crop_las=str(out_crop_las) if out_crop_las else None,
            out_raster=str(out_raster) if out_raster else None,
            out_h5=out_h5,
        )


    except Exception:
        _trace()
        return None


# =========================================================
#                         MAIN
# =========================================================

def main():
    df_table = read_table(Path(TABLE_PATH))
    if df_table.empty:
        print("Table is empty or could not be read.")
        return

    las_files = find_las_files(ROOT_DIRS)
    if not las_files:
        print("No LAS/LAZ files found.")
        return

    print(f"Found {len(las_files)} LAS/LAZ files.")

    results: List[JobResult] = []
    search_roots = [Path(r) for r in ROOT_DIRS]

    # Simple sequential processing (no multiprocessing headaches)
    for las_path in tqdm(las_files, desc="Processing", unit="file"):
        res = process_one(las_path, df_table, search_roots)
        if res is not None:
            results.append(res)

    if not results:
        print("No overlaps found in any LAS.")
        return

    out_root = Path(OUTPUT_DIR)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "batch_summary.csv"
    pd.DataFrame([r.__dict__ for r in results]).to_csv(summary_path, index=False)
    print(f"[WRITE] Summary -> {summary_path}")


if __name__ == "__main__":
    main()
