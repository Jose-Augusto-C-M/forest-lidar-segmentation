# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 16:53:10 2025

@author: joms0005
"""

import rasterio
import h5py
import numpy as np
from pathlib import Path


def tif_to_hdf5(tif_path, h5_path, dataset_name="image",
                compression="gzip", compression_level=4):
    """
    Converts a GeoTIFF (.tif) to an HDF5 (.h5) file.
    """

    tif_path = r"E:/MATCHES_overlap - Copy/MATCHES_overlap_las_2/outputs_logs_fusion/raster18.tif"
    h5_path = r"E:/MATCHES_overlap - Copy/MATCHES_overlap_las_2/outputs_logs_fusion/raster18_file.h5"

    print(f"Reading:  {tif_path}")
    with rasterio.open(tif_path) as src:
        data = src.read()  # shape: (bands, height, width)
        meta = src.meta.copy()
        transform = src.transform
        crs = src.crs

    print(f"Writing: {h5_path}")
    with h5py.File(h5_path, "w") as h5f:

        # Write dataset
        dset = h5f.create_dataset(
            dataset_name,
            data=data,
            compression=compression,
            compression_opts=compression_level
        )

        # Write metadata as dataset attributes
        dset.attrs["height"] = data.shape[1]
        dset.attrs["width"] = data.shape[2]
        dset.attrs["bands"] = data.shape[0]
        dset.attrs["dtype"] = str(data.dtype)
        dset.attrs["driver"] = meta.get("driver", "")
        dset.attrs["nodata"] = meta.get("nodata", np.nan)
        dset.attrs["count"] = meta.get("count", data.shape[0])

        # GeoTransform
        dset.attrs["transform_a"] = transform.a
        dset.attrs["transform_b"] = transform.b
        dset.attrs["transform_c"] = transform.c
        dset.attrs["transform_d"] = transform.d
        dset.attrs["transform_e"] = transform.e
        dset.attrs["transform_f"] = transform.f

        # CRS
        if crs:
            dset.attrs["crs_wkt"] = crs.to_wkt()
        else:
            dset.attrs["crs_wkt"] = ""

    print("Done.")


def main():
    # --------------------------------------------
    # 🔧 EDIT THESE PATHS FOR YOUR SYSTEM
    # --------------------------------------------

    tif_path = r"C:\Users\YourName\path\to\input.tif"
    h5_path = r"C:\Users\YourName\path\to\output.h5"

    # --------------------------------------------

    tif_to_hdf5(
        tif_path=tif_path,
        h5_path=h5_path,
        dataset_name="image",
        compression="gzip",
        compression_level=4
    )


if __name__ == "__main__":
    main()
