import rasterio
from rasterio.warp import reproject, Resampling
from affine import Affine
import numpy as np
import xarray as xr

# -------------------------------------------------------------------
# INPUTS
# -------------------------------------------------------------------
bcw_tif = "/capacity/occr_davin/trigny/chari_P2/Jung_PA/BiodiversityCarbonWater/10km/minshort_speciestargetswithPA_carbon__water__esh10km_repruns10_ranked.tif"

# TEMPLATE NETCDF for target grid
template_nc = "/capacity/occr_davin/mguzman/chari_P2_review/data/control_wdpa17_2015_hist.nc"

out_nc = "/capacity/occr_davin/mguzman/chari_P2_review/data/PA_masks/bcw_biome_withPA.nc"

# -------------------------------------------------------------------
# READ INPUT RASTER (Mollweide)
# -------------------------------------------------------------------
with rasterio.open(bcw_tif) as src:
    data = src.read(1).astype(np.float32)
    src_transform = src.transform
    src_crs = src.crs
    src_nodata = src.nodata

# Replace nodata with NaN
data[data == src_nodata] = np.nan

# -------------------------------------------------------------------
# Mask to top 30%
data = (data < 30).astype(np.float32)
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# SET TARGET GRID USING TEMPLATE
# -------------------------------------------------------------------
tpl = xr.open_dataset(template_nc)
lon = tpl.lon.values
lat = tpl.lat.values

res_lon = np.abs(lon[1] - lon[0])  # ~0.5°
res_lat = np.abs(lat[1] - lat[0])  # ~0.5°

dst_transform = Affine(res_lon, 0, lon.min() - res_lon/2,
                       0, -res_lat, lat.max() + res_lat/2)

dst_height = len(lat)
dst_width = len(lon)
dst_crs = "EPSG:4326"

# Allocate output grid
dst_data = np.full((dst_height, dst_width), np.nan, dtype=np.float32)

# -------------------------------------------------------------------
# REPROJECT WITH AREA‑WEIGHTED AGGREGATION
# -------------------------------------------------------------------
reproject(
    source=data,
    destination=dst_data,
    src_transform=src_transform,
    src_crs=src_crs,
    dst_transform=dst_transform,
    dst_crs=dst_crs,
    resampling=Resampling.average,  # <-- AREA-WEIGHTED AGGREGATION
    src_nodata=np.nan,
    dst_nodata=np.nan
)

# -------------------------------------------------------------------
# SAVE TO NETCDF
# -------------------------------------------------------------------
da = xr.DataArray(
    dst_data,
    dims=("lat", "lon"),
    coords={"lat": lat, "lon": lon},
    name="fraction"
)

ds = xr.Dataset({"fraction": da})
ds.attrs["description"] = "10km NatureMap layer aggregated to 0.5° Plate Carree using area-weighted averaging."
ds.to_netcdf(out_nc)

print("Wrote:", out_nc)
