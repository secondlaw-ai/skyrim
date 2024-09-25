import xarray as xr
from typing import Dict, Optional
from loguru import logger


def load_ifs_grib_data(
    file_path: str, filter_by_keys: Optional[Dict] = None
) -> xr.DataArray:
    """
    Load and process GRIB data from a file, including:
    surface, height-specific, meanSea, entireAtmosphere, pressure level data.

    NOTE:
        latitude: 90 to -90
            e.g. array([ 90.  ,  89.75,  89.5 , ..., -89.5 , -89.75, -90.  ])
        longitude: -180 to 180
            e.g. array([-180.  , -179.75, -179.5 , ...,  179.25,  179.5 ,  179.75])

    Args:
        file_path (str): Path to the GRIB file.
        filter_by_keys (Optional[Dict]): Additional filter keys for data loading.

    Returns:
        xr.DataArray: Processed data combining available variables.

    Raises:
        ValueError: If no data could be loaded from any source.
    """

    def load_dataset(keys: Dict) -> Optional[xr.Dataset]:
        try:
            return xr.open_dataset(
                file_path,
                engine="cfgrib",
                backend_kwargs={
                    "filter_by_keys": keys,
                    "read_keys": ["shortName", "values", "valid_time", "step"],
                },
            )
        except Exception as e:
            print(f"Error loading data with filter {keys}: {e}")
            return None

    def process_data(da: xr.DataArray, drop_var: Optional[str] = None) -> xr.DataArray:
        da = da.to_array() if isinstance(da, xr.Dataset) else da
        if drop_var:
            da = da.drop_vars(drop_var)
        da = da.assign_coords(
            time=da.time + da.step
        )  # overwrite time coord which originally holds forecast start time
        da = (
            da.swap_dims({"step": "time"})
            if "step" in da.dims
            else da.expand_dims("time", axis=1)
        )
        return da.drop_vars(["step", "valid_time"])

    filter_by_keys = filter_by_keys or {}
    data_arrays = []

    # Load and process height-specific and surface data
    logger.info("Loading height-specific and surface data")
    for level_type, levels in [
        ("heightAboveGround", [2, 10, 100]),
        ("surface", [None]),
        ("meanSea", [None]),
        ("entireAtmosphere", [None]),
    ]:
        keys = {**filter_by_keys, "typeOfLevel": level_type}
        for level in levels:
            if level is not None:
                keys["level"] = level
            if ds := load_dataset(keys):
                logger.debug(f"Loaded data with keys: {keys}")
                if level in [10, 100]:
                    # NOTE:
                    # there is an issue: we call the api with 10u, it return u10
                    # similarly, we call with 100u, it returns u100
                    # so we need to rename the variable to match the expected name
                    # Rename variables by appending 'm'
                    # no problem with t2m, for some GOD KNOWS WHY
                    ds = ds.rename({var: f"{var}m" for var in ds.data_vars})
                    logger.debug(f"Renamed variables: {ds.data_vars}")
                data_arrays.append(process_data(ds, drop_var=level_type))

    # Load and process pressure level data
    logger.info("Loading pressure level data")
    if ds_pressure := load_dataset({**filter_by_keys, "typeOfLevel": "isobaricInhPa"}):
        for var in ds_pressure.data_vars:
            for level in ds_pressure.isobaricInhPa.values.reshape(-1):
                if "isobaricInhPa" not in ds_pressure[var].dims:
                    # if fetched for single pressure level, no index was set for isobaricInhPa
                    #  (number, step, isobaricInhPa, latitude, longitude) vs (number, step, latitude, longitude)
                    logger.debug(f"Adding isobaricInhPa coordinate to {var}")
                    da = (
                        ds_pressure[var]
                        .assign_coords(variable=f"{var}{int(level)}")
                        .expand_dims("variable", axis=0)
                    )
                else:
                    da = (
                        ds_pressure[var]
                        .sel(isobaricInhPa=level)
                        .assign_coords(variable=f"{var}{int(level)}")
                        .expand_dims("variable", axis=0)
                    )
                data_arrays.append(process_data(da, drop_var="isobaricInhPa"))

    if not data_arrays:
        raise ValueError("No data could be loaded from any source")

    # return data_arrays
    ifs_da = xr.concat(data_arrays, dim="variable")
    ifs_da.name = None
    return ifs_da
