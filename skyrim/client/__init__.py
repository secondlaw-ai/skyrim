import xarray as xr


# TODO: move Prediction class to common level and import here


def read_forecast(zarr_store_path: str, forecast_id: str):
    """Read forecast from a zarr_store"""
    return xr.open_dataset(zarr_store_path + "/" + forecast_id, engine="zarr")
