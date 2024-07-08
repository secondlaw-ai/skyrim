import xarray as xr
import numpy as np
import torch
from datetime import datetime
from typing import Any
from earth2mip import initial_conditions, time_loop
from loguru import logger


def run_basic_inference(
    model: time_loop.TimeLoop,
    n: int,
    data_source: Any,
    time: datetime,
    x: str | xr.DataArray | None = None,
):
    """Run a basic inference"""
    if x is None:
        x = initial_conditions.get_initial_condition_for_model(model, data_source, time)
        logger.info(f"Fetching initial conditions from data source")
    else:
        logger.info(f"Using provided initial conditions")
        if isinstance(x, str):
            x = torch.Tensor(xr.open_dataarray(x).values[-1:]).to(model.device)
        else:
            x = torch.Tensor(x.values[-1:]).to(model.device)

    arrays = []
    times = []
    for k, (time, data, _) in enumerate(model(time, x)):
        arrays.append(data.cpu().numpy())
        times.append(time)
        if k == n:
            break

    stacked = np.stack(arrays)
    coords = dict(lat=model.grid.lat, lon=model.grid.lon)
    coords["channel"] = model.out_channel_names
    coords["time"] = times
    return xr.DataArray(stacked, dims=["time", "history", "channel", "lat", "lon"], coords=coords)


def estimate_pressure_hpa(elevation_m):
    """
    Estimate atmospheric pressure at a given elevation using the Barometric Formula.

    :param elevation_m: Elevation in meters.
    :return: Atmospheric pressure in Pascals.
    """
    P0 = 101325  # Sea level standard atmospheric pressure (Pa)
    L = 0.0065  # Standard temperature lapse rate (K/m)
    T0 = 288.15  # Standard temperature at sea level (K)
    g = 9.80665  # Acceleration due to gravity (m/s^2)
    M = 0.0289644  # Molar mass of Earth's air (kg/mol)
    R = 8.31447  # Universal gas constant (J/(mol·K))

    P = P0 * (1 - (L * elevation_m) / T0) ** (g * M / (R * L))
    return P / 100  # Convert Pa to hPa


def perturb_initial_conditions(initial_conditions, channel, lat, lon, value):
    """
    Perturb the initial conditions by setting a specific value for a given channel at a specific lat/lon.

    Parameters:
    - initial_conditions (xr.DataArray): The initial conditions dataset to be perturbed.
    - channel (str): Variable name to modify, e.g. 't2m'
    - lat (float): Latitude where the value should be set.
    - lon (float): Longitude where the value should be set.
    - value (float): New value to set at the specified location.
    """

    # Ensure longitude is within the expected range
    if lon < 0:
        lon += 360

    # Select the nearest latitude and longitude if exact values are not present
    closest_lat = initial_conditions.sel(lat=lat, method="nearest").lat.item()
    closest_lon = initial_conditions.sel(lon=lon, method="nearest").lon.item()

    # Set the new value
    initial_conditions[channel].loc[dict(lat=closest_lat, lon=closest_lon)] = value
    return initial_conditions
