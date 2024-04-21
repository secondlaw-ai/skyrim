from datetime import datetime
import xarray as xr
from typing import Any
from earth2mip import initial_conditions, time_loop
import numpy as np
import torch
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
    return xr.DataArray(
        stacked, dims=["time", "history", "channel", "lat", "lon"], coords=coords
    )


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
