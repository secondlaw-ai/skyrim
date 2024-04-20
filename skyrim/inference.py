from datetime import datetime
import xarray
from typing import Any, Optional
from earth2mip import initial_conditions, regrid, time_loop
import numpy as np
import torch
from loguru import logger


def run_basic_inference(
    model: time_loop.TimeLoop,
    n: int,
    data_source: Any,
    time: datetime,
    x: str | xarray.DataArray | None = None,
):
    """Run a basic inference"""
    if x is None:
        x = initial_conditions.get_initial_condition_for_model(model, data_source, time)
        logger.info(f"Fetching initial conditions from data source")
    else:
        logger.info(f"Using provided initial conditions")
        if isinstance(x, str):
            x = torch.Tensor(xarray.open_dataarray(x).values[-1:]).to(model.device)
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
    return xarray.DataArray(
        stacked, dims=["time", "history", "channel", "lat", "lon"], coords=coords
    )
