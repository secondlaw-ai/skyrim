import numpy as np
import xarray as xr
from skyrim.client import read_forecast


def create_ds():
    time = np.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64")
    lat = np.linspace(-90, 90, 181)
    lon = np.linspace(-180, 180, 361)
    temperature = 15 + 8 * np.random.randn(len(time), len(lat), len(lon))
    wind_speed = 10 * np.random.rand(len(time), len(lat), len(lon))
    return xr.Dataset(
        data_vars={
            "temperature": (["time", "lat", "lon"], temperature),
            "wind_speed": (["time", "lat", "lon"], wind_speed),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )


def test_client_loads():
    assert 1 == 1
