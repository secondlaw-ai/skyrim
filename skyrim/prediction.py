import os
import datetime
from pathlib import Path

from loguru import logger
import xarray as xr


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


class GlobalPrediction:
    filepath: str = None
    prediction: xr.Dataset | xr.DataArray = None

    def __init__(self, source):
        if isinstance(source, str):
            self.filepath = source
            self.prediction = xr.open_dataarray(source).squeeze()

        elif isinstance(source, xr.Dataset) or isinstance(source, xr.DataArray):
            self.filepath = None
            self.prediction = source.squeeze()  # get rid of the dimensions with size 1
        else:
            raise ValueError("Invalid source type.")

    @property
    def coords(self):
        return self.prediction.coords

    @property
    def size(self):
        return self.prediction.size

    @property
    def channel(self):
        return self.prediction.channel

    def slice(
        self,
        lat: slice | None = None,
        lon: slice | None = None,
        channel: str | None = None,
        n_step: slice | None = None,
    ):
        """
        Returns a slice of the dataset according to specified dimensions.
        Handles wrapping for longitude if slice.start > slice.stop.
        Only slices across dimensions that are not None.
        """

        if channel is None:
            data = self.prediction
        else:
            assert channel in self.channels, f"Variable {channel} not found in dataset."
            data = self.prediction[channel]

        # slice lat if specified
        if lat:
            data = data.sel(lat=lat)

        if lon:
            data = data.sel(lon=lon)

        if n_step and "time" in data.dims:
            data = data.isel(time=n_step)

    @property
    def model_channel_parser(self, channel: str):
        raise NotImplementedError

    def point(
        self,
        lat: float,  # [90.0,..., -90.0] or [-90.0,..., 90.0]
        lon: float,  # [0.0, 359.8]
        channel: str,  # e.g. v1000, u800
        n_step: int | None = 1,
    ):
        """
        Returns the value of the variable at the specified point.
        """
        # Convert negative longitude to positive in a 0-360 system
        if lon < 0:
            lon = 360 + lon
        assert channel in self.channel, f"Variable {channel} not found in dataset."
        data = self.prediction.sel(lat=lat, lon=lon, channel=channel).isel(time=n_step)
        return data.item()

    def point_wind_uv(
        self,
        lat: float,
        lon: float,
        pressure_level: int = 1000,
        n_step: int | None = 1,
    ):
        u_channel = f"u{pressure_level}"
        v_channel = f"v{pressure_level}"
        u = self.point(lat=lat, lon=lon, channel=u_channel, n_step=n_step)
        v = self.point(lat=lat, lon=lon, channel=v_channel, n_step=n_step)
        return u, v


class PanguPrediction(GlobalPrediction):
    def __init__(self, source):
        super().__init__(source)
        self.model_name = "pangu"

    def __repr__(self) -> str:
        return f"PanguPrediction({self.prediction.shape})"


class FourcastNetPrediction(GlobalPrediction):
    def __init__(self, source):
        super().__init__(source)
        self.model_name = "fcn"


class FourcastNetV2Prediction(GlobalPrediction):
    def __init__(self, source):
        super().__init__(source)
        self.model_name = "fcnv2_sm"


class DLWPPrediction(GlobalPrediction):
    def __init__(self, source):
        super().__init__(source)
        self.model_name = "dlwp"


class GraphcastPrediction(GlobalPrediction):
    def __init__(self, source):
        if isinstance(source, str):
            self.filepath = source
            self.prediction = xr.open_dataset(source).squeeze()

        elif isinstance(source, xr.Dataset) or isinstance(source, xr.DataArray):
            self.filepath = None
            self.prediction = source.squeeze()  # get rid of the dimensions with size 1
        self.model_name = "graphcast"

    def channel(self):
        raise NotImplementedError
