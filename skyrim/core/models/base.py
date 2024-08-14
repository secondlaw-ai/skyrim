import time
import datetime
import xarray as xr
from typing import Literal, Callable, List
from pathlib import Path
from loguru import logger
from earth2mip import schema
from ...libs.ic import get_data_source
from ...common import generate_forecast_id, save_forecast
from .utils import run_basic_inference


def adjust_lead_time(lead_time: int, step_size: int = 6):
    """Adjust lead time to the nearest multiple of step_size"""
    return max(step_size, (lead_time // step_size) * step_size)


class GlobalModel:
    def __init__(
        self,
        model_name: str,
        ic_source: Literal["cds", "gfs", "ifs"] = "cds",
    ):
        clock = time.time()
        self.model_name = model_name
        self.ic_source = ic_source

        logger.debug(f"Building {model_name} model...")
        self.model = self.build_model()

        logger.debug(f"Building {self.ic_source} data source...")
        self.data_source = self.build_datasource()
        logger.success(f"Initialized {model_name} in {time.time() - clock:.1f} seconds")

    def build_model(self):
        """
        Build or load the model configuration and components.
        """
        raise NotImplementedError

    def build_datasource(self):
        """
        Build or load the data source configuration and components.
        """
        return get_data_source(
            self.model.in_channel_names,
            initial_condition_source=schema.InitialConditionSource(self.ic_source),
        )

    def release_model(self):
        """
        Release the model resources.
        """
        # TODO: add functionality to release model resources
        raise NotImplementedError

    @property
    def time_step(self):
        raise NotImplementedError

    def time_steps(self, lead_time: int):
        """Calculate time steps given lead time in h"""
        lead_time = adjust_lead_time(lead_time, step_size=6)
        logger.debug(f"Lead time adjusted to nearest multiple of 6: {lead_time} hours")
        n_steps = int(lead_time // (self.model.time_step.total_seconds() / 3600))
        logger.debug(f"Number of prediction steps: {n_steps}")
        return n_steps

    @property
    def in_channel_names(self):
        raise NotImplementedError

    @property
    def out_channel_names(self):
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name={self.model_name})"

    def predict_one_step(
        self,
        start_time: datetime.datetime,
        initial_condition: str | xr.DataArray | None = None,
    ) -> xr.DataArray:
        # if initial_condition is None, it is fetched from the self.ic_source
        return run_basic_inference(
            model=self.model,
            n=1,
            data_source=self.data_source,
            time=start_time,
            x=initial_condition,
        )

    def forecast(
        self,
        start_time: datetime.datetime,
        n_steps: int = 3,
        channels: List[str] = [],
    ):
        """
        Return full concated forecast for the channels of interest
        i.e., for all steps starting from the ic as xr.DataArray
        if channels is empty, return all channels

        NOTE:   expect minor output discrapancies between this method and rollout.
                for some models, there are other different models for different lead times,
                e.g. pangu6, pangu24
        """

        da = run_basic_inference(
            model=self.model,
            n=n_steps,
            data_source=self.data_source,
            time=start_time,
            x=None,
        )
        return da.sel(channel=channels) if channels else da

    def rollout(
        self,
        start_time: datetime.datetime,
        n_steps: int = 3,
        save: bool = True,
        save_config: dict = {},
    ) -> tuple[xr.DataArray, list[str]]:
        # return final prediction and paths of the intermediate predictions
        # TODO: add functionality to rollout from a given initial condition
        pred, output_paths, source = None, [], self.ic_source
        forecast_id = save_config.get("forecast_id", generate_forecast_id())
        save_config.update({"forecast_id": forecast_id})
        for n in range(n_steps):
            pred = self.predict_one_step(start_time, initial_condition=pred)
            pred_time = start_time + self.time_step
            if save:
                output_path = save_forecast(
                    pred,
                    self.model_name,
                    start_time,
                    pred_time,
                    source,
                    config=save_config,
                )
                output_paths.append(output_path)
            start_time, source = pred_time, "file"
            logger.success(f"Rollout step {n+1}/{n_steps} completed")
        return pred, output_paths


class GlobalPrediction:
    filepath: Path | None = None
    prediction: xr.DataArray | None = None

    def __init__(self, source: str | xr.DataArray, model_name: str = ""):
        self.model = model_name
        if isinstance(source, (str, Path)):
            self.filepath = Path(source)
            self.prediction = xr.open_dataarray(source).squeeze()
        elif isinstance(source, xr.DataArray):
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
    def channels(self):
        return self.prediction.channel

    def __repr__(self) -> str:
        # This shows the filepath if it exists or the type and size of the prediction if not
        source_info = (
            self.filepath
            if self.filepath
            else f"{type(self.prediction).__name__} with shape {self.prediction.shape}"
        )
        return f"GlobalPrediction(model={self.model},source={source_info})"

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
            data = self.prediction.sel(channel=channel)

        # slice lat if specified
        if lat:
            data = data.sel(lat=lat)

        # slice lon if specified
        if lon:
            data = data.sel(lon=lon)

        if n_step and "time" in data.dims:
            data = data.isel(time=n_step)
        return data

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
        assert channel in self.channels, f"Variable {channel} not found in dataset."

        if (
            lat not in self.prediction.coords["lat"].values
            or lon not in self.prediction.coords["lon"].values
        ):
            lat = self.prediction.sel(lat=lat, method="nearest").lat.item()
            lon = self.prediction.sel(lon=lon, method="nearest").lon.item()
            logger.warning(
                f"Exact coordinates not found. Using nearest values: Lat {lat}, Lon {lon}"
            )

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

    def wind_speed(
        self,
        lat: float,
        lon: float,
        pressure_level: int,
        n_step: int | None = 1,
    ):
        # NOTE: pressure level is in hPa
        # TODO: add functionality to estimate pressure from height
        u, v = self.point_wind_uv(lat, lon, pressure_level, n_step)
        return (u**2 + v**2) ** 0.5

    def surface_wind_speed(self, lat: float, lon: float, n_step: int | None = 1):
        return self.wind_speed(lat, lon, pressure_level=1000, n_step=n_step)


class GlobalPredictionRollout:
    def __init__(self, rollout: list[str | Path | xr.DataArray]):
        self.rollout = [GlobalPrediction(source) for source in rollout]
        self.time_steps = [r.prediction.time.values[-1] for r in self.rollout]

    def __repr__(self):
        return f"<GlobalPredictionRollout with {len(self.rollout)} predictions, last times: {self.time_steps}>"

    def wind_speed(
        self,
        lat: float,
        lon: float,
        pressure_level: int,
        n_step: int | None = 1,
    ):
        # NOTE: pressure level is in hPa
        return [
            pred.wind_speed(lat, lon, pressure_level, n_step) for pred in self.rollout
        ]

    def surface_wind_speed(
        self,
        lat: float,
        lon: float,
        n_step: int | None = 1,
    ):
        return self.wind_speed(lat, lon, pressure_level=1000, n_step=n_step)
