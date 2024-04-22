import time
import datetime
from typing import Literal
from pathlib import Path
from loguru import logger
import xarray as xr

OUTPUT_DIR = Path(__file__).parent.parent.parent.resolve() / Path("./outputs")

if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir()
    logger.success(f"Created output directory: {OUTPUT_DIR}")


class GlobalModel:
    def __init__(self, model_name: str):
        clock = time.time()
        self.model_name = model_name
        self.model = self.build_model()
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
        raise NotImplementedError

    @property
    def time_step(self):
        raise NotImplementedError

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
        initial_condition: str | Path | None = None,
    ) -> xr.DataArray | xr.Dataset:
        raise NotImplementedError

    def rollout(
        self,
        start_time: datetime.datetime,
        n_steps: int = 3,
        save: bool = True,
    ) -> tuple[xr.DataArray | xr.Dataset, list[str]]:
        # it does not make sense to keep all the results in the memory
        # return final pred and list of paths of the saved predictions
        # TODO: add functionality to rollout from a given initial condition
        # TODO: support other sources than cds, e.g. ifs, gfs, file, etc

        pred, output_paths, source = None, [], "cds"
        for n in range(n_steps):
            pred = self.predict_one_step(start_time, initial_condition=pred)
            pred_time = start_time + self.time_step
            if save:
                output_path = self.save_output(pred, start_time, pred_time, source)
                start_time, source = pred_time, "file"
                output_paths.append(output_path)
            logger.success(f"Rollout step {n+1}/{n_steps} completed")
        return pred, output_paths

    def save_output(
        self,
        pred: xr.DataArray | xr.Dataset,
        start_time: datetime.datetime,
        pred_time: datetime.datetime,
        source: Literal["cds", "file"] = "cds",
        output_dir=OUTPUT_DIR,
    ):
        # e.g.:
        # filename = "pangu__20180101_00:00__20180101_06:00.nc"
        # output_path = "./outputs/pangu/pangu__20180101_00:00__20180101_06:00.nc"

        filename = (
            f"{self.model_name}" + "__"
            f"{source}__"
            f"{start_time.strftime('%Y%m%d_%H:%M')}"
            + "__"
            + f"{pred_time.strftime('%Y%m%d_%H:%M')}.nc"
        )
        output_path = OUTPUT_DIR / self.model_name / filename

        logger.info(f"Saving outputs to {output_path}")
        if not output_path.parent.exists():
            logger.info(
                f"Creating parent directory to save outputs: {output_path.parent}"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)

        pred.to_netcdf(output_path)
        logger.success(f"outputs saved to {output_path}")
        return output_path


class GlobalPrediction:
    filepath: Path = None
    prediction: xr.Dataset | xr.DataArray = None

    def __init__(self, source):
        if isinstance(source, (str, Path)):
            self.filepath = Path(source)
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
    
    def __repr__(self) -> str:
        # This shows the filepath if it exists or the type and size of the prediction if not
        source_info = self.filepath if self.filepath else f"{type(self.prediction).__name__} with shape {self.prediction.shape}"
        return f"{self.__class__.__name__}(source={source_info})"

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
        pressure_level: int = 1000, 
        n_step: int | None = 1,
    ):
        # NOTE: pressure level is in hPa
        # TODO: add functionality to estimate pressure from height
        u, v = self.point_wind_uv(lat, lon, pressure_level, n_step)
        return (u ** 2 + v ** 2) ** 0.5
    

class GlobalPredictionRollout:
    def __init__(self, rollout: list[str | Path | xr.DataArray]):
        self.rollout = [GlobalPrediction(source) for source in rollout]
        self.time_steps = [r.prediction.time.values[-1] for r in self.rollout]
    
    def wind_speed(
        self,
        lat: float,
        lon: float,
        pressure_level: int = 1000, 
        n_step: int | None = 1,
    ):
        # NOTE: pressure level is in hPa
        return [pred.wind_speed(lat, lon, pressure_level, n_step) for pred in self.rollout]
    
    
    
