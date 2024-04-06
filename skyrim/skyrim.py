from pathlib import Path
from loguru import logger
import xarray as xr
from dataclasses import dataclass, field

from .models import PanguWeather, GraphCast

MODEL_CLASS_MAP = {
    "panguweather": PanguWeather,
    "graphcast": GraphCast,
}


def initiazlize_model(
    model_name: str, date: str, time: int, lead_time: int, file: None
):
    model = MODEL_CLASS_MAP[model_name](date, time, lead_time, file)
    logger.success(f"Model {model_name} initialized")
    return model


class Skyrim:
    def __init__(
        self, model_name: str, date: str, time: int = 12, lead_time: int = 24, file=None
    ) -> None:

        self.model = initiazlize_model(model_name, date, time, lead_time, file)

    def predict(self):
        self.model.predict()

    def predict_local(self):
        pass

    def predict_from_file(self):
        pass


@dataclass(slots=True)
class ModelPrediction:
    filepath: str
    model_name: str = field(init=False)
    date: str = field(init=False)
    time: int = field(init=False)
    lead_time: int = field(init=False)
    prediction: xr.Dataset = field(init=False)

    def __post_init__(self):
        self.model_name = Path(self.filepath).name
        parts = self.model_name.split("__")
        self.date = parts[0].split("=")[1]
        self.time = int(parts[1].split("=")[1].split(":")[0])
        self.lead_time = int(parts[2].split(".")[0])
        self.prediction = xr.open_dataset(self.filepath)

    def __repr__(self):
        return f"ModelPrediction({self.filepath})"

    @property
    def variables(self):
        return self.prediction.data_vars

    @property
    def coords(self):
        return self.prediction.coords

    @property
    def size(self):
        return self.prediction.sizes

    def slice(
        self, latitude: slice, longitude: slice, variables: list | str | None = None
    ):
        # TODO: check if this is the case for other models, e.g. GraphCast

        # handle wrapping of longitudes if the slice crosses the 0/360 boundary
        if longitude.start > longitude.stop:
            # Slice from start to 360 and 0 to stop, then concatenate
            lon_slice_1 = self.prediction.sel(longitude=slice(longitude.start, 360))
            lon_slice_2 = self.prediction.sel(longitude=slice(0, longitude.stop))
            result = xr.concat([lon_slice_1, lon_slice_2], dim="longitude")
        else:
            result = self.prediction.sel(longitude=longitude)

        # slice by latitudez
        result = result.sel(latitude=latitude)

        # select variables if specified
        if variables:
            if isinstance(variables, list):
                result = result[variables]
            else:  # assuming str
                result = result[[variables]]

        return result

    def point(
        self,
        latitude: float,
        longitude: float,
        isobaricInhPa: float | None = None,
        variable: str | None = None,
        step: int | None = 1,  # not sure if this exists in all the models
    ):

        # Handle case where no specific variable is defined: select across all variables
        if variable is None:
            data_selection = self.prediction.sel(
                latitude=latitude, longitude=longitude, method="nearest"
            )
            if step is not None and "step" in data_selection.dims:
                data_selection = data_selection.isel(step=step)
            return data_selection

        if "isobaricInhPa" in self.prediction[variable].dims:
            return self._point_pressure_var(
                latitude=latitude,
                longitude=longitude,
                isobaricInhPa=isobaricInhPa,
                variable=variable,
                step=step,
            )
        else:
            return self._point_surface_var(
                latitude=latitude,
                longitude=longitude,
                variable=variable,
                step=step,
            )

    def _point_surface_var(
        self,
        latitude: float,
        longitude: float,
        variable: str,
        step: int | None = None,
    ):
        data_selection = self.prediction[variable].sel(
            latitude=latitude,
            longitude=longitude,
            method="nearest",
        )
        if step is not None and "step" in data_selection.dims:
            data_selection = data_selection.isel(step=step)
        return data_selection

    def _point_pressure_var(
        self,
        latitude: float,
        longitude: float,
        isobaricInhPa: float,
        variable: str,
        step: int | None = None,
    ):
        data_selection = self.prediction[variable].sel(
            latitude=latitude,
            longitude=longitude,
            isobaricInhPa=isobaricInhPa,
            method="nearest",
        )
        if step is not None and "step" in data_selection.dims:
            data_selection = data_selection.isel(step=step)
        return data_selection
