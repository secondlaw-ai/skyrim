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
        self,
        variable: str | None = None,
        latitude: slice | None = None,
        longitude: slice | None = None,
        isobaricInhPa: slice | None = None,
        step: slice | None = None,
    ):
        """
        Returns a slice of the dataset according to specified dimensions.
        Handles wrapping for longitude if slice.start > slice.stop.
        Only slices across dimensions that are not None.
        """
        # Start with the whole dataset or a specific variable
        if variable is None:
            data = self.prediction
        else:
            data = self.prediction[variable]

        # Handle longitude wrap-around
        if longitude and longitude.start > longitude.stop:
            lon_slice_1 = data.sel(longitude=slice(longitude.start, 360))
            lon_slice_2 = data.sel(longitude=slice(0, longitude.stop))
            data = xr.concat([lon_slice_1, lon_slice_2], dim="longitude")
        elif longitude:
            data = data.sel(longitude=longitude)

        # Apply latitude slice if specified
        if latitude:
            data = data.sel(latitude=latitude)

        # Apply isobaricInhPa slice if specified and relevant
        if isobaricInhPa and "isobaricInhPa" in data.dims:
            data = data.sel(isobaricInhPa=isobaricInhPa)

        if step and "step" in data.dims:
            data = data.sel(step=step)

        return data

    def point(
        self,
        latitude: float,
        longitude: float,
        isobaricInhPa: float | None = None,
        variable: str | None = None,  # if None, select across all variables
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
