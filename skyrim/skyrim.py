import datetime
import os
import time
import dotenv
import xarray as xr
from .models import FoundationModel

from loguru import logger


dotenv.load_dotenv()

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


class Skyrim:
    def __init__(self, model_name: str):
        self.model = FoundationModel(model_name=model_name)
        
    def predict(self, start_time: datetime.datetime, n_steps:int = 1):
        # TODO: add saving functionality
        return self.model.predict(start_time=start_time, n_steps=n_steps)
        
        
# @dataclass(slots=True)
# class ModelPrediction:
#     filepath: str
#     model_name: str = field(init=False)
#     date: str = field(init=False)
#     time: int = field(init=False)
#     lead_time: int = field(init=False)
#     input: str = field(init=False)
#     prediction: xr.Dataset = field(init=False)

#     def __post_init__(self):
#         self.model_name = Path(self.filepath).name
#         parts = self.model_name.split("__")
#         self.date = parts[0].split("=")[1]
#         self.time = int(parts[1].split("=")[1].split(":")[0])
#         self.lead_time = int(parts[2].split(".")[0])
#         self.input = parts[3].split("=")[1]
#         self.prediction = xr.open_dataset(self.filepath)

#     def __repr__(self):
#         return f"ModelPrediction({self.filepath})"

#     @property
#     def variables(self):
#         return list(self.prediction.data_vars)

#     @property
#     def coords(self):
#         return list(self.prediction.coords)

#     @property
#     def size(self):
#         return self.prediction.sizes

#     def slice(
#         self,
#         variable: str | None = None,
#         latitude: slice | None = None,
#         longitude: slice | None = None,
#         isobaricInhPa: slice | None = None,
#         step: slice | None = None,
#     ):
#         """
#         Returns a slice of the dataset according to specified dimensions.
#         Handles wrapping for longitude if slice.start > slice.stop.
#         Only slices across dimensions that are not None.
#         """
#         # Start with the whole dataset or a specific variable
#         if variable is None:
#             data = self.prediction
#         else:
#             data = self.prediction[variable]

#         # Convert negative longitude to positive in a 0-360 system
#         # TODO: check if this is the case for all the models
#         if longitude.start < 0:
#             longitude.start = 360 + longitude

#         # Handle longitude wrap-around
#         if longitude and longitude.start > longitude.stop:
#             lon_slice_1 = data.sel(longitude=slice(longitude.start, 360))
#             lon_slice_2 = data.sel(longitude=slice(0, longitude.stop))
#             data = xr.concat([lon_slice_1, lon_slice_2], dim="longitude")
#         elif longitude:
#             data = data.sel(longitude=longitude)

#         # Apply latitude slice if specified
#         if latitude:
#             data = data.sel(latitude=latitude)

#         # Apply isobaricInhPa slice if specified and relevant
#         if isobaricInhPa and "isobaricInhPa" in data.dims:
#             data = data.sel(isobaricInhPa=isobaricInhPa)

#         if step and "step" in data.dims:
#             data = data.sel(step=step)

#         return data

#     def point(
#         self,
#         latitude: float,
#         longitude: float,
#         isobaricInhPa: float | None = None,
#         variable: str | None = None,  # if None, select across all variables
#         step: int | None = 1,  # not sure if this exists in all the models
#     ):

#         # Convert negative longitude to positive in a 0-360 system
#         # TODO: check if this is the case for all the models
#         if longitude < 0:
#             longitude = 360 + longitude

#         # Handle case where no specific variable is defined: select across all variables
#         if variable is None:
#             data_selection = self.prediction.sel(
#                 latitude=latitude, longitude=longitude, method="nearest"
#             )
#             if step is not None and "step" in data_selection.dims:
#                 data_selection = data_selection.isel(step=step)
#             return data_selection

#         if "isobaricInhPa" in self.prediction[variable].dims:
#             return self._point_pressure_var(
#                 latitude=latitude,
#                 longitude=longitude,
#                 isobaricInhPa=isobaricInhPa,
#                 variable=variable,
#                 step=step,
#             )
#         else:
#             return self._point_surface_var(
#                 latitude=latitude,
#                 longitude=longitude,
#                 variable=variable,
#                 step=step,
#             )

#     def _point_surface_var(
#         self,
#         latitude: float,
#         longitude: float,
#         variable: str,
#         step: int | None = None,
#     ):
#         data_selection = self.prediction[variable].sel(
#             latitude=latitude,
#             longitude=longitude,
#             method="nearest",
#         )
#         if step is not None and "step" in data_selection.dims:
#             data_selection = data_selection.isel(step=step)
#         return data_selection

#     def _point_pressure_var(
#         self,
#         latitude: float,
#         longitude: float,
#         isobaricInhPa: float,
#         variable: str,
#         step: int | None = None,
#     ):
#         data_selection = self.prediction[variable].sel(
#             latitude=latitude,
#             longitude=longitude,
#             isobaricInhPa=isobaricInhPa,
#             method="nearest",
#         )
#         if step is not None and "step" in data_selection.dims:
#             data_selection = data_selection.isel(step=step)
#         return data_selection

#     def point_wind_uv(
#         self,
#         latitude: float,
#         longitude: float,
#         isobaricInhPa: float,
#         step: int | None = None,
#     ):
#         u = self.point(
#             latitude=latitude,
#             longitude=longitude,
#             isobaricInhPa=isobaricInhPa,
#             variable="u",
#             step=step,
#         )
#         v = self.point(
#             latitude=latitude,
#             longitude=longitude,
#             isobaricInhPa=isobaricInhPa,
#             variable="v",
#             step=step,
#         )
#         return {"u": u, "v": v}

#     def plot(self):
#         # TODO: look into availabe 2D visualization libraries
#         # e.g. climetlab
#         pass


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
