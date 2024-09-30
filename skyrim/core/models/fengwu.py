import time
import datetime
from loguru import logger
import xarray as xr
from typing import List
from earth2studio.models.px import FengWu
from earth2studio.data import GFS, IFS, CDS
from earth2studio.io import XarrayBackend
import earth2studio.run as run
from .base import GlobalModel


# fmt: off
CHANNELS =  [
    "u10m", "v10m", "t2m", "msl", "z50", "z100", "z150", "z200", "z250", "z300",
    "z400", "z500", "z600", "z700", "z850", "z925", "z1000", "q50", "q100", "q150",
    "q200", "q250", "q300", "q400", "q500", "q600", "q700", "q850", "q925", "q1000",
    "u50", "u100", "u150", "u200", "u250", "u300", "u400", "u500", "u600", "u700", 
    "u850", "u925", "u1000", "v50", "v100", "v150", "v200", "v250", "v300", "v400",
    "v500", "v600", "v700", "v850", "v925", "v1000", "t50", "t100", "t150", "t200",
    "t250", "t300", "t400", "t500", "t600", "t700", "t850", "t925", "t1000",
]
# fmt: on


class FengwuModel(GlobalModel):
    """
    From: 
    https://github.com/NVIDIA/earth2studio/blob/68dd00bd76be8abc90badd39d0f51f26294ce526/earth2studio/models/px/fengwu.py#L113-L125
    
        FengWu (operational) weather model consists of single auto-regressive model with
        a time-step size of 6 hours. FengWu operates on 0.25 degree lat-lon grid (south-pole
        including) equirectangular grid with 69 atmospheric/surface variables. This model
        uses two time-steps as an input.

    - https://arxiv.org/abs/2304.02948
    - https://github.com/OpenEarthLab/FengWu
    """
    model_name = "fengwu"

    def __init__(self, *args, **kwargs):
        super().__init__(self.model_name, *args, **kwargs)

    def build_model(self):
        return FengWu.load_model(FengWu.load_default_package())

    def build_datasource(self):
        if self.ic_source == "gfs":
            return GFS()
        elif self.ic_source == "ifs":
            return IFS()
        elif self.ic_source == "cds":
            return CDS()

    @property
    def time_step(self):
        return datetime.timedelta(hours=6)

    @property
    def in_channel_names(self):
        # TODO: add the input channel names
        pass

    @property
    def out_channel_names(self):
        # TODO: add the output channel names
        pass

    def forecast(
        self,
        start_time: datetime.datetime,
        n_steps: int,
        channels: List[str] = [],
    ) -> xr.DataArray:

        io = XarrayBackend({})
        io = run.deterministic(
            time=[start_time],
            nsteps=n_steps,
            prognostic=self.model,
            data=self.data_source,
            io=io,
        )
        # TODO: this transformation takes forever, need to optimize
        ts = time.time()
        da = io.root.squeeze().to_array()
        logger.debug(f"io to xr.DataArray {time.time() - ts:.1f} seconds")
        # returned da has the following structure (i.e., earth2studio style):
        # >> da.dims
        # ('variable', 'lead_time', 'lat', 'lon')
        # >> da.coords
        # Coordinates:
        #     time       datetime64[ns] 2024-04-01
        #   * lead_time  (lead_time) timedelta64[ns] 00:00:00 06:00:00 ... 1 days 00:00:00
        #   * lat        (lat) float64 90.0 89.75 89.5 89.25 ... -89.25 -89.5 -89.75 -90.0
        #   * lon        (lon) float64 0.0 0.25 0.5 0.75 1.0 ... 359.0 359.2 359.5 359.8
        #   * variable   (variable) object 'z50' 'z100' 'z150' ... 'v10m' 'msl' 'tp'

        # arrange the dataarray in the format of time, channel, lat, lon, i.e. skyrim style
        da = (
            da.rename({"variable": "channel"})
            .assign_coords(time=lambda x: x.time + x.lead_time)
            .swap_dims({"lead_time": "time"})
            .drop_vars("lead_time")
            .transpose("time", "channel", "lat", "lon")
            .astype("float32")
        )

        return da.sel(channel=channels) if channels else da

    def rollout(self) -> tuple[xr.DataArray, list[str]]:
        raise NotImplementedError

    def predict_one_step(self) -> xr.DataArray:
        raise NotImplementedError
