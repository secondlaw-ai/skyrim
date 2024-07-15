from typing import List
import datetime
import xarray as xr
import jax.numpy as jnp
import numpy as np
import earth2mip.networks.graphcast as graphcast
from pathlib import Path
from earth2mip import registry
from earth2mip.initial_conditions import cds, get_initial_condition_for_model
from ...common import generate_forecast_id, save_forecast
from loguru import logger
from .base import GlobalModel


# fmt: off
# TODO: check tp06 - this is tisr? https://codes.ecmwf.int/grib/param-db/212
CHANNELS = ["z50", "z100", "z150", "z200", "z250", "z300", "z400", "z500", "z600", "z700",
            "z850", "z925", "z1000", "q50", "q100", "q150", "q200", "q250", "q300", "q400",
            "q500", "q600", "q700", "q850", "q925", "q1000", "t50", "t100", "t150", "t200",
            "t250", "t300", "t400", "t500", "t600", "t700", "t850", "t925", "t1000", "u50",
            "u100", "u150", "u200", "u250", "u300", "u400", "u500", "u600", "u700", "u850",
            "u925", "u1000", "v50", "v100", "v150", "v200", "v250", "v300", "v400", "v500",
            "v600", "v700", "v850", "v925", "v1000", "w50", "w100", "w150", "w200", "w250",
            "w300", "w400", "w500", "w600", "w700", "w850", "w925", "w1000", "u10m", "v10m",
            "t2m", "msl","tp06",
            ]
# fmt: on

CHANNEL_MAP = [
    ("specific_humidity", "q"),
    ("geopotential", "z"),
    ("temperature", "t"),
    ("u_component_of_wind", "u"),
    ("v_component_of_wind", "v"),
    ("vertical_velocity", "w"),
    ("2m_temperature", "t2m"),
    ("10m_u_component_of_wind", "u10m"),
    ("10m_v_component_of_wind", "v10m"),
    ("mean_sea_level_pressure", "msl"),
    ("toa_incident_solar_radiation", "tp06"),
]


class GraphcastModel(GlobalModel):
    # TODO: check rollout implementation
    model_name = "graphcast"

    def __init__(self, *args, **kwargs):
        super().__init__(self.model_name, *args, **kwargs)

    def build_model(self):
        return graphcast.load_time_loop_operational(
            registry.get_model("e2mip://graphcast")
        )

    @property
    def time_step(self):
        return self.model.time_step

    @property
    def in_channel_names(self):
        return self.model.in_channel_names

    @property
    def out_channel_names(self):
        return self.model.out_channel_names

    def _to_global_da(self, ds: xr.Dataset) -> xr.DataArray:
        """Convert graphcast dataset to our global dataarray (da) format consistent with other models."""
        lvar_map, sfc_map = CHANNEL_MAP[:6], CHANNEL_MAP[6:]
        lvar_dss, sfc_dss = [], []
        ds = ds.squeeze(dim="batch")
        for name, code in lvar_map:
            channels = [f"{code}{l}" for l in list(ds[name].level.values)]
            x = ds[name]
            x["level"] = channels
            x = x.rename({"level": "channel"})
            lvar_dss.append(x)
        for name, code in sfc_map:
            x = ds[name]
            x["channel"] = code
            x = x.expand_dims("channel")
            sfc_dss.append(x)

        global_da = (
            xr.concat(lvar_dss + sfc_dss, dim="channel")
            .transpose("time", "channel", "lat", "lon")
            .as_numpy()
        )
        global_da.name = None
        return global_da

    def _predict_one_step(
        self,
        start_time: datetime.datetime,
        initial_condition: tuple | None = None,
    ) -> xr.DataArray:
        # TODO: initial condition
        # NOTE: this only supports graphcast operational model
        # some info about stepper:
        # https://github.com/NVIDIA/earth2mip/blob/86b11fe4ba2f19641802112e8b0ba6b962123130/earth2mip/time_loop.py#L114-L122
        self.stepper = self.model.stepper
        if initial_condition is None:
            initial_condition = get_initial_condition_for_model(
                time_loop=self.model,
                data_source=self.data_source,
                time=start_time,
            )

            state = self.stepper.initialize(initial_condition, start_time)
            logger.debug(f"IC fetched - state[0]: {state[0]}")
            # state : with lenght 3
            # state[0] Timestamp('2024-05-01 00:00:00')
            # state[1] <xarray.Dataset>  time: 2, level: 13, batch: 1, lat: 721, lon: 1440
            # state[2] Array([0, 0], dtype=uint32)
        else:
            state = initial_condition
        state, output = self.stepper.step(state)
        logger.debug(f"state[0]: {state[0]}")
        return state

    def forecast(
        self,
        start_time: datetime.datetime,
        n_steps: int = 4,
        channels: List[str] = [],
    ) -> xr.DataArray:
        times = [start_time + i * self.time_step for i in range(n_steps + 1)]
        state, das = None, []
        for n in range(n_steps):
            state = self._predict_one_step(start_time, initial_condition=state)
            logger.success(f"Forecast step {n+1}/{n_steps} completed")

            da = self._to_global_da(
                state[1] if n == 0 else state[1].isel(time=-1).expand_dims("time")
            )
            if channels:
                da = da.sel(channel=channels)
            das.append(da)
        # re-set the time coords: as graphcast by default holds delta_t's
        # prediction, i.e. current state, timestamp value is hold at state[0]
        return xr.concat(das, dim="time").assign_coords(time=times)

    def rollout(
        self,
        start_time: datetime.datetime,
        n_steps: int = 3,
        save: bool = True,
        save_config: dict = {},
    ) -> tuple[xr.DataArray, list[str]]:
        # it does not make sense to keep all the results in the memory
        # return final pred and list of paths of the saved predictions
        # TODO: add functionality to rollout from a given initial condition
        times = [start_time + i * self.time_step for i in range(n_steps + 1)]

        pred, output_paths, source = None, [], self.ic_source
        forecast_id = save_config.get("forecast_id", generate_forecast_id())
        save_config.update({"forecast_id": forecast_id})
        for n in range(n_steps):
            pred = self._predict_one_step(start_time, initial_condition=pred)
            pred_time = start_time + self.time_step
            if save:
                output_path = save_forecast(
                    self._to_global_da(pred[1]).assign_coords(
                        time=[start_time, pred_time]
                    ),
                    self.model_name,
                    start_time,
                    pred_time,
                    source,
                    config=save_config,
                )
                output_paths.append(output_path)
            start_time, source = pred_time, "file"
            logger.success(f"Rollout step {n+1}/{n_steps} completed")
        # re-set the time coords: as graphcast by default holds delta_t's
        return self._to_global_da(pred[1]).assign_coords(time=times[-2:]), output_paths
