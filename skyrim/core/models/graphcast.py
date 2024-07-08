import datetime
from pathlib import Path
import xarray as xr
from loguru import logger
from earth2mip import registry
from earth2mip.initial_conditions import cds, get_initial_condition_for_model
import earth2mip.networks.graphcast as graphcast
from .base import GlobalModel, GlobalPrediction

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


def to_global_prediction(ds: xr.Dataset) -> xr.DataArray:
    """Convert graphcast dataset to our dataarray global prediction format consistent with other models."""
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
    return xr.concat(lvar_dss + sfc_dss, dim="channel").transpose("time", "channel", "lat", "lon")


class GraphcastModel(GlobalModel):
    # TODO: check rollout implementation
    model_name = "graphcast"

    def __init__(self, *args, **kwargs):
        super().__init__(self.model_name, *args, **kwargs)

    def build_model(self):
        return graphcast.load_time_loop_operational(registry.get_model("e2mip://graphcast"))

    @property
    def time_step(self):
        return self.model.time_step

    @property
    def in_channel_names(self):
        return self.model.in_channel_names

    @property
    def out_channel_names(self):
        return self.model.out_channel_names

    def predict_one_step(
        self,
        start_time: datetime.datetime,
        initial_condition: str | Path | None = None,
    ) -> xr.DataArray:
        # TODO: initial condition
        # NOTE: this only works for graphcast operational model
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

        else:
            state = initial_condition
        state, output = self.stepper.step(state)
        return to_global_prediction(state[1])


class GraphcastPrediction(GlobalPrediction):
    # TODO: to be able to use the same GlobalPrediction interface, we need to map graph
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
