import datetime
from pathlib import Path
import xarray as xr

from earth2mip import registry
from earth2mip.initial_conditions import cds
import earth2mip.networks.dlwp as dlwp
from .base import GlobalModel, GlobalPrediction
from .utils import run_basic_inference


class DLWPModel(GlobalModel):
    """
    n_history_levels: int = 2
    grid.lat: list of length 721, [90, 89.75, 89.50, ..., -89.75, -90]
    grid.lon: list of length 1440, [0.0, 0.25, ..., 359.75]
    in_channel_names: list of length 7, ['t850', 'z1000', 'z700', 'z500', 'z300', 'tcwv', 't2m']
    out_channel_names: list of length 7, ['t850', 'z1000', 'z700', 'z500', 'z300', 'tcwv', 't2m']
    """
    
    model_name = "dlwp"

    def __init__(self, *args, **kwargs):
        super().__init__(self.model_name, *args, **kwargs)

    def build_model(self):
        return dlwp.load(registry.get_model("e2mip://dlwp"))

    @property
    def time_step(self):
        return self.model.time_step

    @property
    def in_channel_names(self):
        return self.model.in_channel_names

    @property
    def out_channel_names(self):
        return self.model.out_channel_names
