import datetime
from pathlib import Path
import xarray as xr

from earth2mip import registry
from earth2mip.initial_conditions import cds
import earth2mip.networks.dlwp as dlwp
from .base import GlobalModel, GlobalPrediction
from .utils import run_basic_inference


class DLWPModel(GlobalModel):
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
