import datetime
from pathlib import Path
import xarray as xr

from earth2mip import registry
from earth2mip.initial_conditions import cds
import earth2mip.networks.fcn as fcn
from .base import GlobalModel, GlobalPrediction
import sys
from .utils import run_basic_inference

# 
# fmt: off
# https://github.com/NVIDIA/earth2mip/blob/86b11fe4ba2f19641802112e8b0ba6b962123130/earth2mip/networks/fcn.py#L28-L55
CHANNELS = ["u10m", "v10m", "t2m", "sp", "msl", "t850", "u1000", "v1000", "z1000", "u850", 
            "v850", "z850", "u500", "v500", "z500", "t500", "z50", "r500", "r850", "tcwv", 
            "u100m", "v100m", "u250", "v250", "z250","t250"]
# fmt: on

class FourcastnetModel(GlobalModel):
    model_name = "fourcastnet"
    
    def __init__(self, *args, **kwargs):
        super().__init__(self.model_name, *args, **kwargs)

    def build_model(self):
        return fcn.load(registry.get_model("e2mip://fcn"))
    
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
    ) -> xr.DataArray | xr.Dataset:
        return run_basic_inference(
            model=self.model,
            n=1,
            data_source=self.data_source,
            time=start_time,
            x=initial_condition,
        )


class FourcastnetPrediction(GlobalPrediction):
    def __init__(self, source):
        super().__init__(source)
        self.model_name = "fourcastnet"
        
