import datetime
from pathlib import Path
import xarray as xr

from earth2mip import registry
from earth2mip.initial_conditions import cds
import earth2mip.networks.fcnv2_sm as fcnv2_sm
from .base import GlobalModel, GlobalPrediction
from .utils import run_basic_inference

# fmt: off
    # https://github.com/NVIDIA/earth2mip/blob/86b11fe4ba2f19641802112e8b0ba6b962123130/earth2mip/networks/fcnv2_sm.py#L37-L111
CHANNELS = [ "u10m", "v10m", "u100m", "v100m", "t2m", "sp", "msl", "tcwv", "u50", "u100",
            "u150","u200", "u250", "u300", "u400", "u500", "u600", "u700", "u850", "u925",
            "u1000", "v50", "v100", "v150", "v200", "v250", "v300", "v400", "v500", "v600",
            "v700", "v850", "v925", "v1000", "z50", "z100", "z150", "z200", "z250", "z300",
            "z400", "z500", "z600", "z700", "z850", "z925", "z1000", "t50", "t100", "t150",
            "t200", "t250", "t300", "t400", "t500", "t600", "t700", "t850", "t925", "t1000",
            "r50", "r100", "r150", "r200", "r250", "r300", "r400", "r500", "r600", "r700",
            "r850", "r925","r1000",
    ]
# fmt: on
class FourcastnetV2Model(GlobalModel):
    model_name = "fourcastnet_v2"
    
    def __init__(self):
        super().__init__(self.model_name)

    def build_model(self):
        return fcnv2_sm.load(registry.get_model("e2mip://fcnv2_sm"))

    def build_datasource(self):
        return cds.DataSource(self.model.in_channel_names)

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


class FourcastnetV2Prediction(GlobalPrediction):
    def __init__(self, source):
        super().__init__(source)
        self.model_name = "fourcastnet_v2"
        