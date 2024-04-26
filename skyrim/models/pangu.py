import datetime
from pathlib import Path
import xarray as xr

from earth2mip import registry
from earth2mip.initial_conditions import cds
import earth2mip.networks.pangu as pangu
from .base import GlobalModel, GlobalPrediction
from .utils import run_basic_inference

# fmt: off
CHANNELS = ["z1000", "z925", "z850", "z700", "z600","z500", "z400", "z300", "z250", "z200",
            "z150", "z100", "z50", "q1000", "q925", "q850", "q700", "q600", "q500", "q400",
            "q300", "q250", "q200", "q150", "q100", "q50", "t1000", "t925", "t850", "t700",
            "t600", "t500", "t400", "t300", "t250", "t200", "t150", "t100", "t50", "u1000",
            "u925", "u850", "u700", "u600", "u500", "u400", "u300", "u250", "u200", "u150",
            "u100", "u50", "v1000", "v925", "v850", "v700", "v600", "v500", "v400", "v300",
            "v250", "v200", "v150", "v100", "v50", "msl", "u10m", "v10m", "t2m",
    ]
# fmt: on

class PanguModel(GlobalModel):
    """
    This class implements the Pangu model for environmental predictions using the Earth-2 MIP framework.

    The Pangu model is based on advanced deep learning algorithms designed for high-resolution weather and climate data prediction.

    Official implementation:
    https://github.com/198808xc/Pangu-Weather

    Reference Paper:
    https://www.nature.com/articles/s41586-023-06185-3

    Parameters:
    - model_name (str): Name of the model, default is 'pangu'.
    """
    
    model_name = "pangu"

    def __init__(self):
        super().__init__(self.model_name)

    def build_model(self):
        return pangu.load(registry.get_model("e2mip://pangu"))

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


class PanguPrediction(GlobalPrediction):
    def __init__(self, source):
        super().__init__(source)
        self.model_name = "pangu"


