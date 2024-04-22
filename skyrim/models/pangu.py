import datetime
from pathlib import Path
import xarray as xr

from earth2mip import registry
from earth2mip.initial_conditions import cds
import earth2mip.networks.pangu as pangu
from .base import GlobalModel, GlobalPrediction
from .utils import run_basic_inference


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

    def __init__(self, model_name: str = "pangu"):
        super().__init__(model_name)

    def build_model(self):
        return pangu.load(registry.get_model("e2mip://pangu"))

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


class PanguPrediction(GlobalPrediction):
    def __init__(self, source):
        super().__init__(source)
        self.model_name = "pangu"

    def __repr__(self) -> str:
        return f"PanguPrediction({self.prediction.shape})"
