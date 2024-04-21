import datetime
from pathlib import Path
import xarray as xr

from earth2mip import registry
from earth2mip.initial_conditions import cds, get_initial_condition_for_model
import earth2mip.networks.graphcast as graphcast
from .base import GlobalModel, GlobalPrediction


class GraphcastModel(GlobalModel):
    def __init__(self, model_name: str = "graphcast"):
        super().__init__(model_name)

    def build_model(self):
        return graphcast.load_time_loop_operational(
            registry.get_model("e2mip://graphcast")
        )

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

        # NOTE: this only works for graphcast operational model
        # some info about stepper:
        # https://github.com/NVIDIA/earth2mip/blob/86b11fe4ba2f19641802112e8b0ba6b962123130/earth2mip/time_loop.py#L114-L122

        self.stepper = self.model.stepper
        x = get_initial_condition_for_model(
            time_loop=self.model,
            data_source=self.data_source,
            time=start_time,
        )

        state = self.stepper.initialize(x, start_time)
        state, output = self.stepper.step(state)
        # output.shape: torch.Size([1, 83, 721, 1440])
        # len(state): 3,
        # state[0]: Timestamp('2018-01-02 06:00:00')
        return state[1]

    def rollout(
        self, start_time: datetime.datetime, n_steps: int = 3, save: bool = True
    ) -> tuple[xr.DataArray | xr.Dataset, list[str]]:
        raise NotImplementedError


class GraphcastPrediction(GlobalPrediction):
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
