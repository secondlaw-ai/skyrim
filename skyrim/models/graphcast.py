import datetime
from pathlib import Path
import xarray as xr
from loguru import logger

from earth2mip import registry
from earth2mip.initial_conditions import cds, get_initial_condition_for_model
import earth2mip.networks.graphcast as graphcast
from .base import GlobalModel, GlobalPrediction

# fmt: off

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


class GraphcastModel(GlobalModel):
    # TODO: implement rollout

    def __init__(self, model_name: str = "graphcast", **kwargs):
        super().__init__(model_name, **kwargs)

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

    def predict_one_step(
        self,
        start_time: datetime.datetime,
        initial_condition: str | Path | None = None,
    ) -> xr.DataArray | xr.Dataset:

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
        # output.shape: torch.Size([1, 83, 721, 1440])
        # len(state): 3,
        # state[0]: Timestamp('2018-01-02 06:00:00')
        # return state[1]
        return state

    def rollout(
        self, start_time: datetime.datetime, n_steps: int = 3, save: bool = True
    ) -> tuple[xr.DataArray | xr.Dataset, list[str]]:
        # TODO:
        pred, output_paths, source = None, [], "cds"
        for n in range(n_steps):
            # returns a state tuple
            pred = self.predict_one_step(start_time, initial_condition=pred)
            pred_time = start_time + self.time_step
            if save:
                # pred[1] is the xr.DataSet that we want to save for now
                # we should first using channel names to map this DataSet to our regular DataArray
                output_path = self.save_output(pred[1], start_time, pred_time, source)
                start_time, source = pred_time, "file"
                output_paths.append(output_path)
            logger.success(f"Rollout step {n+1}/{n_steps} completed")
        return pred, output_paths


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
