import os
import datetime
from pathlib import Path

import time
from loguru import logger
import xarray as xr

from earth2mip import inference_ensemble, registry
from earth2mip.initial_conditions import cds, get_initial_condition_for_model
from earth2mip.model_registry import Package

import earth2mip.networks.dlwp as dlwp
import earth2mip.networks.pangu as pangu
import earth2mip.networks.fcnv2_sm as fcnv2_sm
import earth2mip.networks.fcn as fcn
import earth2mip.networks.graphcast as graphcast
from typing import Literal
from .inference import run_basic_inference

OUTPUT_DIR = Path(__file__).parent.parent.resolve() / Path("./outputs")
if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir()
    logger.success(f"Created output directory: {OUTPUT_DIR}")

MODEL_LOADERS = {
    "pangu": pangu.load,
    "fcnv2_sm": fcnv2_sm.load,
    "graphcast": graphcast.load_time_loop_operational,
    "fcn": fcn.load,
    "dlwp": dlwp.load,
}


class NumericalModel:
    def __init__(self, model_name: str):
        raise NotImplementedError

    def predict(self, start_time: datetime.datetime, n_steps: int = 1):
        # NOTE:
        # this could be called predict_local, or predict_regional, or predict_point
        # we should align the interface somehow
        raise NotImplementedError


class IFS(NumericalModel):
    # TODO: add docstring with links to some info
    pass


class GFS(NumericalModel):
    # TODO: add docstring with links to some info
    pass


class FoundationModel:
    def __init__(self, model_name: str):

        self.model_name = model_name

        # get the registry path
        self.registry_path = registry_path = (
            f"e2mip://{model_name}"
            if model_name == "graphcast"
            else f"e2mip://{model_name}"
        )
        # load model package
        self.model_package = registry.get_model(registry_path)
        logger.info(f"Fetching {model_name} model package from {self.registry_path}")

        # load model into memory
        clock = time.time()
        self.model = MODEL_LOADERS[self.model_name](self.model_package)
        logger.success(
            f"Loaded {self.model_name} model in {time.time()-clock:.1f} seconds"
        )

        # load model data source
        self.data_source = cds.DataSource(self.model.in_channel_names)

    @staticmethod
    def list_available_models() -> list:
        """Returns a list of all available model names that can be loaded."""
        return list(MODEL_LOADERS.keys())

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
        # TODO:
        # - add saving functionality?
        # - add functionality to flush the loaded data_source from cache

        if self.model_name != "graphcast":
            return run_basic_inference(
                model=self.model,
                n=1,
                data_source=self.data_source,
                time=start_time,
                x=initial_condition,
            )
        else:
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
        self,
        start_time: datetime.datetime,
        n_steps: int = 3,
        save: bool = True,
    ) -> tuple[xr.DataArray | xr.Dataset, list[str]]:
        # it does not make sense to keep all the results in the memory
        # return final pred and list of paths of the saved predictions
        # TODO: add functionality to rollout from a given initial condition

        pred, output_paths, source = None, [], "cds"
        for n in range(n_steps):
            pred = self.predict_one_step(start_time, initial_condition=pred)
            pred_time = start_time + self.time_step
            if save:
                output_path = self.save_output(pred, start_time, pred_time, source)
                start_time, source = pred_time, "file"
                output_paths.append(output_path)
            logger.success(f"Rollout step {n+1}/{n_steps} completed")
        return pred, output_paths

    def save_output(
        self,
        pred: xr.DataArray | xr.Dataset,
        start_time: datetime.datetime,
        pred_time: datetime.datetime,
        source: Literal["cds", "file"] = "cds",
        output_dir=OUTPUT_DIR,
    ):
        # e.g.:
        # filename = "pangu__20180101_00:00__20180101_06:00.nc"
        # output_path = "./outputs/pangu/pangu__20180101_00:00__20180101_06:00.nc"

        filename = (
            f"{self.model_name}" + "__"
            f"{source}__"
            f"{start_time.strftime('%Y%m%d_%H:%M')}"
            + "__"
            + f"{pred_time.strftime('%Y%m%d_%H:%M')}.nc"
        )
        output_path = OUTPUT_DIR / self.model_name / filename

        logger.info(f"Saving outputs to {output_path}")
        if not output_path.parent.exists():
            logger.info(
                f"Creating parent directory to save outputs: {output_path.parent}"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)

        pred.to_netcdf(output_path)
        logger.success(f"outputs saved to {output_path}")
        return output_path


class FoundationEnsemble:
    def __init__(self, model_names: list):
        # maybe add ensemble weights that is learned?
        # how to load and flush the models from gpu?
        pass

    def predict(start_time, n_step: int = 1):
        pass
