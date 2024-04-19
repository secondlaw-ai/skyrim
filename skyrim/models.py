import os
import datetime
from pathlib import Path

import time
from loguru import logger
import xarray as xr

from earth2mip import inference_ensemble, registry
from earth2mip.initial_conditions import cds, get_initial_condition_for_model

import earth2mip.networks.dlwp as dlwp
import earth2mip.networks.pangu as pangu
import earth2mip.networks.fcnv2_sm as fcnv2_sm
import earth2mip.networks.fcn as fcn
import earth2mip.networks.graphcast as graphcast

OUTPUT_DIR = Path(__file__).parent.parent / Path("./outputs")
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

class FoundationModel:
    def __init__(self, model_name: str):
        
        self.model_name = model_name
                
        # get the registry path
        self.registry_path = registry_path = (
            f"gs://dm_{model_name}" if model_name == "graphcast" else f"e2mip://{model_name}"
        )
        # load model package
        self.model_package = registry.get_model(registry_path)
        logger.info(f"Fetching {model_name} model package from {self.registry_path}")
        
        # load model into memory
        clock = time.time()
        self.model = MODEL_LOADERS[self.model_name](self.model_package)
        logger.success(f"Loaded {self.model_name} model in {time.time()-clock:.1f} seconds")
        
        # load model data source
        self.data_source = cds.DataSource(self.model.in_channel_names)
        
    @staticmethod
    def list_available_models() -> list:
        """Returns a list of all available model names that can be loaded."""
        return list(MODEL_LOADERS.keys())
    
    @property
    def time_step(self):
        return self.model.time_step
    
    def predict(self, start_time: datetime.datetime, n_steps: int = 1) -> xr.DataArray | xr.Dataset:
        # TODO: 
        # - add multistep functionality for graphcast
        # - check if n_steps=1 always maps to 6 hours
        # - add saving functionality
        # - add functionality to flush the loaded data_source from cache
        
        if self.model_name != "graphcast": 
            return inference_ensemble.run_basic_inference(
                model = self.model,
                n =  n_steps,
                data_source = self.data_source,
                time = start_time
            )
        else:
            # NOTE: this only works for graphcast operational model
            # some info about stepper: 
            # https://github.com/NVIDIA/earth2mip/blob/86b11fe4ba2f19641802112e8b0ba6b962123130/earth2mip/time_loop.py#L114-L122
            
            self.stepper = self.model.stepper
            x = get_initial_condition_for_model(
                time_loop = self.model,
                data_source = self.data_source,
                time = start_time,
            )
            
            state = self.stepper.initialize(x, start_time)
            state, output = self.stepper.step(state)
            # output.shape: torch.Size([1, 83, 721, 1440])
            # len(state): 3, 
            # state[0]: Timestamp('2018-01-02 06:00:00')
            return state[1]

class FoundationEnsemble:
    def __init__(self, model_names: list):
        # maybe add ensemble weights that is learned?
        # how to load and flush the models from gpu? 
        pass
    def predict(start_time, n_step:int = 1):
        pass
    
        



