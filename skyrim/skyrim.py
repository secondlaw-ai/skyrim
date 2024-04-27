import datetime
import os
from dotenv import load_dotenv
from loguru import logger
from .models import MODEL_FACTORY
from .models.ensemble import GlobalEnsemblePrediction, GlobalEnsemble
from earth2mip.schema import InitialConditionSource
load_dotenv()
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def wrap_prediction(model_name, source):
    if isinstance(model_name, str):
        if model_name in MODEL_FACTORY:
            return MODEL_FACTORY[model_name][1](source)
        else:
            raise ValueError(f"Model name {model_name} is not supported.")
    elif isinstance(model_name, list):
        return GlobalEnsemblePrediction(source)
    else:
        raise ValueError("Invalid model name. Must be a string or a list of strings.")
    
class Skyrim:
    def __init__(self, *models: str, ic_provider: InitialConditionSource = InitialConditionSource.cds):
        for model in models:
            if not model in MODEL_FACTORY:
                raise ValueError(f"Model {model} is not supported.")
            
        if len(models) > 1:
            logger.info(f"Initializing ensemble model with {models}")
            self.model_name = models
            self.model = GlobalEnsemble(models)

        logger.debug(f"Initializing {model} model with IC from {ic_provider}")
        self.model_name = model
        self.model = MODEL_FACTORY[model][0](ic_provider=ic_provider)

    def __repr__(self) -> str:
        return f"Skyrim({self.model_name})"

    @staticmethod
    def list_available_models():
        return list(MODEL_FACTORY.keys())

    def predict(
        self,
        date: str,  # YYYMMDD, e.g. 20180101
        time: str,  # HHMM, e.g. 0300, 1400, etc
        lead_time: int = 6,  # in hours 0-24, will be clipped to nearest 6 multiple
        save: bool = False
    ):
        # TODO: output dir should be configurable, currently hardcoded to outputs/{model_name}/*
        # TODO: add checks for date and time format
        # Create datetime object using date and time arguments as start_time
        year = int(date[:4])
        month = int(date[4:6])
        day = int(date[6:8])
        hour = int(time[:2])
        minute = int(time[2:4])
        start_time = datetime.datetime(year, month, day, hour, minute)
        # Adjust lead_time to nearest multiple of 6
        lead_time = max(6, (lead_time // 6) * 6)
        logger.debug(f"Lead time adjusted to nearest multiple of 6: {lead_time} hours")
        # Calculate n_steps by dividing lead_time by the model's time_step
        n_steps = int(lead_time // (self.model.time_step.total_seconds() / 3600))
        logger.debug(f"Number of prediction steps: {n_steps}")
        # Rollout predictions
        pred, output_paths = self.model.rollout(
            start_time=start_time,
            n_steps=n_steps,
            save=save,
        )
        # You might want to do something with pred or output_paths here
        logger.debug("Prediction completed successfully")
        return wrap_prediction(self.model_name, pred), output_paths
