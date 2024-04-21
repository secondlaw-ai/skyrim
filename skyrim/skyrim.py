import datetime
import os
import dotenv
from .models import (
    PanguModel,
    PanguPrediction,
    FourcastnetV2Model,
    FourcastnetV2Prediction,
    DLWPModel,
    DLWPPrediction,
    GraphcastModel,
    GraphcastPrediction,
    FourcastnetModel,
    FourcastnetPrediction,
)
from loguru import logger


dotenv.load_dotenv()
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

MODEL_FACTORY = {
    "pangu": (PanguModel, PanguPrediction),
    "fourcastnet": (FourcastnetModel, FourcastnetPrediction),
    "fourcastnet_v2": (FourcastnetV2Model, FourcastnetV2Prediction),
    "dlwp": (DLWPModel, DLWPPrediction),
    "graphcast": (GraphcastModel, GraphcastPrediction),
}


def wrap_prediction(model_name, source):
    if model_name in MODEL_FACTORY:
        return MODEL_FACTORY[model_name][1](source)
    else:
        raise ValueError(f"Model name {model_name} is not supported.")


class Skyrim:
    def __init__(self, model_name: str):
        self.model_name = model_name
        if model_name in MODEL_FACTORY:
            logger.debug(f"Initializing model {model_name}")
            self.model = MODEL_FACTORY[model_name][0]()
        else:
            raise ValueError(f"Model name {model_name} is not supported.")

    @staticmethod
    def list_available_models(self):
        return list(MODEL_FACTORY.keys())

    def predict(
        self,
        date: str,  # HHMM, e.g. 0300, 1400, etc
        time: str,  # YYYMMDD, e.g. 20180101
        lead_time: int = 6,  # in hours 0-24, will be clipped to nearest 6 multiple
        save: bool = False,
    ):

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
