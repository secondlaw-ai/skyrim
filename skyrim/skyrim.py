import datetime
import os
import time
import dotenv
import xarray as xr
from dataclasses import dataclass, field
from .models import FoundationModel
from .prediction import (
    PanguPrediction,
    FourcastNetPrediction,
    FourcastNetV2Prediction,
    DLWPPrediction,
    GraphcastPrediction,
)
from .utils import timeit
from loguru import logger


dotenv.load_dotenv()

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

model_classes = {
    "pangu": PanguPrediction,
    "fcn": FourcastNetPrediction,
    "fcnv2_sm": FourcastNetV2Prediction,
    "dlwp": DLWPPrediction,
    "graphcast": GraphcastPrediction,
}


def wrap_prediction(model_name, source):
    if model_name in model_classes:
        return model_classes[model_name](source)
    else:
        raise ValueError(f"Model name {model_name} is not supported.")


class Skyrim:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = FoundationModel(model_name=model_name)

    @timeit
    def predict(
        self,
        date: str,  # HHMM, e.g. 0300, 1400, etc
        time: str,  # YYYMMDD, e.g. 20180101
        lead_time: int = 6,  # in hours 0-24, will be clipped to nearest 6 multiple
        save: bool = True,
    ):

        # should have more easy to use interface compare to FoundationModel

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

        # Calculate n_steps by dividing lead_time by the model's time_step (assuming time_step is given in hours)
        n_steps = lead_time // self.model.time_step
        logger.debug(f"Number of prediction steps: {n_steps}")

        # Rollout predictions
        pred, output_paths = self.model.rollout(
            start_time=start_time,
            n_steps=n_steps,
            save=save,
        )

        # You might want to do something with pred or output_paths here
        logger.success("Prediction completed successfully")
        return pred, output_paths
