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

    def predict(self, start_time: datetime.datetime, n_steps: int = 1):
        # TODO: add saving functionality
        # TODO: add wrapper around the predictions to be able to play with the models

        return wrap_prediction(
            self.model.predict(start_time=start_time, n_steps=n_steps)
        )
