import datetime
from dotenv import load_dotenv
from loguru import logger
from typing import Literal
from pathlib import Path
from .models import MODEL_FACTORY
from .models.ensemble import GlobalEnsemblePrediction, GlobalEnsemble

load_dotenv()


def wrap_prediction(model_names, source):
    if len(model_names) > 1:
        return GlobalEnsemblePrediction(source)
    return MODEL_FACTORY[model_names[0]][1](source)


class Skyrim:
    def __init__(
        self,
        *model_names: str,
        ic_source: Literal["cds", "gfs", "ifs"] = "cds",
    ):
        # TODO: device of the model should be configurable

        missing_names = [name for name in model_names if name not in MODEL_FACTORY]
        if missing_names:
            raise ValueError(f"Invalid model name(s): {missing_names}")

        self.model_names = model_names
        self.ic_source = ic_source

        if len(model_names) > 1:
            logger.info(f"Initializing ensemble model with {model_names}")
            self.model = GlobalEnsemble(model_names)

        else:
            self.model = MODEL_FACTORY[model_names[0]][0](ic_source=ic_source)
        logger.debug(
            f"Initialized {self.model} model with initial conditions from {ic_source}"
        )

    def __repr__(self) -> str:
        return f"Skyrim(models={self.model_names},ic={self.ic_source})"

    @staticmethod
    def list_available_models():
        return list(MODEL_FACTORY.keys())

    def predict(
        self,
        date: str,  # YYYMMDD, e.g. 20180101
        time: str,  # HHMM, e.g. 0300, 1400, etc
        lead_time: int = 6,  # in hours 0-24, will be clipped to nearest 6 multiple
        save: bool = False,
        save_config: dict = {},
    ):
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
            save_config=save_config,
        )
        # You might want to do something with pred or output_paths here
        logger.debug("Prediction completed successfully")
        return wrap_prediction(self.model_names, pred), output_paths
