import datetime
import xarray as xr
from dotenv import load_dotenv
from loguru import logger
from typing import Literal, Iterator
from .models import MODELS
from .models.base import GlobalModel, adjust_lead_time, GlobalPrediction
from .models.ensemble import GlobalEnsemble


load_dotenv()


class Skyrim:
    def __init__(
        self,
        *model_names: str,
        ic_source: Literal["cds", "gfs", "ifs"] = "cds",
    ):
        # TODO: device of the model should be configurable

        missing_names = [name for name in model_names if name not in MODELS]
        if missing_names:
            raise ValueError(f"Invalid model name(s): {missing_names}")
        self.model_names = model_names
        self.ic_source = ic_source
        self.model: GlobalEnsemble | GlobalModel
        if len(model_names) > 1:
            logger.info(f"Initializing ensemble model with {model_names}")
            self.model = GlobalEnsemble(model_names)
        else:
            self.model = MODELS[model_names[0]](ic_source=ic_source)
        logger.debug(
            f"Initialized {self.model} model with initial conditions from {ic_source}"
        )

    def __repr__(self) -> str:
        return f"Skyrim(models={self.model_names},ic={self.ic_source})"

    @staticmethod
    def list_available_models():
        return list(MODELS.keys())

    def forecast(
        self,
        start_time: datetime.datetime,
        lead_time: int = 6,
        save: bool = False,
        save_config: dict = {},
    ) -> xr.DataArray | list[xr.DataArray] | list[str]:
        """
        Predict a forecast (multiple snapshots leading to the final shapshot.)
        """
        lead_time = adjust_lead_time(lead_time, steps=6)
        logger.debug(f"Lead time adjusted to nearest multiple of 6: {lead_time} hours")
        n_steps = int(lead_time // (self.model.time_step.total_seconds() / 3600))
        logger.debug(f"Number of prediction steps: {n_steps}")
        start_time = start_time.replace(second=0, microsecond=0)
        if not save:
            return GlobalPrediction(
                self.model.predict_all_steps(start_time=start_time, n_steps=n_steps),
                model_name=self.model_names,
            )

        _, output_paths = self.model.rollout(
            start_time=start_time, n_steps=n_steps, save=True, save_config=save_config
        )
        logger.debug("Prediction completed successfully")
        return output_paths

    def predictions(
        self, start_time: datetime.datetime, lead_time: int = 6
    ) -> Iterator[xr.DataArray]:
        """Step through predictions"""
        for pred in self.model.predict_steps(start_time, lead_time=lead_time):
            yield GlobalPrediction(pred, model_name=self.model_names)

    def predict(
        self,
        date: str,  # YYYMMDD, e.g. 20180101
        time: str,  # HHMM, e.g. 0300, 1400, etc
        lead_time: int = 6,  # in hours 0-24, will be clipped to nearest 6 multiple
        save: bool = False,
        save_config: dict = {},
    ):
        """
        Predict a single snapshot, optionally save all intermediary step snapshots.
        """
        year = int(date[:4])
        month = int(date[4:6])
        day = int(date[6:8])
        hour = int(time[:2])
        minute = int(time[2:4])
        start_time = datetime.datetime(year, month, day, hour, minute)

        # Adjust lead_time to nearest multiple of 6
        lead_time = adjust_lead_time(lead_time, steps=6)
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
        return GlobalPrediction(pred, model_name=self.model_names), output_paths
