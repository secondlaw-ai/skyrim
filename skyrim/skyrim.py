from loguru import logger
from .models import PanguWeather, GraphCast

MODEL_CLASS_MAP = {
    "panguweather": PanguWeather,
    "graphcast": GraphCast,
}


def initiazlize_model(
    model_name: str, date: str, time: int, lead_time: int, file: None
):
    model = MODEL_CLASS_MAP[model_name](date, time, lead_time, file)
    logger.success(f"Model {model_name} initialized")
    return model


class Skyrim:
    def __init__(
        self, model_name: str, date: str, time: int = 12, lead_time: int = 24, file=None
    ) -> None:

        self.model = initiazlize_model(model_name, date, time, lead_time, file)

    def predict(self):
        self.model.predict()

    def predict_local(self, date, time: int, lead_time: int, lat: float, lon: float):
        pass

    def predict_from_file(self, file_path: str):
        pass
