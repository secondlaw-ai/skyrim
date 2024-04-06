import os
from pathlib import Path
import requests
from loguru import logger
from ai_models.model import load_model

OUTPUT_DIR = Path("./results")
if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir()
logger.debug(f"Output directory: {OUTPUT_DIR}")

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
if not CHECKPOINT_DIR.exists():
    CHECKPOINT_DIR.mkdir()
logger.debug(f"Checkpoint directory: {CHECKPOINT_DIR}")

BASE_CONFIG = {
    "models": False,
    "model": "YourModelName",  # Replace with your model name
    "debug": False,  # Enable debug mode if needed
    "verbose": 0,  # Verbosity level (increase for more verbose output)
    "retrieve_requests": False,  # Print MARS requests to stdout
    "archive_requests": None,  # Filename to save MARS archive requests
    "requests_extra": None,  # Key-value pairs to extend requests
    "json": False,  # Dump requests in JSON format
    "dump_provenance": None,  # Filename to dump provenance information
    "input": "cds",  # Input source ('mars', 'file', 'cds')
    "file": None,  # File to use if input is 'file'
    "output": "file",  # Output destination (e.g., 'file')
    "date": "-1",  # Analysis start date ('-1' for yesterday)
    "time": 12,  # Analysis start time
    "assets": os.environ.get("AI_MODELS_ASSETS", "."),  # Path to assets directory
    "assets_sub_directory": False,  # Load assets from model-named subdirectory
    "assets_list": False,  # List the assets used by the model
    "download_assets": False,  # Download assets if not present
    "path": "path/to/your/output/file",  # Output file path
    "fields": False,  # Show input fields required by the model
    "expver": None,  # Experiment version
    "class_": None,  # 'class' metadata for output
    "metadata": {},  # Additional metadata as key=value pairs
    "num_threads": 1,  # Number of computation threads
    "lead_time": 240,  # Forecast length in hours
    "hindcast_reference_year": None,  # Year for hindcast-like outputs
    "staging_dates": None,  # Dates for staging hindcast-like outputs
    "only_gpu": False,  # Fail if GPU not available
    "deterministic": False,  # Ensure deterministic computations on GPU
    "model_version": "latest",  # Model version ('latest' by default)
    "remote_execution": os.environ.get("AI_MODELS_REMOTE", "0")
    == "1",  # Enable remote execution based on config
}

model_args = []  # Additional model arguments not covered by cfg


class BaseModel:
    def __init__(self, date: str, time: int, lead_time: int, file: None) -> None:
        self.config = BASE_CONFIG
        self.config["date"] = date
        self.config["time"] = time
        self.config["lead_time"] = lead_time
        self.config["assets"] = CHECKPOINT_DIR

        if file:
            self.config["input"] = "file"
            self.config["file"] = file
            self.from_file = True

    def predict(self):
        raise NotImplementedError

    def download_checkpoint(self):
        raise NotImplementedError


class PanguWeather(BaseModel):
    def __init__(self, date: str, time: int, lead_time: int, file: None) -> None:
        super().__init__(date, time, lead_time, file)
        self.config["model"] = "panguweather"
        self.output_path = (
            OUTPUT_DIR
            / "panguweather/"
            / f"date={date}__time={time:02d}:00__{lead_time}.grib"
        )
        self.config["path"] = str(self.output_path)

        if not Path(self.output_path).exists():
            Path(self.output_path.parent).mkdir(parents=True, exist_ok=True)

        self.download_checkpoint()
        self.model = load_model(self.config["model"], **self.config, model_args=[])

    def download_checkpoint(self):
        download_url_base = (
            "https://get.ecmwf.int/repository/test-data/ai-models/pangu-weather/"
        )
        download_files = ["pangu_weather_24.onnx", "pangu_weather_6.onnx"]
        for filename in download_files:
            file_path = CHECKPOINT_DIR / filename
            if not file_path.exists():
                download_url = f"{download_url_base}{filename}"
                logger.info(f"Downloading {filename}...")
                response = requests.get(download_url)
                response.raise_for_status()  # Raises an HTTPError if the response was an error
                with open(file_path, "wb") as f:
                    f.write(response.content)
                logger.succes(f"Downloaded {filename} to {file_path}")
            else:
                logger.info(f"{filename} already exists.")

    def predict(self):
        self.model.run()


class GraphCast(BaseModel):
    pass


class FourCastNet(BaseModel):
    pass


class FourCastNetV2(BaseModel):
    pass


print(__file__)
