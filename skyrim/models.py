import os
from pathlib import Path
import requests
from loguru import logger
from ai_models.model import load_model

OUTPUT_DIR = Path(__file__).parent.parent / Path("./results")
if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir()
    logger.success(f"Created output directory: {OUTPUT_DIR}")

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
if not CHECKPOINT_DIR.exists():
    CHECKPOINT_DIR.mkdir()
    logger.success(f"Checkpoint directory: {CHECKPOINT_DIR}")

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

    DOWNLOAD_URL_BASE = None
    DOWNLOAD_FILES = None

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

    @classmethod
    def download_checkpoint(cls):
        if cls.DOWNLOAD_URL_BASE is None or cls.DOWNLOAD_FILES is None:
            raise ValueError("Download URL base or files not set in the subclass")

        for filename in cls.DOWNLOAD_FILES:
            file_path = CHECKPOINT_DIR / filename
            if not file_path.exists():
                download_url = f"{cls.DOWNLOAD_URL_BASE}{filename}"
                if not file_path.parent.exists():
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Downloading {filename}...")
                response = requests.get(download_url)
                response.raise_for_status()
                with open(file_path, "wb") as f:
                    f.write(response.content)
                logger.success(f"Downloaded {filename} to {file_path}")
            else:
                logger.info(f"{filename} already exists.")


class PanguWeather(BaseModel):
    DOWNLOAD_URL_BASE = (
        "https://get.ecmwf.int/repository/test-data/ai-models/pangu-weather/"
    )
    DOWNLOAD_FILES = ["pangu_weather_24.onnx", "pangu_weather_6.onnx"]

    def __init__(self, date: str, time: int, lead_time: int, file: None) -> None:
        super().__init__(date, time, lead_time, file)
        self.config["model"] = "panguweather"
        self.output_path = (
            OUTPUT_DIR
            / "panguweather/"
            / f"date={date}__time={time:02d}:00__{lead_time}__input={self.config['input']}.grib"
        )
        self.config["path"] = str(self.output_path)

        if not Path(self.output_path).exists():
            Path(self.output_path.parent).mkdir(parents=True, exist_ok=True)

        self.download_checkpoint()
        self.model = load_model(self.config["model"], **self.config, model_args=[])

    def predict(self):
        self.model.run()
        return self.output_path


class GraphCast(BaseModel):

    DOWNLOAD_URL_BASE = "https://storage.googleapis.com/dm_graphcast/"
    DOWNLOAD_FILES = [
        "params/GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - pressure levels 13 - mesh 2to6 - precipitation output only.npz",
        "stats/diffs_stddev_by_level.nc",
        "stats/mean_by_level.nc",
        "stats/stddev_by_level.nc",
    ]

    def __init__(self, date: str, time: int, lead_time: int, file: None) -> None:
        super().__init__(date, time, lead_time, file)
        self.config["model"] = "graphcast"
        self.output_path = (
            OUTPUT_DIR
            / "graphcast/"
            / f"date={date}__time={time:02d}:00__{lead_time}__input={self.config['input']}.grib"
        )
        self.config["path"] = str(self.output_path)

        if not Path(self.output_path).exists():
            Path(self.output_path.parent).mkdir(parents=True, exist_ok=True)

        self.download_checkpoint()
        self.model = load_model(self.config["model"], **self.config, model_args=[])

    def predict(self):
        self.model.run()
        return self.output_path


class FourCastNet(BaseModel):
    DOWNLOAD_URL_BASE = (
        "https://get.ecmwf.int/repository/test-data/ai-models/fourcastnet/0.0/"
    )

    DOWNLOAD_FILES = [
        "backbone.ckpt",
        "precip.ckpt",
        "global_means.npy",
        "global_stds.npy",
    ]

    def __init__(self, date: str, time: int, lead_time: int, file: None) -> None:
        super().__init__(date, time, lead_time, file)
        self.config["model"] = "fourcastnet"
        self.output_path = (
            OUTPUT_DIR
            / "fourcastnet/"
            / f"date={date}__time={time:02d}:00__{lead_time}__input={self.config['input']}.grib"
        )
        self.config["path"] = str(self.output_path)

        if not Path(self.output_path).exists():
            Path(self.output_path.parent).mkdir(parents=True, exist_ok=True)

        self.download_checkpoint()
        self.model = load_model(self.config["model"], **self.config, model_args=[])

    def predict(self):
        self.model.run()
        return self.output_path

    pass


class FourCastNetV2(BaseModel):
    DOWNLOAD_URL_BASE = (
        "https://get.ecmwf.int/repository/test-data/ai-models/fourcastnetv2/small/"
    )
    DOWNLOAD_FILES = ["weights.tar", "global_means.npy", "global_stds.npy"]

    def __init__(self, date: str, time: int, lead_time: int, file: None) -> None:
        super().__init__(date, time, lead_time, file)
        self.config["model"] = "fourcastnetv2-small"
        self.output_path = (
            OUTPUT_DIR
            / "fourcastnetv2-small/"
            / f"date={date}__time={time:02d}:00__{lead_time}__input={self.config['input']}.grib"
        )
        self.config["path"] = str(self.output_path)

        if not Path(self.output_path).exists():
            Path(self.output_path.parent).mkdir(parents=True, exist_ok=True)

        self.download_checkpoint()
        self.model = load_model(self.config["model"], **self.config, model_args=[])

    def predict(self):
        self.model.run()
        return self.output_path

    pass
