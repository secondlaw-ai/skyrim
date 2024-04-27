import os
from modal import App, Image, gpu
from skyrim.utils import ensure_cds_loaded

ensure_cds_loaded()
CDSAPI_KEY = os.getenv("CDSAPI_KEY")
CDSAPI_URL = os.getenv("CDSAPI_URL")
APP_NAME = "skyrim-dev-sample"

if not CDSAPI_KEY or not CDSAPI_URL:
    raise Exception("Missing credentials for CDS")

# Setup for Skyrim Image
image = (
    Image.from_registry("nvcr.io/nvidia/modulus/modulus:23.11")
    .run_commands("git clone https://github.com/secondlaw-ai/skyrim")
    .workdir("/skyrim")
    .run_commands("pip install -r requirements.txt")
    .env(
        {
            "CDSAPI_KEY": CDSAPI_KEY,
            "CDSAPI_URL": CDSAPI_URL,
        }
    )
)
app = App(APP_NAME)


@app.function(
    gpu=gpu.A10G(), container_idle_timeout=240 * 2, timeout=60 * 15, image=image
)
def run_inference(model_name: str, lead_time: int, date: str):
    from skyrim import Skyrim

    model = Skyrim(model_name)
    pred = model.predict(
        date="20240420",
        time="0000",
        lead_time=6,
        save=True,
    )
    print("Completed!")


@app.local_entrypoint()
def main(date: str = "20240420", model: str = "pangu", lead_time: int = 6):
    """
    args:
    date: str
        Date in YYYYMMDD format
    model: str
        Name of the model to run inference
    lead_time: int
        Lead time in hours
    Run inference for the given model, lead_time and date
    Example: `modal run forecast.py --model pangu --lead_time 24 --date 20240420`
    will run a forecast for 2024-04-21 starting from the initial conditions
    retrieved at 2024-04-21 00:00:00 in ERA5 CDS reanalysis data
    """
    run_inference.remote(model, lead_time, date)
