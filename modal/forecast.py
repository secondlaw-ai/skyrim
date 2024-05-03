import os
import secrets
import subprocess
from modal import App, Image, gpu, Volume, forward
from dotenv import load_dotenv

load_dotenv()
CDSAPI_KEY = os.getenv("CDSAPI_KEY")
CDSAPI_URL = os.getenv("CDSAPI_URL")
APP_NAME = "skyrim-dev-forecast"

if not CDSAPI_KEY or not CDSAPI_URL:
    raise Exception("Missing credentials for CDS")


image = (
    Image.from_registry("nvcr.io/nvidia/modulus/modulus:23.11")
    .run_commands("git clone https://github.com/secondlaw-ai/skyrim")
    .workdir("/skyrim")
    .run_commands("pip install -r requirements.txt")
    .env(
        {
            "CDSAPI_KEY": CDSAPI_KEY,
            "CDSAPI_URL": CDSAPI_URL,
            "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID", ""),
            "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY", ""),
            "AWS_DEFAULT_REGION": "eu-west-1",
        }
    )
)
app = App(APP_NAME)
vol = Volume.from_name("forecasts", create_if_missing=True)


@app.function(
    gpu=gpu.A10G(),
    container_idle_timeout=240 * 2,
    timeout=60 * 15,
    image=image,
    volumes={"/skyrim/outputs": vol},
)
def run_inference(
    model_name: str,
    lead_time: int,
    date: str,
    out: str = "/skyrim/outputs",
):
    from skyrim.core import Skyrim

    model = Skyrim(model_name)
    pred, output_paths = model.predict(
        date=date,
        time="0000",
        lead_time=lead_time,
        save=True,
        save_config={"output_dir": out},
    )
    vol.commit()
    print("Saved forecasts!")


analysis_image = (
    Image.debian_slim()
    .pip_install("python-dotenv", "jupyterlab")
    .env(
        {
            "CDSAPI_KEY": CDSAPI_KEY,
            "CDSAPI_URL": CDSAPI_URL,
        }
    )
)


@app.function(
    volumes={"/skyrim/outputs": vol},
    image=analysis_image,
    timeout=60 * 60 * 2,  # 2 hour timeout
    memory=(2048, 4096),  # no need more than 4GB
)
def run_analysis():
    vol.reload()
    token = secrets.token_urlsafe(13)
    with forward(8888) as tunnel:
        url = tunnel.url + "/?token=" + token
        print(f"Starting Jupyter at {url}")
        subprocess.run(
            [
                "jupyter",
                "lab",
                "--no-browser",
                "--allow-root",
                "--ip=0.0.0.0",
                "--port=8888",
                "--LabApp.allow_origin='*'",
                "--LabApp.allow_remote_access=1",
            ],
            env={**os.environ, "JUPYTER_TOKEN": token, "SHELL": "/bin/bash"},
            stderr=subprocess.DEVNULL,
        )


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
    Example: `modal run modal/forecast.py --model pangu --lead-time 12 --date 20240420`
    will run a forecast for 2024-04-21 starting from the initial conditions
    retrieved at 2024-04-21 00:00:00 in ERA5 CDS reanalysis data.

    Next day forecast is about 2GB of data. See https://modal.com/pricing for storage costs.
    You can immediately start analysing the forecast by running `modal run modal/forecast.py:run_analysis`
    This will start a JupyterLab server in a modal container that you can access with the provided URL.
    Alternatively, if you want to work with the data locally, you can run `modal volume get forecasts /skyrim/outputs/[model_name]/[filename] .[your_local_path]`
    Once you are done with the analysis, you can delete the volume with `modal volume rm forecasts /[model_name] -r`
    """
    run_inference.remote(model, lead_time, date)
