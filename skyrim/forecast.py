import click
import subprocess
from dotenv import load_dotenv
from datetime import datetime, timedelta
from pathlib import Path
from skyrim.common import AVAILABLE_MODELS

yesterday = (datetime.now() - timedelta(days=1)).date().isoformat().replace("-", "")

load_dotenv()


# NOTE:
# ERA5 variables are mean values for previous hour,
# i.e. 13:01 to 14:00 are labelled as "14:00"


def run_forecast(
    model_name: str,
    date: str,
    time: str,
    lead_time: int,
    list_models: bool,
    initial_conditions: str,
    output_dir: str,
    filter_vars: str,
):
    # for the local inference setup you need core packages imported here:
    from skyrim.core import Skyrim
    from skyrim.core.utils import ensure_cds_loaded

    if list_models:
        available_models = Skyrim.list_available_models()
        print("Available models:", available_models)
        exit()
    # initialize the model
    ensure_cds_loaded()
    model = Skyrim(model_name, ic_source=initial_conditions)
    pred, output_paths = model.predict(
        date=date,
        time=time,
        lead_time=lead_time,
        save=True,
        save_config={
            "output_dir": output_dir or str(Path.cwd() / "outputs"),
            "filter_vars": (
                filter_vars.split(",") if bool(filter_vars) else []
            ),  # TODO: sanitize vars
        },
    )
    return output_paths


@click.command()
@click.option(
    "--model_name",
    "-m",
    type=click.Choice(AVAILABLE_MODELS, case_sensitive=False),
    default="pangu",
    help="Select model",
)
@click.option("--date", "-d", type=str, default=yesterday, help="YYYYMMDD")
@click.option("--time", "-t", type=str, default="0000", help="HHMM")
@click.option(
    "--lead_time", "-l", type=int, default=6, help="Lead time in hours, int 0-24"
)
@click.option(
    "--list_models", "-lm", is_flag=True, help="List all available models and exit"
)
@click.option(
    "--initial_conditions",
    "-ic",
    type=click.Choice(["cds", "ifs", "gfs"], case_sensitive=False),
    default="gfs",
    help="Initial conditions provider.",
)
@click.option(
    "--output_dir",
    "-o",
    type=str,
    default="",
    help="Output directory, can be local or s3 path (e.g. s3://my-path/)",
)
@click.option(
    "--filter_vars",
    "-f",
    type=str,
    default="",
    help="Filter variables such as t2m (temperature) before saving forecasts.",
)
@click.option(
    "--modal",
    "-mo",
    is_flag=True,
    help="If set, will run on modal.",
)
def main(
    model_name: str,
    date: str,
    time: str,
    lead_time: int,
    list_models: bool,
    initial_conditions: str,
    output_dir: str,
    filter_vars: str,
    modal: bool,
):
    if modal:
        return subprocess.run(
            [
                "modal",
                "run",
                "skyrim/modal/forecast.py",
                "--model-name",
                model_name,
                "--date",
                date,
                "--time",
                time,
                "--lead-time",
                str(lead_time),
                "--initial-conditions",
                initial_conditions,
                "--output-dir",
                output_dir,
                "--filter-vars",
                filter_vars,
            ],
            check=True,
        )

    return run_forecast(
        model_name,
        date,
        time,
        lead_time,
        list_models,
        initial_conditions,
        output_dir,
        filter_vars,
    )


if __name__ == "__main__":
    main()
