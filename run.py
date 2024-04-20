import argparse
import os
import datetime
from loguru import logger
from skyrim.models import FoundationModel
from skyrim.utils import ensure_cds_loaded
from dotenv import load_dotenv, dotenv_values

load_dotenv()

# NOTE:
# ERA5 variables are mean values for previous hour,
# i.e. 13:01 to 14:00 are labelled as "14:00"


OUTPUT_DIR = "./outputs"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        "-m",
        type=str,
        choices=["pangu", "fcnv2_sm", "graphcast", "fcn", "dlwp"],
        default="pangu",
    )
    parser.add_argument(
        "--start_time",
        "-s",
        type=str,
        default="20180101",
    )
    parser.add_argument(
        "--n_steps",
        "-n",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--list_models",
        "-lm",
        action="store_true",
        help="List all available models and exit",
    )

    args = parser.parse_args()

    if args.list_models:
        available_models = FoundationModel.list_available_models()
        print("Available models:", available_models)
        exit()

    # Convert start time string to datetime object
    start_time = datetime.datetime.strptime(args.start_time, "%Y%m%d")

    # initialize the model
    ensure_cds_loaded()
    model = FoundationModel(model_name=args.model_name)

    # set prediction initial state time
    # the input state is fetched from cds
    pred = model.predict(start_time=start_time)

    # save the prediction
    pred_datetime = start_time + model.time_step
    output_meta = (
        f"{start_time.strftime('%Y%m%d_%H:%M')}"
        + "__"
        + f"{pred_datetime.strftime('%Y%m%d_%H:%M')}"
    )

    model_name = model.model_name
    os.makedirs(f"{OUTPUT_DIR}/{model_name}", exist_ok=True)
    output_path = f"{OUTPUT_DIR}/{model_name}/{model_name}__{output_meta}.nc"
    pred.squeeze().to_netcdf(output_path)
    logger.success(f"outputs saved to {output_path}")
