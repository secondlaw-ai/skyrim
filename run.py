import argparse
from skyrim import Skyrim
from skyrim.utils import ensure_cds_loaded
from dotenv import load_dotenv

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
        choices=["pangu", "fourcastnet", "fourcastnet_v2", "graphcast", "dlwp"],
        default="pangu",
    )
    parser.add_argument(
        "--date",
        "-s",
        type=str,
        default="20180101",
        help="YYYYMMDD",
    )
    parser.add_argument(
        "--time",
        "-t",
        type=str,
        default="0000",
        help="HHMM",
    )
    parser.add_argument(
        "--lead_time",
        "-l",
        type=int,
        default=6,
        help="Lead time in hours, int 0-24",
    )
    parser.add_argument(
        "--list_models",
        "-lm",
        action="store_true",
        help="List all available models and exit",
    )

    args = parser.parse_args()

    if args.list_models:
        available_models = Skyrim.list_available_models()
        print("Available models:", available_models)
        exit()

    # initialize the model
    ensure_cds_loaded()
    model = Skyrim(model_name=args.model_name)

    # NOTE: the input state is fetched from cds by default
    pred = model.predict(
        date=args.date,
        time=args.time,
        lead_time=args.lead_time,
        save=True,
    )
