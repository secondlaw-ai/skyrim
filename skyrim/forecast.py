import argparse
from skyrim.core import Skyrim
from skyrim.core.utils import ensure_cds_loaded
from dotenv import load_dotenv

load_dotenv()

# NOTE:
# ERA5 variables are mean values for previous hour,
# i.e. 13:01 to 14:00 are labelled as "14:00"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_names",
        "-m",
        type=str,
        nargs="+",
        choices=["pangu", "fourcastnet", "fourcastnet_v2", "graphcast", "dlwp"],
        default=["pangu"],  # Default is now a list with one item
    )
    parser.add_argument(
        "--date",
        "-s",
        type=str,
        default="20240421",
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
    parser.add_argument(
        "--initial_conditions",
        "-ic",
        type=str,
        choices=["cds", "ifs", "gfs"],
        default="cds",
        help="Initial conditions provider.",
    )

    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="/skyrim/outputs",
        help="Output directory, can be local or s3 path (e.g. s3://my-path/)",
    )

    args = parser.parse_args()

    if args.list_models:
        available_models = Skyrim.list_available_models()
        print("Available models:", available_models)
        exit()
    # initialize the model
    ensure_cds_loaded()
    model = Skyrim(*args.model_names, ic_source=args.initial_conditions)
    # NOTE: the input state is fetched from cds by default
    pred = model.predict(
        date=args.date,
        time=args.time,
        lead_time=args.lead_time,
        save=True,
        save_config={
            'output_dir':args.output_dir
        }
    )

if __name__ == "__main__":
    main()
