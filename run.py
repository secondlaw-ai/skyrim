import os
from loguru import logger
from skyrim.models import FoundationModel
from dotenv import load_dotenv, dotenv_values
import datetime

# Load environment variables from .env file
env_vars = dotenv_values(".env")

# Print the contents of the .env file
for key, value in env_vars.items():
    print(f"{key}: {value}")

load_dotenv()

OUTPUT_DIR = "./outputs"

# NOTE:
# ERA5 variables are mean values for previous hour,
# i.e. 13:01 to 14:00 are labelled as "14:00"

logger.info(f"CDSAPI_URL: {os.environ.get('CDSAPI_URL')}")
logger.info(f"CDSAPI_KEY: {os.environ.get('CDSAPI_KEY')}")

logger.success("imports successful")

available_models = FoundationModel.list_available_models()
print(available_models)


for model_name in available_models:
    # initialize the model
    model = FoundationModel(model_name = model_name)

    # set prediction initial state time
    # the input state is fetched from cds 
    start_time = datetime.datetime(2018,1,1)
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
    pred.to_netcdf(output_path)
    logger.success(f"outputs saved to {output_path}")
