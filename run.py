from loguru import logger
from skyrim import Skyrim
from dataclasses import asdict

# NOTE:
# ERA5 variables are mean values for previous hour,
# i.e. 13:01 to 14:00 are labelled as "14:00"


logger.success("imports successful")
model = Skyrim(
    # model_name="graphcast",
    # model_name="panguweather",
    # model_name="fourcastnetv2",
    model_name="fourcastnet",
    date="20240324",  # str YYYYMMDD
    time=12,  # int 0-23
    lead_time=12,  # hours
)
pred = model.predict()
logger.info(f"Prediction file saved to {pred.filepath}")
logger.info(f"Date: {pred.date}")
logger.info(f"Time: {pred.time}")
logger.info(f"Lead time: {pred.lead_time}")
logger.info(f"Input: {pred.input}")
