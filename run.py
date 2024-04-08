from skyrim import Skyrim

model = Skyrim(
    model_name="panguweather",
    date="20240324",  # str YYYYMMDD
    time=12,  # int 0-23
    lead_time=6,  # hours
)
model.predict()
