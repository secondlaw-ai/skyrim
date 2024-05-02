import xarray as xr
import pandas as pd
import modal
import subprocess


"""
[forecast_id]/[zarr_store]
"""


def write_forecast(forecast_id: str, forecast: xr.Dataset):
    forecast.to_zarr(f"s3://s3-bucket-name/{forecast_id}/forecast.zarr", mode="a")


def read_forecast(forecast_id: str):
    return xr.open_dataset("s3://s3-bucket-name/forecast.nc")


app = modal.App()  # Note: prior to April 2024, "app" was called "stub"

s3_bucket_name = "s3-bucket-name"  # Bucket name not ARN.
s3_access_credentials = modal.Secret.from_dict(
    {
        "AWS_ACCESS_KEY_ID": "...",
        "AWS_SECRET_ACCESS_KEY": "...",
    }
)


@app.function(
    volumes={
        "/my-mount": modal.CloudBucketMount(
            s3_bucket_name, secret=s3_access_credentials
        )
    }
)
def f():
    subprocess.run(["ls", "/my-mount"])
