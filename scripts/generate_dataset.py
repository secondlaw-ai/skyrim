import pandas as pd
import gc
import torch
from loguru import logger
from skyrim.common import save_forecast
from datetime import datetime, timedelta
from skyrim.core import Skyrim
from typing import TypeAlias


IC_START_HOURS = {0, 12, 18, 24}  # dissemination is with about 6h delay.
Location: TypeAlias = list[tuple[float, float]]


def clear_memory():
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_max_memory_cached()
    gc.collect()


def generate_large_forecast_dataset(
    dataset_run_id: str,
    locations: list[Location],
    start_time: datetime,
    end_time: datetime,
    lead_time: int,
    model: str,
    ic: str,
    channels: list[str],
    output_dir: str = "s3://skyrim-dev/",
    ic_start_hour: int = 0,
):
    assert ic_start_hour in IC_START_HOURS
    model = Skyrim(model, ic_source=ic)
    lats, lons = [lat for lat, _ in locations], [lon for _, lon in locations]
    pred_start_dates = [start_time + timedelta(days=step) for step in range((end_time - start_time).days + 1)]
    for i, start in enumerate(pred_start_dates, 1):
        logger.debug(f"Generating forecast for {i} / {len(pred_start_dates)}")
        pred_start_time = start.replace(hour=ic_start_hour)
        for pred in model.predictions(start_time=start, lead_time=lead_time):
            # breakpoint()
            save_forecast(
                (pred.prediction.sel(lat=lats, lon=lons, method="nearest") if len(locations) else pred.prediction)
                .sel(channel=channels)
                .assign_coords(pred_start=("time", [pred_start_time for _ in range(pred.prediction.time.size)]))
                .set_xindex("pred_start"),
                model,
                start,
                start + timedelta(hours=lead_time),
                ic,
                {
                    "forecast_id": dataset_run_id,
                    "file_type": "zarr",
                    "output_dir": output_dir,
                },
            )
        gc.collect()


def generate_forecast_dataset(
    dataset_run_id: str,
    locations: list[Location],
    start_time: datetime,
    end_time: datetime,
    lead_time: int,
    model: str,
    ic: str,
    channels: list[str],
    output_dir: str = "s3://skyrim-dev/",
    ic_start_hour: int = 0,
):
    model = Skyrim(model, ic_source=ic)
    lats, lons = [lat for lat, _ in locations], [lon for _, lon in locations]
    pred_start_dates = [start_time + timedelta(days=step) for step in range((end_time - start_time).days + 1)]
    assert ic_start_hour in IC_START_HOURS
    for i, start in enumerate(pred_start_dates, 1):
        logger.debug(f"Generating forecast for {i} / {len(pred_start_dates)}")
        start = start.replace(hour=ic_start_hour)
        pred = model.forecast(
            start_time=start,
            lead_time=lead_time,
        )
        save_forecast(
            (pred.prediction.sel(lat=lats, lon=lons, method="nearest") if len(locations) else pred.prediction)
            .sel(channel=channels)
            .assign_coords(pred_start=("time", [start for _ in range(pred.prediction.time.size)]))
            .set_xindex("pred_start"),
            model,
            start,
            start + timedelta(hours=lead_time),
            ic,
            {
                "forecast_id": dataset_run_id,
                "file_type": "zarr",
                "output_dir": output_dir,
            },
        )
        del pred


def generate_for_uk_farms():
    df = pd.read_csv("./notebooks/20230405_wind_generators_uk.csv")
    locations = df.sort_values("Project Capacity (MW)", ascending=False).head(100)[["lat", "lon"]].to_records(index=False)
    ts = lambda: datetime.now().isoformat(timespec="minutes").replace(":", "_")
    # generate 3 days for pangu and then for fourcastnet_v2
    models = ["pangu", "fourcast_v2", "fourcast"]
    xp_id = ts()
    start_time = datetime(2024, 3, 1)
    end_time = datetime(2024, 6, 30)
    lead_time = 12  # hours
    ic = "ifs"
    for m in models:
        run_id = xp_id + "__" + m
        logger.debug(f"Starting run for model {m}")
        generate_forecast_dataset(
            run_id,
            locations,
            start_time,
            end_time,
            lead_time,
            m,
            ic,
            ["u10m", "v10m", "t2m", "u1000", "v1000"],
        )
        logger.debug(f"Completed run for model {m}")
        clear_memory()


def generate():
    ts = lambda: datetime.now().isoformat(timespec="minutes").replace(":", "_")
    # generate 3 days for pangu and then for fourcastnet_v2
    # models = ["fourcastnet_v2", "fourcastnet", "pangu"]
    # models = ["pangu", "fourcastnet_v2", "fourcastnet"]
    models = ["fourcastnet_v2"]
    ic = "ifs"
    ic_start_hour = 0
    start_time = datetime(2024, 4, 4)  # march 1 has an issue in IFS with w param, double check.
    end_time = datetime(2024, 4, 6)
    lead_time = 48
    make_xp_id = lambda model: model + "__" + f"{ts()}__{ic}__{str(start_time.date()).replace('-','')}_{str(end_time.date()).replace('-','')}"
    for m in models:
        run_id = make_xp_id(m)
        logger.debug(f"Starting run for model {m}")
        args_ = run_id, [], start_time, end_time, lead_time, m, ic, ["u10m", "v10m", "t2m", "u1000", "v1000"]
        kwargs_ = dict(ic_start_hour=ic_start_hour)
        if m != "graphcast":
            generate_forecast_dataset(*args_, **kwargs_)
        else:
            generate_large_forecast_dataset(*args_, **kwargs_)
        logger.debug(f"Completed run for model {m}")
        clear_memory()


if __name__ == "__main__":
    generate()
