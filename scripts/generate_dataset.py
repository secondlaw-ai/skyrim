import pandas as pd
import gc
import torch
from loguru import logger
from skyrim.common import save_forecast
from datetime import datetime, timedelta
from skyrim.core import Skyrim


def clear_memory():
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_max_memory_cached()
    gc.collect()

def generate_forecast_dataset(dataset_run_id, locations, start_time, end_time, lead_time, model, ic, channels, output_dir='s3://skyrim-dev/'):
    model = Skyrim(model, ic_source=ic)
    lats, lons = [lat for lat, _ in locations], [lon for _, lon in locations]
    pred_start_dates = [start_time + timedelta(days=step) for step in range((end_time - start_time).days + 1)]
    for i, start in enumerate(pred_start_dates, 1):
        logger.debug(f'Generating forecast for {i} / {len(pred_start_dates)}')
        pred, _ = model.predict(
            date=str(start.date()).replace('-',''),
            time='0000',
            lead_time=lead_time,
            save=False
        )
        save_forecast(
            pred.prediction.sel(lat=lats,lon=lons, method='nearest').sel(channel=channels),
            model,
            start,
            start + timedelta(hours=lead_time),
            ic,
            {
                'forecast_id': dataset_run_id,
                'file_type':'zarr',
                'output_dir': output_dir,
            }
        )
        del pred


if __name__ == '__main__':
    df =pd.read_csv('./notebooks/20230405_wind_generators_uk.csv')
    locations = df.sort_values('Project Capacity (MW)',ascending=False).head(100)[['lat', 'lon']].to_records(index=False)
    ts = lambda : datetime.now().isoformat(timespec='minutes').replace(':', '_')
    # generate 3 days for pangu and then for fourcastnet_v2
    models = ['pangu', 'fourcastnet_v2']
    xp_id = ts()
    start_time = datetime(2024,6,1)
    end_time = datetime(2024,6,5)
    lead_time = 24 # hours
    ic = 'ifs'
    for m in models:
        run_id = xp_id + '__' + m
        logger.debug(f'Starting run for model {m}')
        generate_forecast_dataset(run_id, locations, start_time, end_time, lead_time, m, ic, ['u1000', 'v1000'])
        logger.debug(f'Completed run for model {m}')
        clear_memory()