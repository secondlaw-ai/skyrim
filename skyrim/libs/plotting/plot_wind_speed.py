import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt
from skyrim.models.base import (
    GlobalPredictionRollout,
)


def plot_wind_speed(
    rollout_snapshots: list,
    lat: float,
    lon: float,
    pressure_level: int = 1000,
    n_step: int = 1,
):
    """
    Plot the wind speed from a list of rollout_snapshot paths over time using a line plot with markers.

    Parameters:
        rollout_snapshots: List of data snapshots or paths to the snapshots for prediction.
        lat: Latitude of the location for which to predict wind speed.
        lon: Longitude of the location for which to predict wind speed.
        pressure_level: Atmospheric pressure level in hPa to consider for wind speed.
        n_step: Time step index to fetch the wind speed (default is 1, first future time step).
    """
    # Disable loguru warning for this model
    logger.disable("skyrim.models.base")

    # Create a GlobalPredictionRollout object and retrieve wind speed and time steps
    rollout = GlobalPredictionRollout(rollout_snapshots)
    wind_speed = rollout.wind_speed(
        lat=lat, lon=lon, pressure_level=pressure_level, n_step=n_step
    )
    formatted_time = pd.to_datetime(rollout.time_steps).strftime("%Y-%m-%d %H:%M")

    # Plotting using a line plot with markers
    plt.figure(figsize=(12, 6))
    plt.plot(
        formatted_time,
        wind_speed,
        "--s",
        linewidth=1,
        markersize=8,
        markerfacecolor="blue",
    )
    plt.xticks(rotation=45, fontsize=8)
    plt.ylabel("Wind Speed [m/s]")
    plt.xlabel("Date/time")
    plt.grid(True)  # Optionally add grid for better readability
    plt.show()
