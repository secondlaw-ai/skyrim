import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt
from skyrim.core.models.base import (
    GlobalPredictionRollout,
)


def plot_wind_speed(
    rollout_snapshots: list,
    lat: float,
    lon: float,
    pressure_level: int = 1000,
    n_step: int = 1,
    ax=None,
    show=True,
    figsize=(12, 6),
    **kwargs,
):
    """
    Plot the wind speed from a list of rollout_snapshot paths over time using a line plot with markers.
    Allows for additional plotting on the same figure if desired, with adjustable figure size.

    Parameters:
        rollout_snapshots: List of data snapshots or paths to the snapshots for prediction.
        lat: Latitude of the location for which to predict wind speed.
        lon: Longitude of the location for which to predict wind speed.
        pressure_level: Atmospheric pressure level in hPa to consider for wind speed.
        n_step: Time step index to fetch the wind speed (default is 1, first future time step).
        ax (matplotlib.axes.Axes, optional): Axis object to plot on. If None, creates a new figure.
        show (bool): Whether to show the plot immediately. If False, plot can be further customized.
        figsize (tuple): Figure size in inches, given as (width, height).
    """
    # Disable loguru warning for this model
    logger.disable("skyrim.models.base")

    # If no axis is provided, create a new figure and axis with specified figure size
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Create a GlobalPredictionRollout object and retrieve wind speed and time steps
    rollout = GlobalPredictionRollout(rollout_snapshots)
    wind_speed = rollout.wind_speed(
        lat=lat, lon=lon, pressure_level=pressure_level, n_step=n_step
    )
    formatted_time = pd.to_datetime(rollout.time_steps).strftime("%Y-%m-%d %H:%M")

    # Plotting using a line plot with markers on the provided or new axis
    ax.plot(
        formatted_time,
        wind_speed,
        "--*",
        linewidth=1,
        markersize=8,
        markerfacecolor="blue",
        **kwargs,
    )
    ax.set_xticklabels(formatted_time, rotation=90, fontsize=8)
    ax.set_ylabel("Wind Speed [m/s]")
    ax.set_xlabel("Date/time")
    ax.grid(True)  # Optionally add grid for better readability

    # Show plot if requested
    if show:
        plt.show()

    # Return the axes object for further use
    return ax
