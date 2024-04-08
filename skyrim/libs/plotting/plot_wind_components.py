import matplotlib.pyplot as plt
import numpy as np


def plot_wind_components(data_dict, wind_farm_name):
    """
    Plot bar charts for the u components, v components, and the vectorial sum of u and v components
    of wind speed for each date in a single row with three columns.

    Parameters:
    - data_dict: A dictionary with dates as keys and dictionaries with 'u' and 'v' wind components as values.
    - wind_farm_name: Name of the wind farm.

    Examples:
    # Example usage:
    >>> data_dict = {
        '2022-03-22': {'u': 5, 'v': 3},
        '2022-03-23': {'u': -2, 'v': 4},
        '2022-03-24': {'u': 1, 'v': -1},
    }
    >>> wind_farm_name = 'Farm 1'
    >>> plot_wind_components(data_dict, wind_farm_name)
    """
    dates = list(data_dict.keys())
    u_values = [data_dict[date]["u"] for date in dates]
    v_values = [data_dict[date]["v"] for date in dates]
    overall_wind = [np.sqrt(u**2 + v**2) for u, v in zip(u_values, v_values)]

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # 1 row, 3 columns

    # Plot u components
    axs[0].bar(dates, u_values, color="skyblue")
    axs[0].set_title("U Components")
    axs[0].set_ylabel("Wind Speed (m/s)")
    axs[0].set_xlabel("Date")
    axs[0].tick_params(labelrotation=45)

    # Plot v components
    axs[1].bar(dates, v_values, color="lightgreen")
    axs[1].set_title("V Components")
    axs[1].set_xlabel("Date")
    axs[1].tick_params(labelrotation=45)

    # Plot vectorial sum of u and v components
    axs[2].bar(dates, overall_wind, color="salmon")
    axs[2].set_title("Overall Wind Speed")
    axs[2].set_xlabel("Date")
    axs[2].tick_params(labelrotation=45)

    fig.suptitle(f"Wind Speed Components at {wind_farm_name}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
