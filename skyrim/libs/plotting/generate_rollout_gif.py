import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import imageio
import os
from pathlib import Path
from tqdm.auto import tqdm
from typing import Literal
from IPython.display import HTML
from loguru import logger

ProjectionType = Literal["Orthographic", "PlateCarree", "Mollweide", "Robinson"]

def generate_rollout_gif(
    output_paths: list[str | Path],
    channel: str,
    output_dir: str | Path,
    cmap: str = "coolwarm",
    projection: str = "Orthographic",
):
    """
    Creates a GIF from a list of NetCDF files using xarray and Cartopy for plotting.
    Each file is visualized on a 3D-like globe using a configurable projection.

    Parameters:
    - output_paths: List of paths to the output NetCDF files.
    - channel: Name of the data channel to plot (e.g., 't2m').
    - output_dir: Directory path where the GIF should be saved.
    - cmap: Colormap to use for the plots.
    - projection: The Cartopy projection to use for plotting.

    # Example usage:
    generate_rollout_gif(output_paths = output_paths,
                         channel='t2m',
                         output_dir='.',
                         projection='Orthographic')
    """

    # Initialize lists to store all data for computing global statistics
    all_data = []

    # Load data and append to list
    for file_path in output_paths:
        with xr.open_dataarray(file_path) as ds:
            data = ds.sel(channel=channel).isel(time=-1).squeeze()
            all_data.append(data.values.flatten())

    # Concatenate all data into a single array and compute global statistics
    all_data = np.concatenate(all_data)
    mean_val = np.mean(all_data)
    std_val = np.std(all_data)
    global_min = mean_val - 3 * std_val
    global_max = mean_val + 3 * std_val

    # create gif_path including model name and simulation time span
    model_name = os.path.basename(output_paths[0]).split("__")[0]
    start_time = os.path.basename(output_paths[0]).split("__")[2]
    end_time = os.path.basename(output_paths[-1]).split("__")[3].replace(".nc", "")
    gif_path = (
        Path(output_dir) / f"{model_name}_{start_time}_to_{end_time}_{channel}.gif"
    )

    # Check if the gif already exists
    if gif_path.exists():
        print(f"GIF already exists: {gif_path}")
        return gif_path

    images = []
    for file_path in tqdm(output_paths, desc="Generating GIF Frames"):
        ds = xr.open_dataarray(file_path)
        data = ds.sel(channel=channel).isel(time=-1).squeeze()

        projection_class = getattr(ccrs, projection)
        plt.figure(figsize=(10, 10))
        ax = plt.axes(
            projection=projection_class(central_longitude=0, central_latitude=0)
        )
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=":")

        img = data.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            add_colorbar=False,
            vmin=global_min,
            vmax=global_max,
        )
        plt.colorbar(
            img, ax=ax, orientation="vertical", pad=0.02, aspect=50, extend="both"
        )
        ax.set_title(f"{channel.upper()} at Time Step: {ds.time[-1].values}")

        temp_png_path = f"temp_{os.path.basename(file_path)}.png"
        plt.savefig(temp_png_path, bbox_inches="tight", dpi=300)
        plt.close()

        images.append(imageio.imread(temp_png_path))
        os.remove(temp_png_path)

    imageio.mimsave(gif_path, images, duration=1, loop=0)
    print(f"GIF saved to {gif_path}")
    return gif_path


def visualize_rollout(
    output_paths: list[str | Path],
    channels: str,
    output_dir: str | Path,
    cmap="coolwarm",
    projection="Orthographic",
):
    """
    Generates and displays GIFs for specified variables from NetCDF files.

    Parameters:
    - output_paths: List of paths to the output NetCDF files.
    - channels: List of variable names to plot and generate GIFs.
    - cmap: Colormap to use for the plots.
    - projection: The Cartopy projection to use for plotting.
    """
    gif_paths = []
    for channel in channels:
        gif_path = generate_rollout_gif(
            output_paths=output_paths,
            channel=channel,
            output_dir=output_dir,
            cmap=cmap,
            projection=projection,
        )
        gif_paths.append(gif_path)
    logger.debug(f"Generated GIFs: {gif_paths}")
    return display_gifs_side_by_side(gif_paths)


def display_gifs_side_by_side(gif_paths):
    """Display given GIFs side by side in a Jupyter Notebook."""
    if not gif_paths:
        return HTML("<p>No GIFs provided.</p>")

    num_gifs = len(gif_paths)
    width_percent = 100 / num_gifs

    img_tags = " ".join(
        f'<img src="{path}" style="width: {width_percent}%; margin-right: 2px;">'
        for path in gif_paths[:-1]
    )
    img_tags += f'<img src="{gif_paths[-1]}" style="width: {width_percent}%;">'  # Last image without right margin

    html_str = f"""
    <div style="display:flex; justify-content:space-between;">
        {img_tags}
    </div>
    """
    return HTML(html_str)


# Example usage:
# output_paths = ['path_to_netCDF1.nc', 'path_to_netCDF2.nc']
# channels = ['t2m', 'u10m', 'v10m']
# visualize_rollout(output_paths, channels)
