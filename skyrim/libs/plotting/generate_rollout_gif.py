import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import imageio
import os
from tqdm.auto import tqdm
from typing import Literal
from IPython.display import HTML

ProjectionType = Literal["Orthographic", "PlateCarree", "Mollweide", "Robinson"]


def generate_rollout_gif(
    output_paths,
    variable_name,
    gif_path,
    cmap="coolwarm",
    projection: ProjectionType = "Orthographic",
):
    """
    Creates a GIF from a list of NetCDF files using xarray and Cartopy for plotting.
    Each file is visualized on a 3D-like globe using a configurable projection.

    Parameters:
    - output_paths: List of paths to the output NetCDF files.
    - variable_name: Name of the variable to plot (e.g., 't2m').
    - gif_path: Path where the GIF should be saved.
    - cmap: Colormap to use for the plots.
    - projection (ProjectionType): The Cartopy projection to use for plotting.

    # Example usage:
    generate_rollout_gif(output_paths = output_paths,
                               variable_name='t2m',
                               gif_path='output_animation_ortho.gif',
                               projection='Orthographic')

    """
    images = []
    for file_path in tqdm(output_paths, desc="Generating GIF Frames"):
        # Load the dataset
        ds = xr.open_dataarray(file_path)

        # Extract the variable at the last time step and squeeze out single dimensions
        data = ds.sel(channel=variable_name).isel(time=-1).squeeze()

        # Create a figure and set up the geographic projection based on the input parameter
        projection_class = getattr(ccrs, projection)
        plt.figure(figsize=(10, 10))
        ax = plt.axes(
            projection=projection_class(central_longitude=0, central_latitude=0)
        )
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=":")

        # Set manual limits if needed (particularly useful for specific variables like 't2m')
        vmin = data.min().values if variable_name != "t2m" else 220
        vmax = data.max().values if variable_name != "t2m" else 320

        # Plot the data using the specified colormap and no colorbar
        img = data.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            add_colorbar=False,
            vmin=vmin,
            vmax=vmax,
        )

        # Add a colorbar manually with fixed location and size
        plt.colorbar(
            img, ax=ax, orientation="vertical", pad=0.02, aspect=50, extend="both"
        )
        ax.set_title(f"{variable_name.upper()} at Time Step: {ds.time[-1].values}")

        # Save the figure to a temporary PNG file with a tight layout
        temp_png_path = f"temp_{os.path.basename(file_path)}.png"
        plt.savefig(temp_png_path, bbox_inches="tight", dpi=300)  # High resolution
        plt.close()

        # Load the image and append to the list
        images.append(imageio.imread(temp_png_path))

        # Remove the temporary file
        os.remove(temp_png_path)

    # Create the GIF
    imageio.mimsave(
        gif_path, images, duration=1, loop=0
    )  # duration controls the display time for each frame

    print(f"GIF saved to {gif_path}")


import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import imageio
from tqdm import tqdm
import os


def visualize_rollout(
    output_paths, channels, cmap="coolwarm", projection="Orthographic"
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
    for variable_name in channels:
        gif_path = f"{variable_name}_animation.gif"
        generate_rollout_gif(
            output_paths=output_paths,
            variable_name=variable_name,
            gif_path=gif_path,
            cmap=cmap,
            projection=projection,
        )
        gif_paths.append(gif_path)

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
