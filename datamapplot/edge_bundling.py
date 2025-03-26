import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import RegularGridInterpolator
from datashader.bundling import hammer_bundle


def _hex_to_rgb(hex_color):
    """
    Helper function.

    Converts a hex color string to an RGB tuple.

    Parameters:
    - hex_color (str): A hex color string, e.g., '#FF5733' or '#FF5733FF' for RGBA.

    Returns:
    - rgb (tuple): A tuple of integers representing the RGB values, e.g., (255, 87, 51) or (255, 87, 51, 255) for RGBA.
    """
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 6:
        return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))  # RGB
    elif len(hex_color) == 8:
        return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4, 6))  # RGBA


def _rgb_to_hex(rgb):
    """
    Helper function.

        Converts an RGB or RGBA tuple to a hex color string using f-strings.

        Parameters:
        - rgb (tuple): A tuple of integers representing the RGB values, e.g., (255, 87, 51) or (255, 87, 51, 255) for RGBA.

        Returns:
        - hex_color (str): A hex color string, e.g., '#FF5733' or '#FF5733FF' for RGBA.
    """
    rgb = tuple(int(30 * round(c / 30)) for c in rgb)
    if len(rgb) == 3:
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
    elif len(rgb) == 4:
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}{rgb[3]:02x}"


def _interpolate_colors(tree, points, colors, target_points, nn=100):
    """
    Helper function.

    Build a color field and sample colors for target points
    """
    grid_x = np.linspace(points[:, 0].min(), points[:, 0].max(), 100)
    grid_y = np.linspace(points[:, 1].min(), points[:, 1].max(), 100)
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y, indexing="ij")
    grid_points = np.vstack([grid_xx.ravel(), grid_yy.ravel()]).T

    print("constructing color grid")

    colors = np.array([_hex_to_rgb(color) for color in colors])
    distances, indices = tree.kneighbors(grid_points, n_neighbors=nn)
    weights = 1 / np.maximum(distances, 1e-6) ** 1  # Avoid division by zero
    weights /= weights.sum(axis=1)[:, None]  #
    grid_colors = np.einsum("ij,ijk->ik", weights, colors[indices])
    grid_colors = grid_colors.reshape(len(grid_x), len(grid_y), 3)

    # Interpolation function
    color_interpolator = RegularGridInterpolator((grid_x, grid_y), grid_colors)
    print("querying color grid")
    segment_colors = color_interpolator(target_points)
    return [_rgb_to_hex(segment_colors[i]) for i in range(len(segment_colors))]


def bundle_edges(data_map_coords, color_list, n_neighbors=10, color_map_nn=100):
    """
    Use hammer edge bundling on nearest neighbors

    Parameters:
    - data_map_coords (np.ndarray): an ND array of shape (n_samples,2)
        The 2D coordinates of the data points
    - color_list (list): list of strings of length n_samples
        A list of hex-string colors, one per sample, for colouring the data points
    - n_neighbors (int): number of neighbors to use build the KNN graph for edge bundling
    - color_map_nn (int): number of neighbors to consider when coloring line segments

    Returns:
    - segments (np.ndarray): an ND array of shape (N,4) where each row is a line segment in format [x0,y0,x1,y1]
    - segment_colors (list): a list of hex-colors of length N. Each row corresponds to the color of the line segment at the same index.
    """

    bundle_points = data_map_coords
    nbrs = NearestNeighbors(
        n_neighbors=max(n_neighbors, color_map_nn), algorithm="ball_tree"
    ).fit(bundle_points)
    _, indices = nbrs.kneighbors(bundle_points, n_neighbors=n_neighbors)

    edges = []
    for i, neighbors in enumerate(indices):
        for neighbor in neighbors:
            if i != neighbor:  # Do not include self
                edges.append([i, neighbor])

    bundle_points = pd.DataFrame({"x": data_map_coords.T[0], "y": data_map_coords.T[1]})
    edges = pd.DataFrame(
        {
            "source": [edge[0] for edge in edges],
            "target": [edge[1] for edge in edges],
        }
    )
    # Perform edge bundling
    bundled = hammer_bundle(bundle_points, edges, use_dask=False)

    print("parsing lines")
    bundled["is_valid"] = bundled.isnull().all(axis=1).cumsum()
    bundled = bundled.dropna()
    x = bundled["x"][:-1].values
    y = bundled["y"][:-1].values
    x1 = bundled["x"][1:].values
    y1 = bundled["y"][1:].values
    is_valid = bundled["is_valid"][1:].values == bundled["is_valid"][:-1].values
    segments = np.array([x, y, x1, y1]).T[is_valid]

    midpoints = np.array(
        [(segments[:, 0] + segments[:, 2]) / 2, (segments[:, 1] + segments[:, 3]) / 2]
    ).T

    # Compute line segment color based on midpoint
    print("interpolating colors")
    segment_colors = _interpolate_colors(nbrs, data_map_coords, color_list, midpoints)

    return segments, segment_colors
