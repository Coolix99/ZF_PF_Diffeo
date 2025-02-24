import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import os
import logging
logger = logging.getLogger(__name__)


def movie_temporal_evolution(interpolated_nodes, triangles, save_path=None, scale_unit="µm"):
    """
    Creates a movie of the temporal evolution of interpolated reference geometries.

    Args:
        interpolated_nodes (dict): Dictionary containing nodes at each time step.
        triangles (ndarray): Triangle connectivity for the mesh.
        save_path (str, optional): Path to save the video. If None, displays interactively.
        scale_unit (str): Unit for the scale bar.
    """
    # Get unique sorted time steps (keys are already integers)
    times = sorted(interpolated_nodes.keys())

    # Determine plot limits
    all_nodes = np.vstack(list(interpolated_nodes.values()))
    min_x, max_x = all_nodes[:, 0].min(), all_nodes[:, 0].max()
    min_y, max_y = all_nodes[:, 1].min(), all_nodes[:, 1].max()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(min_x - 10, max_x + 10)
    ax.set_ylim(min_y - 10, max_y + 10)
    ax.set_title(f"Time: {times[0]} hpf", fontsize=14)

    # Initialize plot
    nodes=interpolated_nodes[times[0]]
    x = nodes[:, 0]
    y = nodes[:, 1]
    triang = tri.Triangulation(x, y, triangles)
    mesh_plot = ax.triplot(triang, color="blue", alpha=0.6)[0]

    def update(frame):
        """Update function for animation."""
        time = times[frame]
        
        nodes = interpolated_nodes[time]
        triang = tri.Triangulation(nodes[:, 0], nodes[:, 1], triangles)
        ax.clear()
        ax.set_xlim(min_x - 10, max_x + 10)
        ax.set_ylim(min_y - 10, max_y + 10)
        ax.triplot(triang, color="blue", alpha=0.6)

        ax.set_title(f"Time: {time} hpf", fontsize=14)
        return mesh_plot,


    anim = animation.FuncAnimation(fig, update, frames=len(times), blit=False, interval=200)

    if save_path:
        anim.save(save_path, writer="ffmpeg", fps=5, dpi=200)
    else:
        plt.show()

def load_interpolated_data(directory):
    """
    Loads interpolated nodes and last mesh (triangles) from a given directory.

    Args:
        directory (str): Path to the directory containing interpolated data.

    Returns:
        interpolated_nodes (dict): Dictionary of interpolated node positions for each time step.
        last_mesh (tuple): (nodes, triangles) from the last time step.
    """
    nodes_file = os.path.join(directory, "interpolated_nodes.npz")
    mesh_file = os.path.join(directory, "last_mesh.npz")

    if not os.path.exists(nodes_file) or not os.path.exists(mesh_file):
        raise FileNotFoundError(f"Missing interpolated data files in {directory}")

    # Load interpolated nodes
    interpolated_data = np.load(nodes_file, allow_pickle=True)
    interpolated_nodes = {int(k): v for k, v in interpolated_data.items()}  # Convert keys back to int

    # Load last mesh (nodes & triangles)
    mesh_data = np.load(mesh_file)
    last_mesh = (mesh_data["nodes"], mesh_data["triangles"])

    return interpolated_nodes, last_mesh

def show_temporal_mesh_evolution(directory):
    """
    Loads interpolated reference geometries from a directory and creates an animation.

    Args:
        directory (str): Path to the directory containing temporal interpolation results.
    """
    logger.info(f"Loading interpolated temporal data from: {directory}")
    
    try:
        interpolated_nodes, (_,triangles) = load_interpolated_data(directory)
        movie_temporal_evolution(interpolated_nodes, triangles)
    except FileNotFoundError as e:
        logger.error(str(e))

def load_interpolated_hist_data(directory):
    """
    Loads interpolated nodes, last mesh (triangles), and interpolated histograms.

    Args:
        directory (str): Path to the directory containing interpolated data.

    Returns:
        interpolated_nodes (dict): Dictionary of interpolated node positions for each time step.
        triangles (ndarray): Triangle connectivity for the mesh.
        interpolated_histograms (dict): Dictionary of interpolated histogram values per feature key.
    """
    nodes_file = os.path.join(directory, "interpolated_nodes.npz")
    mesh_file = os.path.join(directory, "last_mesh.npz")
    hist_file = os.path.join(directory, "interpolated_histograms.npz")

    if not os.path.exists(nodes_file) or not os.path.exists(mesh_file) or not os.path.exists(hist_file):
        raise FileNotFoundError(f"Missing required data files in {directory}")

    # Load interpolated nodes
    interpolated_data = np.load(nodes_file, allow_pickle=True)
    interpolated_nodes = {int(k): v for k, v in interpolated_data.items()}  # Convert keys back to int

    # Load last mesh (nodes & triangles)
    mesh_data = np.load(mesh_file)
    triangles = mesh_data["triangles"]

    # Load interpolated histogram data
    hist_data = np.load(hist_file, allow_pickle=True)
    interpolated_histograms = {k: v.item() if isinstance(v, np.ndarray) and v.dtype == object else v 
                               for k, v in hist_data.items()}  # Convert keys properly

    return interpolated_nodes, triangles, interpolated_histograms

def movie_temporal_hist_evolution(directory, save_path=None, scale_unit="µm"):
    """
    Creates a movie of the temporal evolution of interpolated reference geometries with histogram overlays.

    Args:
        directory (str): Path to the directory containing temporal interpolation results.
        save_path (str, optional): Path to save the video. If None, displays interactively.
        scale_unit (str): Unit for the scale bar.
    """
    logger.info(f"Loading interpolated data from: {directory}")

    try:
        interpolated_nodes, triangles, interpolated_histograms = load_interpolated_hist_data(directory)
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    times = sorted(interpolated_nodes.keys())

    # Determine plot limits
    all_nodes = np.vstack(list(interpolated_nodes.values()))
    min_x, max_x = all_nodes[:, 0].min(), all_nodes[:, 0].max()
    min_y, max_y = all_nodes[:, 1].min(), all_nodes[:, 1].max()

    # Extract all available feature keys
    feature_keys = list(interpolated_histograms.keys())

    fig, axes = plt.subplots(1, len(feature_keys), figsize=(8 * len(feature_keys), 8))
    if len(feature_keys) == 1:
        axes = [axes]  # Ensure axes is always iterable

    # Initialize plots
    mesh_plots = []
    contour_plots = []

    for ax, feature_key in zip(axes, feature_keys):
        ax.set_xlim(min_x - 10, max_x + 10)
        ax.set_ylim(min_y - 10, max_y + 10)
        ax.set_title(f"{feature_key} - Time: {times[0]} hpf", fontsize=14)

        nodes = interpolated_nodes[times[0]]
        values = interpolated_histograms[feature_key][str(times[0])]

        triangulation = tri.Triangulation(nodes[:, 0], nodes[:, 1], triangles)
        contour = ax.tricontourf(triangulation, values, cmap="viridis", levels=100)

        mesh_plot = ax.triplot(triangulation, color="black", alpha=0.5)

        mesh_plots.append(mesh_plot)
        contour_plots.append(contour)

    def update(frame):
        """Update function for animation."""
        time = times[frame]

        for ax, feature_key, contour, mesh_plot in zip(axes, feature_keys, contour_plots, mesh_plots):
            ax.clear()
            ax.set_xlim(min_x - 10, max_x + 10)
            ax.set_ylim(min_y - 10, max_y + 10)

            nodes = interpolated_nodes[time]
            values = interpolated_histograms[feature_key][str(time)]

            triangulation = tri.Triangulation(nodes[:, 0], nodes[:, 1], triangles)
            contour = ax.tricontourf(triangulation, values, cmap="viridis", levels=100)

            ax.triplot(triangulation, color="black", alpha=0.5)
            ax.set_title(f"{feature_key} - Time: {time} hpf", fontsize=14)

        return contour_plots + mesh_plots

    anim = animation.FuncAnimation(fig, update, frames=len(times), blit=False, interval=200)

    if save_path:
        anim.save(save_path, writer="ffmpeg", fps=5, dpi=200)
    else:
        plt.show()

