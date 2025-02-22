import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm


def add_scale_bar(ax, nodes, scale_unit="µm"):
    """
    Adds a scale bar to the plot based on the node size.
    
    Args:
        ax: Matplotlib axis.
        nodes: Numpy array of shape (N, 2) containing mesh nodes.
        scale_unit (str): Unit for the scale bar.
    """
    min_x, max_x = nodes[:, 0].min(), nodes[:, 0].max()
    bar_size = round(((max_x - min_x) * 0.1)/5, 0)*5  # 10% of mesh width
    fontprops = fm.FontProperties(size=12)
    
    scalebar = AnchoredSizeBar(ax.transData,
                               bar_size,
                               f"{bar_size} {scale_unit}",
                               loc="lower right",
                               bbox_to_anchor=(-0.05, -0.05),  # Moves it slightly left and down
                               bbox_transform=ax.transAxes,  # Ensures movement is relative to the axis
                               pad=0.1,
                               color="black",
                               frameon=False,
                               size_vertical=1,
                               fontproperties=fontprops)
    ax.add_artist(scalebar)

def plot_all_reference_meshes(base_dir, scale_unit="µm"):
    """
    Plots all reference meshes stored in subfolders of base_dir.

    Args:
        base_dir (str): Path to the directory containing reference geometry folders.
        scale_unit (str): Unit for the scale bar.
    """
    subfolders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    
    num_meshes = len(subfolders)
    cols = min(4, num_meshes)  # Max 4 columns
    rows = (num_meshes // cols) + (num_meshes % cols > 0)  # Determine row count

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    axes = axes.flatten()

    for idx, folder in enumerate(subfolders):
        npz_file = os.path.join(base_dir, folder, f"{folder}_ref.npz")
        
        if not os.path.exists(npz_file):
            print(f"Skipping {folder}, no reference mesh found.")
            continue
        
        # Load mesh data
        data = np.load(npz_file)
        nodes, triangles = data["nodes"], data["triangles"]

        # Plot mesh
        ax = axes[idx]
        triang = tri.Triangulation(nodes[:, 0], nodes[:, 1], triangles)
        ax.triplot(triang, color="blue", alpha=0.6)
        ax.scatter(nodes[:, 0], nodes[:, 1], color="red", s=5)
        
        # Title from folder name (category keys)
        ax.set_title(folder.replace("_", " "), fontsize=12)
        ax.set_aspect("equal")
        ax.axis("off")

        # Add scale bar
        add_scale_bar(ax, nodes, scale_unit)

    # Hide unused subplots if any
    for i in range(num_meshes, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

def plot_all_reference_data(base_dir, data_to_value_function, scale_unit="µm", vmin=None, vmax=None, separate_windows=False):
    """
    Plots all reference meshes and overlays selected histogram data with uniform color scaling.
    
    Args:
        base_dir (str): Directory containing reference geometries.
        data_to_value_function: Function to extract values from histogram data.
        scale_unit (str): Unit for the scale bar.
        vmin (float, optional): Minimum value for color scaling.
        vmax (float, optional): Maximum value for color scaling.
        separate_windows (bool, optional): If True, each mesh is plotted in a separate window.
    """
    subfolders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    num_meshes = len(subfolders)

    if num_meshes == 0:
        print("No valid data found.")
        return

    all_plot_values = []  
    hist_data_map = {}  

    # First Pass: Collect histogram data and compute global min/max
    for folder in subfolders:
        npz_hist_file = os.path.join(base_dir, folder, "histogram_data.npz")
        
        if os.path.exists(npz_hist_file):
            hist_data = np.load(npz_hist_file, allow_pickle=True)
            hist_data_map[folder] = hist_data
            all_plot_values.append(data_to_value_function(hist_data))
            
    if not all_plot_values:
        print(f"No valid data found.")
        return

    all_plot_values = np.concatenate(all_plot_values)
    if vmin is None:
        vmin = np.min(all_plot_values)
    if vmax is None:
        vmax = np.max(all_plot_values)

    # Single figure setup if not using separate windows
    if not separate_windows:
        cols = min(4, num_meshes)
        rows = (num_meshes // cols) + (num_meshes % cols > 0)
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 7 * rows), squeeze=False)
        axes = axes.flatten()
    else:
        axes = [None] * num_meshes  

    # Second Pass: Plot the meshes
    for idx, folder in enumerate(subfolders):
        npz_mesh_file = os.path.join(base_dir, folder, f"{folder}_ref.npz")

        if not os.path.exists(npz_mesh_file):
            print(f"Skipping {folder}, no reference mesh found.")
            continue

        mesh_data = np.load(npz_mesh_file)
        nodes, triangles = mesh_data["nodes"], mesh_data["triangles"]

        hist_data = hist_data_map.get(folder, None)
        plot_values = data_to_value_function(hist_data) if hist_data else None
       
        if separate_windows:
            fig, ax = plt.subplots(figsize=(7, 7))
        else:
            ax = axes[idx]

        triang = tri.Triangulation(nodes[:, 0], nodes[:, 1], triangles)
        if plot_values is not None:
            ax.tricontourf(triang, plot_values, cmap="viridis", levels=100, vmin=vmin, vmax=vmax)
        else:
            ax.triplot(triang, color="blue", alpha=0.6)

        ax.set_title(folder.replace("_", " "), fontsize=12)
        ax.set_aspect("equal")
        ax.axis("off")

        # Add scale bar
        add_scale_bar(ax, nodes, scale_unit)

        if separate_windows:
            plt.subplots_adjust(left=0.15)
            plt.show()

    # Colorbar and layout adjustments for single-figure mode
    if not separate_windows:
        cbar_ax = fig.add_axes([0.15, 0.04, 0.7, 0.02])
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=vmin, vmax=vmax))
        cbar = plt.colorbar(sm, cax=cbar_ax, orientation="horizontal")
        cbar.set_label('feature_key', fontsize=14)

        for i in range(num_meshes, len(axes)):
            fig.delaxes(axes[i])

        plt.subplots_adjust(hspace=0.2, wspace=0.2, bottom=0.1, top=0.97)
        plt.show()
