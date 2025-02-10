import logging
logger = logging.getLogger(__name__)

def get_xy_lims(avgPolygons):
    min_val, max_val = float('inf'), float('-inf')  # Initialize with extreme values
    
    # Iterate through all polygons in avgPolygons
    for _, polygon in avgPolygons.items():
        x_values = polygon[0, :]  # Extract x-coordinates
        y_values = polygon[1, :]  # Extract y-coordinates
        
        # Find the min and max of both x and y
        min_val = min(min_val, np.min(x_values), np.min(y_values))
        max_val = max(max_val, np.max(x_values), np.max(y_values))
    
    return min_val, max_val
def movie_grid_time_evolution_with_data(interpolated_nodes, triangles, data_dict, feature_key, save_path=None, feature_name=None, show_axes=True,scale_unit='',num_levels=20,vlim=None):
    if feature_name is None:
        feature_name = feature_key

    # Get unique times sorted
    times = sorted(list(set([key[0] for key in interpolated_nodes.keys()])))

    # Compute the 10th and 90th percentile limits for the given feature_key across all times and categories
    feature_values = []
    for (time, category), data in data_dict.items():
        if feature_key in data:
            feature_values.extend(data[feature_key])

    if vlim is None:
        vmin = np.percentile(feature_values, 10)
        vmax = np.percentile(feature_values, 90)
    else:
        vmin,vmax=vlim

    # Create the figure and axes
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Set font sizes for axes labels and titles
    plt.rcParams.update({'font.size': 14})

    # Set axis labels and titles
    if show_axes:
        axs[0].set_title("Development Mesh", fontsize=18)
        axs[0].set_xlabel("X Coordinates", fontsize=14)
        axs[0].set_ylabel("Y Coordinates", fontsize=14)
        axs[1].set_title("Regeneration Mesh", fontsize=18)
        axs[1].set_xlabel("X Coordinates", fontsize=14)
        axs[1].set_ylabel("Y Coordinates", fontsize=14)

    # Get the mesh limits for both conditions
    min_val, max_val = get_mesh_lims(
        {key: nodes for key, nodes in interpolated_nodes.items() if key[1] == 'Development'},
        {key: nodes for key, nodes in interpolated_nodes.items() if key[1] == 'Regeneration'}
    )
    min_val -= 10
    max_val += 10
    axs[0].set_xlim(min_val, max_val)
    axs[0].set_ylim(min_val, max_val)
    axs[1].set_xlim(min_val, max_val)
    axs[1].set_ylim(min_val, max_val)

    # Create a color bar for the data feature with adjusted limits
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = plt.colorbar(sm, ax=axs, orientation='vertical', fraction=0.05)
    cbar.set_label(f'{feature_name}', fontsize=16)

    def add_scale_bar(ax):
        def round_to_one_significant_digit(value):
            if value == 0:
                return 0
            return round(value, -int(math.floor(math.log10(abs(value)))))
        """Add a scale bar to the plot if axes are switched off."""
        # Choose size of the scale bar (adjust this based on your mesh size)
        bar_size = round_to_one_significant_digit((max_val - min_val) * 0.1) # Adjust bar size based on mesh size
        fontprops = fm.FontProperties(size=12)
        scalebar = AnchoredSizeBar(ax.transData,
                                   bar_size,  # Length of the scale bar
                                   f'{bar_size:.1f} {scale_unit}',  # Label
                                   'lower right',  # Position of the scale bar
                                   pad=0.1,
                                   color='black',
                                   frameon=False,
                                   size_vertical=1,
                                   fontproperties=fontprops)

        ax.add_artist(scalebar)

    def update(frame):
        time = times[frame]
        """Update the plots for each frame of the animation"""
        # Clear the axes before each update
        axs[0].clear()
        axs[1].clear()

        
        axs[0].set_title(f"t = {time} hpf", fontsize=18)
        axs[1].set_title(f"t = {time} hpf", fontsize=18)

        # Set axis limits
        axs[0].set_xlim(min_val, max_val)
        axs[0].set_ylim(min_val, max_val)
        axs[1].set_xlim(min_val, max_val)
        axs[1].set_ylim(min_val, max_val)

        # Get the current node positions and feature data for development and regeneration
        dev_nodes = interpolated_nodes[(time, 'Development')]
        reg_nodes = interpolated_nodes[(time, 'Regeneration')]
        dev_feature_data = data_dict[(time, 'Development')][feature_key]
        reg_feature_data = data_dict[(time, 'Regeneration')][feature_key]

        # Create triangulations for development and regeneration
        dev_triang = tri.Triangulation(dev_nodes[:, 0], dev_nodes[:, 1], triangles['Development'])
        reg_triang = tri.Triangulation(reg_nodes[:, 0], reg_nodes[:, 1], triangles['Regeneration'])

        # Plot the development mesh with filled contours
        axs[0].tricontourf(dev_triang, dev_feature_data, cmap='viridis', vmin=vmin, vmax=vmax,levels=num_levels)
        #axs[0].triplot(dev_triang, 'ko-')  # Mesh outline in black

        # Plot the regeneration mesh with filled contours
        axs[1].tricontourf(reg_triang, reg_feature_data, cmap='viridis', vmin=vmin, vmax=vmax,levels=num_levels)
        #axs[1].triplot(reg_triang, 'ko-')  # Mesh outline in black

        if not show_axes:
            axs[0].axis('off')
            axs[1].axis('off')

            # Add a scale bar
            add_scale_bar(axs[0])
            add_scale_bar(axs[1])

        return axs

    # Create the animation
    anim = FuncAnimation(fig, update, frames=len(times), blit=False, interval=200)

    if save_path:
        # Save the animation as a video file
        anim.save(save_path, writer='ffmpeg', fps=8,dpi=200)
    else:
        # Display the animation
        plt.show()


def show_mesh_series():
    interpolated_nodes, triangles,_=load_interpolated_data(AvgShape_path)
    movie_grid_time_evolution(interpolated_nodes, triangles)

def movie_time_evolution(avgPolygons, save_path=None):
    # Get unique times and sort them
    times = sorted(list(set([key[0] for key in avgPolygons.keys()])))
    
    # Normalize the time values for color mapping
    min_time, max_time = min(times), max(times)
    norm = plt.Normalize(min_time, max_time)
    cmap = cm.viridis  # Color map based on time
    
    fig, ax = plt.subplots(figsize=(10, 10))

    # Initialize plot elements
    line_dev, = ax.plot([], [], 'g-', lw=2, label="Development")
    line_reg, = ax.plot([], [], 'b--', lw=2, label="Regeneration")
    
    # Set axis labels
    ax.set_xlabel("X Coordinates")
    ax.set_ylabel("Y Coordinates")
    ax.set_title("Time Evolution of Polygons")
    
    # Set fixed axis limits (adjust based on the polygon sizes)
    min_val, max_val=get_xy_lims(avgPolygons)
    ax.set_xlim(min_val-10, max_val+10)
    ax.set_ylim(min_val-10, max_val+10)
    
    # Create a color bar for the time evolution
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(times)
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Time in hpf')

    def init():
        """Initial setup of the plot."""
        return line_dev, line_reg

    def update(frame):
        """Update the plot for each frame of the animation."""
        time = times[frame]
        
        # Get the polygons corresponding to the current time step
        polygons_to_plot = [(condition, polygon) for (t, condition), polygon in avgPolygons.items() if t == time]

        print(f"Time: {time}, Polygons to Plot: {len(polygons_to_plot)}")  # Debugging line

        # Loop over the polygons for this time step and update the plot
        for condition, polygon in polygons_to_plot:
            x, y = polygon[0, :], polygon[1, :]  # Extract x and y coordinates
            color = cmap(norm(time))

            print(f"Condition: {condition}, X: {x[:5]}, Y: {y[:5]}")  # Debugging line to print first few points
            
            if condition == 'Development':
                line_dev.set_data(x, y)  # Update development polygon data
                line_dev.set_color(color)  # Color it based on the current time
                
            elif condition == 'Regeneration':
                line_reg.set_data(x, y)  # Update regeneration polygon data
                line_reg.set_color(color)  # Color it based on the current time

        # Update the title to reflect the current time
        ax.set_title(f"Time: {time} hpf")

        return line_dev, line_reg

    # Create the animation without blit for proper frame updating
    anim = FuncAnimation(fig, update, frames=len(times), init_func=init, blit=False, interval=100)

    if save_path:
        # Save the animation as a video file
        anim.save(save_path, writer='ffmpeg', fps=2)
    else:
        # Display the animation
        plt.show()

def plot_mesh_movement(nodes, triangles, boundary_indices, title):
    """
    Helper function to plot mesh movement for debugging purposes.
    """
    plt.figure(figsize=(6, 6))
    plt.triplot(nodes[:, 0], nodes[:, 1], triangles, color='blue')
    plt.scatter(nodes[boundary_indices, 0], nodes[boundary_indices, 1], color='red', label='Boundary Points')
    plt.title(title)
    plt.legend()
    plt.show()


def movie_grid_time_evolution(interpolated_nodes, triangles, save_path=None):
    # Get unique times sorted
    times = sorted(list(set([key[0] for key in interpolated_nodes.keys()])))
   
    # Create the figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Set axis labels and titles
    ax1.set_title("Development Mesh")
    ax1.set_xlabel("X Coordinates")
    ax1.set_ylabel("Y Coordinates")
    ax2.set_title("Regeneration Mesh")
    ax2.set_xlabel("X Coordinates")
    ax2.set_ylabel("Y Coordinates")

    # Get the mesh limits for both conditions
    min_val, max_val = get_mesh_lims(
        {key: nodes for key, nodes in interpolated_nodes.items() if key[1] == 'Development'},
        {key: nodes for key, nodes in interpolated_nodes.items() if key[1] == 'Regeneration'}
    )
    min_val -= 10
    max_val += 10
    ax1.set_xlim(min_val, max_val)
    ax1.set_ylim(min_val, max_val)
    ax2.set_xlim(min_val, max_val)
    ax2.set_ylim(min_val, max_val)

    def update(frame):
        time = times[frame]
        """Update the plots for each frame of the animation"""
        # Clear the axes before each update
        ax1.clear()
        ax2.clear()

        # Set titles again after clearing
        ax1.set_title(f"Development Mesh at Time {time}")
        ax2.set_title(f"Regeneration Mesh at Time {time}")
        
        ax1.set_xlabel("X Coordinates")
        ax1.set_ylabel("Y Coordinates")
        ax2.set_xlabel("X Coordinates")
        ax2.set_ylabel("Y Coordinates")

        # Set axis limits
        ax1.set_xlim(min_val, max_val)
        ax1.set_ylim(min_val, max_val)
        ax2.set_xlim(min_val, max_val)
        ax2.set_ylim(min_val, max_val)

        # Get the current node positions for development and regeneration
        dev_nodes = interpolated_nodes[(time, 'Development')]
        reg_nodes = interpolated_nodes[(time, 'Regeneration')]

        # Plot the development mesh
        ax1.triplot(dev_nodes[:, 0], dev_nodes[:, 1], triangles['Development'], color='blue')

        # Plot the regeneration mesh
        ax2.triplot(reg_nodes[:, 0], reg_nodes[:, 1], triangles['Regeneration'], color='blue')

        return ax1, ax2

    # Create the animation
    anim = FuncAnimation(fig, update, frames=len(times), blit=False, interval=200)

    if save_path:
        # Save the animation as a video file
        anim.save(save_path, writer='ffmpeg', fps=2)
    else:
        # Display the animation
        plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.animation as animation
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
    bar_size = round((max_x - min_x) * 0.1, 1)  # 10% of mesh width
    fontprops = fm.FontProperties(size=12)
    
    scalebar = AnchoredSizeBar(ax.transData,
                               bar_size,
                               f"{bar_size} {scale_unit}",
                               loc="lower right",
                               pad=0.1,
                               color="black",
                               frameon=False,
                               size_vertical=1,
                               fontproperties=fontprops)
    ax.add_artist(scalebar)

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
    ax.set_title("Temporal Evolution of Reference Geometries", fontsize=14)

    # Initialize plot
    nodes=interpolated_nodes[times[0]]
    x = nodes[:, 0]
    y = nodes[:, 1]
    triang = tri.Triangulation(x, y, triangles)
    mesh_plot = ax.triplot(triang, color="blue", alpha=0.6)[0]

    def update(frame):
        """Update function for animation."""
        time = times[frame]
        nodes = interpolated_nodes[time]  # Use int key directly

        # Update triangulation
        triang.set_mask(np.zeros(triangles.shape[0], dtype=bool))  # Reset mask
        triang.x = nodes[:, 0]
        triang.y = nodes[:, 1]
        
        # Update mesh plot
        mesh_plot.set_data(triang.x, triang.y)

        ax.set_title(f"Time: {time} hpf", fontsize=14)
        return mesh_plot,

    anim = animation.FuncAnimation(fig, update, frames=len(times), blit=False, interval=200)

    if save_path:
        anim.save(save_path, writer="ffmpeg", fps=5, dpi=200)
    else:
        plt.show()

import os
import pickle

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
