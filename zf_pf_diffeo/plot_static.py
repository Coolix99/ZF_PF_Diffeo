

def show_data_movie():
    data_dict=load_data_dict(AvgShape_path)
    interpolated_nodes, triangles,boundaries=load_interpolated_data(AvgShape_path)
    movie_grid_time_evolution_with_data(interpolated_nodes, triangles,data_dict, 'thickness_avg',feature_name=r'Thickness $[\mathrm{\mu m}]$',show_axes=False,scale_unit=r"$\mathrm{\mu m}$",vlim=(0,60),save_path='/home/max/Videos/thickness.mp4')#
    movie_grid_time_evolution_with_data(interpolated_nodes, triangles,data_dict, 'gauss_curvature_avg',feature_name=r'Gauss curvature $[1/\mu m^2]$',show_axes=False,scale_unit=r"$\mathrm{\mu m}$",vlim=(-1.5e-4,1.5e-4),save_path='/home/max/Videos/gauss_curvature.mp4')

def plot_grid_time_series_with_data(interpolated_nodes, triangles, data_dict, feature_key, times_to_plot, feature_name=None, show_axes=True, scale_unit='', num_levels=20, vlim=None, save_path=None):
    if feature_name is None:
        feature_name = feature_key

    # Compute the 10th and 90th percentile limits for the given feature_key across all times and categories
    feature_values = []
    for (time, category), data in data_dict.items():
        if feature_key in data:
            feature_values.extend(data[feature_key])

    if vlim is None:
        vmin = np.percentile(feature_values, 10)
        vmax = np.percentile(feature_values, 90)
    else:
        vmin, vmax = vlim

    # Get the mesh limits for both conditions
    min_val, max_val = get_mesh_lims(
        {key: nodes for key, nodes in interpolated_nodes.items() if key[1] == 'Development'},
        {key: nodes for key, nodes in interpolated_nodes.items() if key[1] == 'Regeneration'}
    )
    min_val -= 10
    max_val += 10

    num_times = len(times_to_plot)

    # Create the figure and axes
    fig, axs = plt.subplots(2, num_times, figsize=(4 * num_times, 8), squeeze=False)

    # Function to add scale bar
    def add_scale_bar(ax):
        def round_to_one_significant_digit(value):
            if value == 0:
                return 0
            return round(value, -int(math.floor(math.log10(abs(value)))))

        bar_size = round_to_one_significant_digit((max_val - min_val) * 0.1)
        fontprops = fm.FontProperties(size=12)
        scalebar = AnchoredSizeBar(ax.transData,
                                   bar_size,
                                   f'{bar_size:.1f} {scale_unit}',
                                   'lower right',
                                   pad=0.1,
                                   color='black',
                                   frameon=False,
                                   size_vertical=1,
                                   fontproperties=fontprops)
        ax.add_artist(scalebar)

    # Loop over times
    for idx, time in enumerate(times_to_plot):
        # Plot Development
        ax_dev = axs[0, idx]
        dev_nodes = interpolated_nodes[(time, 'Development')]
        dev_feature_data = data_dict[(time, 'Development')][feature_key]
        dev_triang = tri.Triangulation(dev_nodes[:, 0], dev_nodes[:, 1], triangles['Development'])
        ax_dev.tricontourf(dev_triang, dev_feature_data, cmap='viridis', vmin=vmin, vmax=vmax, levels=num_levels)
        ax_dev.set_xlim(min_val, max_val)
        ax_dev.set_ylim(min_val, max_val)
        ax_dev.set_title(f'Development t = {time} hpf')
        if show_axes:
            pass    
        else:
            ax_dev.axis('off')
            add_scale_bar(ax_dev)

        # Plot Regeneration
        ax_reg = axs[1, idx]
        reg_nodes = interpolated_nodes[(time, 'Regeneration')]
        reg_feature_data = data_dict[(time, 'Regeneration')][feature_key]
        reg_triang = tri.Triangulation(reg_nodes[:, 0], reg_nodes[:, 1], triangles['Regeneration'])
        ax_reg.tricontourf(reg_triang, reg_feature_data, cmap='viridis', vmin=vmin, vmax=vmax, levels=num_levels)
        ax_reg.set_xlim(min_val, max_val)
        ax_reg.set_ylim(min_val, max_val)
        ax_reg.set_title(f'Regeneration t = {time} hpf')
        if show_axes:
            pass
        else:
            ax_reg.axis('off')
            add_scale_bar(ax_reg)

    # Adjust layout to make room for the color bar below the plots
    plt.tight_layout(rect=[0, 0.1, 1, 1])  # Leave more space at the bottom for the color bar

    # Create a horizontal color bar below the plots
    cbar_ax = fig.add_axes([0.2, 0.1, 0.6, 0.02])  # Adjust the position and size as needed
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(f'{feature_name}', fontsize=16)

    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()

def show_data_series():
    data_dict = load_data_dict(AvgShape_path)
    interpolated_nodes, triangles, boundaries = load_interpolated_data(AvgShape_path)
    times_to_plot = [84,96,120,126,132,144]  # User-specified times
    plot_grid_time_series_with_data(
        interpolated_nodes,
        triangles,
        data_dict,
        feature_key='thickness_avg',
        times_to_plot=times_to_plot,
        feature_name=r'Thickness $[\mathrm{\mu m}]$',
        show_axes=False,
        scale_unit=r"$\mathrm{\mu m}$",
        vlim=(0, 60),
        save_path='/home/max/Pictures/thickness.png'
    )
    plot_grid_time_series_with_data(
        interpolated_nodes,
        triangles,
        data_dict,
        feature_key='gauss_curvature_avg',
        times_to_plot=times_to_plot,
        feature_name=r'Gauss curvature $\mathrm{[1/\mu m^2]}$',
        show_axes=False,
        scale_unit=r"$\mathrm{\mu m}$",
        vlim=(-1.5e-4,1.0e-4),
        save_path='/home/max/Pictures/gauss_curvature.png'
    )

def get_n():
    FlatFin_folder_list = os.listdir(FlatFin_path)
    FlatFin_folder_list = [item for item in FlatFin_folder_list if os.path.isdir(os.path.join(FlatFin_path, item))]
    
    data_list = []
    for FlatFin_folder in FlatFin_folder_list:
        FlatFin_dir_path = os.path.join(FlatFin_path, FlatFin_folder)
        MetaData = get_JSON(FlatFin_dir_path)
        
        if 'Thickness_MetaData' not in MetaData:
            continue
        
        MetaData = MetaData['Thickness_MetaData']
        
        data_list.append({
            'folder_name': FlatFin_folder,
            'file_name': MetaData['Surface file'],
            'condition': MetaData['condition'],
            'time in hpf': MetaData['time in hpf'],
            'genotype': MetaData['genotype'],
            'experimentalist': MetaData['experimentalist']
        })
        
    # Store the metadata in a pandas DataFrame
    df = pd.DataFrame(data_list)

    # Count the total number of measurements for 'Regeneration' and 'Development'
    regen_count = df[df['condition'] == 'Regeneration'].shape[0]
    dev_count = df[df['condition'] == 'Development'].shape[0]
    
    print(f"Number of 'Regeneration' measurements: {regen_count}")
    print(f"Number of 'Development' measurements: {dev_count}")
    
    # Group by 'condition' and 'time in hpf' and count the number of occurrences in each group
    grouped_counts = df.groupby(['condition', 'time in hpf']).size().reset_index(name='count')
    
    print("\nGrouped counts by condition and time in hpf:")
    print(grouped_counts)

def plot_time_evolution(avgPolygons):
    # Initialize the plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Normalize the time values for color mapping
    times = np.array([key[0] for key in avgPolygons.keys()])
    min_time, max_time = times.min(), times.max()
    norm = plt.Normalize(min_time, max_time)
    cmap = cm.viridis  # Colormap for time-based coloring

    for (time, condition), polygon in avgPolygons.items():
        # Extract x and y coordinates
        x, y = polygon[0, :], polygon[1, :]
        
        # Choose color based on time and line style based on condition
        color = cmap(norm(time))
        if condition == 'Development':
            linestyle = '-'  # Solid line for Development
        elif condition == 'Regeneration':
            linestyle = '--'  # Dashed line for Regeneration
        
        # Plot the polygon
        ax.plot(x, y, linestyle=linestyle, color=color, label=f'Time: {time}, {condition}')

    # Create a colorbar to represent the time evolution
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(times)
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Time in hpf')
    
    # Set plot labels and title
    ax.set_title("Time Evolution of Polygons")
    ax.set_xlabel("X Coordinates")
    ax.set_ylabel("Y Coordinates")
    
    # Show plot
    plt.show()

# Plotting function to visualize the mesh
def plot_mesh(nodes, triangles, ax, title):
    ax.triplot(nodes[:, 0], nodes[:, 1], triangles, color='blue')
    ax.scatter(nodes[:, 0], nodes[:, 1], color='red', s=10)
    ax.set_title(title)
    ax.set_aspect('equal')



def plot_grouped_polygons_and_average(df, all_poly, all_coeff, harmonics=2):
    grouped = df.groupby(['time in hpf', 'condition'])

    for group, indices in grouped.groups.items():
        group_polygons = [all_poly[df['folder_name'].iloc[i]] for i in indices]
        group_coeffs = [all_coeff[df['folder_name'].iloc[i]] for i in indices]

        # Compute the average Fourier coefficients for the group
        avg_coeff = np.mean(group_coeffs, axis=0)
        avg_coeff = spatial_efd.AverageCoefficients(group_coeffs)
        print(group_coeffs)
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        # Plot individual polygons (centered by subtracting the centroid)
        for coeff in group_coeffs:
            x, y = spatial_efd.inverse_transform(coeff, harmonic=harmonics,n_coords=n_coords)
            ax.plot(x, y, 'r', alpha=0.3, label='Individual Polygons smooth ' if not ax.lines else "")
        
        for polygon in group_polygons:
            x, y = polygon[0, :], polygon[1, :]
            centroid = centroid_Polygon(x, y)
            x_centered = x - centroid[0]
            y_centered = y - centroid[1]
            ax.plot(x_centered, y_centered, 'g', alpha=0.3, label='Individual Polygons' if not ax.lines else "")
        
        # Inverse transform the average coefficients to reconstruct the averaged polygon
        xt, yt = spatial_efd.inverse_transform(avg_coeff, harmonic=harmonics,n_coords=n_coords)
        
        # Plot the averaged polygon (centered at the origin)
        ax.plot(xt, yt, 'b', label='Averaged Polygon')

        ax.legend()
        ax.set_title(f"Group: {group}")
        plt.show()


###new

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

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

def plot_all_reference_data(base_dir, data_to_value_function, scale_unit="µm",vmin=None,vmax=None):
    """
    Plots all reference meshes and overlays selected histogram data with uniform color scaling.

    Args:
        base_dir (str): Directory containing reference geometries.
        data_to_value_function
        scale_unit (str): Unit for the scale bar.
    """
    subfolders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

    num_meshes = len(subfolders)
    cols = min(4, num_meshes)  # Max 4 columns
    rows = (num_meshes // cols) + (num_meshes % cols > 0)  # Calculate rows

    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 7 * rows), squeeze=False)
    axes = axes.flatten()

    all_plot_values = []  # Collect all feature values for global scaling
    hist_data_map = {}  # Cache for histogram data

    # First Pass: Load histogram data once and collect global min/max values
    for folder in subfolders:
        npz_hist_file = os.path.join(base_dir, folder, "histogram_data.npz")
        
        if os.path.exists(npz_hist_file):
            hist_data = np.load(npz_hist_file, allow_pickle=True)
            hist_data_map[folder] = hist_data  # Cache the loaded data

            all_plot_values.append(data_to_value_function(hist_data))

    if not all_plot_values:
        print(f"No valid data found.")
        return

    # Compute global min/max scaling for colorbar
    all_plot_values = np.concatenate(all_plot_values)
    if vmin is None:
        vmin=np.min(all_plot_values)
        print('vmin',vmin)
    if vmax is None:
        vmax =np.max(all_plot_values)
        print('vmax',vmax)

    # Second Pass: Plot using cached data
    for idx, folder in enumerate(subfolders):
        npz_mesh_file = os.path.join(base_dir, folder, f"{folder}_ref.npz")

        if not os.path.exists(npz_mesh_file):
            print(f"Skipping {folder}, no reference mesh found.")
            continue

        # Load mesh data
        mesh_data = np.load(npz_mesh_file)
        nodes, triangles = mesh_data["nodes"], mesh_data["triangles"]

        # Retrieve cached histogram data
        hist_data = hist_data_map.get(folder, None)
        plot_values = None
        if hist_data :
            plot_values = data_to_value_function(hist_data)

        # Plot reference mesh
        ax = axes[idx]
        triang = tri.Triangulation(nodes[:, 0], nodes[:, 1], triangles)
        if plot_values is not None:
            ax.tricontourf(triang, plot_values, cmap="viridis", levels=100, vmin=vmin, vmax=vmax)
        else:
            ax.triplot(triang, color="blue", alpha=0.6)

        # Title from folder name (category keys)
        ax.set_title(folder.replace("_", " "), fontsize=12)
        ax.set_aspect("equal")
        ax.axis("off")

        # Add scale bar
        add_scale_bar(ax, nodes, scale_unit)

    # Create a single colorbar below all plots
    cbar_ax = fig.add_axes([0.15, 0.04, 0.7, 0.02])  # Position the color bar
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label('feature_key', fontsize=14)

    # Hide unused subplots if any
    for i in range(num_meshes, len(axes)):
        fig.delaxes(axes[i])

    #plt.tight_layout()
    plt.subplots_adjust(hspace=0.2)  
    plt.subplots_adjust(wspace=0.2) 
    plt.subplots_adjust(bottom=0.1) 
    plt.subplots_adjust(top=0.97) 
    plt.show()
