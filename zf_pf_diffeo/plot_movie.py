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

