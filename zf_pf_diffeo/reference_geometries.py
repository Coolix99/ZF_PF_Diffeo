import os
import pandas as pd
import pyvista as pv
import numpy as np
import spatial_efd
import gmsh
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
import pickle 
from typing import List

from boundary import getBoundary


def getCoeff(polygon, harmonics):
    # Extract x and y coordinates from the polygon (2xN)
    x, y = polygon[0, :], polygon[1, :]

    # Calculate Fourier descriptors
    X,Y=spatial_efd.CloseContour(x,y)
    coeffs = spatial_efd.CalculateEFD(X, Y, harmonics=harmonics)
    
    return coeffs

def get_avg_Shapes(df, all_coeff, harmonics=2):
    grouped = df.groupby(['time in hpf', 'condition'])

    avgCoeffs={}
    avgPolygons={}

    for group, indices in grouped.groups.items():
        group_coeffs = [all_coeff[df['folder_name'].iloc[i]] for i in indices]

        # Compute the average Fourier coefficients for the group
        avg_coeff = spatial_efd.AverageCoefficients(group_coeffs)
        avgCoeffs[group] = avg_coeff
       
        # Inverse transform the average coefficients to reconstruct the averaged polygon
        xt, yt = spatial_efd.inverse_transform(avg_coeff, harmonic=harmonics,n_coords=n_coords)
        avgPolygons[group]=np.stack((xt,yt))

    return avgCoeffs,avgPolygons   


def interpolatePolygons(avgPolygons):
    def linear_interpolate(polygon1, polygon2, t1, t2, t):
        """Interpolate linearly between two polygons for a given time t."""
        factor = (t - t1) / (t2 - t1)
        interpolated_polygon = polygon1 + factor * (polygon2 - polygon1)
        return interpolated_polygon

    # Separate polygons by condition
    conditions = {'Development': {}, 'Regeneration': {}}
    
    for (time, condition), polygon in avgPolygons.items():
        conditions[condition][time] = polygon

    # Interpolate polygons for both conditions
    for condition, poly_dict in conditions.items():
        times = sorted(poly_dict.keys())  # Sort time steps
        min_time, max_time = times[0], times[-1]
        
        # Create new polygons for missing times
        for t in range(min_time, max_time):
            if t not in poly_dict:
                # Find the closest surrounding polygons for interpolation
                t1 = max([ti for ti in times if ti <= t])
                t2 = min([ti for ti in times if ti > t])
                
                polygon1 = poly_dict[t1]
                polygon2 = poly_dict[t2]
                
                # Interpolate between polygon1 and polygon2 for time t
                interpolated_polygon = linear_interpolate(polygon1, polygon2, t1, t2, t)
                
                # Add the interpolated polygon to the dictionary
                avgPolygons[(t, condition)] = interpolated_polygon

    # Ensure the avgPolygons dictionary now has entries for all time steps
    return avgPolygons



def get_boundary_indices(nodes, boundary_points):
    """
    Find the indices in the `nodes` array that correspond to the `boundary_points`.
    This uses a proximity check to account for floating-point differences.
    """
    boundary_indices = []
    for boundary_point in boundary_points:
        # Find the index of the node that is closest to the boundary point
        distances = np.linalg.norm(nodes - boundary_point, axis=1)
        closest_index = np.argmin(distances)
        boundary_indices.append(closest_index)
    
    return np.array(boundary_indices)

def get_last_time_meshes(avgPolygons, mesh_size=0.1):
    # Find the maximum time for each condition
    last_dev_time = max([time for (time, condition) in avgPolygons if condition == 'Development'])
    last_reg_time = max([time for (time, condition) in avgPolygons if condition == 'Regeneration'])
    
    # Extract the corresponding polygons
    dev_polygon = avgPolygons[(last_dev_time, 'Development')]
    reg_polygon = avgPolygons[(last_reg_time, 'Regeneration')]
    
    # Generate meshes for both polygons
    dev_nodes, dev_triangles = generate_2d_mesh(dev_polygon.T, mesh_size)
    reg_nodes, reg_triangles = generate_2d_mesh(reg_polygon.T, mesh_size)
    
    # Find the indices of the original boundary points in the new nodes array
    dev_boundary_indices = get_boundary_indices(dev_nodes, dev_polygon.T)
    reg_boundary_indices = get_boundary_indices(reg_nodes, reg_polygon.T)
    
    # Return meshes and boundary indices
    return (dev_nodes, dev_triangles, dev_boundary_indices), (reg_nodes, reg_triangles, reg_boundary_indices)


def compute_element_stiffness_matrix(vertices):
    # Compute the area of the triangle (vertices are in counterclockwise order)
    x1, y1 = vertices[0]
    x2, y2 = vertices[1]
    x3, y3 = vertices[2]
    
    area = 0.5 * np.abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

    # Compute the gradients of the linear basis functions
    b1, c1 = y2 - y3, x3 - x2
    b2, c2 = y3 - y1, x1 - x3
    b3, c3 = y1 - y2, x2 - x1

    B = np.array([[b1, b2, b3], [c1, c2, c3]])

    # Local stiffness matrix
    local_K = (B.T @ B) / (4 * area)

    return local_K

def assemble_global_stiffness_matrix(nodes, triangles):
    num_nodes = len(nodes)
    K = sp.lil_matrix((num_nodes, num_nodes))  # Initialize global stiffness matrix

    for triangle in triangles:
        # Extract the vertices of the triangle
        vertices = nodes[triangle]

        # Compute the local stiffness matrix for this element
        local_K = compute_element_stiffness_matrix(vertices)

        # Assemble the local stiffness matrix into the global matrix
        for i, vi in enumerate(triangle):
            for j, vj in enumerate(triangle):
                K[vi, vj] += local_K[i, j]

    return K.tocsr()

def solve_laplace(nodes, triangles, boundary_indices, boundary_displacement):
    """
    Solves the Laplace equation using the finite element method to propagate
    boundary displacements to interior nodes.
    """
    num_nodes = len(nodes)

    # Assemble the global stiffness matrix
    K = assemble_global_stiffness_matrix(nodes, triangles)

    # Initialize the displacement vector for all nodes
    displacement = np.zeros((num_nodes, 2))

    # Apply boundary displacements
    displacement[boundary_indices] = boundary_displacement

    # Separate free nodes (interior) from boundary nodes
    free_indices = np.setdiff1d(np.arange(num_nodes), boundary_indices)

    # Solve for displacement at interior nodes using Laplace equation
    for dim in range(2):  # x and y dimensions
        # Construct the right-hand side (forcing) vector
        rhs = -K[free_indices, :][:, boundary_indices] @ displacement[boundary_indices, dim]
        
        # Solve for interior displacements
        K_free = K[free_indices, :][:, free_indices]  # Submatrix for free nodes
        displacement[free_indices, dim] = spsolve(K_free, rhs)

    return displacement


def interpolate_mesh_over_time(avgPolygons, last_time_meshes):
    # Initialize the interpolated nodes dictionary for both conditions
    interpolated_nodes = {}

    # Define the conditions
    conditions = ['Development', 'Regeneration']

    # Loop over each condition (Development and Regeneration)
    for condition in conditions:
        # Extract the corresponding nodes, triangles, and boundary indices
        nodes, triangles, boundary_indices = last_time_meshes[condition]

        # Get the time steps for the current condition
        times = sorted([time for (time, cond) in avgPolygons if cond == condition])

        # Initialize the interpolation with the last time step
        interpolated_nodes[(times[-1], condition)] = nodes.copy()

        # Step backward from the last time to the first, computing displacement at each step
        for t_idx in range(len(times) - 2, -1, -1):
            current_time = times[t_idx]

            # Get the corresponding polygon for the current time step
            polygon = avgPolygons[(current_time, condition)].T

            # Compute the displacement for the boundary nodes
            boundary_displacement = polygon - nodes[boundary_indices]

            # Solve Laplace equation to propagate the displacement to the interior nodes
            displacement = solve_laplace(nodes, triangles, boundary_indices, boundary_displacement)

            # Apply the displacement to the nodes
            nodes += displacement

            # Store the interpolated node positions
            interpolated_nodes[(current_time, condition)] = nodes.copy()

    return interpolated_nodes


def get_mesh_lims(interpolated_dev_nodes, interpolated_reg_nodes):
    min_val, max_val = float('inf'), float('-inf')  # Initialize with extreme values
    
    # Iterate through all time steps for development nodes
    for nodes in interpolated_dev_nodes.values():
        x_values = nodes[:, 0]  # Extract x-coordinates
        y_values = nodes[:, 1]  # Extract y-coordinates
        
        # Find the min and max of both x and y
        min_val = min(min_val, np.min(x_values), np.min(y_values))
        max_val = max(max_val, np.max(x_values), np.max(y_values))
    
    # Iterate through all time steps for regeneration nodes
    for nodes in interpolated_reg_nodes.values():
        x_values = nodes[:, 0]  # Extract x-coordinates
        y_values = nodes[:, 1]  # Extract y-coordinates
        
        # Find the min and max of both x and y
        min_val = min(min_val, np.min(x_values), np.min(y_values))
        max_val = max(max_val, np.max(x_values), np.max(y_values))
    
    return min_val, max_val



def save_interpolated_data(interpolated_nodes, triangles,boundaries, AvgShape_path):
    # Ensure the directory exists
    if not os.path.exists(AvgShape_path):
        os.makedirs(AvgShape_path)
    
    # Define file paths
    nodes_file = os.path.join(AvgShape_path, "interpolated_nodes.pkl")
    triangles_file = os.path.join(AvgShape_path, "triangles.pkl")
    boundaries_file = os.path.join(AvgShape_path, "boundaries.pkl")

    # Save interpolated_nodes
    with open(nodes_file, 'wb') as f:
        pickle.dump(interpolated_nodes, f)
    
    # Save triangles
    with open(triangles_file, 'wb') as f:
        pickle.dump(triangles, f)

    # Save triangles
    with open(boundaries_file, 'wb') as f:
        pickle.dump(boundaries, f)

    print(f"Data saved successfully to {AvgShape_path}")

def load_interpolated_data(AvgShape_path):
    # Define file paths
    nodes_file = os.path.join(AvgShape_path, "interpolated_nodes.pkl")
    triangles_file = os.path.join(AvgShape_path, "triangles.pkl")
    boundaries_file = os.path.join(AvgShape_path, "boundaries.pkl")

    # Load interpolated_nodes
    with open(nodes_file, 'rb') as f:
        interpolated_nodes = pickle.load(f)
    
    # Load triangles
    with open(triangles_file, 'rb') as f:
        triangles = pickle.load(f)

    # Load triangles
    with open(boundaries_file, 'rb') as f:
        boundaries = pickle.load(f)


    print(f"Data loaded successfully from {AvgShape_path}")
    
    return interpolated_nodes, triangles,boundaries

def create_mesh_series():
    FlatFin_folder_list = os.listdir(FlatFin_path)
    FlatFin_folder_list = [item for item in FlatFin_folder_list if os.path.isdir(os.path.join(FlatFin_path, item))]
    
    data_list = []
    all_coeff = {}
    all_poly = {}
    
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
        
        surface_file_name = MetaData['Surface file']
        mesh_3d = pv.read(os.path.join(FlatFin_dir_path, surface_file_name))
        polygon = getPolygon(mesh_3d)
        coeff = getCoeff(polygon, n_harmonics)


        # x, y = polygon[0, :], polygon[1, :]
        # centroid = centroid_Polygon(x, y)
        # x_centered = x - centroid[0]
        # y_centered = y - centroid[1]
        #plt.plot(x_centered, y_centered, 'g', alpha=0.3, label='Individual Polygons')
        #xt, yt = spatial_efd.inverse_transform(coeff, harmonic=n_harmonics)
        #plt.plot(xt, yt, 'b', label='smooth Polygon')
        #print(coeff)
        x0=np.sum(coeff[:,2])
        y0=np.sum(coeff[:,0])
        #print(np.sum(coeff,axis=0))
        #plt.scatter((x0,x0), (y0,y0), label='start')
        theta=np.arctan2(y0, x0)
        coeff_shift=shift_coeff(coeff,-theta)
        #xt, yt = spatial_efd.inverse_transform(coeff_shift, harmonic=n_harmonics)
        #plt.plot(xt, yt, 'g', label='shift Polygon')
        # x0=np.sum(coeff_shift[:,2])
        # y0=np.sum(coeff_shift[:,0])
        #print(np.sum(coeff_shift,axis=0))
        #plt.scatter((x0,x0), (y0,y0), label='start')
       
        #plt.show()
        

        all_coeff[FlatFin_folder] = coeff_shift
        all_poly[FlatFin_folder] = polygon

    # Store the metadata in a pandas DataFrame
    df = pd.DataFrame(data_list)
    
    # Plot grouped polygons and the averaged Fourier descriptors
    #plot_grouped_polygons_and_average(df, all_poly, all_coeff,harmonics=n_harmonics)
    avgCoeffs,avgPolygons=get_avg_Shapes(df,all_coeff,n_harmonics)
    #plot_time_evolution(avgPolygons)
    interpolatePolygons(avgPolygons)
    #plot_time_evolution(avgPolygons)
    #movie_time_evolution(avgPolygons)

    (dev_nodes, dev_triangles, dev_boundary_indices), (reg_nodes, reg_triangles, reg_boundary_indices) = get_last_time_meshes(avgPolygons, mesh_size=1000.0)
    last_time_meshes = {
        'Development': (dev_nodes, dev_triangles, dev_boundary_indices),
        'Regeneration': (reg_nodes, reg_triangles, reg_boundary_indices)
    }
    triangles = {
        'Development': dev_triangles,
        'Regeneration': reg_triangles
    }
    boundaries = {
        'Development': dev_boundary_indices,
        'Regeneration': reg_boundary_indices
    }
    interpolated_nodes=interpolate_mesh_over_time(avgPolygons, last_time_meshes)
    #save_interpolated_data(interpolated_nodes, triangles,boundaries, AvgShape_path)
    

def filter_outliers(data, n_std=3):
    mean = np.mean(data)
    std = np.std(data)
    filtered = np.where(np.abs(data - mean) > n_std * std, np.nan, data)
    return filtered

def transfer_data(nodes, triangle,boundary, mesh:pv.PolyData):
    #print(mesh.point_data)
    polygon = getPolygon(mesh)
    centroid = centroid_Polygon(polygon[0, :], polygon[1, :])
    
    coeff = getCoeff(polygon, n_harmonics)
    x0=np.sum(coeff[:,2])
    y0=np.sum(coeff[:,0])

    theta=np.arctan2(y0, x0)
    coeff_shift=shift_coeff(coeff,-theta)
    xt, yt = spatial_efd.inverse_transform(coeff_shift, harmonic=n_harmonics,n_coords=n_coords)


    # Compute the displacement for the boundary nodes
    boundary_displacement = np.stack((xt, yt)).T - nodes[boundary]
    # Solve Laplace equation to propagate the displacement to the interior nodes
    displacement = solve_laplace(nodes, triangle, boundary, boundary_displacement)
    # Apply the displacement to the nodes
    nodes_displaced =nodes.copy() + displacement + centroid


    # fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    # ax.triplot(nodes[:, 0], nodes[:, 1], triangle, color='blue', label='Mesh')
    # ax.triplot(nodes_displaced[:, 0], nodes_displaced[:, 1], triangle, color='red', label='Mesh displaced')
    # #ax.plot(polygon[0, :], polygon[1, :], 'g-', label='Polygon', lw=2)
    # #ax.plot(xt, yt, 'r-', label='Reconstructed Shape', lw=2)
    # ax.triplot(mesh.point_data['coord_1'], mesh.point_data['coord_2'], mesh.faces.reshape(-1, 4)[:, 1:], color='green', label='Hist mesh')
    # ax.legend()
    # ax.set_aspect('equal')
    # plt.show()

    mesh_coords = np.stack((mesh.point_data['coord_1'], mesh.point_data['coord_2']), axis=1)
    kdtree = cKDTree(mesh_coords)

    # Find the nearest point in the original mesh for each displaced node
    distances, indices = kdtree.query(nodes_displaced)

    # Step 5: Create the data dictionary
    data_dict = {}
    for key in ['mean_curvature_avg', 'mean_curvature_std', 'gauss_curvature_avg', 'gauss_curvature_std', 'thickness_avg', 'thickness_std', 'mean2-gauss']:
        # Extract the point data for the given key
        raw_data = np.array(mesh.point_data[key])

        # Filter out outliers in the raw data
        filtered_data = filter_outliers(raw_data, n_std=4)

        # Create a mask for non-outlier points
        valid_indices = ~np.isnan(filtered_data)  # Get valid (non-outlier) indices

        # Create mesh coordinates only for non-outlier points
        valid_coords_1 = np.array(mesh.point_data['coord_1'])[valid_indices]
        valid_coords_2 = np.array(mesh.point_data['coord_2'])[valid_indices]
        mesh_coords = np.stack((valid_coords_1, valid_coords_2), axis=1)

        # Ensure we have valid data points for KDTree
        if len(mesh_coords) == 0:
            raise ValueError(f"No valid points left for mesh after filtering outliers for key: {key}")

        # Create KDTree using the valid coordinates
        kdtree = cKDTree(mesh_coords)

        # Find the nearest point in the original mesh for each displaced node
        distances, indices = kdtree.query(nodes_displaced)

        # Map the found indices back to the original filtered data
        data_dict[key] = np.array(filtered_data[valid_indices])[indices]

    # Return the data dictionary with outliers filtered out
    return data_dict

def linear_interpolate_data(data_dict):
    # Step 1: Organize the data by category
    categories = set(category for _, category in data_dict.keys())
    
    # Step 2: Loop over each category
    for category in categories:
        # Extract all times for the current category
        category_times = sorted([time for time, cat in data_dict.keys() if cat == category])

        # Find the full range of times (smallest to largest)
        min_time = min(category_times)
        max_time = max(category_times)
        
        # Generate the full list of times that should be present
        full_times = list(range(min_time, max_time + 1))

        # Identify missing times
        missing_times = [time for time in full_times if (time, category) not in data_dict]

        # Step 3: Interpolate for the missing times
        for missing_time in missing_times:
            # Find the nearest previous and next times that have data
            previous_time = max([t for t in category_times if t < missing_time], default=None)
            next_time = min([t for t in category_times if t > missing_time], default=None)

            if previous_time is not None and next_time is not None:
                # Get the data for the previous and next times
                data_prev = data_dict[(previous_time, category)]
                data_next = data_dict[(next_time, category)]

                # Interpolate each data field
                interpolated_data = {}
                for key in data_prev.keys():
                    value_prev = data_prev[key]
                    value_next = data_next[key]

                    # Linear interpolation for each field
                    interpolated_value = value_prev + (value_next - value_prev) * (
                        (missing_time - previous_time) / (next_time - previous_time)
                    )

                    interpolated_data[key] = interpolated_value

                # Store the interpolated data in the dictionary
                data_dict[(missing_time, category)] = interpolated_data

            else:
                print(f"Warning: Cannot interpolate data for time {missing_time} in category {category}")

    return data_dict

def save_data_dict(data_dict, save_path, file_name="data_dict.pkl"):
    # Ensure the directory exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Define the full file path
    file_path = os.path.join(save_path, file_name)

    # Save the data_dict using pickle
    with open(file_path, 'wb') as f:
        pickle.dump(data_dict, f)

    print(f"Data dictionary saved successfully to {file_path}")

def load_data_dict(save_path, file_name="data_dict.pkl"):
    # Define the full file path
    file_path = os.path.join(save_path, file_name)

    # Load the data_dict using pickle
    with open(file_path, 'rb') as f:
        data_dict = pickle.load(f)

    print(f"Data dictionary loaded successfully from {file_path}")
    return data_dict


def create_data_series():
    interpolated_nodes, triangles,boundaries=load_interpolated_data(AvgShape_path)
    times, categories, meshs=getData()
    
    data_dict = {}
    # Loop through each mesh, time, and category
    for i, (time, category, mesh) in enumerate(zip(times, categories, meshs)):
        # Find the corresponding node and triangle data
        nodes = interpolated_nodes.get((time, category))
        triangle = triangles.get(category)
        boundary = boundaries.get(category)

        # Check if nodes and triangle exist for this time and category
        if nodes is not None and triangle is not None:
            # Call the transfer_data function
            data = transfer_data(nodes, triangle,boundary, mesh)
            
            # Store the resulting data dict
            data_dict[(time, category)] = data


        else:
            print(f"Warning: No interpolation found for time {time} and category {category}")

    print(len(data_dict))

    #debug_mesh_time_evolution(interpolated_nodes, triangles, data_dict, 'gauss_curvature_avg',vlim=(-2e-4,2e-4))
    linear_interpolate_data(data_dict)
    print(len(data_dict))
    save_data_dict(data_dict,AvgShape_path)

##########new


#n_harmonics=10

def getPolygon(mesh_3d: pv.PolyData):
    boundary_indices = getBoundary(mesh_3d)
    
    bc1 = mesh_3d.point_data['coord_1'][boundary_indices]
    bc2 = mesh_3d.point_data['coord_2'][boundary_indices]
    
    centroid = centroid_Polygon(bc1, bc2)
    b = np.stack((bc1, bc2))

    cross_prod = (bc1[0] - centroid[0]) * (bc2[1] - centroid[1]) - (bc2[0] - centroid[1]) * (bc1[1] - centroid[0])
    if cross_prod <0:
        b=b[:,::-1]
    
    return b

def centroid_Polygon(x, y):
    # Ensure that the polygon is closed by appending the first point to the end
    x = np.append(x, x[0])
    y = np.append(y, y[0])

    # Calculate the signed area (A) of the polygon
    A = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])

    # Calculate the centroid coordinates
    C_x = (1 / (6 * A)) * np.sum((x[:-1] + x[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1]))
    C_y = (1 / (6 * A)) * np.sum((y[:-1] + y[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1]))

    return np.array([C_x, C_y])

def shift_coeff(coeff, theta):
    coeff_shifted = coeff.copy()  # Create a copy to avoid modifying the original coefficients
    num_orders = coeff.shape[0]   # Assuming coeff has shape (n_harmonics, 4) for 2x2 matrices
    
    # Loop through each order n
    for n in range(0, num_orders):  
        # Get the phase shift matrix for order n
        cos_ntheta = np.cos((n+1)* theta)
        sin_ntheta = np.sin((n+1)* theta)
        phase_shift_matrix = np.array([[cos_ntheta, sin_ntheta],
                                       [-sin_ntheta, cos_ntheta]])
        
        # Reshape the coefficients to a 2x2 matrix
        coeff_matrix = coeff[n].reshape(2, 2).T
        
        # Apply the phase shift matrix
        coeff_shifted[n] = (phase_shift_matrix @ coeff_matrix).T.flatten()  # Flatten back to a 1D array
    
    return coeff_shifted

def create_2dGeometry(mesh_3d: pv.PolyData,n_harmonics):
    polygon = getPolygon(mesh_3d)
    coeff = getCoeff(polygon, n_harmonics)

    # x, y = polygon[0, :], polygon[1, :]
    # centroid = centroid_Polygon(x, y)
    # x_centered = x - centroid[0]
    # y_centered = y - centroid[1]
    #plt.plot(x_centered, y_centered, 'g', alpha=0.3, label='Individual Polygons')
    #xt, yt = spatial_efd.inverse_transform(coeff, harmonic=n_harmonics)
    #plt.plot(xt, yt, 'b', label='smooth Polygon')
    #print(coeff)
    x0=np.sum(coeff[:,2])
    y0=np.sum(coeff[:,0])
    #print(np.sum(coeff,axis=0))
    #plt.scatter((x0,x0), (y0,y0), label='start')
    theta=np.arctan2(y0, x0)
    coeff_shift=shift_coeff(coeff,-theta)

    return coeff_shift

def generate_2d_mesh(boundary_points, mesh_size=0.1):
    # Initialize Gmsh
    gmsh.initialize()

    # Create a new Gmsh model
    gmsh.model.add("2D Mesh")

    # Define boundary points as Gmsh points
    point_tags = []
    for i, (x, y) in enumerate(boundary_points):
        point_tag = gmsh.model.geo.addPoint(x, y, 0, mesh_size)
        point_tags.append(point_tag)
    
    # Create boundary lines from points and close the loop
    line_tags = []
    num_points = len(point_tags)
    for i in range(num_points):
        # Create a line between consecutive points
        line_tag = gmsh.model.geo.addLine(point_tags[i], point_tags[(i + 1) % num_points])
        line_tags.append(line_tag)

    # Create a curve loop from the boundary lines
    curve_loop = gmsh.model.geo.addCurveLoop(line_tags)

    # Create a plane surface bounded by the curve loop
    surface = gmsh.model.geo.addPlaneSurface([curve_loop])

    # Synchronize the CAD representation with the Gmsh model
    gmsh.model.geo.synchronize()

    # Generate the 2D mesh
    gmsh.model.mesh.generate(2)

    # Retrieve mesh data (vertices and triangles)
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    nodes = np.array(node_coords).reshape(-1, 3)[:, :2]  # Extract x, y coordinates only

    element_types, element_tags, element_node_tags = gmsh.model.mesh.getElements(2)
    triangles = np.array(element_node_tags[0]).reshape(-1, 3) - 1  # Convert to 0-based indexing

    # Finalize the gmsh session
    gmsh.finalize()

    return nodes, triangles



def create_refGeometry(meshes3d:List[pv.PolyData],n_harmonics=10,n_coords=40):
    all_coeff=[]
    for mesh in meshes3d:
        coeff=create_2dGeometry(mesh,n_harmonics)
        all_coeff.append(coeff)

    avg_coeff = spatial_efd.AverageCoefficients(all_coeff)
    xt, yt = spatial_efd.inverse_transform(avg_coeff, harmonic=n_harmonics,n_coords=n_coords)
    avgPolygon=np.stack((xt,yt))

    avg_mesh=generate_2d_mesh(avgPolygon.T)

    return avg_coeff,avg_mesh  

