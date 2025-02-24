import pyvista as pv
import numpy as np
import gmsh
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
from typing import List
from scipy.interpolate import interp1d
import logging
from spatial_efd import AverageCoefficients, inverse_transform,CloseContour,CalculateEFD

from zf_pf_diffeo.boundary import getBoundary


logger = logging.getLogger(__name__)


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

def getCoeff(polygon, harmonics):
    # Extract x and y coordinates from the polygon (2xN)
    x, y = polygon[0, :], polygon[1, :]

    # Calculate Fourier descriptors
    X, Y = CloseContour(x, y)
    coeffs = CalculateEFD(X, Y, harmonics=harmonics)

   

    return coeffs

def remove_duplicates_preserve_order(boundary_indices):
    """Removes duplicates from a 1D NumPy array while preserving order."""
    boundary_indices = np.asarray(boundary_indices)  # Ensure it's a NumPy array

    unique_values, idx = np.unique(boundary_indices, return_index=True)

    # Ensure idx is always an array, and sort it to preserve order
    if idx.ndim == 0:  # If idx is a scalar, wrap it in an array
        idx = np.array([idx])

    return boundary_indices[np.sort(idx)]  # Extract unique values in order

def getPolygon(mesh_3d: pv.PolyData,debug=False):
    """
    Extracts the boundary polygon from a 3D mesh.

    Args:
        mesh_3d (pv.PolyData): Input 3D mesh.

    Returns:
        np.ndarray: Extracted 2D polygon as an array of shape (2, N).
    """

    boundary_indices = getBoundary(mesh_3d)
    unique_boundary_indices = remove_duplicates_preserve_order(boundary_indices)

    bc1 = mesh_3d.point_data['coord_1'][unique_boundary_indices]
    bc2 = mesh_3d.point_data['coord_2'][unique_boundary_indices]
    
    centroid = centroid_Polygon(bc1, bc2)
    polygon = np.stack((bc1, bc2))


    cross_prod = (bc1[0] - centroid[0]) * (bc2[1] - centroid[1]) - (bc2[0] - centroid[1]) * (bc1[1] - centroid[0])
    if cross_prod < 0:
        polygon = polygon[:, ::-1]  # Reverse order if necessary

    logger.debug(f"Extracted polygon with {polygon.shape[1]} points.")
    return polygon

def centroid_Polygon(x, y):
    """
    Computes the centroid of a closed polygon.

    Args:
        x (np.ndarray): X-coordinates of polygon vertices.
        y (np.ndarray): Y-coordinates of polygon vertices.

    Returns:
        np.ndarray: Centroid (Cx, Cy) of the polygon.
    """
    x = np.append(x, x[0])
    y = np.append(y, y[0])

    A = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])

    if A == 0:
        logger.warning("Polygon area is zero, centroid calculation may be invalid.")
        return np.array([np.mean(x), np.mean(y)])

    C_x = (1 / (6 * A)) * np.sum((x[:-1] + x[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1]))
    C_y = (1 / (6 * A)) * np.sum((y[:-1] + y[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1]))

    return np.array([C_x, C_y])

def shift_coeff(coeff, theta):
    """
    Rotates Fourier coefficients by an angle theta.

    Args:
        coeff (np.ndarray): Fourier coefficients of shape (n_harmonics, 4).
        theta (float): Rotation angle in radians.

    Returns:
        np.ndarray: Rotated Fourier coefficients.
    """
    coeff_shifted = coeff.copy()
    num_orders = coeff.shape[0]

    for n in range(num_orders):
        cos_ntheta = np.cos((n+1) * theta)
        sin_ntheta = np.sin((n+1) * theta)
        phase_shift_matrix = np.array([[cos_ntheta, sin_ntheta], [-sin_ntheta, cos_ntheta]])

        coeff_matrix = coeff[n].reshape(2, 2).T
        coeff_shifted[n] = (phase_shift_matrix @ coeff_matrix).T.flatten()

    logger.debug(f"Shifted Fourier coefficients by {theta:.4f} radians.")
    return coeff_shifted

def create_2dGeometry(mesh_3d: pv.PolyData, n_harmonics: int):
    """
    Computes Fourier coefficients for a 2D representation of a 3D mesh boundary.

    Args:
        mesh_3d (pv.PolyData): Input 3D mesh.
        n_harmonics (int): Number of Fourier harmonics to use.

    Returns:
        np.ndarray: Shifted Fourier coefficients.
    """
    polygon = getPolygon(mesh_3d)
    coeff = getCoeff(polygon, n_harmonics)
    


    x0, y0 = np.sum(coeff[:, 2]), np.sum(coeff[:, 0])
    theta = np.arctan2(y0, x0)
    coeff_shift = shift_coeff(coeff, -theta)
    logger.info(f"Computed 2D Fourier representation for mesh with {polygon.shape[1]} boundary points.")
    return coeff_shift

def generate_2d_mesh(boundary_points: np.ndarray, mesh_size=100):
    """
    Generates a 2D triangular mesh from boundary points.

    Args:
        boundary_points (np.ndarray): Shape (N, 2), coordinates of boundary points.
        mesh_size (float): Desired mesh element size.

    Returns:
        tuple: (nodes, triangles), where:
            - nodes (np.ndarray): Shape (M, 2), mesh node coordinates.
            - triangles (np.ndarray): Shape (K, 3), triangle connectivity.
    """
    gmsh.initialize()
    gmsh.model.add("2D Mesh")
   
    point_tags = [gmsh.model.geo.addPoint(x, y, 0, mesh_size) for x, y in boundary_points[:-1,:]]
    line_tags = [gmsh.model.geo.addLine(point_tags[i], point_tags[(i+1) % len(point_tags)]) for i in range(len(point_tags))]
   
    curve_loop = gmsh.model.geo.addCurveLoop(line_tags)
    gmsh.model.geo.addPlaneSurface([curve_loop])

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    nodes = np.array(node_coords).reshape(-1, 3)[:, :2]

    element_types, element_tags, element_node_tags = gmsh.model.mesh.getElements(2)
    triangles = np.array(element_node_tags[0]).reshape(-1, 3) - 1

    gmsh.finalize()

    logger.info(f"Generated 2D mesh with {len(nodes)} nodes and {len(triangles)} triangles.")
    return nodes, triangles

def create_refGeometry(meshes3d: List[pv.PolyData], n_harmonics=10, n_coords=40):
    """
    Computes a representative 2D geometry from a set of 3D meshes.

    Args:
        meshes3d (List[pv.PolyData]): List of 3D meshes.
        n_harmonics (int): Number of Fourier harmonics for shape representation.
        n_coords (int): Number of coordinates for inverse transformation.

    Returns:
        tuple: (avg_coeff, (nodes, triangles, boundary_indices))
            - avg_coeff (np.ndarray): Averaged Fourier coefficients.
            - nodes (ndarray): Nodes of the averaged 2D mesh.
            - triangles (ndarray): Triangular connectivity.
            - boundary_indices (ndarray): Indices of boundary nodes.
    """
    all_coeff = [create_2dGeometry(mesh, n_harmonics) for mesh in meshes3d]
    avg_coeff = AverageCoefficients(all_coeff)
    xt, yt = inverse_transform(avg_coeff, harmonic=n_harmonics, n_coords=n_coords)
    avgPolygon = np.stack((xt, yt))

    nodes, triangles = generate_2d_mesh(avgPolygon.T)
    boundary_indices = get_boundary_indices(nodes, avgPolygon.T)  # Extract boundary indices

    logger.info(f"Created reference geometry using {len(meshes3d)} meshes.")
    return avg_coeff, (nodes, triangles, boundary_indices)

def interpolate_coefficients(ref_coeffs):
    """
    Interpolates Fourier coefficients over time.

    Args:
        ref_coeffs (dict): Dictionary mapping time -> Fourier coefficients.

    Returns:
        dict: Interpolated coefficients for missing time steps.
    """
    times = np.array(sorted(ref_coeffs.keys()))
    coeff_array = np.stack([ref_coeffs[t] for t in times])

    interpolated_coeffs = {}
    
    for i in range(coeff_array.shape[1]):
        for j in range(coeff_array.shape[2]):
            interp_func = interp1d(times, coeff_array[:, i, j], kind="linear", fill_value="extrapolate")
            #for t in times:
            for t in range(min(times), max(times) + 1):
                if t not in interpolated_coeffs:
                    interpolated_coeffs[t] = np.zeros_like(coeff_array[0])
                interpolated_coeffs[t][i, j] = interp_func(t)

    return interpolated_coeffs

def propagate_mesh_backward(last_mesh, interpolated_coeffs):
    """
    Propagates the last time step's mesh backward using Laplacian deformation.

    Args:
        last_mesh (tuple): (nodes, triangles) of the last time mesh.
        interpolated_coeffs (dict): Interpolated Fourier coefficients over time.
    Returns:
        dict: Interpolated node positions for each time step.
    """
    nodes, triangles,boundary_indices = last_mesh
    times=np.array(list(interpolated_coeffs.keys()))
    interpolated_nodes = {times[-1]: nodes.copy()}
    
    for t_idx in range(len(times) - 2, -1, -1):
        current_time = times[t_idx]
        
        # Compute displacement based on Fourier coefficient changes
        xt, yt = inverse_transform(interpolated_coeffs[current_time], harmonic=10, n_coords=40)
        target_polygon = np.stack((xt, yt)).T
        
        boundary_displacement = target_polygon - nodes[boundary_indices]
        
        # Solve Laplacian equation to propagate deformation
        displacement = solve_laplace(nodes, triangles, boundary_indices, boundary_displacement)
        nodes += displacement

        interpolated_nodes[current_time] = nodes.copy()
    
    return interpolated_nodes

def create_TempRefGeometry(ref_coeffs,times):
    # Interpolate Fourier coefficients over time (moved to reference_geometries.py)
    interpolated_coeffs = interpolate_coefficients(ref_coeffs)

    # Generate last time step's mesh
    last_time = max(times)
    last_coeff = interpolated_coeffs[last_time]
    xt, yt = inverse_transform(last_coeff, harmonic=10, n_coords=40)
    nodes, triangles = generate_2d_mesh(np.stack((xt, yt)).T, mesh_size=100)
    avgPolygon = np.stack((xt, yt))
    boundary_indices = get_boundary_indices(nodes, avgPolygon.T)  # Extract boundary indices

    last_mesh = nodes, triangles,boundary_indices
    # Propagate backward through time (moved to reference_geometries.py)
    interpolated_nodes = propagate_mesh_backward(last_mesh, interpolated_coeffs)
    return interpolated_nodes,last_mesh

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

def transfer_data_to_reference(nodes, triangles,boundaries, mesh_3d:pv.PolyData,n_harmonics=10,n_coords=40):
    """
    Transfers point data from a deformed 3D surface to the reference 2D shape.

    Args:
        nodes (ndarray): 2D reference nodes.
        triangles (ndarray): Triangulation of the reference mesh.
        boundary (ndarray): Indices of the boundary nodes.
        mesh_3d (pv.PolyData): The original 3D mesh.

    Returns:
        dict: A dictionary mapping feature names to transported data.
    """
    # Extract boundary polygon from 3D mesh
    polygon = getPolygon(mesh_3d)
    centroid = centroid_Polygon(polygon[0, :], polygon[1, :])
    
    coeff = getCoeff(polygon, n_harmonics)
    x0=np.sum(coeff[:,2])
    y0=np.sum(coeff[:,0])

    theta=np.arctan2(y0, x0)
    coeff_shift=shift_coeff(coeff,-theta)
    xt, yt = inverse_transform(coeff_shift, harmonic=n_harmonics,n_coords=n_coords)


    # Compute the displacement for the boundary nodes
    boundary_displacement = np.stack((xt, yt)).T - nodes[boundaries]
    # Solve Laplace equation to propagate the displacement to the interior nodes
    displacement = solve_laplace(nodes, triangles, boundaries, boundary_displacement)
    # Apply the displacement to the nodes
    nodes_displaced =nodes.copy() + displacement + centroid

    # Build KDTree for nearest neighbor search
    mesh_coords = np.column_stack((mesh_3d.point_data["coord_1"], mesh_3d.point_data["coord_2"]))

  
    kdtree = cKDTree(nodes_displaced)

    # Find nearest neighbors
    distances, indices = kdtree.query(mesh_coords)
    return indices
    
